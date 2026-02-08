# GPU Setup: PrimeIntellect.ai

Migration checklist for running MoE Emergence training on PrimeIntellect.

## Why PrimeIntellect

- RTX 4090 at $0.32/hr — $80 budget gets ~250 GPU-hours (far more than needed)
- 15 free hours of RTX compute for new signups
- Persistent storage survives instance termination (checkpoints safe)
- SSH access, pre-built PyTorch containers, no lock-in
- CLI tool (`prime`) built on `uv`

## GPU Choice

RTX 4090 (24GB VRAM, $0.32/hr) is the right pick. GPT-2 small + 8 experts is well
under 24GB. A100 ($0.79/hr) is overkill for this project's scale.

Verify current rates in the PrimeIntellect dashboard before booking - pricing and
free-credit promos can change.

## Budget Math

| Run               | Preset    | Max Steps         | Est. Time | Est. Cost |
| ----------------- | --------- | ----------------- | --------- | --------- |
| Dense shakedown   | shakedown | 100               | ~2 min    | ~$0.01    |
| MoE shakedown     | shakedown | 100               | ~3 min    | ~$0.02    |
| Dense baseline    | dense     | 5000              | ~1.5 hr   | ~$0.48    |
| MoE main          | moe-main  | 10000             | ~4 hr     | ~$1.28    |
| No-LB ablation    | no-lb     | 2000 (early-stop) | ~1 hr     | ~$0.32    |
| Top-2 directional | top2      | 3000              | ~1.5 hr   | ~$0.48    |

**Total estimate: ~$2.60** (plus idle/setup time). Well within budget even with
generous overhead. Times are rough — actual throughput depends on data loading.

---

## Pre-Flight (Local)

- [ ] Create PrimeIntellect account at https://www.primeintellect.ai
- [ ] Claim free 15h RTX compute credit
- [ ] Add billing / payment method
- [ ] Generate SSH key if needed (`ssh-keygen -t ed25519`)
- [ ] Register public key in PrimeIntellect profile settings
- [ ] Get W&B API key from https://wandb.ai/authorize (will need it on the instance)

## Install CLI (Local)

```bash
uv tool install prime
prime config set-api-key        # interactive, keeps key out of shell history
prime config set-ssh-key-path   # point to your private key
prime config view               # verify
```

## Create Persistent Storage

Persistent storage keeps checkpoints, datasets, and logs across instance restarts.

1. Go to **Instances > Storage** tab in the dashboard
2. Click **Create Disk**
3. Pick a provider + datacenter (remember this — instances must match)
4. Set size: **100 GB** (MoE snapshots are ~1GB each and accumulate across runs since
   only `.pt` files are auto-pruned; total artifacts across all runs are ~82GB class)
5. Wait for status to show **Active**

After shakedown runs pass the gate, consider deleting `checkpoints/shake-*` to
free roughly 8GB before starting budgeted runs.

## Launch Instance

```bash
# see what's available
prime availability list --gpu-type RTX_4090

# create pod (pick one in the SAME datacenter as your disk)
prime pods create --name moe-train
```

During creation in the dashboard: attach your persistent disk when prompted
("Add Shared Filesystem" button).

After deploy, note the **mount path** from instance details (e.g., `/mnt/shared`).

## Connect

```bash
prime pods ssh moe-train
```

If the name does not resolve, use `prime pods list` to find the pod ID and connect
with `prime pods ssh <pod-id>`.

## Instance Setup (One-Time)

Run these once after first SSH:

```bash
# check CUDA is working
nvidia-smi

# clone repo to persistent storage so it survives restarts
cd /mnt/shared   # or whatever the mount path is
git clone https://github.com/sumitdotml/moe-emergence.git
cd moe-emergence

# install dependencies
pip install uv
uv sync
uv pip install -e .

# configure W&B
pip install wandb
wandb login   # paste API key

# verify imports work
uv run python -c "from moe_emergence.train import train; print('ok')"
```

## Cache Directory

The project expects cached data in `.cache/` at the repo root (not `~/.cache/`).
HuggingFace datasets will download here automatically. On persistent storage this
means datasets survive instance restarts too.

Verify the env var is set if needed:

```bash
export HF_HOME=$(pwd)/.cache
```

## Run Training

Follow the run order from the training plan:

```bash
# 1. shakedown (mandatory gate)
uv run python -m moe_emergence.train --preset shakedown --run-name shake-dense
uv run python -m moe_emergence.train --preset shakedown --run-name shake-moe --moe-layers 8 9 10 11

# 2. dense baseline
uv run python -m moe_emergence.train --preset dense --run-name dense-baseline

# 3. MoE main run
uv run python -m moe_emergence.train --preset moe-main --run-name moe-main

# 4. no-LB ablation (will early-stop on collapse)
uv run python -m moe_emergence.train --preset no-lb --run-name no-lb-ablation

# 5. top-2 directional (optional, budget permitting)
uv run python -m moe_emergence.train --preset top2 --run-name top2-directional
```

## Resume After Interruption

If the instance dies mid-run, relaunch and resume:

```bash
# reconnect
prime pods ssh moe-train
cd /mnt/shared/moe-emergence

# resume from latest checkpoint (sort -V orders by step number)
uv run python -m moe_emergence.train --preset moe-main --run-name moe-main \
  --resume $(ls checkpoints/moe-main/ckpt-step-*.pt | sort -V | tail -1)
```

Checkpoints are on persistent storage, so they survive instance termination.

## Download Results

After training completes, pull artifacts locally:

```bash
# from your local machine
scp -r <user>@<host>:/mnt/shared/moe-emergence/checkpoints/ ./checkpoints/
scp -r <user>@<host>:/mnt/shared/moe-emergence/checkpoints/*/metrics.jsonl ./local-metrics/
```

Or use `rsync` for incremental transfers:

```bash
rsync -avz <user>@<host>:/mnt/shared/moe-emergence/checkpoints/ ./checkpoints/
```

W&B metrics sync automatically if online mode is working.

## Teardown

```bash
# terminate instance when done (stops billing)
prime pods terminate moe-train

# persistent disk keeps running — delete it when fully done to stop storage charges
# (do this from the dashboard after downloading everything)
```

## Troubleshooting

| Issue               | Fix                                                                |
| ------------------- | ------------------------------------------------------------------ |
| OOM on 4090         | `--batch-size 1 --grad-accum-steps 8` (same effective batch)       |
| Still OOM           | `--block-size 256` (halves sequence memory)                        |
| W&B offline         | Runs still work, logs to local JSONL. Sync later with `wandb sync` |
| Disk not attachable | Instance must be in same provider + datacenter as disk             |
| Slow data loading   | First run downloads datasets (~30MB). Cached after that            |

## References

- [PrimeIntellect Docs](https://docs.primeintellect.ai)
- [PrimeIntellect CLI](https://github.com/PrimeIntellect-ai/prime-cli)
- [Persistent Storage Guide](https://docs.primeintellect.ai/tutorials-storage/create-persistent-storage)
- [RTX 4090 Pricing](https://www.primeintellect.ai/compute/nvidia-rtx4090)
