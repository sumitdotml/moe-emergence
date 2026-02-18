# GPU Setup: PrimeIntellect.ai

Migration checklist for running MoE Emergence training on PrimeIntellect.

## Why PrimeIntellect

- Aggregated GPU marketplace with competitive RTX 4090 options
- Live availability + pricing via CLI (`prime availability list`)
- New-user promo credits exist but can change at any time (check current offer in app)
- Persistent storage survives instance termination (checkpoints safe)
- SSH access, pre-built PyTorch containers, no lock-in
- CLI tool (`prime`) built on `uv`

## GPU Choice

RTX 4090 (24GB VRAM) is the right pick. GPT-2 small + 8 experts is well under 24GB
for this project, and A100-class options are usually unnecessary for these runs.

Verify current rates in the PrimeIntellect dashboard before booking - pricing and
free-credit promos can change.

## Budget Math

| Run               | Preset    | Max Steps         | Est. Time | Cost Formula            |
| ----------------- | --------- | ----------------- | --------- | ----------------------- |
| Dense shakedown   | shakedown | 100               | ~2 min    | `(2/60) * live_rate`    |
| MoE shakedown     | shakedown | 100               | ~3 min    | `(3/60) * live_rate`    |
| Dense baseline    | dense     | 5000              | ~1.5 hr   | `1.5 * live_rate`       |
| MoE main          | moe-main  | 10000             | ~4 hr     | `4.0 * live_rate`       |
| No-LB ablation    | no-lb     | 2000 (early-stop) | ~1 hr     | `1.0 * live_rate`       |
| Top-2 directional | top2      | 3000              | ~1.5 hr   | `1.5 * live_rate`       |

Use `live_rate` from the exact RTX 4090 row returned by `prime availability list`.
Total runtime budget for planned runs is roughly ~8 hours (+ setup/idle overhead).

---

## Pre-Flight (Local)

- [ ] Create PrimeIntellect account at https://www.primeintellect.ai
- [ ] Claim any active signup promo/credit currently shown in the app
- [ ] Add billing / payment method
- [ ] Generate SSH key if needed (`ssh-keygen -t ed25519`)
- [ ] Register public key in PrimeIntellect profile settings
- [ ] Get W&B API key from https://wandb.ai/authorize (will need it on the instance)

## Install CLI (Local)

```bash
uv tool install prime
prime login                     # preferred auth flow
# optional alternative if needed:
# prime config set-api-key
prime config set-ssh-key-path   # point to your private key
prime config view               # verify auth + ssh key path
```

## Create Persistent Storage

Persistent storage keeps checkpoints, datasets, and logs across instance restarts.
CLI path (recommended):

```bash
# pick a storage option in your target region/datacenter
prime availability disks --regions united_states

# create disk (example size: 100GB)
prime disks create --id <disk-option-id> --size 100 --name moe-emergence

# confirm disk ID + status
prime disks list
```

Notes:
- Choose disk location first, then schedule GPUs in the same location.
- Disks are billed continuously until you terminate them.

## Launch Instance

```bash
# find 4090 capacity compatible with your existing disk location
prime availability list --gpu-type RTX_4090 --disks <disk-id>

# create pod and attach disk
prime pods create --id <gpu-option-id> --name moe-train --disks <disk-id>
```

After deploy, note:
- Pod ID (`prime pods list`)
- Mount path / attached disk info (`prime pods status <pod-id>`)

## Connect

```bash
prime pods list
prime pods ssh <pod-id>
```

## Instance Setup (One-Time)

Run these once after first SSH:

```bash
# check CUDA is working
nvidia-smi

# clone repo to persistent storage so it survives restarts
cd /mnt/shared   # replace with actual mounted disk path from pod status
git clone https://github.com/sumitdotml/moe-emergence.git
cd moe-emergence

# install dependencies
uv sync

# configure W&B
uv run wandb login --verify   # paste API key once

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

## W&B Preflight (Mandatory)

Before long runs, confirm W&B is online and receiving metrics:

```bash
uv run python -m moe_emergence.train \
  --preset shakedown \
  --run-name wandb-preflight \
  --device cuda \
  --max-steps 2 \
  --eval-every 1 \
  --save-every 1
```

Check your W&B project page immediately for:
- `train/loss`, `eval/loss`, `eval/perplexity`
- domain metrics (`eval/loss_code`, `eval/loss_math`, `eval/loss_prose`)

If W&B auth/network fails, training still continues and local metrics are written to:
`checkpoints/<run-name>/metrics.jsonl`.

## Run Training

Follow the run order from the training plan:

```bash
# 1. shakedown (mandatory gate)
uv run python -m moe_emergence.train --preset shakedown --run-name shake-dense --device cuda
uv run python -m moe_emergence.train --preset shakedown --run-name shake-moe --device cuda --moe-layers 8 9 10 11

# 2. dense baseline
uv run python -m moe_emergence.train --preset dense --run-name dense-baseline --device cuda

# 3. MoE main run
uv run python -m moe_emergence.train --preset moe-main --run-name moe-main --device cuda

# 4. no-LB ablation (will early-stop on collapse)
uv run python -m moe_emergence.train --preset no-lb --run-name no-lb-ablation --device cuda

# 5. top-2 directional (optional, budget permitting)
uv run python -m moe_emergence.train --preset top2 --run-name top2-directional --device cuda
```

## Resume After Interruption

If the instance dies mid-run, relaunch and resume:

```bash
# reconnect
prime pods ssh <pod-id>
cd /mnt/shared/moe-emergence

# resume from latest checkpoint (sort -V orders by step number)
uv run python -m moe_emergence.train --preset moe-main --run-name moe-main \
  --device cuda \
  --resume $(ls checkpoints/moe-main/ckpt-step-*.pt | sort -V | tail -1)
```

Checkpoints are on persistent storage, so they survive instance termination.

## Download Results

After training completes, pull artifacts locally:

```bash
# from your local machine; get host/port from: prime pods status <pod-id>
scp -P <port> -r <user>@<host>:/mnt/shared/moe-emergence/checkpoints/ ./checkpoints/
scp -P <port> -r <user>@<host>:/mnt/shared/moe-emergence/checkpoints/*/metrics.jsonl ./local-metrics/
```

Or use `rsync` for incremental transfers:

```bash
rsync -avz -e "ssh -p <port>" \
  <user>@<host>:/mnt/shared/moe-emergence/checkpoints/ ./checkpoints/
```

W&B metrics sync automatically if online mode is working.

## Teardown

```bash
# terminate instance when done (stops billing)
prime pods terminate <pod-id>

# disk keeps billing until terminated
prime disks terminate <disk-id>
```

## Troubleshooting

| Issue               | Fix                                                                |
| ------------------- | ------------------------------------------------------------------ |
| OOM on 4090         | `--batch-size 1 --grad-accum-steps 8` (same effective batch)       |
| Still OOM           | `--block-size 256` (halves sequence memory)                        |
| W&B auth failure    | Run `uv run wandb login --verify`, then rerun preflight             |
| W&B offline         | Runs still work, logs to local JSONL. Sync later with `wandb sync` |
| Disk not attachable | Instance must be in same provider + datacenter as disk             |
| Slow data loading   | First run downloads datasets (~30MB). Cached after that            |

## References

- [PrimeIntellect Docs](https://docs.primeintellect.ai)
- [PrimeIntellect CLI](https://github.com/PrimeIntellect-ai/prime-cli)
- [Persistent Storage Guide](https://docs.primeintellect.ai/tutorials-storage/create-persistent-storage)
- [RTX 4090 Pricing](https://www.primeintellect.ai/compute/nvidia-rtx4090)
