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

All 1x RTX 4090 options across regions are **the same price (~$0.61/hr)** but differ
in vCPUs and RAM. Compare regions before committing:

```bash
prime availability list --gpu-type RTX4090_24GB --gpu-count 1 -o json | python3 -c "
import json, sys
data = json.loads(sys.stdin.read())['gpu_resources']
for x in sorted(data, key=lambda x: -int(str(x['memory_gb']).split('-')[0])):
    print(f\"{x['id']}  {x['location']:4s}  stock={x['stock_status']:6s}  \${x['price_value']:.2f}/hr  vCPUs={x['vcpus']:>5s}  RAM={x['memory_gb']}GB\")
"
```

Pick the region with the best specs. Watch the **stock level** — "Low" means the GPU
may not actually be available when you try to deploy, even if it shows in the listing.
"High" stock is more reliable.

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
- [ ] Add billing / payment method (disk and pod creation will fail without this)
- [ ] Generate SSH key if needed (`ssh-keygen -t ed25519`)
- [ ] Register public key under **Keys & Secrets** in the PrimeIntellect dashboard
- [ ] Get W&B API key from https://wandb.ai/authorize (will need it on the instance)

## Install CLI (Local)

```bash
uv tool install prime
prime login                                   # preferred auth flow
# optional alternative if needed:
# prime config set-api-key
prime config set-ssh-key-path ~/.ssh/id_ed25519   # point to your private key
prime config view                                 # verify auth + ssh key path
```

## Create Persistent Storage

Persistent storage keeps checkpoints, datasets, and logs across instance restarts.

**Important:** Runpod uses ~40GB of disk for the container filesystem. Only the
remainder is usable volume. For example, 300GB disk = ~260GB usable. The cost
difference is negligible (~$0.03/hr for 300GB vs 100GB), so size up.

```bash
# list disk options (omit --regions to see all)
prime availability disks

# create disk — use --yes to skip interactive prompt
prime disks create --id <disk-option-id> --size 300 --name moe-emergence --yes

# confirm disk ID + status (wait for ACTIVE)
prime disks get <disk-id>
```

Notes:
- Choose disk location first, then schedule GPUs in the same provider + datacenter.
- Disks are billed continuously (~$0.033/hr for 300GB) until you terminate them.
- Verify GPU availability in your chosen region *before* creating the disk —
  "Low" stock regions may have no GPUs when you actually try to deploy.

## Launch Instance

```bash
# find 4090 capacity compatible with your existing disk location
prime availability list --gpu-type RTX4090_24GB --disks <disk-id>

# create pod — all flags needed to avoid interactive prompts
prime pods create \
  --id <gpu-option-id> \
  --name moe-train \
  --disk-size 120 \
  --disks <disk-id> \
  --image cuda_12_4_pytorch_2_4 \
  --yes
```

Available images include `cuda_12_4_pytorch_2_4`, `cuda_12_1_pytorch_2_2`, and
others. Pick the latest PyTorch + CUDA combo.

After deploy, note:
- Pod ID (`prime pods list`)
- Mount path for persistent disk (`prime pods status <pod-id>` — typically `/workspace`)

## Connect

```bash
prime pods list
prime pods ssh <pod-id>
```

## Instance Setup (One-Time)

Run these once after first SSH:

```bash
# install tmux (not pre-installed on runpod images)
apt-get update && apt-get install -y tmux

# check CUDA is working
nvidia-smi

# clone repo to persistent storage so it survives restarts
cd /workspace   # persistent disk mount path (confirm via prime pods status)
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

## Use tmux for Training

**Always run training inside tmux.** Training runs take 30–60+ minutes. If your SSH
connection drops without tmux, the run dies and you lose progress since the last
checkpoint.

```bash
# start a new tmux session
tmux new -s train

# if reconnecting after disconnect:
tmux attach -t train
```

Run all training commands inside the tmux session. You can detach with `Ctrl-b d`
and reattach later from a new SSH connection.

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
cd /workspace/moe-emergence

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
scp -P <port> -r root@<host>:/workspace/moe-emergence/checkpoints/ ./checkpoints/
scp -P <port> -r root@<host>:/workspace/moe-emergence/checkpoints/*/metrics.jsonl ./local-metrics/
```

Or use `rsync` for incremental transfers:

```bash
rsync -avz -e "ssh -p <port>" \
  root@<host>:/workspace/moe-emergence/checkpoints/ ./checkpoints/
```

W&B metrics sync automatically if online mode is working.

## Teardown

```bash
# terminate instance when done (stops GPU billing)
prime pods terminate <pod-id>

# disk keeps billing (~$0.033/hr) until terminated
prime disks terminate <disk-id> --yes
```

## Troubleshooting

| Issue               | Fix                                                                |
| ------------------- | ------------------------------------------------------------------ |
| OOM on 4090         | `--batch-size 1 --grad-accum-steps 8` (same effective batch)       |
| Still OOM           | `--block-size 256` (halves sequence memory)                        |
| W&B auth failure    | Run `uv run wandb login --verify`, then rerun preflight             |
| W&B offline         | Runs still work, logs to local JSONL. Sync later with `wandb sync` |
| Disk not attachable | Disk and pod must be same provider + datacenter                    |
| No GPU found        | "Low" stock region sold out. Delete disk, recreate in another region |
| Payment required    | Add payment method at dashboard/billing before creating resources   |
| Slow data loading   | First run downloads datasets (~30MB). Cached after that            |

## References

- [PrimeIntellect Docs](https://docs.primeintellect.ai)
- [PrimeIntellect CLI](https://github.com/PrimeIntellect-ai/prime-cli)
- [Persistent Storage Guide](https://docs.primeintellect.ai/tutorials-storage/create-persistent-storage)
- [RTX 4090 Pricing](https://www.primeintellect.ai/compute/nvidia-rtx4090)
