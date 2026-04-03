# RunPod Setup Guide — AME-2 Locomotion Trainer

Complete setup for training AME-2 on RunPod using the `nvcr.io/nvidia/isaac-lab:2.3.2` Docker image.

---

## 1. Create RunPod Pod

- **Template/Image:** `nvcr.io/nvidia/isaac-lab:2.3.2`
- **Environment variable:** `ACCEPT_EULA` = `true` (required — Isaac Sim won't start without it)
- **GPU:** Any NVIDIA GPU with 24GB+ VRAM (RTX 4090, A100, RTX 6000 Ada, etc.)
- **Disk:** 50GB+ container disk

Once the pod is running, open a terminal.

---

## 2. Install & Train (copy-paste this entire block)

The training script supports two robots via `--robot`:

| Robot | Flag | Joints | Description |
|-------|------|--------|-------------|
| ANYmal-D | `--robot anymal_d` (default) | 12 | Quadruped, 4 legs |
| PF_TRON1A | `--robot tron1` | 6 | Biped, 2 legs |

### ANYmal-D (default)

```bash
#!/bin/bash
set -e

# ── Clone locomotion_trainer ──
cd /workspace
git clone https://github.com/Safe-Sentinel-Inc/locomotion_trainer.git
cd locomotion_trainer

# ── Install Isaac Lab v2.2.1 ──
# The image bundles rsl-rl-lib 3.1.2 which is incompatible.
# Isaac Lab v2.2.1 pins the correct versions.
git clone https://github.com/isaac-sim/IsaacLab.git /workspace/IsaacLab
cd /workspace/IsaacLab && git checkout v2.2.1
/isaac-sim/python.sh -m pip install -e source/isaaclab
/isaac-sim/python.sh -m pip install -e source/isaaclab_rl

# ── Downgrade rsl-rl-lib (3.1.2 → 2.3.3) ──
/isaac-sim/python.sh -m pip install rsl-rl-lib==2.3.3

# ── Install this repo's network package ──
cd /workspace/locomotion_trainer
/isaac-sim/python.sh -m pip install -e .

# ── Launch training (ANYmal-D) ──
mkdir -p logs_v43
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 WANDB_MODE=disabled \
    nohup /isaac-sim/python.sh scripts/train_ame2_direct.py \
    --robot anymal_d \
    --num_envs 2048 --seed 42 --log_dir logs_v43 --headless \
    > logs_v43/train.log 2>&1 &

echo ""
echo "========================================="
echo " Training launched (PID: $!)"
echo " Monitor: tail -f logs_v43/train.log"
echo "========================================="
```

### PF_TRON1A (biped)

**Prerequisite:** Convert the PF_TRON1A URDF to USD before first run:

```bash
# Convert URDF → USD (run once)
/isaac-sim/python.sh -m isaaclab.app.tools.convert_urdf \
    --urdf_path path/to/pf_tron1a.urdf \
    --output_path data/robots/pf_tron1a/pf_tron1a.usd \
    --fix_base False --make_instanceable
```

Then update the USD path in `ame2_direct/tron1_asset.py` if it differs from the default `data/robots/pf_tron1a/pf_tron1a.usd`.

```bash
# ── Launch training (TRON1 biped) ──
mkdir -p logs_tron1
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 WANDB_MODE=disabled \
    nohup /isaac-sim/python.sh scripts/train_ame2_direct.py \
    --robot tron1 \
    --num_envs 2048 --seed 42 --log_dir logs_tron1 --headless \
    > logs_tron1/train.log 2>&1 &

echo ""
echo "========================================="
echo " TRON1 training launched (PID: $!)"
echo " Monitor: tail -f logs_tron1/train.log"
echo "========================================="
```

---

## 3. Phase 0 — Pretrain MappingNet (optional, ~1hr GPU)

Before launching Phase 1 Teacher PPO, you can pretrain the MappingNet on synthetic terrain using Warp raytracing. This produces a checkpoint consumed by Phase 1/2 via `--mapping_ckpt`.

```bash
# Install warp dependency
/isaac-sim/python.sh -m pip install "warp-lang>=1.0"

# Run MappingNet pretraining (50K steps, ~1hr on H100 / ~2hr on A100)
cd /workspace/locomotion_trainer
mkdir -p logs outputs/mapping

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
    nohup /isaac-sim/python.sh scripts/train_mapping.py \
    --num_steps 50000 --batch_size 64 \
    --save_path logs/mapping_net.pt \
    --output_dir outputs/mapping \
    > logs/mapping_train.log 2>&1 &

echo "MappingNet training launched (PID: $!)"
echo "Monitor: tail -f logs/mapping_train.log"
```

**Monitor:**

```bash
# Live log stream
tail -f /workspace/locomotion_trainer/logs/mapping_train.log

# Check latest step/loss
grep "Step" /workspace/locomotion_trainer/logs/mapping_train.log
```

Loss is beta-NLL and will go **negative** as the model converges — this is expected (it means confident, accurate predictions with well-calibrated uncertainty).

**Output:**
- `logs/mapping_net.pt` — trained MappingNet weights
- `outputs/mapping/mapping_*.png` — visualizations every 20 steps (noisy input, prediction, uncertainty, ground truth)
- `outputs/mapping/mapping_loss.png` — loss curve

**Legacy mode** (no Warp dependency):

```bash
/isaac-sim/python.sh scripts/train_mapping.py \
    --no_warp --num_steps 50000 --batch_size 64 \
    --save_path logs/mapping_net.pt
```

---

## 4. Monitor Phase 1 Training

```bash
# Live log stream
tail -f /workspace/locomotion_trainer/logs_v43/train.log

# Quick status (iterations, timesteps, ETA)
grep -E "Iteration|Total timesteps|ETA|Mean reward" \
    /workspace/locomotion_trainer/logs_v43/train.log | tail -12
```

Checkpoints are saved every 50 iterations to `logs_v43/model_*.pt`.

---

## 5. Resume from Checkpoint

If the pod restarts or you want to continue from a saved checkpoint, re-run the install block above but replace the training launch with:

```bash
# ANYmal-D resume
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 WANDB_MODE=disabled \
    nohup /isaac-sim/python.sh scripts/train_ame2_direct.py \
    --robot anymal_d \
    --num_envs 2048 --seed 42 --log_dir logs_v43 \
    --resume logs_v43/model_800.pt --headless \
    > logs_v43/train.log 2>&1 &
```

```bash
# TRON1 resume
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 WANDB_MODE=disabled \
    nohup /isaac-sim/python.sh scripts/train_ame2_direct.py \
    --robot tron1 \
    --num_envs 2048 --seed 42 --log_dir logs_tron1 \
    --resume logs_tron1/model_800.pt --headless \
    > logs_tron1/train.log 2>&1 &
```

Replace `model_800.pt` with the latest checkpoint file. The `--robot` flag must match the robot used when the checkpoint was created.

---

## 6. Multi-GPU (if your pod has multiple GPUs)

Add `--robot tron1` or `--robot anymal_d` as needed.

```bash
NUM_GPUS=$(nvidia-smi -L | wc -l)

# ANYmal-D multi-GPU
PYTHONUNBUFFERED=1 WANDB_MODE=disabled \
    nohup torchrun --nproc_per_node=$NUM_GPUS \
    --master_addr=127.0.0.1 --master_port=29500 \
    scripts/train_ame2_direct.py \
    --robot anymal_d \
    --num_envs 1200 --seed 42 --log_dir logs_v43 --headless \
    > logs_v43/train.log 2>&1 &

# TRON1 multi-GPU
PYTHONUNBUFFERED=1 WANDB_MODE=disabled \
    nohup torchrun --nproc_per_node=$NUM_GPUS \
    --master_addr=127.0.0.1 --master_port=29500 \
    scripts/train_ame2_direct.py \
    --robot tron1 \
    --num_envs 1200 --seed 42 --log_dir logs_tron1 --headless \
    > logs_tron1/train.log 2>&1 &
```

1200 envs/GPU matches the paper spec. For 4x GPUs that's 4800 total environments.

---

## Robot Comparison

| Parameter | ANYmal-D | PF_TRON1A |
|-----------|----------|-----------|
| Type | Quadruped | Biped |
| Joints | 12 (3 per leg) | 6 (3 per leg) |
| Legs | 4 | 2 |
| Mass | 50 kg | 18.5 kg |
| Standing height | 0.6 m | 0.65 m |
| Action scale | 0.5 | 0.25 |
| Actor prop dim | 48 | 30 |
| Contact dim | 13 | 7 |
| Local scan (MappingNet) | 31x51 @ 4cm | 31x31 @ 4cm |
| Policy map (encoder) | 14x36 @ 8cm | 13x18 @ 8cm |
| Obs flat dim | 3140 | 1478 |
| `--robot` flag | `anymal_d` | `tron1` |
| Log directory | `logs_v43/` | `logs_tron1/` |

---

## Version Pinning

| Component    | Version              | Why                                           |
|--------------|----------------------|-----------------------------------------------|
| Docker image | `nvcr.io/nvidia/isaac-lab:2.3.2` | Bundles Isaac Sim 5.1 + Python 3.11 |
| Isaac Lab    | v2.2.1 (isaaclab 0.45.9) | Matches "Isaac Lab 0.46.x" requirement   |
| rsl-rl-lib   | 2.3.3                | Bundled 3.1.2 has breaking API changes        |
| ame2         | 0.1.0                | This repo (`pip install -e .`)                |
