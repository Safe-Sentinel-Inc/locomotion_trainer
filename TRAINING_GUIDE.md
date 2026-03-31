# Training Guide

Full setup and training instructions for AME-2 legged locomotion on NVIDIA GPUs.

## Hardware Requirements

- **Minimum**: NVIDIA RTX 3090 (24GB VRAM) — up to 2048 parallel envs
- **Recommended**: NVIDIA H100 / A100 (80GB VRAM) — up to 8192 parallel envs
- **CPU**: 8+ cores recommended (physics stepping is CPU-assisted)
- **RAM**: 32GB+
- **OS**: Ubuntu 22.04 (tested), any Linux with NVIDIA drivers

## Clone & Setup on a New Machine

```bash
git clone https://github.com/Safe-Sentinel-Inc/locomotion_trainer.git
cd locomotion_trainer
```

## Dependencies

**CRITICAL: Python 3.10 is required.** Isaac Sim's PhysX extensions are compiled for `cp310` only. Python 3.11+ will fail at runtime with `omni.physx` resolution errors even though `pip install` succeeds.

If your system Python is 3.11+, create a 3.10 venv:

```bash
# Ubuntu: install Python 3.10 if not present
sudo apt-get install -y python3.10 python3.10-venv python3.10-dev

# Create and activate venv
python3.10 -m venv ~/ame2_venv
source ~/ame2_venv/bin/activate
pip install --upgrade pip
```

### 1. System packages

```bash
sudo apt-get update
sudo apt-get install -y libsm6 libxt6 libice6 libxext6 libxrender1 \
    libgl1-mesa-glx libglib2.0-0 vulkan-tools libvulkan1 libvulkan-dev
```

### 2. Isaac Sim 4.5 (physics simulator)

```bash
pip install isaacsim-rl isaacsim-replicator isaacsim-extscache-physics isaacsim-extscache-kit-sdk
```

Accept the NVIDIA EULA on first run:
```bash
echo "Yes" | python3 -c "import isaacsim"
```

### 3. Isaac Lab v2.1.0 (from source)

`isaaclab==2.1.0` is NOT on PyPI. You must install from the GitHub tag. The `flatdict==4.0.1` dependency requires `setuptools<71` to build:

```bash
# Install setuptools that supports pkg_resources (needed by flatdict)
pip install "setuptools<71"

# Install flatdict without build isolation first (workaround for build issue)
pip install flatdict==4.0.1 --no-build-isolation

# Clone Isaac Lab v2.1.0 and install from source
git clone --depth 1 --branch v2.1.0 https://github.com/isaac-sim/IsaacLab.git /tmp/IsaacLab
pip install --no-build-isolation -e /tmp/IsaacLab/source/isaaclab
pip install -e /tmp/IsaacLab/source/isaaclab_assets
pip install -e /tmp/IsaacLab/source/isaaclab_rl
```

### 4. RSL-RL v2.3.3 (from GitHub)

Not on PyPI either. Install from the tag:

```bash
pip install git+https://github.com/leggedrobotics/rsl_rl.git@v2.3.3
```

### 5. This project

```bash
cd locomotion_trainer
pip install -e ".[dev]"
```

### 6. Verify installation

```bash
# Quick test (no Isaac Sim needed)
python3 scripts/train_mapping.py

# Unit tests
pytest scripts/test_ame2.py -v
```

### Quick dependency summary

| Package | Version | Source |
|---------|---------|--------|
| Python | 3.10.x | **Required** (3.11+ breaks PhysX) |
| isaacsim-* | 4.5.0 | PyPI |
| isaaclab | 0.36.21 (v2.1.0) | GitHub tag (source install) |
| isaaclab_assets | 0.2.2 | GitHub tag (source install) |
| isaaclab_rl | 0.1.4 | GitHub tag (source install) |
| rsl-rl-lib | 2.3.3 | GitHub tag |
| torch | 2.5.1 | Installed by isaaclab |

## Training

### Single-GPU Training

```bash
# RTX 3090 (24GB) — 2048 envs
CUDA_VISIBLE_DEVICES=0 python3 scripts/train_ame2_direct.py \
    --num_envs 2048 --seed 42 --log_dir logs/gpu0 --headless

# A100/H100 (80GB) — 4096 envs (best throughput/memory balance)
CUDA_VISIBLE_DEVICES=0 python3 scripts/train_ame2_direct.py \
    --num_envs 4096 --seed 42 --log_dir logs/gpu0 --headless
```

### Resume from checkpoint

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/train_ame2_direct.py \
    --num_envs 4096 --seed 42 --log_dir logs/gpu0 \
    --resume logs/gpu0/model_1200.pt --headless
```

### Multi-GPU Distributed Training (gradient-synced, paper-faithful)

Uses `torchrun` to launch one process per GPU. Each rank gets its own GPU via `CUDA_VISIBLE_DEVICES`, runs independent Isaac Sim environments, then model parameters are averaged via NCCL `all_reduce` after each PPO update. This matches the paper's training setup.

```bash
# 4 GPUs, 1200 envs each = 4800 total (paper spec)
PYTHONUNBUFFERED=1 torchrun --nproc_per_node=4 \
    --master_addr=127.0.0.1 --master_port=29500 \
    scripts/train_ame2_direct.py \
    --num_envs 1200 --seed 42 --log_dir logs/distributed \
    --resume model_2200.pt --headless

# Long-running (survives SSH disconnect)
PYTHONUNBUFFERED=1 nohup torchrun --nproc_per_node=4 \
    --master_addr=127.0.0.1 --master_port=29500 \
    scripts/train_ame2_direct.py \
    --num_envs 1200 --seed 42 --log_dir logs/distributed \
    --resume model_2200.pt --headless > logs/distributed/train.log 2>&1 &
```

**How the distributed training works:**

1. `torchrun` sets `WORLD_SIZE`, `RANK`, `LOCAL_RANK` env vars per process
2. Each rank sets `CUDA_VISIBLE_DEVICES=<LOCAL_RANK>` so Isaac Sim sees only 1 GPU
3. Each rank runs its own Isaac Sim instance + env + PPO independently
4. After each PPO update, `all_reduce(SUM) / world_size` averages parameters across all ranks
5. Every 200 iterations, a sync check verifies all ranks have identical parameter checksums
6. Only rank 0 saves checkpoints and logs metrics

**Why not use rsl_rl's built-in multi-GPU?**

rsl_rl's `OnPolicyRunner` has distributed support, but it requires each rank to see all GPUs (`device=cuda:N`). Isaac Sim cannot create environments when multiple processes on the same node all see all GPUs — only 1 of 4 ranks completes `simulation_start`. The workaround is `CUDA_VISIBLE_DEVICES` per rank, which means each rank uses `cuda:0`, conflicting with rsl_rl's device check. So we handle distributed sync manually.

**Choosing envs per GPU:**

| Envs/GPU | Total (4 GPU) | VRAM/GPU | Iter time | Iters/hr | Samples/hr | Wall-clock to 80K |
|----------|---------------|----------|-----------|----------|------------|-------------------|
| 1200 | 4800 | ~20GB | ~5.8s | ~620 | ~71M | ~5.2 days |
| 2400 | 9600 | ~30GB | ~10s | ~360 | ~83M | ~9 days |
| 4800 | 19200 | ~50GB | ~20s | ~180 | ~83M | ~18 days |

1200/GPU matches the paper and gives fastest wall-clock. Higher envs give slightly more samples/hr but much slower iteration speed. Use 1200 unless training is unstable and needs richer gradients.

### Multi-GPU Independent Seeds (no sync)

If you don't need gradient sync and want to run multiple independent experiments:

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/train_ame2_direct.py --seed 42 --log_dir logs/gpu0 --headless &
CUDA_VISIBLE_DEVICES=1 python3 scripts/train_ame2_direct.py --seed 43 --log_dir logs/gpu1 --headless &
CUDA_VISIBLE_DEVICES=2 python3 scripts/train_ame2_direct.py --seed 44 --log_dir logs/gpu2 --headless &
CUDA_VISIBLE_DEVICES=3 python3 scripts/train_ame2_direct.py --seed 45 --log_dir logs/gpu3 --headless &
```

### Watchdog (auto-restarts on CUDA crashes)

```bash
./watchdog.sh 0 42 v57 --num_envs 4096
```

## Training time estimates

| GPU | Mode | Envs | Iters/hr | Time to 80K iters |
|-----|------|------|----------|--------------------|
| RTX 3090 (24GB) | single | 2048 | ~150 | ~22 days |
| H100 (80GB) | single | 4096 | ~250 | ~13 days |
| 4×A100 (80GB) | distributed | 4800 total | ~620 | ~5.2 days |
| 4×A100 (80GB) | distributed | 19200 total | ~180 | ~18 days |
| 4×A100 (80GB) | independent seeds | 8192/GPU | ~110/GPU | ~30 days (pick best) |

## Monitoring

### Checkpoints

Saved every 50 iterations to `<log_dir>/model_<iter>.pt`.

```bash
ls -lt logs/distributed/model_*.pt | head -5
```

### GPU utilization

```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
```

### TensorBoard

```bash
tensorboard --logdir logs/distributed
```

### Training log

```bash
tail -f logs/distributed/train.log
```

### Verify distributed sync

In the training log, look for `[Sync Check]` lines every 200 iterations:
```
[Sync Check it 2400] param_sums=['-1429.9628', '-1429.9628', '-1429.9628', '-1429.9628'] → SYNCED
```

All ranks should show identical parameter sums. If they diverge, NCCL communication has failed.

## Reward weight overrides

All reward weights can be overridden from the CLI:

```bash
python3 scripts/train_ame2_direct.py --num_envs 4096 --headless \
    --w_position_tracking 5 \
    --w_vel_toward_goal 10 \
    --w_undesired_contacts -0.5
```

See `ame2_direct/config.py` for all available `w_*` parameters.

## Troubleshooting

### `omni.physx` can't be satisfied / platform incompatible `cp310`

You're running Python 3.11+. Isaac Sim's PhysX extensions only exist for Python 3.10. Create a Python 3.10 venv (see top of this guide).

### `flatdict` fails to build (`No module named 'pkg_resources'`)

Install with `pip install "setuptools<71"` first, then `pip install flatdict==4.0.1 --no-build-isolation`.

### `isaaclab==2.1.0` not found on PyPI

It's not published to PyPI. Install from source using the `v2.1.0` git tag (see step 3 above).

### Isaac Sim hangs with multi-GPU `torchrun`

Isaac Sim cannot start simulation when multiple processes on the same node all see all GPUs. The training script handles this by setting `CUDA_VISIBLE_DEVICES` per rank. If you're writing a custom launcher, ensure each process only sees its own GPU.

### `Device 'cuda:0' does not match expected device for local rank`

The rsl_rl `OnPolicyRunner` detects `WORLD_SIZE` env var and tries its own multi-GPU init, which conflicts with our manual approach. The training script temporarily hides `WORLD_SIZE` during runner creation. If you see this error, ensure the env var masking is in place.

### `libhdx.so: cannot open shared object file`

Set `LD_LIBRARY_PATH` — though in most venv installs this is not needed.

### `VkResult: ERROR_INCOMPATIBLE_DRIVER`

Missing NVIDIA Vulkan ICD. Non-fatal for headless training — PhysX uses CUDA directly.

### CUDA OOM

Reduce `--num_envs`. Guideline: ~10MB VRAM per env on A100, ~12MB on RTX 3090.
