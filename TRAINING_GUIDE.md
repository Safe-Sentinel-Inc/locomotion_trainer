# Training Guide

Full setup and training instructions for AME-2 legged locomotion on NVIDIA GPUs.

## Hardware Requirements

- **Minimum**: NVIDIA RTX 3090 (24GB VRAM) — up to 2048 parallel envs
- **Recommended**: NVIDIA H100 / A100 (80GB VRAM) — up to 8192 parallel envs
- **CPU**: 8+ cores recommended (physics stepping is CPU-assisted)
- **RAM**: 32GB+
- **OS**: Ubuntu 22.04 (tested), any Linux with NVIDIA drivers

## Dependencies

### 1. System packages

```bash
sudo apt-get update
sudo apt-get install -y libsm6 libxt6 libice6 libxext6 libxrender1 \
    libgl1-mesa-glx libglib2.0-0 vulkan-tools libvulkan1 libvulkan-dev
```

### 2. Python (3.10 required)

```bash
python3 --version  # must be 3.10.x
pip install --upgrade pip setuptools
```

### 3. Isaac Sim 4.5 (physics simulator)

```bash
pip install isaacsim-rl isaacsim-replicator isaacsim-extscache-physics isaacsim-extscache-kit-sdk
```

Accept the NVIDIA EULA on first run:
```bash
echo "Yes" | python3 -c "import isaacsim"
```

### 4. Isaac Lab 2.1.0 (robot learning framework)

```bash
pip install --no-build-isolation isaaclab==2.1.0
```

### 5. Isaac Lab extensions (from source, must match v2.1.0)

```bash
git clone https://github.com/isaac-sim/IsaacLab.git /tmp/IsaacLab
cd /tmp/IsaacLab && git checkout v2.1.0
pip install -e /tmp/IsaacLab/source/isaaclab_assets
pip install -e /tmp/IsaacLab/source/isaaclab_rl
```

### 6. This project

```bash
git clone https://github.com/PRB-R-NINE-T/locomotion_trainer.git
cd locomotion_trainer
pip install -e ".[dev]"
```

### 7. Verify installation

```bash
# Quick test (no Isaac Sim needed)
python3 scripts/train_mapping.py

# Unit tests
pytest scripts/test_ame2.py -v
```

## Training

### Environment variable setup

Isaac Sim needs library paths set. Add to your shell or prefix every command:

```bash
export USD_LIBS="$(python3 -c "import site; print(site.getusersitepackages())")/omni/data/Kit/Isaac-Sim/4.5/exts/3/omni.usd.libs-1.0.1+d02c707b.lx64.r.cp310/bin"
export PHYSX_BINS="$(find $(python3 -c "import site; print(site.getusersitepackages())")/isaacsim/extsPhysics -name bin -type d | tr '\n' ':')"
export LD_LIBRARY_PATH="$USD_LIBS:$PHYSX_BINS$LD_LIBRARY_PATH"
```

### Phase 0: MappingNet pretraining (no Isaac Sim needed)

```bash
# Quick demo (100 steps, ~30s)
python3 scripts/train_mapping.py

# Full pretraining
python3 scripts/train_mapping.py --num_steps 50000 --batch_size 64 --save_path logs/mapping_net.pt
```

Output: `logs/mapping_net.pt`

### Phase 1: Teacher PPO (main training)

```bash
# RTX 3090 (24GB) — 2048 envs
CUDA_VISIBLE_DEVICES=0 python3 scripts/train_ame2_direct.py \
    --num_envs 2048 --seed 42 --log_dir logs/gpu0 --headless

# H100/A100 (80GB) — 4096 envs (best throughput/memory balance)
CUDA_VISIBLE_DEVICES=0 python3 scripts/train_ame2_direct.py \
    --num_envs 4096 --seed 42 --log_dir logs/gpu0 --headless

# H100/A100 (80GB) — 8192 envs (max throughput, ~95% VRAM)
CUDA_VISIBLE_DEVICES=0 python3 scripts/train_ame2_direct.py \
    --num_envs 8192 --seed 42 --log_dir logs/gpu0 --headless
```

### Resume from checkpoint

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/train_ame2_direct.py \
    --num_envs 4096 --seed 42 --log_dir logs/gpu0 \
    --resume logs/gpu0/model_1200.pt --headless
```

### Long-running training (survives SSH disconnect)

```bash
nohup bash -c 'CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
    python3 scripts/train_ame2_direct.py \
    --num_envs 4096 --headless --log_dir logs/gpu0' > logs/gpu0/train.log 2>&1 &
```

### Watchdog (auto-restarts on CUDA crashes)

```bash
./watchdog.sh 0 42 v57 --num_envs 4096
```

### Multi-GPU (independent seeds)

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/train_ame2_direct.py --seed 42 --log_dir logs/gpu0 --headless &
CUDA_VISIBLE_DEVICES=1 python3 scripts/train_ame2_direct.py --seed 43 --log_dir logs/gpu1 --headless &
```

## Training time estimates

| GPU | Envs | Iters/hr | Time to 80K iters |
|-----|------|----------|--------------------|
| RTX 3090 (24GB) | 2048 | ~150 | ~22 days |
| H100 (80GB) | 4096 | ~250 | ~13 days |
| H100 (80GB) | 8192 | ~143 | ~23 days* |

*8192 envs processes more samples per iteration but is memory-bandwidth limited, so fewer iterations per hour. 4096 is the sweet spot on H100.

## Monitoring

### Checkpoints

Saved every 50 iterations to `logs/gpu0/model_<iter>.pt`.

```bash
ls -lt logs/gpu0/model_*.pt | head -5
```

### GPU utilization

```bash
nvidia-smi
```

### TensorBoard

```bash
tensorboard --logdir logs/gpu0
```

### Training log (if using nohup)

```bash
tail -f logs/gpu0/train.log
```

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

### `libhdx.so: cannot open shared object file`
Set `LD_LIBRARY_PATH` as described in the environment variable setup section above.

### `RayCasterCfg.__init__() got an unexpected keyword argument`
You have a version mismatch between isaaclab and this code. This repo requires isaaclab 2.1.0.

### `VkResult: ERROR_INCOMPATIBLE_DRIVER`
Missing NVIDIA Vulkan ICD. Non-fatal for headless training — PhysX uses CUDA directly.

### CUDA OOM
Reduce `--num_envs`. Guideline: ~5MB VRAM per env.
