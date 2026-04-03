# Dependencies & Installation Guide

## Docker Image

Use the official NVIDIA Isaac Lab Docker image:

```
nvcr.io/nvidia/isaac-lab:2.3.2
```

### Required Environment Variable

Set the EULA acceptance variable when running the container:

```bash
docker run -e ACCEPT_EULA=true ... nvcr.io/nvidia/isaac-lab:2.3.2
```

---

## Installation Steps

All commands use the Isaac Sim bundled Python (`/isaac-sim/python.sh`).

### 1. Clone the locomotion_trainer repository

```bash
git clone https://github.com/Safe-Sentinel-Inc/locomotion_trainer.git
cd locomotion_trainer
```

### 2. Install Isaac Lab v2.2.1

The bundled Isaac Sim 5.1 ships with `rsl-rl-lib==3.1.2`, which is incompatible with this codebase. Isaac Lab v2.2.1 (isaaclab 0.45.9) is the correct version to match the "Isaac Lab 0.46.x" requirement in the README.

```bash
git clone https://github.com/isaac-sim/IsaacLab.git /workspace/IsaacLab
cd /workspace/IsaacLab
git checkout v2.2.1
```

Install the core Isaac Lab packages:

```bash
/isaac-sim/python.sh -m pip install -e source/isaaclab
/isaac-sim/python.sh -m pip install -e source/isaaclab_rl
```

### 3. Downgrade rsl-rl-lib

The bundled `rsl-rl-lib==3.1.2` in Isaac Sim 5.1 is too new (introduces `obs_groups` / TensorDict API). Downgrade to the version expected by `isaaclab_rl==0.2.4`:

```bash
/isaac-sim/python.sh -m pip install rsl-rl-lib==2.3.3
```

### 4. Install the locomotion_trainer (ame2) package

```bash
cd /path/to/locomotion_trainer
/isaac-sim/python.sh -m pip install -e .
```

---

## Launch Training

```bash
cd /path/to/locomotion_trainer

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 WANDB_MODE=disabled \
    /isaac-sim/python.sh scripts/train_ame2_direct.py \
    --num_envs 2048 --seed 42 --log_dir logs_v43 --headless
```

Set `WANDB_MODE=disabled` unless you have a Weights & Biases API key configured (`wandb login`).

---

## Version Summary

| Component       | Version       | Notes                                       |
|-----------------|---------------|---------------------------------------------|
| Docker image    | `nvcr.io/nvidia/isaac-lab:2.3.2` | Must set `ACCEPT_EULA=true`     |
| Isaac Sim       | 5.1.0         | Bundled in the Docker image                 |
| Isaac Lab       | v2.2.1 (0.45.9) | Installed from source                    |
| isaaclab_rl     | 0.2.4         | Installed from Isaac Lab v2.2.1 source      |
| rsl-rl-lib      | 2.3.3         | Downgraded from bundled 3.1.2               |
| ame2            | 0.1.0         | From this repository (`pip install -e .`)   |
| Python          | 3.11          | Bundled with Isaac Sim                      |
