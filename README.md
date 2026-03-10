# AME-2: Unofficial Reimplementation

> **This is an independent reimplementation, not the official code.**
> The original paper and official results are from ETH Zurich RSL.
> This repo reconstructs the architecture and training pipeline from the paper description.

Unofficial reimplementation of **AME-2** — a three-phase teacher-student RL pipeline for
agile and generalized legged locomotion, based on:

> Chong Zhang, Victor Klemm, Fan Yang, Marco Hutter (ETH Zurich RSL)
> "AME-2: Agile and Generalized Legged Locomotion via Attention-Based Neural Map Encoding"
> arXiv:2601.08485

Tested on **ANYmal-D** (quadruped) with Isaac Lab **Direct Workflow** (DirectRLEnv).

---

## Repository Structure

```
ame2/                       # Network package (pip install -e .)
├── networks/
│   ├── ame2_model.py       # Pure PyTorch: MappingNet, AME2Encoder, AME2Policy,
│   │                       #   AsymmetricCritic, StudentLoss, LSIO
│   └── rslrl_wrapper.py    # RSL-RL wrapper: AME2ActorCritic

ame2_direct/                # Direct Workflow environment (current training)
├── __init__.py             # Exports AME2DirectEnv, AME2DirectEnvCfg, AME2DirectWrapper
├── config.py               # V43 Paper-Faithful configuration (Table I + Table VI)
├── env.py                  # DirectRLEnv implementation (~960 lines)
└── wrapper.py              # RSL-RL compatible wrapper (obs TensorDict)

scripts/
├── train_ame2_direct.py    # Phase 1 Teacher PPO training (current)
├── play_record.py          # Per-env tiled camera video recording
├── play_record_global.py   # Global bird's-eye camera recording (experimental)
├── train_ame2.py           # Legacy ManagerBased training launcher
├── train_mapping.py        # Phase 0: MappingNet standalone pretraining
└── test_ame2.py            # Unit tests (19 tests, no Isaac Sim required)

paper/                      # Paper PDF + MinerU-parsed markdown
```

---

## Architecture Overview

### Teacher Network (Phase 1)

```
Teacher Actor (48D → 12D):
  TeacherPropEncoder: prop(48D) → MLP(256→128) → prop_emb(128D)
  MappingNet: raw_scan(31×51) → UNet denoiser → elevation + uncertainty + gate
  AME2Encoder: policy_map(14×36) → CNN → positional_emb → fusion_MLP
               → cross-attention(16 heads, d=64, Q=prop_emb) → map_emb(192D)
  AME2Policy decoder: cat(map_emb, prop_emb)(320D) → MLP(512→256→12)

Critic (55D → 1D):
  prop(45D) + critic_cmd(5D) + nav_extra(5D)
  critic_cmd = [x_rel, y_rel, sin(d_yaw), cos(d_yaw), t_remaining]
  MLP(256→256→128→1)
```

### Observation Layout

| Group | Dim | Content |
|-------|-----|---------|
| `prop` (actor) | 48 | base_vel(3) + ang_vel(3) + gravity(3) + q(12) + dq(12) + act(12) + cmd_actor(3) |
| `prop_critic` | 55 | base_vel(3) + ang_vel(3) + gravity(3) + q(12) + dq(12) + act(12) + critic_cmd(5) + nav_extra(5) |
| `height_scan` | 1581 | MappingNet input: 31×51 @ 4cm |
| `policy_map` | 504 | Policy map: 14×36 @ 8cm |

---

## Training (Phase 1 — Teacher PPO)

### Configuration (V43, Paper Table I + Table VI)

| Parameter | Value | Source |
|-----------|-------|--------|
| Physics dt | 1/200s | Paper |
| Control frequency | 50 Hz (decimation=4) | Paper |
| Episode length | 20s | Paper |
| Goal distance | [2m, 6m] annulus | Paper "avg 4m" |
| num_envs | 2048 (paper: 4800) | RTX 3090 limit |
| PPO epochs | 4 | Table VI |
| mini_batches | 16 (paper: 3) | OOM workaround |
| steps_per_env | 24 | Table VI |
| Entropy coef | 0.004 → 0.001 decay | Table VI |

### Rewards (Paper Table I, all ×dτ=0.02)

**Task:**
- position_tracking (100): `1/(1+0.25d²)`, last 4s only
- heading_tracking (50): last 2s only
- moving_to_goal (5): binary, v_toward > 0.3 m/s
- standing_at_goal (5): d < 0.5m AND v < 0.2 m/s

**Regularization:**
- early_termination (-500), undesired_contacts (-5), ang_vel_roll (-0.1)
- joint_reg (-0.001), action_rate (-0.01), link_contact_forces (-0.00001)
- link_acceleration (-0.001)

**Simulation Fidelity:**
- joint_pos_limits (-1000), joint_vel_limits (-1), joint_torque_limits (-1)

### Termination Conditions

- `bad_orientation`: projected_gravity_z > -0.5 (>60° tilt), grace 20 steps
- `high_thigh_acceleration`: >500 m/s² (crash detection), grace 50 steps
- `stagnation`: <0.5m displacement in 5s AND goal_dist > 0.5m

### Curricula (Paper Sec.IV-D.3)

- **Heading**: face-goal → random yaw over first 20% iterations
- **Perception noise**: 0 → max over first 20% iterations
- **Terrain**: automatic (IsaacLab, success → harder, fail → easier)

### Launch Training

```bash
# Single GPU (RTX 3090, 2048 envs)
CUDA_VISIBLE_DEVICES=0 python scripts/train_ame2_direct.py \
    --num_envs 2048 --seed 42 --log_dir logs_v43l/gpu0 --headless

# Resume from checkpoint
CUDA_VISIBLE_DEVICES=0 python scripts/train_ame2_direct.py \
    --num_envs 2048 --seed 42 --log_dir logs_v43l/gpu0 \
    --resume logs_v43/gpu0/model_800.pt --headless
```

### Record Video

```bash
# Per-env tiled view (4 envs, 2×2 grid)
CUDA_VISIBLE_DEVICES=4 python scripts/play_record.py \
    --checkpoint logs_v43l/gpu0/model_1000.pt \
    --num_envs 4 --num_steps 500 --headless --output v43l_record.mp4
```

---

## Version History

### V43l (current) — Paper-Faithful + Anti-Crawl

- `w_base_height = 0.0` — removed, causes "stand still" exploit
- `w_undesired_contacts = -5.0` — 5× paper value to penalize knee crawling
- Resume from V43j model_800

### V43j — Environment Stabilization

Key fixes from V43a-V43j debug:
1. **replicate_physics=True** — prevents robot collision across envs (ROUGH_TERRAINS_CFG has 200 tiles but 2048 envs)
2. **terrain_oob removed** — not in paper, 200 tiles < 2048 envs causes all OOB
3. **base_collision removed** — terrain_origin_z unreliable as height reference on rough terrain
4. **bad_orientation simplified** — `projected_gravity_z > -0.5` (>60° tilt), grace 20 steps
5. **thigh_acc threshold 500** — normal walking jitter ~50 m/s², crash >500 m/s²
6. **mini_batches=16** — RTX 3090 OOM with paper's 3
7. **init_at_random_ep_len=False** — prevents stagnation false trigger at step 0

### V43k — Knee Crawl Fix (failed)

- Added `w_base_height=5.0` to encourage standing → created "stand still" exploit
- Robot discovered standing still gives net positive reward (+base_height - stagnation)

### V42 → V43 — Match Paper Exactly

- V42 "stand still" exploit: 8s episode + 0.8m goal = no need to walk
- Removed all non-paper rewards (bias_goal, anti_stall, etc.)
- link_contact_forces threshold: 490N (body weight), not 1N

---

## Important Lessons

- **replicate_physics=True is mandatory** with ROUGH_TERRAINS_CFG: only 200 tiles but 2048+ envs → multiple robots per tile collide without it
- **Termination thresholds need careful tuning**: too aggressive → false positives → episodes too short → no learning
- **terrain_origin_z is not a reliable height reference**: rough terrain has ±0.3m+ local variation
- **ANYmal-D HFE/KFE joints have ±540° limits** (effectively unlimited) — only HAA has real limits (±35-45°)
- **Don't simplify paper configuration**: episode length, goal distance, PPO params all matter
- **Positive rewards create exploit risk**: `w_base_height > 0` makes standing still more profitable than walking
- **RTX 3090 limit**: 2048 envs max (4096 causes PhysX OOM)
- **carb Mutex crash**: Omniverse crashes every 20-45 min, must use checkpoint resume
- **PYTHONUNBUFFERED=1**: required for nohup log output (otherwise 64KB buffer delay)

---

## Dependencies

| Dependency | Purpose |
|-----------|---------|
| `torch >= 2.0` | All network computations |
| `tensordict >= 0.3` | Observation TensorDict assembly |
| **Isaac Sim 5.0** | Physics simulation |
| **Isaac Lab 0.46.x** | DirectRLEnv, terrain, actuators, sensors |
| **RSL-RL** (`isaaclab_rl`) | PPO runner |

`ame2.networks` is fully importable without Isaac Sim.

---

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{zhang2025ame2,
  title   = {{AME-2}: Agile and Generalized Legged Locomotion via
              Attention-Based Neural Map Encoding},
  author  = {Zhang, Chong and Klemm, Victor and Yang, Fan and Hutter, Marco},
  year    = {2025},
  url     = {https://arxiv.org/abs/2601.08485}
}
```

---

## Disclaimer

This repository is an **independent reimplementation** created for research and learning purposes.
It is not affiliated with, endorsed by, or the official release of the ETH Zurich RSL authors.
Results may differ from those reported in the original paper.

---

## License

Apache-2.0
