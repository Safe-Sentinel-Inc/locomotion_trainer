"""AME-2 Phase 1 Teacher Training — V42 Unified Plan.

3-5x faster than manager-based env (no Python manager overhead).
Based on anymal_c_env.py Direct Workflow pattern + IsaacLab RSL-RL train.py pattern.

V42: AME-2 command + 2209 bias/stall + Risky Terrains curriculum
     Phase 1 bootstrap: no fallen start, no upright reward.

Usage (single GPU):
    /path/to/python scripts/train_ame2_direct.py --num_envs 2048 --headless

Usage (multi-GPU, different seeds):
    CUDA_VISIBLE_DEVICES=0 /path/to/python scripts/train_ame2_direct.py \
        --num_envs 2048 --seed 42 --log_dir logs_v42/gpu0 --headless &
    CUDA_VISIBLE_DEVICES=3 /path/to/python scripts/train_ame2_direct.py \
        --num_envs 2048 --seed 43 --log_dir logs_v42/gpu3 --headless &
"""

from __future__ import annotations

import argparse
import sys

# ── Isaac Sim MUST be imported before anything else ──────────────────────────
import isaacsim  # noqa: F401 — initializes the Isaac Sim app

from isaaclab.app import AppLauncher

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="AME-2 V42 Teacher Training (Phase 1)")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num_envs",         type=int,   default=2048,
                    help="Number of parallel environments")
parser.add_argument("--max_iterations",   type=int,   default=80_000,
                    help="Total PPO training iterations [stated: 80k]")
parser.add_argument("--num_mini_batches", type=int,   default=48,
                    help="Mini-batches per PPO update")
parser.add_argument("--seed",             type=int,   default=42)
parser.add_argument("--log_dir",          type=str,   default="logs_v42")
parser.add_argument("--resume",           type=str,   default=None,
                    help="Checkpoint .pt path to resume from")
args_cli, _ = parser.parse_known_args()

# Force headless unless explicitly disabled
if not hasattr(args_cli, "headless"):
    args_cli.headless = True

# Launch Isaac Sim app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Post-launch imports ───────────────────────────────────────────────────────
import os
import time
import torch

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
from rsl_rl.runners import OnPolicyRunner

# AME-2 modules
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ame2_direct import AME2DirectEnv, AME2DirectEnvCfg, AME2DirectWrapper  # noqa: E402

# AME-2 network + policy config
try:
    from ame2.networks.rslrl_wrapper import AME2ActorCritic
    from ame2.networks.ame2_model import anymal_d_policy_cfg
except ImportError:
    from robot_lab.tasks.manager_based.locomotion.ame2.networks.rslrl_wrapper import AME2ActorCritic
    from robot_lab.tasks.manager_based.locomotion.ame2.networks.ame2_model import anymal_d_policy_cfg


# ── PPO config ────────────────────────────────────────────────────────────────

def make_runner_cfg(seed: int, num_mini_batches: int, log_dir: str, device: str) -> dict:
    """PPO runner config matching AME-2 Appendix C [stated]."""
    cfg = RslRlOnPolicyRunnerCfg(
        seed=seed,
        device=device,
        num_steps_per_env=24,            # stated Table VI
        max_iterations=args_cli.max_iterations,
        save_interval=999999,  # disabled: outer loop handles checkpointing
        experiment_name="ame2_v42_phase1",
        empirical_normalization=False,
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            actor_hidden_dims=[256, 256, 128],
            critic_hidden_dims=[256, 256, 128],
            activation="elu",
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=2,
            num_mini_batches=num_mini_batches,
            learning_rate=1e-3,
            schedule="adaptive",
            gamma=0.99,                  # stated Table VI
            lam=0.95,                    # stated Table VI
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        obs_groups={
            "policy": ["prop"],
            "critic": ["prop"],
        },
    )
    return cfg.to_dict()


# ── V42 Curriculum State ─────────────────────────────────────────────────────

# Goal radius curriculum levels (expand only when success is stable)
_GOAL_R_LEVELS = [0.8, 1.2, 1.8, 2.5, 3.5, 5.0]
_goal_r_idx     = [0]           # current level index
_goal_radius    = [0.8]         # current goal radius

# Success tracking for goal radius expansion
_success_ema    = [0.0]         # EMA of success rate (position_tracking > threshold)
_pos_track_ema  = [0.0]         # EMA of position_tracking reward

# bias_goal decay state
_bias_active    = [True]
_BIAS_DECAY_ITER = 3000         # decay bias_goal after this many iterations

# v_min curriculum
_V_MIN_BOOTSTRAP = 0.1          # easy threshold for first 3000 iters
_V_MIN_FULL      = 0.3          # paper value after bootstrap
_V_MIN_SWITCH_ITER = 3000

# Heading/noise curriculum
_HEADING_RAMP_FRAC = 0.2        # ramp heading curriculum over 20% of training
_NOISE_RAMP_FRAC   = 0.2        # ramp perception noise over 20% of training

# Goal radius expansion criteria
_GOAL_R_COOLDOWN     = 400      # min iters between expansions
_last_expand_it      = [-400]   # allow first expansion after warmup
_SUCCESS_EMA_THRESH  = 0.6      # require success EMA > 0.6 to expand
_POS_TRACK_RISING    = [0.0]    # previous pos_track_ema for rising check


def update_curricula(env_direct: AME2DirectWrapper, runner: OnPolicyRunner, it: int) -> None:
    """V42 unified curriculum — all scheduling in one place."""
    env = env_direct._env  # underlying AME2DirectEnv
    cfg = env.cfg
    max_it = args_cli.max_iterations

    # ── 1. Heading curriculum: 0→1 over first 20% iters ──
    heading_frac = min(1.0, it / max(1, int(_HEADING_RAMP_FRAC * max_it)))
    env_direct.set_heading_curriculum(heading_frac)

    # ── 2. Perception noise: 0→max over first 20% iters ──
    noise_scale = min(1.0, it / max(1, int(_NOISE_RAMP_FRAC * max_it)))
    env_direct.set_scan_noise_scale(noise_scale)

    # ── 3. moving_to_goal v_min: 0.1 → 0.3 at 3000 iter ──
    if it < _V_MIN_SWITCH_ITER:
        cfg.moving_to_goal_v_min = _V_MIN_BOOTSTRAP
    else:
        cfg.moving_to_goal_v_min = _V_MIN_FULL

    # ── 4. bias_goal decay: when success EMA > 0.5 or after 3000 iters ──
    ep_log = runner.env.extras.get("log", {})
    pos_track = float(ep_log.get("Episode_Reward/position_tracking", 0.0))
    _pos_track_ema[0] = 0.05 * pos_track + 0.95 * _pos_track_ema[0]

    # Success = position_tracking EMA is significant (robot reaching goals)
    _success_ema[0] = 0.05 * (1.0 if pos_track > 0.5 else 0.0) + 0.95 * _success_ema[0]

    if _bias_active[0]:
        if _success_ema[0] > 0.5 or it >= _BIAS_DECAY_ITER:
            # Linearly decay bias_goal weight to 0 over 500 iters
            decay_start = max(it, _BIAS_DECAY_ITER)
            decay_progress = min(1.0, (it - decay_start + 500) / 500)
            # Note: cfg weights are already dt-scaled, so we need the dt-scaled version
            raw_bias = 3.0 * env.step_dt  # original dt-scaled weight
            cfg.w_bias_goal = raw_bias * max(0.0, 1.0 - decay_progress)
            if cfg.w_bias_goal <= 0.0:
                _bias_active[0] = False
                cfg.w_bias_goal = 0.0
                print(f"[V42 Curriculum] it={it}: bias_goal decayed to 0")

    # ── 5. Goal radius curriculum: expand when success is stable ──
    env_direct.set_goal_radius(_goal_radius[0])
    cooldown_ok = (it - _last_expand_it[0]) >= _GOAL_R_COOLDOWN
    pos_rising = _pos_track_ema[0] > _POS_TRACK_RISING[0] - 0.01  # not declining
    _POS_TRACK_RISING[0] = _pos_track_ema[0]

    if (_success_ema[0] > _SUCCESS_EMA_THRESH
            and pos_rising
            and cooldown_ok
            and _goal_r_idx[0] < len(_GOAL_R_LEVELS) - 1):
        _goal_r_idx[0] += 1
        _goal_radius[0] = _GOAL_R_LEVELS[_goal_r_idx[0]]
        _last_expand_it[0] = it
        env_direct.set_goal_radius(_goal_radius[0])
        print(f"[V42 GoalCurr] it={it}: goal_radius→{_goal_radius[0]:.1f}m  "
              f"success_ema={_success_ema[0]:.2f}  pos_track_ema={_pos_track_ema[0]:.3f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(args_cli.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Env ──
    policy_cfg = anymal_d_policy_cfg()     # ANYmal-D PolicyConfig (map_h=14, map_w=36)
    policy_cfg.d_prop_critic = 55          # Extended critic: +5D nav signals
    env_cfg = AME2DirectEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    _goal_radius[0] = env_cfg.goal_pos_range_init
    env = AME2DirectEnv(cfg=env_cfg)

    # ── Wrapper: provides TensorDict obs + 5-tuple step() for RSL-RL runner ──
    rsl_env = AME2DirectWrapper(env, device=device)

    # ── AME-2 network ──
    ame2_net = AME2ActorCritic(
        obs=None,
        obs_groups={},
        num_actions=12,
        ame2_cfg=policy_cfg,
        is_student=False,
    ).to(device)

    # ── Runner ──
    os.makedirs(args_cli.log_dir, exist_ok=True)
    runner_cfg = make_runner_cfg(args_cli.seed, args_cli.num_mini_batches,
                                 args_cli.log_dir, device)
    runner = OnPolicyRunner(rsl_env, runner_cfg, args_cli.log_dir, device=device)
    runner.alg.policy = ame2_net
    runner.alg.optimizer = torch.optim.Adam(ame2_net.parameters(), lr=1e-3)

    # ── Resume ──
    it_start = 0
    if args_cli.resume:
        print(f"[Resume] {args_cli.resume}")
        ckpt = torch.load(args_cli.resume, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        ame2_net.load_state_dict(state, strict=False)
        if "goal_radius" in ckpt:
            _goal_radius[0] = float(ckpt["goal_radius"])
            # Find matching level index
            for i, r in enumerate(_GOAL_R_LEVELS):
                if r >= _goal_radius[0] - 0.01:
                    _goal_r_idx[0] = i
                    break
            print(f"[Resume] goal_radius={_goal_radius[0]:.1f}m (level {_goal_r_idx[0]})")
        if "success_ema" in ckpt:
            _success_ema[0] = float(ckpt["success_ema"])
        if "pos_track_ema" in ckpt:
            _pos_track_ema[0] = float(ckpt["pos_track_ema"])
        if "bias_active" in ckpt:
            _bias_active[0] = bool(ckpt["bias_active"])
        with torch.no_grad():
            ame2_net.std.fill_(0.5)
        import re
        m = re.search(r"model_(\d+)\.pt", args_cli.resume)
        if m:
            it_start = int(m.group(1)) + 1
            print(f"[Resume] from iteration {it_start}")

    # ── Print config ──
    print(f"\n[AME-2 V42] env={args_cli.num_envs}, iters={args_cli.max_iterations}, "
          f"seed={args_cli.seed}, device={device}")
    print(f"  episode_length={env_cfg.episode_length_s}s  fallen_start={env_cfg.fallen_start_ratio}")
    print(f"  Main rewards: pos_track={100.0} head_track={50.0} move={5.0} stand={5.0} bias={3.0} stall={2.5}")
    print(f"  Log: {args_cli.log_dir}\n")

    t0 = time.time()

    for it in range(it_start, args_cli.max_iterations):
        runner.learn(num_learning_iterations=1, init_at_random_ep_len=(it == it_start))
        update_curricula(rsl_env, runner, it)

        # Save checkpoint every 25 iters
        if it % 25 == 0:
            ckpt_path = os.path.join(args_cli.log_dir, f"model_{it}.pt")
            torch.save({
                "model_state_dict": ame2_net.state_dict(),
                "goal_radius": _goal_radius[0],
                "goal_r_idx": _goal_r_idx[0],
                "success_ema": _success_ema[0],
                "pos_track_ema": _pos_track_ema[0],
                "bias_active": _bias_active[0],
            }, ckpt_path)

        # Logging every 50 iters
        if it % 50 == 0:
            elapsed = time.time() - t0
            it_time = elapsed / max(it - it_start + 1, 1)
            eta_h   = it_time * (args_cli.max_iterations - it) / 3600

            ep_log  = runner.env.extras.get("log", {})
            pos_t   = float(ep_log.get("Episode_Reward/position_tracking", 0.0))
            move    = float(ep_log.get("Episode_Reward/moving_to_goal", 0.0))
            bias    = float(ep_log.get("Episode_Reward/bias_goal", 0.0))
            stall   = float(ep_log.get("Episode_Reward/anti_stall", 0.0))
            head    = float(ep_log.get("Episode_Reward/heading_tracking", 0.0))

            print(
                f"[it {it:6d}] "
                f"terrain={rsl_env.get_terrain_level():.2f}  "
                f"goal_r={_goal_radius[0]:.1f}m  "
                f"pos_t={pos_t:.3f}  move={move:.3f}  bias={bias:.3f}  "
                f"stall={stall:.3f}  head={head:.3f}  "
                f"succ_ema={_success_ema[0]:.2f}  v_min={env.cfg.moving_to_goal_v_min:.1f}  "
                f"iter={it_time:.1f}s  ETA={eta_h:.1f}h"
            )

    print("[AME-2 V42] Done!")
    simulation_app.close()


if __name__ == "__main__":
    main()
