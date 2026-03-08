"""AME-2 Phase 1 Teacher Training — Isaac Lab Direct Workflow.

3-5x faster than manager-based env (no Python manager overhead).
Based on anymal_c_env.py Direct Workflow pattern + IsaacLab RSL-RL train.py pattern.

Usage (single GPU):
    /path/to/python scripts/train_ame2_direct.py --num_envs 4096 --headless

Usage (2 GPUs independently):
    CUDA_VISIBLE_DEVICES=0 /path/to/python scripts/train_ame2_direct.py \
        --num_envs 4096 --seed 42 --log_dir logs_direct/gpu0 --headless &
    CUDA_VISIBLE_DEVICES=1 /path/to/python scripts/train_ame2_direct.py \
        --num_envs 4096 --seed 43 --log_dir logs_direct/gpu1 --headless &

Paper: arXiv:2601.08485 — trained with Isaac Gym + RSL-RL PPO, 8× RTX 4090, ~60 GPU-days.
"""

from __future__ import annotations

import argparse
import sys

# ── Isaac Sim MUST be imported before anything else ──────────────────────────
import isaacsim  # noqa: F401 — initializes the Isaac Sim app

from isaaclab.app import AppLauncher

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="AME-2 Direct Env Teacher Training (Phase 1)")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num_envs",         type=int,   default=4096,
                    help="Number of parallel environments")
parser.add_argument("--max_iterations",   type=int,   default=80_000,
                    help="Total PPO training iterations [stated: 80k]")
parser.add_argument("--num_mini_batches", type=int,   default=96,  # batch≈512, fits 32GB GPU
                    help="Mini-batches per PPO update [stated Table VI: 4]")
parser.add_argument("--seed",             type=int,   default=42)
parser.add_argument("--log_dir",          type=str,   default="logs_direct")
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
        experiment_name="ame2_direct_phase1",
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
            entropy_coef=0.02,           # raised 0.004→0.02: prevent noise collapse at iter<500
            num_learning_epochs=2,       # reduced: 4→2 to limit gradient steps per rollout
            num_mini_batches=num_mini_batches,  # use 96+ to keep batch<512 → fits 32GB GPU
            learning_rate=1e-3,
            schedule="adaptive",
            gamma=0.99,                  # stated Table VI
            lam=0.95,                    # stated Table VI
            desired_kl=0.02,             # raised 0.01→0.02: slow LR adaptation, more exploration room
            max_grad_norm=1.0,
        ),
        # obs_groups required by rsl_rl runner; AME2ActorCritic uses full TensorDict directly
        obs_groups={
            "policy": ["prop"],          # default actor_critic constructed with 48D prop
            "critic": ["prop"],          # replaced post-hoc by AME2ActorCritic
        },
    )
    return cfg.to_dict()


# ── Curriculum state ──────────────────────────────────────────────────────────

_stag_ema        = [0.5]
_goal_radius     = [1.2]    # start at 1.2m (proven working distance)
_last_expand_it  = [-100]   # last iteration goal_r was expanded (rate limiter)
_GOAL_R_MIN      = 0.8
_GOAL_R_MAX      = 5.0
_GOAL_R_STEP     = 0.2      # +0.2m per expansion event
_GOAL_R_COOLDOWN = 400      # cooldown 200→400: slower expansion, more time to consolidate


def update_curricula(env_direct: AME2DirectWrapper, runner: OnPolicyRunner, it: int) -> None:
    """Update all curricula each iteration.

    Curriculum design:
      1. Heading:   face-goal → gradual random over 60% of training
                    (user request: learn walk first, then learn to turn)
      2. Goal dist: expand +0.2m only when stag_ema<0.25 AND 200-iter cooldown
                    (prevents rapid jump from stagnation noise)
      3. Perception noise: 0→max over first 20% iters [stated]
    """
    # Heading curriculum: ramp over 60% of training (not 20%) — give robot time to
    # learn goal-directed walking before it must also learn to turn from random yaw.
    heading_frac = min(1.0, it / max(1, int(0.6 * args_cli.max_iterations)))
    env_direct.set_scan_noise_scale(
        min(1.0, it / max(1, int(0.2 * args_cli.max_iterations)))
    )
    env_direct.set_heading_curriculum(heading_frac)

    # Update stagnation EMA
    ep_log  = runner.env.extras.get("log", {})
    stag    = float(ep_log.get("Episode_Termination/stagnation", 0.5))
    _stag_ema[0] = 0.05 * stag + 0.95 * _stag_ema[0]

    # Goal distance curriculum — expand only when robot is ACTUALLY reaching goals.
    # Bug fix: stag_ema<0.25 only means no stagnation TERMINATIONS (robot moves a tiny
    # bit to dodge the 5s/1m threshold), NOT that it's reaching goals.  Adding
    # goal_fine_avg>3.0 ensures the robot genuinely reaches the close zone before
    # we make goals harder.
    env_direct.set_goal_radius(_goal_radius[0])
    goal_fine_avg = float(ep_log.get("Episode_Reward/goal_fine", 0.0))
    cooldown_ok   = (it - _last_expand_it[0]) >= _GOAL_R_COOLDOWN
    if _stag_ema[0] < 0.25 and goal_fine_avg > 3.0 and _goal_radius[0] < _GOAL_R_MAX and cooldown_ok:
        _goal_radius[0] = min(_goal_radius[0] + _GOAL_R_STEP, _GOAL_R_MAX)
        _last_expand_it[0] = it
        print(f"[GoalCurr] it={it}: goal_radius→{_goal_radius[0]:.1f}m  goal_fine={goal_fine_avg:.1f}")

    # Entropy: keep at 0.02 (fixed, no decay) — decay to 0.004 was counteracting the
    # noise-collapse fix and being overridden every iter anyway.
    runner.alg.entropy_coef = 0.02

    # (lin_vel_tracking replaced by tanh position tracking in v14 — no decay needed)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(args_cli.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Env ──
    policy_cfg = anymal_d_policy_cfg()     # ANYmal-D PolicyConfig (map_h=14, map_w=36)
    policy_cfg.d_prop_critic = 55          # Extended critic: +5D nav signals (see env.py)
    env_cfg = AME2DirectEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = AME2DirectEnv(cfg=env_cfg)

    # ── Wrapper: provides TensorDict obs + 5-tuple step() for RSL-RL runner ──
    rsl_env = AME2DirectWrapper(env, device=device)

    # ── AME-2 network ──
    # obs=None is safe: AME2ActorCritic.__init__ does not inspect obs
    # obs_groups={}: runner routes TensorDict obs directly to actor_critic
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
    # Replace default MLP actor_critic with AME2ActorCritic (post-hoc, before any training)
    # NOTE: RSL-RL PPO stores the policy as self.policy (NOT self.actor_critic)
    runner.alg.policy = ame2_net
    runner.alg.optimizer = torch.optim.Adam(ame2_net.parameters(), lr=1e-3)

    # ── Resume ──
    if args_cli.resume:
        print(f"[Resume] {args_cli.resume}")
        ckpt = torch.load(args_cli.resume, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        ame2_net.load_state_dict(state, strict=False)
        # Reset action noise std to 1.0: if std collapsed (<0.7) before walking was learned,
        # the policy is stuck in a "stand-still" local optimum.  Restoring exploration lets
        # the policy rediscover locomotion before entropy regularization takes over.
        with torch.no_grad():
            ame2_net.std.fill_(1.0)
        print(f"[Resume] loaded ✓  (std reset to 1.0)")


    # ── Speed benchmark: print iter time before training ──
    print(f"\n[AME-2 Direct] env={args_cli.num_envs}, iters={args_cli.max_iterations}, "
          f"seed={args_cli.seed}, device={device}")
    print(f"  Log: {args_cli.log_dir}\n")

    t0 = time.time()

    # Resume: detect starting iteration from latest checkpoint
    it_start = 0
    if args_cli.resume:
        import re
        m = re.search(r"model_(\d+)\.pt", args_cli.resume)
        if m:
            it_start = int(m.group(1)) + 1
            print(f"[Resume] Resuming from iteration {it_start}")

    for it in range(it_start, args_cli.max_iterations):
        runner.learn(num_learning_iterations=1, init_at_random_ep_len=(it == it_start))
        update_curricula(rsl_env, runner, it)

        # Save checkpoint every 25 iters (outer loop manages, not RSL-RL internal)
        if it % 25 == 0:
            ckpt_path = os.path.join(args_cli.log_dir, f"model_{it}.pt")
            torch.save({"model_state_dict": ame2_net.state_dict()}, ckpt_path)

        if it % 50 == 0:
            elapsed   = time.time() - t0
            it_time   = elapsed / max(it - it_start + 1, 1)
            eta_days  = it_time * (args_cli.max_iterations - it) / 86400
            ep_log = runner.env.extras.get("log", {})
            goal_c = float(ep_log.get("Episode_Reward/goal_coarse", 0.0))
            goal_f = float(ep_log.get("Episode_Reward/goal_fine", 0.0))
            stag_r = float(ep_log.get("Episode_Termination/stagnation", 1.0))
            print(
                f"[it {it:6d}] "
                f"terrain={rsl_env.get_terrain_level():.2f}  "
                f"stag_ema={_stag_ema[0]:.3f}  "
                f"goal_r={_goal_radius[0]:.1f}m  "
                f"goal_c={goal_c:.1f}  goal_f={goal_f:.2f}  "
                f"stag_rate={stag_r:.2f}  "
                f"iter={it_time:.1f}s  ETA={eta_days:.1f}d"
            )

    print("[AME-2 Direct] Done!")
    simulation_app.close()


if __name__ == "__main__":
    main()
