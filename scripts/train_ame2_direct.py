"""AME-2 Teacher Training — True Distributed.

Uses RSL-RL's built-in distributed training (gradient sync via all_reduce)
instead of manual parameter averaging.

Usage:
    LD_PRELOAD=/lib/x86_64-linux-gnu/libnccl.so.2 torchrun --nproc_per_node=8 \
        scripts/train_ame2_direct.py --num_envs 1200 --seed 42 \
        --log_dir logs/distributed --headless
"""

from __future__ import annotations

import argparse
import os
import sys

# ── Isaac Sim MUST be imported before anything else ──────────────────────────
import isaacsim  # noqa: F401

from isaaclab.app import AppLauncher

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="AME-2 Teacher Training (True Distributed)")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num_envs",         type=int,   default=2048,
                    help="Number of parallel environments per GPU")
parser.add_argument("--max_iterations",   type=int,   default=80_000,
                    help="Total PPO training iterations")
parser.add_argument("--seed",             type=int,   default=42)
parser.add_argument("--log_dir",          type=str,   default="logs_v43")
parser.add_argument("--resume",           type=str,   default=None,
                    help="Checkpoint .pt path to resume from")
# ── Reward weight overrides (all w_* from config, auto-registered) ──
for _wn in [
    "w_position_tracking", "w_arrival", "w_heading_tracking",
    "w_moving_to_goal", "w_standing_at_goal",
    "w_bias_goal", "w_anti_stall", "w_upward",
    "w_goal_coarse", "w_goal_fine",
    "w_vel_toward_goal", "w_position_approach",
    "w_base_height", "w_feet_air_time", "w_anti_stagnation", "w_lin_vel_z_l2",
    "w_early_termination", "w_undesired_contacts",
    "w_ang_vel_xy_l2", "w_joint_reg_l2", "w_action_rate_l2",
    "w_link_contact_forces", "w_link_acceleration",
    "w_joint_pos_limits", "w_joint_vel_limits", "w_joint_torque_limits",
]:
    parser.add_argument(f"--{_wn}", type=float, default=None)
args_cli, _ = parser.parse_known_args()

if not hasattr(args_cli, "headless"):
    args_cli.headless = True

# ── Multi-GPU: each rank sees only its own GPU via CUDA_VISIBLE_DEVICES ──
_world_size = int(os.environ.get("WORLD_SIZE", "1"))
_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
_rank = int(os.environ.get("RANK", "0"))
_is_distributed = _world_size > 1
if _is_distributed:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_local_rank)

print(f"[Rank {_rank}] Creating AppLauncher (CVD={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')})...", flush=True)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print(f"[Rank {_rank}] AppLauncher ready", flush=True)

# ── Post-launch imports ───────────────────────────────────────────────────────
import os
import re
import time
import torch
import torch.distributed as dist

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
from rsl_rl.runners import OnPolicyRunner

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ame2_direct import AME2DirectEnv, AME2DirectEnvCfg, AME2DirectWrapper  # noqa: E402

try:
    from ame2.networks.rslrl_wrapper import AME2ActorCritic
    from ame2.networks.ame2_model import anymal_d_policy_cfg
except ImportError:
    from robot_lab.tasks.manager_based.locomotion.ame2.networks.rslrl_wrapper import AME2ActorCritic
    from robot_lab.tasks.manager_based.locomotion.ame2.networks.ame2_model import anymal_d_policy_cfg


# ── PPO config (Paper Table VI) ──────────────────────────────────────────────

def make_runner_cfg(seed: int, log_dir: str, device: str) -> dict:
    """PPO runner config — Paper Appendix C, Table VI."""
    cfg = RslRlOnPolicyRunnerCfg(
        seed=seed,
        device=device,
        num_steps_per_env=24,            # Paper Table VI
        max_iterations=args_cli.max_iterations,
        save_interval=50,
        experiment_name="ame2_v46",
        empirical_normalization=False,
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            actor_hidden_dims=[256, 256, 128],
            critic_hidden_dims=[256, 256, 128],
            activation="elu",
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=2.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.004,          # Paper Table VI: 0.004→0.001 (decay handled by update_curricula)
            num_learning_epochs=4,       # Paper Table VI
            num_mini_batches=8,          # 8 mini-batches (stable training)
            learning_rate=1e-3,
            schedule="adaptive",
            gamma=0.99,                  # Paper Table VI
            lam=0.95,                    # Paper Table VI
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
    )
    return cfg.to_dict()


# ── Paper Curricula (Sec.IV-D.3) ────────────────────────────────────────────
_HEADING_RAMP_FRAC = 0.2
_NOISE_RAMP_FRAC   = 0.2
_ENTROPY_START = 0.004
_ENTROPY_END   = 0.001


def update_curricula(env_direct: AME2DirectWrapper, runner: OnPolicyRunner, it: int) -> None:
    """Paper curricula only — heading, noise, entropy decay."""
    max_it = args_cli.max_iterations

    heading_frac = min(1.0, it / max(1, int(_HEADING_RAMP_FRAC * max_it)))
    env_direct.set_heading_curriculum(heading_frac)

    noise_scale = min(1.0, it / max(1, int(_NOISE_RAMP_FRAC * max_it)))
    env_direct.set_scan_noise_scale(noise_scale)

    frac = min(1.0, it / max(1, max_it))
    entropy = _ENTROPY_START + (_ENTROPY_END - _ENTROPY_START) * frac
    runner.alg.entropy_coef = entropy


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1
    device = "cuda:0"  # Each rank sees only 1 GPU via CVD masking
    is_main = rank == 0

    torch.manual_seed(args_cli.seed + rank)

    # ── Env ──
    policy_cfg = anymal_d_policy_cfg()
    policy_cfg.d_prop_critic = 55
    env_cfg = AME2DirectEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    for attr_name in dir(env_cfg):
        if attr_name.startswith('w_') and isinstance(getattr(env_cfg, attr_name), (int, float)):
            val = getattr(args_cli, attr_name, None)
            if val is not None:
                setattr(env_cfg, attr_name, val)
                if is_main:
                    print(f"  [Override] {attr_name} = {val}")
    print(f"[Rank {rank}] Creating env ({args_cli.num_envs} envs)...", flush=True)
    env = AME2DirectEnv(cfg=env_cfg)
    print(f"[Rank {rank}] Env created", flush=True)

    rsl_env = AME2DirectWrapper(env, device=device)

    # ── AME-2 network ──
    ame2_net = AME2ActorCritic(
        obs=None,
        obs_groups={},
        num_actions=12,
        ame2_cfg=policy_cfg,
        is_student=False,
    ).to(device)

    # ── Runner with RSL-RL native distributed ──
    # We already called init_process_group, so we need to prevent the runner
    # from calling it again. We do this by temporarily patching LOCAL_RANK
    # to 0 (matching our CVD-masked cuda:0 device).
    _saved_local_rank = os.environ.get("LOCAL_RANK")
    if is_distributed:
        os.environ["LOCAL_RANK"] = "0"  # match cuda:0 after CVD masking

    os.makedirs(args_cli.log_dir, exist_ok=True)
    runner_cfg = make_runner_cfg(args_cli.seed, args_cli.log_dir, device)
    runner = OnPolicyRunner(rsl_env, runner_cfg, args_cli.log_dir, device=device)

    # Restore LOCAL_RANK
    if _saved_local_rank is not None:
        os.environ["LOCAL_RANK"] = _saved_local_rank

    # Replace runner's policy and optimizer with our AME-2 network
    runner.alg.policy = ame2_net
    runner.alg.optimizer = torch.optim.Adam(ame2_net.parameters(), lr=1e-3)

    # Verify the runner detected distributed mode
    if is_main:
        print(f"  Runner distributed: {runner.is_distributed}, "
              f"world_size: {runner.gpu_world_size}, "
              f"multi_gpu_cfg: {runner.multi_gpu_cfg is not None}")

    # ── Resume ──
    it_start = 0
    if args_cli.resume:
        if is_main:
            print(f"[Resume] {args_cli.resume}")
        ckpt = torch.load(args_cli.resume, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        ame2_net.load_state_dict(state, strict=False)
        with torch.no_grad():
            ame2_net.std.fill_(0.5)
        m = re.search(r"model_(\d+)\.pt", args_cli.resume)
        if m:
            it_start = int(m.group(1)) + 1
            if is_main:
                print(f"[Resume] from iteration {it_start}")

    # ── Print config (rank 0 only) ──
    if is_main:
        total_envs = args_cli.num_envs * world_size
        print(f"\n[AME-2 Teacher — True Distributed]")
        print(f"  GPUs={world_size}  envs/GPU={args_cli.num_envs}  total_envs={total_envs}  "
              f"iters={args_cli.max_iterations}  seed={args_cli.seed}")
        print(f"  episode={env_cfg.episode_length_s}s  "
              f"goal=[{env_cfg.goal_pos_range_min}, {env_cfg.goal_pos_range_max}]m  "
              f"v_min={env_cfg.moving_to_goal_v_min}")
        print(f"  PPO: epochs=4  mini_batches=8  entropy=0.004→0.001")
        print(f"  Distributed: gradient sync via all_reduce (RSL-RL native)")
        print(f"  Rewards (pre-dt): pos={env_cfg.w_position_tracking:.0f} arrival={env_cfg.w_arrival:.0f} "
              f"vel={env_cfg.w_vel_toward_goal:.0f} move={env_cfg.w_moving_to_goal:.0f} "
              f"appr={env_cfg.w_position_approach:.0f} undes={env_cfg.w_undesired_contacts:.0f}")
        print(f"  Log: {args_cli.log_dir}\n")

    # ── W&B logging (rank 0 only) ──
    if is_main:
        import wandb
        wandb.init(
            project="ame2-locomotion",
            name="anymal-d-teacher-8xH100-truedist",
            config={
                "robot": "ANYmal-D",
                "phase": "teacher",
                "gpus": world_size,
                "gpu_type": "H100-80GB",
                "envs_per_gpu": args_cli.num_envs,
                "total_envs": args_cli.num_envs * world_size,
                "mini_batches": 8,
                "learning_epochs": 4,
                "steps_per_env": 24,
                "max_iterations": args_cli.max_iterations,
                "seed": args_cli.seed,
                "distributed": "true_gradient_sync",
            },
            resume="allow",
        )

    # ── Training loop ──
    # Use the runner's native learn() which handles:
    # - rollout collection
    # - advantage computation
    # - PPO update with gradient all_reduce across GPUs
    # - logging and checkpointing (rank 0 only)
    runner.current_learning_iteration = it_start
    t0 = time.time()

    for it in range(it_start, args_cli.max_iterations):
        runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)

        update_curricula(rsl_env, runner, it)

        # Save checkpoint every 50 iters (rank 0 only)
        if it % 50 == 0 and is_main:
            ckpt_path = os.path.join(args_cli.log_dir, f"model_{it}.pt")
            torch.save({
                "model_state_dict": ame2_net.state_dict(),
                "iteration": it,
            }, ckpt_path)

        # Custom logging every 50 iters (rank 0 only)
        if it % 50 == 0 and is_main:
            elapsed = time.time() - t0
            it_time = elapsed / max(it - it_start + 1, 1)
            eta_h   = it_time * (args_cli.max_iterations - it) / 3600

            ep_log  = runner.env.extras.get("log", {})
            pos_t   = float(ep_log.get("Episode_Reward/position_tracking", 0.0))
            move    = float(ep_log.get("Episode_Reward/moving_to_goal", 0.0))
            stand   = float(ep_log.get("Episode_Reward/standing_at_goal", 0.0))
            head    = float(ep_log.get("Episode_Reward/heading_tracking", 0.0))
            appr    = float(ep_log.get("Episode_Reward/position_approach", 0.0))
            vtg     = float(ep_log.get("Episode_Reward/vel_toward_goal", 0.0))
            dxy     = float(ep_log.get("Episode_Goal/terminal_dxy_mean", 0.0))
            succ05  = float(ep_log.get("Episode_Success/pos_0.50m", 0.0))
            succ100 = float(ep_log.get("Episode_Success/pos_1.00m", 0.0))

            print(
                f"[it {it:6d}] "
                f"terrain={rsl_env.get_terrain_level():.2f}  "
                f"dxy={dxy:.2f}m  succ@0.5={succ05:.2f}  succ@1.0={succ100:.2f}  "
                f"pos_t={pos_t:.3f}  move={move:.3f}  stand={stand:.3f}  head={head:.3f}  "
                f"appr={appr:.3f}  vtg={vtg:.3f}  "
                f"iter={it_time:.1f}s  ETA={eta_h:.1f}h"
            )
            wandb.log({
                "terrain_level": rsl_env.get_terrain_level(),
                "terminal_dxy": dxy,
                "success_0.5m": succ05,
                "success_1.0m": succ100,
                "position_tracking": pos_t,
                "moving_to_goal": move,
                "standing_at_goal": stand,
                "heading_tracking": head,
                "position_approach": appr,
                "vel_toward_goal": vtg,
                "iter_time_s": it_time,
                "eta_hours": eta_h,
            }, step=it)

    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()
    if is_main:
        wandb.finish()
        print("[AME-2] Done!")
    simulation_app.close()


if __name__ == "__main__":
    main()
