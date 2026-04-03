"""AME-2 Direct Env — Play / Eval Script (headless + optional video).

Usage (text eval, GPU1, no interrupt to training):
    CUDA_VISIBLE_DEVICES=1 python scripts/play_direct.py \
        --checkpoint logs_direct_v4/model_0.pt \
        --num_envs 1 --num_steps 500 --headless

Usage (video, GPU1):
    CUDA_VISIBLE_DEVICES=1 python scripts/play_direct.py \
        --checkpoint logs_direct_v4/model_0.pt \
        --num_envs 1 --num_steps 500 --video --headless
"""
from __future__ import annotations

import argparse
import sys

import isaacsim  # noqa: F401

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs",   type=int, default=1)
parser.add_argument("--num_steps",  type=int, default=500)
parser.add_argument("--video",      action="store_true")
parser.add_argument("--video_length", type=int, default=500)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── post-launch imports ───────────────────────────────────────────────────────
import os
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ame2_direct import AME2DirectEnv, AME2DirectEnvCfg, AME2DirectWrapper

try:
    from ame2.networks.rslrl_wrapper import AME2ActorCritic
    from ame2.networks.ame2_model import anymal_d_policy_cfg
except ImportError:
    from robot_lab.tasks.manager_based.locomotion.ame2.networks.rslrl_wrapper import AME2ActorCritic
    from robot_lab.tasks.manager_based.locomotion.ame2.networks.ame2_model import anymal_d_policy_cfg

# ── video wrapper (optional) ──────────────────────────────────────────────────
def maybe_wrap_video(env_direct, num_steps):
    if not args_cli.video:
        return env_direct
    try:
        import gymnasium as gym
        from isaaclab.envs.utils.wrappers.gymnasium import IsaacLabGymWrapper
        # Use Isaac Lab's built-in SkipResetWrapper + RecordVideo pattern
        video_dir = os.path.join(_ROOT, "videos")
        os.makedirs(video_dir, exist_ok=True)
        print(f"[Video] Saving to {video_dir}")
    except Exception as e:
        print(f"[Video] Could not set up video wrapper: {e}. Running text-only.")
    return env_direct


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Env ──────────────────────────────────────────────────────────────────
    policy_cfg = anymal_d_policy_cfg()
    policy_cfg.d_prop_critic = 55

    env_cfg = AME2DirectEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.terrain.max_init_terrain_level = 0  # flat terrain for eval
    env_cfg.episode_length_s = 20.0

    env = AME2DirectEnv(cfg=env_cfg)
    wrapper = AME2DirectWrapper(env, device=device)

    # ── Network ──────────────────────────────────────────────────────────────
    net = AME2ActorCritic(
        obs=None, obs_groups={}, num_actions=12,
        ame2_cfg=policy_cfg, is_student=False,
    ).to(device)

    ckpt = torch.load(args_cli.checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    net.load_state_dict(state, strict=False)
    net.eval()
    print(f"[Play] Loaded checkpoint: {args_cli.checkpoint}")

    # ── Run ──────────────────────────────────────────────────────────────────
    obs_td = wrapper.get_observations()

    print(f"\n{'Step':>6}  {'d_xy(m)':>8}  {'vel_fwd':>8}  {'rew':>8}  {'term':>6}")
    print("-" * 50)

    total_rew = 0.0
    for step in range(args_cli.num_steps):
        with torch.no_grad():
            actions = net.act_inference(obs_td)

        obs_td, rew, dones, info = wrapper.step(actions)
        total_rew += rew.mean().item()

        # Diagnostics
        goal_xy_b = env._get_goal_xy_body()
        d_xy      = torch.norm(goal_xy_b, dim=-1).mean().item()
        vel_xy    = env._robot.data.root_lin_vel_b[:, :2]
        to_goal   = goal_xy_b / (torch.norm(goal_xy_b, dim=-1, keepdim=True) + 1e-8)
        v_fwd     = (vel_xy * to_goal).sum(-1).mean().item()
        done_rate = dones.float().mean().item()

        if step % 20 == 0:
            print(f"{step:>6}  {d_xy:>8.3f}  {v_fwd:>8.3f}  {rew.mean().item():>8.3f}  {done_rate:>6.2f}")

        if simulation_app.is_running() is False:
            break

    print(f"\nTotal reward over {args_cli.num_steps} steps: {total_rew:.2f}")
    simulation_app.close()


if __name__ == "__main__":
    main()
