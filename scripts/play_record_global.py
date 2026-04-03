"""AME-2 Direct Env — Global Bird's-Eye Video Recording.

Single camera overlooking all robots on shared terrain (like ATOM01 AMP style).

Usage:
    CUDA_VISIBLE_DEVICES=4 python scripts/play_record_global.py \
        --checkpoint logs_v43/gpu0/model_500.pt \
        --num_envs 16 --num_steps 500 --headless \
        --output v43_global.mp4
"""
from __future__ import annotations

import argparse
import sys

import isaacsim  # noqa: F401

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs",   type=int, default=16)
parser.add_argument("--num_steps",  type=int, default=500)
parser.add_argument("--output",     type=str, default="v43_global.mp4")
parser.add_argument("--width",      type=int, default=1920)
parser.add_argument("--height",     type=int, default=1080)
parser.add_argument("--cam_height", type=float, default=15.0, help="Camera height above terrain")
parser.add_argument("--cam_dist",   type=float, default=20.0, help="Camera horizontal distance")
parser.add_argument("--cam_angle",  type=float, default=55.0, help="Camera elevation angle (degrees)")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── post-launch imports ───────────────────────────────────────────────────────
import os
import math
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.utils import configclass

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


_CAM_IDENTITY = (1.0, 0.0, 0.0, 0.0)


def _set_camera_lookat(cam_prim_path: str, eye, target, up=(0, 0, 1)):
    """Position camera via USD Matrix4d lookat."""
    import omni.usd
    from pxr import UsdGeom, Gf

    eye    = Gf.Vec3d(*eye)
    target = Gf.Vec3d(*target)
    up     = Gf.Vec3d(*up)

    fwd   = (target - eye).GetNormalized()
    right = Gf.Cross(fwd, up).GetNormalized()
    new_up = Gf.Cross(right, fwd).GetNormalized()

    mat = Gf.Matrix4d()
    mat.SetRow(0, Gf.Vec4d(right[0],   right[1],   right[2],   0))
    mat.SetRow(1, Gf.Vec4d(new_up[0],  new_up[1],  new_up[2],  0))
    mat.SetRow(2, Gf.Vec4d(-fwd[0],   -fwd[1],    -fwd[2],     0))
    mat.SetRow(3, Gf.Vec4d(eye[0],     eye[1],     eye[2],      1))

    stage = omni.usd.get_context().get_stage()
    prim  = stage.GetPrimAtPath(cam_prim_path)
    if not prim.IsValid():
        print(f"[Camera] Prim not found: {cam_prim_path}")
        return
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(mat)
    print(f"[Camera] eye={tuple(eye)} → target={tuple(target)}")


def _add_distant_light():
    """Add a bright directional light."""
    import omni.usd
    from pxr import UsdLux, Gf, Sdf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    light_path = Sdf.Path("/World/RecordLight")
    if not stage.GetPrimAtPath(light_path).IsValid():
        light = UsdLux.DistantLight.Define(stage, light_path)
        light.CreateIntensityAttr(3000.0)
        light.CreateAngleAttr(1.0)
        xf = UsdGeom.Xformable(light.GetPrim())
        xf.ClearXformOpOrder()
        rot = Gf.Matrix4d()
        rot.SetRotate(Gf.Rotation(Gf.Vec3d(0, 1, 0), -45))
        rot2 = Gf.Matrix4d()
        rot2.SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), 45))
        xf.AddTransformOp().Set(rot * rot2)
        print("[Camera] Added distant light")


def _add_global_camera(width, height):
    """Create a standalone camera prim at /World/GlobalCamera."""
    import omni.usd
    from pxr import Sdf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    cam_path = "/World/GlobalCamera"
    if not stage.GetPrimAtPath(cam_path).IsValid():
        UsdGeom.Camera.Define(stage, Sdf.Path(cam_path))
    print(f"[Camera] Created global camera at {cam_path}")
    return cam_path


@configclass
class AME2GlobalRecordEnvCfg(AME2DirectEnvCfg):
    """Config with a single TiledCamera on env_0 (we'll reposition it globally)."""
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_0/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=_CAM_IDENTITY,
            convention="world",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=8.0,          # wider FOV for overview
            focus_distance=400.0,
            horizontal_aperture=36.0,
            clipping_range=(0.1, 500.0),
        ),
        width=args_cli.width,
        height=args_cli.height,
    )


class AME2DirectEnvGlobalRecord(AME2DirectEnv):
    """Env with single TiledCamera for global bird's-eye recording."""

    cfg: AME2GlobalRecordEnvCfg

    def _setup_scene(self):
        super()._setup_scene()
        self._camera = TiledCamera(self.cfg.tiled_camera)
        self.scene.sensors["tiled_camera"] = self._camera

    def _get_dones(self):
        terminated, truncated = super()._get_dones()
        terminated[:] = False  # no early resets for clean video
        return terminated, truncated

    def get_frame(self) -> np.ndarray | None:
        data = self._camera.data.output.get("rgb")
        if data is None:
            return None
        frame = data[0].cpu().numpy()  # only env_0's camera
        if frame.dtype != np.uint8:
            frame = (frame * 255).clip(0, 255).astype(np.uint8)
        return frame[:, :, :3].copy()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    policy_cfg = anymal_d_policy_cfg()
    policy_cfg.d_prop_critic = 55

    env_cfg = AME2GlobalRecordEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.terrain.max_init_terrain_level = 0  # flat terrain for clean video
    env_cfg.episode_length_s = args_cli.num_steps / 50.0 + 5.0

    # Override camera resolution
    env_cfg.tiled_camera = TiledCameraCfg(
        prim_path="/World/envs/env_0/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=_CAM_IDENTITY,
            convention="world",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=8.0,
            focus_distance=400.0,
            horizontal_aperture=36.0,
            clipping_range=(0.1, 500.0),
        ),
        width=args_cli.width,
        height=args_cli.height,
    )

    env = AME2DirectEnvGlobalRecord(cfg=env_cfg)
    wrapper = AME2DirectWrapper(env, device=device)

    # ── Network ──
    net = AME2ActorCritic(
        obs=None, obs_groups={}, num_actions=12,
        ame2_cfg=policy_cfg, is_student=False,
    ).to(device)

    ckpt = torch.load(args_cli.checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    net.load_state_dict(state, strict=False)
    net.eval()
    print(f"[Record] Loaded: {args_cli.checkpoint}")

    # ── Position global camera ──
    # Compute center of all env origins
    env_origins = env._terrain.env_origins[:args_cli.num_envs].cpu()
    center_x = float(env_origins[:, 0].mean())
    center_y = float(env_origins[:, 1].mean())
    center_z = float(env_origins[:, 2].mean())

    # Camera positioned at an angle looking at the center
    angle_rad = math.radians(args_cli.cam_angle)
    cam_h = args_cli.cam_height
    cam_d = args_cli.cam_dist
    eye = (center_x + cam_d * math.cos(angle_rad),
           center_y - cam_d * 0.3,
           center_z + cam_h)
    target = (center_x, center_y, center_z + 0.5)

    _set_camera_lookat("/World/envs/env_0/Camera", eye, target)
    _add_distant_light()

    print(f"[Record] {args_cli.num_envs} envs, center=({center_x:.1f}, {center_y:.1f}, {center_z:.1f})")
    print(f"[Record] Camera: eye={eye}, target={target}")
    print(f"[Record] Resolution: {args_cli.width}x{args_cli.height}")
    print(f"[Record] Recording {args_cli.num_steps} steps → {args_cli.output}")

    # ── Record loop ──
    obs_td = wrapper.get_observations()
    frames = []

    for step in range(args_cli.num_steps):
        with torch.no_grad():
            actions = net.act_inference(obs_td)
        obs_td, rew, dones, info = wrapper.step(actions)

        frame = env.get_frame()
        if frame is not None:
            frames.append(frame)

        if step % 100 == 0:
            goal_xy_b = env._get_goal_xy_body()
            d_xy = torch.norm(goal_xy_b, dim=-1).mean().item()
            print(f"  step {step:4d}  d_xy={d_xy:.2f}m  frames={len(frames)}")

        if not simulation_app.is_running():
            break

    # ── Save video ──
    if frames:
        import imageio
        out_path = os.path.join(_ROOT, args_cli.output)
        imageio.mimwrite(out_path, frames, fps=50, quality=8)
        print(f"[Record] Saved {len(frames)} frames → {out_path}")
    else:
        print("[Record] No frames captured.")

    simulation_app.close()


if __name__ == "__main__":
    main()
