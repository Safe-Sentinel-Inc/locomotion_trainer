"""AME-2 Direct Env — Headless Video Recording Script.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/play_record.py \
        --checkpoint logs_direct_v4/model_475.pt \
        --num_steps 600 --headless
"""
from __future__ import annotations

import argparse
import sys

import isaacsim  # noqa: F401

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_steps",  type=int, default=600)
parser.add_argument("--output",     type=str, default="record_output.mp4")
parser.add_argument("--width",      type=int, default=1280)
parser.add_argument("--height",     type=int, default=720)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True   # required for TiledCamera

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── post-launch imports ───────────────────────────────────────────────────────
import os
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


# ── Camera-augmented env config ───────────────────────────────────────────────

# Identity quaternion — camera will be repositioned via USD lookat after spawn
_CAM_IDENTITY = (1.0, 0.0, 0.0, 0.0)

@configclass
class AME2RecordEnvCfg(AME2DirectEnvCfg):
    """Same as training config, but with TiledCamera added."""
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=_CAM_IDENTITY,
            convention="world",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            focus_distance=400.0,
            horizontal_aperture=36.0,    # wide FOV (~90°)
            clipping_range=(0.1, 200.0),
        ),
        width=args_cli.width,
        height=args_cli.height,
    )


def _set_camera_lookat(cam_prim_path: str, eye, target, up=(0, 0, 1)):
    """Position camera via USD Matrix4d lookat — bypasses quaternion conventions."""
    import omni.usd
    from pxr import UsdGeom, Gf

    eye    = Gf.Vec3d(*eye)
    target = Gf.Vec3d(*target)
    up     = Gf.Vec3d(*up)

    fwd   = (target - eye).GetNormalized()
    right = Gf.Cross(fwd, up).GetNormalized()
    new_up = Gf.Cross(right, fwd).GetNormalized()

    # USD camera: +X right, +Y up, -Z forward
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
    print(f"[Camera] Positioned at eye={tuple(eye)} → target={tuple(target)}")


def _add_distant_light():
    """Add a bright directional light so the scene isn't dark."""
    import omni.usd
    from pxr import UsdLux, Gf, Sdf

    stage = omni.usd.get_context().get_stage()
    light_path = Sdf.Path("/World/RecordLight")
    if not stage.GetPrimAtPath(light_path).IsValid():
        light = UsdLux.DistantLight.Define(stage, light_path)
        light.CreateIntensityAttr(3000.0)
        light.CreateAngleAttr(1.0)
        from pxr import UsdGeom
        xf = UsdGeom.Xformable(light.GetPrim())
        xf.ClearXformOpOrder()
        # Shine from upper-front-right
        import math
        rot = Gf.Matrix4d()
        rot.SetRotate(Gf.Rotation(Gf.Vec3d(0, 1, 0), -45))
        rot2 = Gf.Matrix4d()
        rot2.SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), 45))
        xf.AddTransformOp().Set(rot * rot2)
        print("[Camera] Added distant light")


class AME2DirectEnvRecord(AME2DirectEnv):
    """Adds TiledCamera sensor to the DirectRLEnv for video recording."""

    cfg: AME2RecordEnvCfg
    _camera_ready: bool = False

    def _setup_scene(self):
        super()._setup_scene()
        self._camera = TiledCamera(self.cfg.tiled_camera)
        self.scene.sensors["tiled_camera"] = self._camera

    def setup_camera_lookat(self, env_origin):
        """Call once after sim start to set exact camera position via USD."""
        ox, oy, oz = float(env_origin[0]), float(env_origin[1]), float(env_origin[2])
        eye    = (ox + 4.0, oy - 3.0, oz + 3.5)   # elevated rear-side
        target = (ox + 0.0, oy + 0.0, oz + 0.6)   # robot body height
        _set_camera_lookat("/World/envs/env_0/Camera", eye, target)
        _add_distant_light()
        self._camera_ready = True

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Disable ALL terminations for clean video recording.

        At early training stages the policy falls frequently → constant resets
        make the video unwatchable. Disable all resets; robot stays even if fallen.
        Episode ends only by episode timeout (episode_length_s).
        """
        terminated, truncated = super()._get_dones()
        terminated[:] = False   # no early resets — show full episode cleanly
        return terminated, truncated

    def get_camera_frames(self) -> np.ndarray | None:
        """Return RGB frame from camera as uint8 (H, W, 3) for env_0."""
        data = self._camera.data.output.get("rgb")
        if data is None:
            return None
        frame = data[0].cpu().numpy()
        if frame.dtype != np.uint8:
            frame = (frame * 255).clip(0, 255).astype(np.uint8)
        # TiledCamera (replicator) outputs top-down (row 0 = top of scene).
        # USD lookat matrix correctly orients camera → no flip needed.
        return frame[:, :, :3].copy()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    policy_cfg = anymal_d_policy_cfg()
    policy_cfg.d_prop_critic = 55

    env_cfg = AME2RecordEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.terrain.max_init_terrain_level = 0   # flat for cleaner video
    # Long episode = no truncation reset during recording
    env_cfg.episode_length_s = args_cli.num_steps / 50.0 + 5.0  # slightly longer than video
    env_cfg.goal_pos_range_init = 2.0  # 2m goal

    env = AME2DirectEnvRecord(cfg=env_cfg)
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
    print(f"[Record] Recording {args_cli.num_steps} steps → {args_cli.output}")

    # ── Record loop ──
    obs_td = wrapper.get_observations()

    # Position camera after first physics step (USD prims now exist)
    env_origin = env._terrain.env_origins[0].cpu()
    env.setup_camera_lookat(env_origin)

    frames = []

    for step in range(args_cli.num_steps):
        with torch.no_grad():
            actions = net.act_inference(obs_td)

        obs_td, rew, dones, info = wrapper.step(actions)

        frame = env.get_camera_frames()
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
        try:
            import imageio
            out_path = os.path.join(_ROOT, args_cli.output)
            imageio.mimwrite(out_path, frames, fps=50, quality=8)
            print(f"[Record] Saved {len(frames)} frames → {out_path}")
        except ImportError:
            # fallback: save as numpy array
            np_path = os.path.join(_ROOT, args_cli.output.replace(".mp4", ".npy"))
            np.save(np_path, np.stack(frames))
            print(f"[Record] imageio not found. Saved frames as {np_path}")
    else:
        print("[Record] No frames captured.")

    simulation_app.close()


if __name__ == "__main__":
    main()
