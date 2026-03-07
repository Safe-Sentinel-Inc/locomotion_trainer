"""AME-2 Direct Environment Config.

Based on anymal_c_env_cfg.py pattern from IsaacLab 0.46.x:
  - observation_space / action_space are int (SpaceType)
  - state_space for critic
  - prim_path format: /World/envs/env_.*/Robot
"""
from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.sim.simulation_cfg import PhysxCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from isaaclab_assets.robots.anymal import ANYMAL_D_CFG, ANYDRIVE_3_SIMPLE_ACTUATOR_CFG  # isort: skip



@configclass
class AME2DirectEnvCfg(DirectRLEnvCfg):
    """AME-2 direct environment configuration (ANYmal-D).

    observation_space = 48   (teacher actor prop)           [stated]
    action_space      = 12   (joint position targets)       [stated]
    state_space       = 0    (asymmetric critic in network)
    """

    # ── Env ─────────────────────────────────────────────────────────────────
    episode_length_s:  float = 20.0
    decimation:        int   = 4        # 50 Hz control                [stated]
    action_scale:      float = 0.5      # joint targets = default + scale*action

    # SpaceType (int → flat Box)
    observation_space: int = 48         # teacher actor prop dim       [stated]
    action_space:      int = 12         # 12-DoF joint position targets
    state_space:       int = 0          # asymmetric critic handled in network
    # critic_prop: 45(base) + 5(critic_cmd) + 5(nav_extra) = 55D (extended from paper's 50D)

    # ── Simulation ───────────────────────────────────────────────────────────
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 200.0,                 # 200 Hz physics               [stated]
        render_interval=4,
        physx=PhysxCfg(
            gpu_max_rigid_patch_count=2**21,  # 2M patches — avoids overflow with 2048+ envs
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # ── Scene ────────────────────────────────────────────────────────────────
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4800,                  # [stated Table VI]
        env_spacing=3.0,
        replicate_physics=False,
    )

    # ── Terrain ──────────────────────────────────────────────────────────────
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=1,
        collision_group=-1,
        debug_vis=False,
    )

    # ── Robot ─────────────────────────────────────────────────────────────────
    robot: ArticulationCfg = ANYMAL_D_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        actuators={"legs": ANYDRIVE_3_SIMPLE_ACTUATOR_CFG},
    )

    # ── Contact Sensor ────────────────────────────────────────────────────────
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    # ── Height Scanners ───────────────────────────────────────────────────────
    # MappingNet scanner: 31×51 @ 4cm                                 [stated]
    height_scanner: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.04,
            size=[1.20, 2.00],  # (31-1)*0.04 × (51-1)*0.04
        ),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # GT policy map scanner: 14×36 @ 8cm                              [stated]
    height_scanner_policy: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.08,
            size=[1.04, 2.80],  # (14-1)*0.08 × (36-1)*0.08
        ),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # ── Goal Command ─────────────────────────────────────────────────────────
    goal_pos_range_init: float = 0.8   # reduced from 1.5: easier early goal-reaching signal
    goal_pos_range_max:  float = 5.0

    # ── Reward Weights ──────────────────────────────────────────────────────
    # v14: Isaac Lab Navigation-style tanh position tracking (Hoeller et al. IROS 2022)
    # Primary goal signal: 1-tanh(d/std) — non-zero everywhere, gradient up to 2*std
    w_goal_coarse:          float = 1.5       # continuous distance gradient: 1-tanh(d/2.0), gradient up to d=4m
    w_goal_fine:            float = 5.0       # 1-tanh(d/0.3): reward only when actually close (<0.5m)
    w_position_tracking:    float = 0.0       # phase 2+: only fires when at goal (d<0.5m)
    w_position_approach:    float = 0.0       # replaced by w_goal_coarse
    w_upright_bonus:        float = 0.3       # stay upright
    w_base_height:          float = 0.0       # disabled: upright_bonus already covers this
    w_feet_air_time:        float = 1.0       # stepping gait
    w_heading_tracking:     float = 0.0       # phase 2+: only fires d<0.5m
    w_moving_to_goal:       float = 0.0       # disabled
    w_vel_toward_goal:      float = 3.0       # PRIMARY signal: walk at 1m/s toward goal → +3.0/step
    w_lin_vel_tracking:     float = 0.0       # disabled
    w_anti_stagnation:      float = 0.5       # per-step: -1 when speed<0.2 AND d>0.5m
    w_standing_at_goal:     float = 0.0       # phase 2+: only fires d<0.5m
    w_early_termination:    float = -2.0      # small penalty — don't fear falling, explore!
    w_undesired_events:     float = 0.0       # disabled
    w_base_roll_rate:       float = 0.0       # disabled: natural during walking
    w_joint_regularization: float = -0.001     # keep joints near default; prevents chaotic flailing
    w_action_smoothness:    float = -0.005     # smooth action changes → stable gait bootstrap [AME-2 Table I]
    w_link_contact_forces:  float = 0.0       # disabled
    w_link_acceleration:    float = 0.0       # disabled
    w_joint_pos_limits:     float = -1.0      # keep: only true hard limit (joint damage)
    w_joint_vel_limits:     float = 0.0       # disabled: let robot move fast
    w_joint_torque_limits:  float = 0.0       # disabled: let robot be strong
