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
    # -- Fallen start (V41) --
    fallen_start_ratio:  float = 0.5    # fraction of envs starting fallen
    fallen_roll_range:   tuple = (-3.14, 3.14)  # full roll randomization
    fallen_pitch_range:  tuple = (-0.5, 0.5)    # moderate pitch

    goal_pos_range_init: float = 0.8   # reduced from 1.5: easier early goal-reaching signal
    goal_pos_range_max:  float = 5.0

    # -- Reward Weights (V39) -------------------------------------------------
    # Navigation rewards
    w_goal_coarse:          float = 1.5       # coarse distance-to-goal signal
    w_goal_fine:            float = 5.0       # fine near-goal signal (< 0.6m)
    w_position_tracking:    float = 2.0       # Eq.(1) last 4s terminal tracking
    w_heading_tracking:     float = 1.0       # Eq.(3) heading at goal
    w_moving_to_goal:       float = 1.0       # Eq.(4) binary walk signal
    w_vel_toward_goal:      float = 15.0       # directional velocity incentive
    w_standing_at_goal:     float = 0.1       # Eq.(5) stand still at goal
    # Stability rewards
    w_upward:               float = 1.0       # robot_lab (1-g_z)^2: upright=4, fallen=0
    w_base_height:          float = 0.0       # disabled
    # Gait shaping penalties
    w_lin_vel_z_l2:         float = -2.0      # penalize vertical bouncing
    w_ang_vel_xy_l2:        float = -0.05     # penalize body pitch/roll rate
    w_action_rate_l2:       float = -0.0005    # penalize action changes
    w_joint_reg_l2:         float = -0.001    # penalize joint deviation from default
    w_undesired_contacts:   float = -0.02     # 7 bad behaviors (slip/stumble/spin)
    w_link_contact_forces:  float = 0.0       # disabled
    w_link_acceleration:    float = -0.00002  # penalize body flailing
    w_joint_pos_limits:     float = -1.0      # hard joint position limits
    w_joint_vel_limits:     float = -0.02     # joint overspeed
    w_joint_torque_limits:  float = -0.02     # torque abuse
    # Termination
    w_early_termination:    float = -2.0      # penalty on terminated episodes
    # Disabled (kept for backward compat, weight=0)
    w_feet_air_time:        float = 0.0       # disabled: was penalizing normal walking
    w_position_approach:    float = 0.0       # disabled: replaced by goal_coarse
    w_anti_stagnation:      float = 0.0       # disabled: was too dominant early training
