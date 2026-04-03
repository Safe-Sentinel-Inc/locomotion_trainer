"""AME-2 Direct Environment Config for PF_TRON1A (6-DOF biped).

Mirrors config.py (ANYmal-D) structure with TRON1-specific dimensions:
  - observation_space = 30   (teacher actor prop, 6 joints)
  - action_space = 6         (joint position targets)
  - Height scanner: 31x31 @ 4cm (MappingNet local grid)
  - Policy scanner: 13x18 @ 8cm (GT policy map)
  - Contact sensor: base + hip + knee + foot (7 links)
  - TRON1_TERRAIN_CFG (reduced difficulty)
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.sim.simulation_cfg import PhysxCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from ame2.terrains import TRON1_TERRAIN_CFG

from .tron1_asset import PF_TRON1A_CFG


@configclass
class TRON1DirectEnvCfg(DirectRLEnvCfg):
    """AME-2 direct environment configuration for PF_TRON1A (biped).

    observation_space = 30   (teacher actor prop: base_vel(3)+hist(24)+cmd(3))
    action_space      = 6    (6-DoF joint position targets)
    state_space       = 0    (asymmetric critic in network)
    """

    # ── Env ─────────────────────────────────────────────────────────────────
    episode_length_s:  float = 20.0
    decimation:        int   = 4       # 50 Hz control
    action_scale:      float = 0.25    # smaller for biped stability

    # SpaceType (int → flat Box)
    observation_space: int = 30        # teacher actor prop: 3+24+3
    action_space:      int = 6         # 6-DoF joint targets
    state_space:       int = 0

    # ── Simulation ───────────────────────────────────────────────────────────
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 200.0,                 # 200 Hz physics
        render_interval=4,
        physx=PhysxCfg(
            gpu_max_rigid_patch_count=2**21,
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
        num_envs=2048,
        env_spacing=4.0,
        replicate_physics=True,
    )

    # ── Terrain ──────────────────────────────────────────────────────────────
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TRON1_TERRAIN_CFG,
        max_init_terrain_level=1,
        collision_group=-1,
        debug_vis=False,
    )

    # ── Robot ─────────────────────────────────────────────────────────────────
    robot = PF_TRON1A_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )

    # ── Contact Sensor ────────────────────────────────────────────────────────
    # Matches all links: base + hip + knee + foot = 7 links
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    # ── Height Scanners ───────────────────────────────────────────────────────
    # MappingNet scanner: 31x31 @ 4cm, center x=0.6m (stated for TRON1)
    height_scanner: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base_Link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.6, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.04,
            size=[1.20, 1.20],  # (31-1)*0.04 x (31-1)*0.04
        ),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # GT policy map scanner: 13x18 @ 8cm, center x=0.32m (stated for TRON1)
    height_scanner_policy: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base_Link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.32, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.08,
            size=[0.96, 1.36],  # (13-1)*0.08 x (18-1)*0.08
        ),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # ── Goal Command ─────────────────────────────────────────────────────────
    fallen_start_ratio:  float = 0.0
    fallen_roll_range:   tuple = (-3.14, 3.14)
    fallen_pitch_range:  tuple = (-0.5, 0.5)

    goal_pos_range_min:  float = 2.0
    goal_pos_range_max:  float = 5.0     # slightly smaller for biped

    moving_to_goal_v_min: float = 0.3

    # ── Reward Weights (same Table I structure, adapted for biped) ───────────
    # All weights RAW; dt=0.02 scaling applied in env __init__.
    w_position_tracking:    float = 100.0
    w_arrival:              float = 0.0
    w_heading_tracking:     float = 50.0
    w_moving_to_goal:       float = 5.0
    w_standing_at_goal:     float = 5.0
    w_bias_goal:            float = 0.0
    w_anti_stall:           float = 0.0
    w_upward:               float = 0.0
    w_goal_coarse:          float = 5.0
    w_goal_fine:            float = 0.0
    w_vel_toward_goal:      float = 0.0
    w_position_approach:    float = 50.0
    w_base_height:          float = 0.0
    w_feet_air_time:        float = 0.0
    w_anti_stagnation:      float = 0.0
    w_lin_vel_z_l2:         float = 0.0
    w_early_termination:    float = -500.0
    w_undesired_contacts:   float = -1.0
    w_ang_vel_xy_l2:        float = -0.1
    w_joint_reg_l2:         float = -0.001
    w_action_rate_l2:       float = -0.01
    w_link_contact_forces:  float = -0.00001
    w_link_acceleration:    float = -0.001
    w_joint_pos_limits:     float = -1000.0
    w_joint_vel_limits:     float = -1.0
    w_joint_torque_limits:  float = -1.0
