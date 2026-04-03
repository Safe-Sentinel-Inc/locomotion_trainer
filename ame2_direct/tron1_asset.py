"""PF_TRON1A (LimX) biped ArticulationCfg for Isaac Lab.

6-DOF biped: 2 legs x 3 joints (hip, knee, ankle).

USD asset must be generated from the PF_TRON1A URDF via Isaac Lab's
``urdf_converter`` tool before training:

    python -m isaaclab.app.tools.convert_urdf \
        --urdf_path path/to/pf_tron1a.urdf \
        --output_path data/robots/pf_tron1a/pf_tron1a.usd \
        --fix_base False \
        --make_instanceable

Update ``PF_TRON1A_USD_PATH`` below once the USD is available.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# ── USD path — update to actual converted USD ───────────────────────────
PF_TRON1A_USD_PATH = "data/robots/pf_tron1a/pf_tron1a.usd"

# ── Actuator config (all 6 joints identical) ─────────────────────────────
TRON1_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=[".*"],
    stiffness=40.0,
    damping=2.5,
    effort_limit=300.0,
)

# ── ArticulationCfg ──────────────────────────────────────────────────────
PF_TRON1A_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=PF_TRON1A_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.65),
        joint_pos={
            # Default standing pose — update from URDF zero configuration
            ".*": 0.0,
        },
    ),
    actuators={
        "legs": TRON1_ACTUATOR_CFG,
    },
)
