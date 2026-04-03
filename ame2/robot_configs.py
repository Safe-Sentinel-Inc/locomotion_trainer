"""Robot-specific configuration for AME-2 multi-robot support.

Each RobotConfig bundles form-factor-specific constants:
  - Joint/leg counts and derived observation dimensions
  - Left-right symmetry permutations and sign flips
  - Body name regex patterns for contact/kinematics queries
  - Physical parameters (mass, standing height, action scale)

Two presets:
  ANYMAL_D_ROBOT  — 12 joints, 4 legs (quadruped)
  TRON1_ROBOT     — 6 joints, 2 legs  (biped)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class RobotConfig:
    """Robot form-factor configuration for AME-2."""

    name: str

    # ── Topology ──────────────────────────────────────────────────────────
    num_joints: int
    num_legs: int

    @property
    def num_contacts(self) -> int:
        """All-link contact dim: base(1) + 3 links per leg (thigh+shank+foot)."""
        return 1 + 3 * self.num_legs

    # ── Derived observation dimensions ────────────────────────────────────
    @property
    def d_hist(self) -> int:
        """Per-step history dim: ang_vel(3) + grav(3) + q(J) + dq(J) + act(J)."""
        return 6 + 3 * self.num_joints

    @property
    def d_prop_raw(self) -> int:
        """Teacher actor prop: base_vel(3) + d_hist + actor_cmd(3)."""
        return 3 + self.d_hist + 3

    @property
    def d_prop_critic(self) -> int:
        """Critic prop: base_vel(3) + d_hist + critic_cmd(5)."""
        return 3 + self.d_hist + 5

    @property
    def d_prop_critic_ext(self) -> int:
        """Extended critic prop: d_prop_critic + nav_extra(5)."""
        return self.d_prop_critic + 5

    # ── Left-right symmetry permutations ──────────────────────────────────
    lr_joint_perm: List[int] = field(default_factory=list)
    lr_joint_sign: List[float] = field(default_factory=list)
    lr_contact_perm: List[int] = field(default_factory=list)

    # ── Body name patterns (regex for find_bodies) ────────────────────────
    base_link_pattern: str = "base"
    foot_link_pattern: str = ".*FOOT"
    thigh_link_pattern: str = ".*THIGH"
    shank_link_pattern: str = ".*SHANK"

    # ── Physical parameters ───────────────────────────────────────────────
    robot_mass_kg: float = 50.0
    standing_height_m: float = 0.6
    action_scale: float = 0.5
    thigh_acc_threshold: float = 100.0


# ══════════════════════════════════════════════════════════════════════════
# ANYmal-D (12 joints, 4 legs)
# ══════════════════════════════════════════════════════════════════════════

ANYMAL_D_ROBOT = RobotConfig(
    name="anymal_d",
    num_joints=12,
    num_legs=4,
    # Joint ordering (Isaac Lab USD traversal):
    #   [LF_HAA, LH_HAA, RF_HAA, RH_HAA,    idx 0-3
    #    LF_HFE, LH_HFE, RF_HFE, RH_HFE,    idx 4-7
    #    LF_KFE, LH_KFE, RF_KFE, RH_KFE]    idx 8-11
    # L-R flip: LF<->RF, LH<->RH
    lr_joint_perm=[2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9],
    # HAA joints negate (abduction axis flips sign); HFE/KFE stay positive
    lr_joint_sign=[-1., -1., -1., -1., 1., 1., 1., 1., 1., 1., 1., 1.],
    # All-link contact: base(0), thigh(1-4), shank(5-8), foot(9-12)
    # Within each group: [LF, LH, RF, RH]. Swap LF<->RF, LH<->RH.
    lr_contact_perm=[0, 3, 4, 1, 2, 7, 8, 5, 6, 11, 12, 9, 10],
    # Body patterns
    base_link_pattern="base",
    foot_link_pattern=".*FOOT",
    thigh_link_pattern=".*THIGH",
    shank_link_pattern=".*SHANK",
    # Physical
    robot_mass_kg=50.0,
    standing_height_m=0.6,
    action_scale=0.5,
    thigh_acc_threshold=100.0,
)


# ══════════════════════════════════════════════════════════════════════════
# PF_TRON1A (6 joints, 2 legs — biped)
# ══════════════════════════════════════════════════════════════════════════

TRON1_ROBOT = RobotConfig(
    name="tron1",
    num_joints=6,
    num_legs=2,
    # Joint ordering (URDF traversal, 2 legs x 3 joints):
    #   [L_hip, R_hip, L_knee, R_knee, L_ankle, R_ankle]   idx 0-5
    # L-R flip: L(0)<->R(1), L(2)<->R(3), L(4)<->R(5)
    lr_joint_perm=[1, 0, 3, 2, 5, 4],
    # Hip joints negate (abduction axis flips sign); knee/ankle stay positive
    lr_joint_sign=[-1., -1., 1., 1., 1., 1.],
    # All-link contact: base(0), thigh(1-2), shank(3-4), foot(5-6)
    # Within each group: [L, R]. Swap L<->R.
    lr_contact_perm=[0, 2, 1, 4, 3, 6, 5],
    # Body patterns (PF_TRON1A URDF)
    base_link_pattern="base_Link",
    foot_link_pattern=".*foot.*Link",
    thigh_link_pattern=".*hip.*Link",
    shank_link_pattern=".*knee.*Link",
    # Physical
    robot_mass_kg=18.508,
    standing_height_m=0.65,
    action_scale=0.25,
    thigh_acc_threshold=80.0,
)
