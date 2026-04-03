"""AME-2 Direct Workflow Environment for PF_TRON1A (6-DOF biped).

Subclass of AME2DirectEnv with overrides for TRON1 body patterns,
observation dimensions, and physical parameters.

Dimension summary (TRON1, J=6, L=2):
  Actor prop:   30D = base_vel(3) + hist(24) + cmd(3)
  d_hist:       24  = ang_vel(3) + grav(3) + q(6) + dq(6) + act(6)
  Contact:       7D = base(1) + thigh(2) + shank(2) + foot(2)
  Critic prop:  37D = base(27) + critic_cmd(5) + nav_extra(5)
  GT map:       (N, 3, 13, 18)
"""

from __future__ import annotations

import math

import torch

from .config_tron1 import TRON1DirectEnvCfg
from .env import AME2DirectEnv


class TRON1DirectEnv(AME2DirectEnv):
    """AME-2 direct environment for PF_TRON1A biped.

    Overrides body-specific methods for 6-DOF, 2-leg morphology.
    """

    cfg: TRON1DirectEnvCfg

    def __init__(self, cfg: TRON1DirectEnvCfg, render_mode: str | None = None, **kwargs):
        # Temporarily override parent's hardcoded body patterns before super().__init__
        # calls _setup_scene and body queries.
        # We need to override __init__ to handle TRON1 body patterns.
        # Call grandparent (DirectRLEnv) __init__ first via super().
        super(AME2DirectEnv, self).__init__(cfg, render_mode, **kwargs)

        n, dev = self.num_envs, self.device

        # ── Body indices (TRON1 patterns) ──
        self._base_cs_id, _     = self._contact_sensor.find_bodies("base_Link")
        self._thigh_cs_ids, _   = self._contact_sensor.find_bodies(".*hip.*Link")
        self._shank_cs_ids, _   = self._contact_sensor.find_bodies(".*knee.*Link")
        self._foot_cs_ids, _    = self._contact_sensor.find_bodies(".*foot.*Link")
        self._non_foot_cs_ids   = list(self._thigh_cs_ids) + list(self._shank_cs_ids)
        self._all_cs_ids        = (
            list(self._base_cs_id) + list(self._thigh_cs_ids)
            + list(self._shank_cs_ids) + list(self._foot_cs_ids)
        )  # 7 links: base(1) + thigh(2) + shank(2) + foot(2)

        _body_names  = self._contact_sensor.body_names
        _foot_names  = [_body_names[i] for i in self._foot_cs_ids]
        _thigh_names = [_body_names[i] for i in self._thigh_cs_ids]
        _shank_names = [_body_names[i] for i in self._shank_cs_ids]
        print(f"[TRON1 ContactOrder] feet:   {_foot_names}")
        print(f"[TRON1 ContactOrder] thighs: {_thigh_names}")
        print(f"[TRON1 ContactOrder] shanks: {_shank_names}")
        print(f"[TRON1 ContactOrder] all_cs_ids={self._all_cs_ids}  (base+thigh+shank+foot, 7 total)")

        _jlims = self._robot.data.joint_pos_limits
        if _jlims.dim() == 3:
            _jlims = _jlims[0]
        _jdef = self._robot.data.default_joint_pos[0]
        print(f"[TRON1 JointLimits] {self._robot.joint_names}")
        for _i, _jn in enumerate(self._robot.joint_names):
            _lo, _hi = float(_jlims[_i, 0]), float(_jlims[_i, 1])
            _dp = float(_jdef[_i])
            print(f"  {_jn:20s}  lo={_lo:+.3f} ({_lo*57.3:+.0f}deg)  hi={_hi:+.3f} ({_hi*57.3:+.0f}deg)  default={_dp:+.3f}")

        # Robot body indices for kinematics
        self._thigh_rb_ids, _   = self._robot.find_bodies(".*hip.*Link")
        self._foot_rb_ids, _    = self._robot.find_bodies(".*foot.*Link")
        self._shank_rb_ids, _   = self._robot.find_bodies(".*knee.*Link")

        # ── Action buffers (6-DOF) ──
        import gymnasium as gym
        n_act = gym.spaces.flatdim(self.single_action_space)
        self._actions           = torch.zeros(n, n_act, device=dev)
        self._prev_actions      = torch.zeros(n, n_act, device=dev)
        self._processed_actions = torch.zeros(n, n_act, device=dev)

        # ── Goal command buffers ──
        self._goal_pos_w   = torch.zeros(n, 2, device=dev)
        self._goal_heading = torch.zeros(n, device=dev)
        self._goal_radius  = float(cfg.goal_pos_range_max)

        # ── Stagnation detection ──
        self._stag_pos  = torch.zeros(n, 2, device=dev)
        self._stag_step = torch.zeros(n, device=dev, dtype=torch.long)

        # ── Contact history ──
        self._prev_nf_contact = torch.zeros(n, len(self._non_foot_cs_ids), device=dev, dtype=torch.bool)
        self._prev_foot_contact = torch.zeros(n, len(self._foot_cs_ids), device=dev, dtype=torch.bool)

        # ── Thigh velocity buffer (2 thighs for biped) ──
        self._prev_thigh_vel = torch.zeros(n, len(self._thigh_rb_ids), 3, device=dev)
        self._prev_body_lin_vel = torch.zeros(n, self._robot.num_bodies, 3, device=dev)

        # ── Joint velocity buffer (6 joints) ──
        self._prev_joint_vel = torch.zeros(n, 6, device=dev)

        # ── Terminated cache ──
        self._terminated = torch.zeros(n, device=dev, dtype=torch.bool)

        # ── Curriculum state ──
        self.heading_curriculum_frac: float = 0.0
        self.scan_noise_scale:        float = 0.0

        # ── Navigation progress buffers ──
        self._prev_d_xy    = torch.zeros(n, device=dev)
        self._prev_yaw_err = torch.zeros(n, device=dev)
        self._reward_d_prev = torch.zeros(n, device=dev)

        # ── Episode sums ──
        self._ep_sums = {k: torch.zeros(n, device=dev) for k in [
            "goal_coarse", "goal_fine", "anti_stagnation",
            "position_tracking", "arrival", "position_approach", "upward",
            "heading_tracking", "moving_to_goal",
            "vel_toward_goal", "lin_vel_z_l2", "standing_at_goal", "early_termination",
            "undesired_contacts", "base_height", "feet_air_time", "ang_vel_xy_l2", "joint_reg_l2",
            "action_rate_l2", "link_contact", "link_acc",
            "jpos_lim", "jvel_lim", "jtau_lim",
            "bias_goal", "anti_stall",
        ]}

        # ── Scale reward weights by dt ──
        _dt = self.step_dt
        for _attr in dir(cfg):
            if _attr.startswith('w_') and isinstance(getattr(cfg, _attr), (int, float)):
                setattr(cfg, _attr, getattr(cfg, _attr) * _dt)
        print(f'[TRON1 V40] Reward weights scaled by dt={_dt:.4f}')

    # ─────────────────────────────────────────────────────────────────────────
    # Observations (30D actor prop, 7D contact, 37D critic_prop)
    # ─────────────────────────────────────────────────────────────────────────

    def _get_observations(self) -> dict:
        actor_cmd = self._get_actor_cmd()   # (N, 3)

        # Teacher actor obs: 30D = base_vel(3)+ang_vel(3)+grav(3)+q(6)+dq(6)+act(6)+cmd(3)
        obs_policy = torch.cat([
            self._robot.data.root_lin_vel_b  * 2.0,                              # 3
            self._robot.data.root_ang_vel_b  * 0.25,                             # 3
            self._robot.data.projected_gravity_b,                                 # 3
            self._robot.data.joint_pos - self._robot.data.default_joint_pos,     # 6
            self._robot.data.joint_vel * 0.05,                                   # 6
            self._prev_actions,                                                   # 6
            actor_cmd,                                                            # 3
        ], dim=-1)  # (N, 30)

        if self.scan_noise_scale > 0.0:
            obs_policy = self._add_obs_noise(obs_policy)

        # GT policy map: (N, 3, 13, 18)
        gt_map_4d = self._get_gt_policy_map()

        # All-link binary contact states: (N, 7)
        all_f      = self._contact_sensor.data.net_forces_w_history[:, 0, self._all_cs_ids, :]
        contact_7  = (torch.norm(all_f, dim=-1) > 1.0).float()
        foot_f     = self._contact_sensor.data.net_forces_w_history[:, 0, self._foot_cs_ids, :]

        # Critic prop: base(27D) + critic_cmd(5D) + nav_extra(5D) = 37D
        prop_base  = obs_policy[:, :27]   # base_vel(3)+ang_vel(3)+grav(3)+q(6)+dq(6)+act(6)
        goal_xy_b  = self._get_goal_xy_body()
        d_yaw_s    = self._get_d_yaw_signed()
        d_yaw      = d_yaw_s.abs()
        t_rem      = (
            (self.max_episode_length - self.episode_length_buf).float()
            * self.step_dt / max(self.max_episode_length_s, 1.0)
        )
        critic_cmd  = torch.stack([
            goal_xy_b[:, 0].clamp(-5.0, 5.0),
            goal_xy_b[:, 1].clamp(-5.0, 5.0),
            torch.sin(d_yaw_s),
            torch.cos(d_yaw_s),
            t_rem,
        ], dim=-1)  # (N, 5)

        # Nav extra (5D)
        vel_xy   = self._robot.data.root_lin_vel_b[:, :2]
        d_xy_raw = torch.norm(goal_xy_b, dim=1)
        to_goal  = goal_xy_b / (d_xy_raw.unsqueeze(1) + 1e-8)
        v_proj = (vel_xy * to_goal).sum(1)
        d_progress = (self._prev_d_xy - d_xy_raw) / max(self.step_dt, 1e-6)
        heading_align_rate = (self._prev_yaw_err - d_yaw) / max(self.step_dt, 1e-6)
        yaw     = self._get_yaw()
        cy, sy  = torch.cos(yaw), torch.sin(yaw)
        vel_w_x = cy * vel_xy[:, 0] - sy * vel_xy[:, 1]
        vel_w_y = sy * vel_xy[:, 0] + cy * vel_xy[:, 1]
        nav_extra = torch.stack([
            (v_proj / 2.0).clamp(-1.0, 1.0),
            d_progress.clamp(-2.0, 2.0) / 2.0,
            heading_align_rate.clamp(-5.0, 5.0) / 5.0,
            (vel_w_x * 2.0).clamp(-4.0, 4.0),
            (vel_w_y * 2.0).clamp(-4.0, 4.0),
        ], dim=-1)  # (N, 5)

        self._prev_d_xy[:]    = d_xy_raw.detach()
        self._prev_yaw_err[:] = d_yaw.detach()

        critic_prop = torch.cat([prop_base, critic_cmd, nav_extra], dim=-1)  # (N, 37)

        # Height scan for logging: 31x31 = 961
        heights     = self._get_height_scan_rel()
        foot_f_flat = foot_f.reshape(self.num_envs, -1)
        obs_priv    = torch.cat([heights, foot_f_flat], dim=-1)

        return {
            "policy":             obs_policy,
            "prop":               obs_policy,
            "map":                gt_map_4d,
            "map_teacher":        gt_map_4d,
            "critic_prop":        critic_prop,
            "contact":            contact_7,
            "teacher_privileged": obs_priv,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Observation noise (30D layout)
    # ─────────────────────────────────────────────────────────────────────────

    def _add_obs_noise(self, obs: torch.Tensor) -> torch.Tensor:
        """Appendix B noise for 30D TRON1 obs."""
        noise           = torch.zeros_like(obs)
        scale           = self.scan_noise_scale
        noise[:, 0:3]   = (torch.rand_like(obs[:, 0:3]) - 0.5) * 0.2 * scale    # lin_vel
        noise[:, 3:6]   = (torch.rand_like(obs[:, 3:6]) - 0.5) * 0.4 * scale    # ang_vel
        noise[:, 6:9]   = (torch.rand_like(obs[:, 6:9]) - 0.5) * 0.1 * scale    # gravity
        noise[:, 9:15]  = (torch.rand_like(obs[:, 9:15]) - 0.5) * 0.02 * scale   # joint_pos (6)
        noise[:, 15:21] = (torch.rand_like(obs[:, 15:21]) - 0.5) * 3.0 * scale   # joint_vel (6)
        noise[:, 21:27] = (torch.rand_like(obs[:, 21:27]) - 0.5) * 0.05 * scale  # prev_actions (6)
        noise[:, 27:30] = (torch.rand_like(obs[:, 27:30]) - 0.5) * 0.1 * scale   # actor_cmd
        return obs + noise

    # ─────────────────────────────────────────────────────────────────────────
    # GT policy map (13x18)
    # ─────────────────────────────────────────────────────────────────────────

    def _get_gt_policy_map(self) -> torch.Tensor:
        """13x18 GT policy map: (N, 3, 13, 18)."""
        hits  = self._height_scanner_policy.data.ray_hits_w   # (N, 234, 3)
        base  = self._robot.data.root_pos_w
        rel   = (hits - base.unsqueeze(1)).clamp(-5.0, 5.0)
        return rel.permute(0, 2, 1).reshape(self.num_envs, 3, 13, 18)

    # ─────────────────────────────────────────────────────────────────────────
    # Height scan (31x31 for TRON1)
    # ─────────────────────────────────────────────────────────────────────────

    def _get_height_scan_rel(self) -> torch.Tensor:
        """31x31 height scan relative to scanner position (N, 961)."""
        scanner_z = self._height_scanner.data.pos_w[:, 2].unsqueeze(1)
        hits_z    = self._height_scanner.data.ray_hits_w[:, :, 2]
        rel_z     = (scanner_z - hits_z - 0.5).clamp(-1.0, 1.0)

        if self.scan_noise_scale > 0.0:
            noise = (torch.rand_like(rel_z) - 0.5) * 2.0 * 0.05 * self.scan_noise_scale
            rel_z = (rel_z + noise).clamp(-1.0, 1.0)
        return rel_z

    # ─────────────────────────────────────────────────────────────────────────
    # Rewards — use TRON1 mass/height
    # ─────────────────────────────────────────────────────────────────────────

    def _get_rewards(self) -> torch.Tensor:
        # Most rewards are inherited from parent (AME2DirectEnv._get_rewards).
        # Override link_contact_forces robot weight to TRON1 mass.
        cfg = self.cfg
        rew = torch.zeros(self.num_envs, device=self.device)

        goal_xy_b = self._get_goal_xy_body()
        d_xy      = torch.norm(goal_xy_b, dim=1)

        # 1. position_tracking
        r_pos = (1.0 / (1.0 + 0.25 * d_xy**2)) * self._t_mask(4.0)
        rew += cfg.w_position_tracking * r_pos
        self._ep_sums["position_tracking"] += cfg.w_position_tracking * r_pos

        # arrival
        arrived = (d_xy < 0.5).float()
        r_arrival = arrived / max(self.max_episode_length, 1)
        rew += cfg.w_arrival * r_arrival
        self._ep_sums["arrival"] += cfg.w_arrival * r_arrival

        # goal_coarse
        r_goal_coarse = 1.0 - torch.tanh(d_xy / 2.0)
        rew += cfg.w_goal_coarse * r_goal_coarse
        self._ep_sums["goal_coarse"] += cfg.w_goal_coarse * r_goal_coarse

        # goal_fine
        r_goal_fine = 1.0 - torch.tanh(d_xy / 0.3)
        rew += cfg.w_goal_fine * r_goal_fine
        self._ep_sums["goal_fine"] += cfg.w_goal_fine * r_goal_fine

        # position_approach
        r_approach = torch.clamp(self._reward_d_prev - d_xy, -0.1, 0.1)
        self._reward_d_prev[:] = d_xy.detach()
        rew += cfg.w_position_approach * r_approach
        self._ep_sums["position_approach"] += cfg.w_position_approach * r_approach

        # upward
        g_b_z = self._robot.data.projected_gravity_b[:, 2]
        r_upright = torch.square(1.0 - g_b_z)
        rew += cfg.w_upward * r_upright
        self._ep_sums["upward"] += cfg.w_upward * r_upright

        # base_height (using TRON1 standing height 0.65m)
        base_z    = self._robot.data.root_pos_w[:, 2]
        terrain_z = self._terrain.env_origins[:, 2]
        r_height  = torch.clamp((base_z - terrain_z) / 0.65, 0.0, 1.0)
        rew += cfg.w_base_height * r_height
        self._ep_sums["base_height"] += cfg.w_base_height * r_height

        # feet_air_time
        foot_contact = torch.norm(
            self._contact_sensor.data.net_forces_w_history[:, 0, self._foot_cs_ids, :], dim=-1
        ) > 1.0
        first_contact = foot_contact & ~self._prev_foot_contact
        last_air = self._contact_sensor.data.last_air_time[:, self._foot_cs_ids]
        r_feet = ((last_air - 0.25) * first_contact.float()).sum(dim=1)
        r_feet = r_feet * (g_b_z < -0.5).float()
        self._prev_foot_contact = foot_contact.clone()
        rew += cfg.w_feet_air_time * r_feet
        self._ep_sums["feet_air_time"] += cfg.w_feet_air_time * r_feet

        # heading_tracking
        d_yaw = self._get_d_yaw()
        r_head = (1.0/(1.0+d_yaw**2)) * self._t_mask(2.0) * torch.sigmoid((0.5 - d_xy) * 10.0)
        rew += cfg.w_heading_tracking * r_head
        self._ep_sums["heading_tracking"] += cfg.w_heading_tracking * r_head

        # moving_to_goal
        vel_xy  = self._robot.data.root_lin_vel_b[:, :2]
        to_goal = goal_xy_b / (d_xy.unsqueeze(1) + 1e-8)
        v_proj  = (vel_xy * to_goal).sum(1)
        speed   = torch.norm(vel_xy, dim=-1)
        v_min   = cfg.moving_to_goal_v_min
        v_max   = 2.0
        cos_ok  = v_proj / (speed + 1e-8) > 0.5
        speed_ok = (speed >= v_min) & (speed <= v_max)
        near_goal = d_xy < 0.5
        r_move  = (near_goal | (cos_ok & speed_ok)).float()
        rew += cfg.w_moving_to_goal * r_move
        self._ep_sums["moving_to_goal"] += cfg.w_moving_to_goal * r_move

        # vel_toward_goal (disabled)
        r_vel  = torch.clamp(v_proj/2.0, -1.0, 1.0) * torch.sigmoid((d_xy - 0.5) * 10.0)
        rew += cfg.w_vel_toward_goal * r_vel
        self._ep_sums["vel_toward_goal"] += cfg.w_vel_toward_goal * r_vel

        # Legacy (disabled)
        vel_nrm = speed
        cos_t = v_proj / (vel_nrm + 1e-8)
        r_bias = (torch.relu(cos_t) * torch.clamp(vel_nrm / 0.5, 0.0, 1.0) * (d_xy > 0.5).float())
        rew += cfg.w_bias_goal * r_bias
        self._ep_sums["bias_goal"] += cfg.w_bias_goal * r_bias
        r_stall = -1.0 * (speed < 0.1).float() * (d_xy > 0.5).float()
        rew += cfg.w_anti_stall * r_stall
        self._ep_sums["anti_stall"] += cfg.w_anti_stall * r_stall
        r_anti_stag = -(speed < 0.2).float() * (d_xy > 0.5).float()
        rew += cfg.w_anti_stagnation * r_anti_stag
        self._ep_sums["anti_stagnation"] += cfg.w_anti_stagnation * r_anti_stag

        # lin_vel_z_l2
        r_lin = self._robot.data.root_lin_vel_b[:, 2].square()
        rew += cfg.w_lin_vel_z_l2 * r_lin
        self._ep_sums["lin_vel_z_l2"] += cfg.w_lin_vel_z_l2 * r_lin

        # standing_at_goal
        r_stand = self._standing_at_goal_reward(d_xy, d_yaw)
        rew += cfg.w_standing_at_goal * r_stand
        self._ep_sums["standing_at_goal"] += cfg.w_standing_at_goal * r_stand

        # early_termination
        r_term = self._terminated.float()
        rew += cfg.w_early_termination * r_term
        self._ep_sums["early_termination"] += cfg.w_early_termination * r_term

        # undesired_contacts
        r_undes = self._undesired_events()
        rew += cfg.w_undesired_contacts * r_undes
        self._ep_sums["undesired_contacts"] += cfg.w_undesired_contacts * r_undes

        # ang_vel_xy_l2
        r_ang_vel_xy = self._robot.data.root_ang_vel_b[:, 0].square()
        rew += cfg.w_ang_vel_xy_l2 * r_ang_vel_xy
        self._ep_sums["ang_vel_xy_l2"] += cfg.w_ang_vel_xy_l2 * r_ang_vel_xy

        # joint_reg
        jvel = self._robot.data.joint_vel
        jtau = self._robot.data.applied_torque
        jacc = (jvel - self._prev_joint_vel) / self.step_dt
        self._prev_joint_vel = jvel.clone()
        r_jreg = (jvel.square().sum(1) + 0.01 * jtau.square().sum(1) + 0.001 * jacc.square().sum(1))
        rew += cfg.w_joint_reg_l2 * r_jreg
        self._ep_sums["joint_reg_l2"] += cfg.w_joint_reg_l2 * r_jreg

        # action_rate_l2
        r_smooth = (self._actions - self._prev_actions).square().sum(1)
        rew += cfg.w_action_rate_l2 * r_smooth
        self._ep_sums["action_rate_l2"] += cfg.w_action_rate_l2 * r_smooth

        # link_contact_forces — TRON1 mass = 18.508 kg
        _robot_weight = 18.508 * 9.81
        nf_forces = self._contact_sensor.data.net_forces_w_history[:, 0, self._non_foot_cs_ids, :]
        fmag      = torch.norm(nf_forces, dim=-1)
        excess    = torch.clamp(fmag - _robot_weight, min=0.0)
        r_lfc     = excess.square().sum(1)
        rew += cfg.w_link_contact_forces * r_lfc
        self._ep_sums["link_contact"] += cfg.w_link_contact_forces * r_lfc

        # link_acceleration
        body_vel = self._robot.data.body_lin_vel_w
        body_acc = (body_vel - self._prev_body_lin_vel) / self.step_dt
        self._prev_body_lin_vel = body_vel.clone()
        r_link_acc = torch.norm(body_acc, dim=-1).sum(1)
        rew += cfg.w_link_acceleration * r_link_acc
        self._ep_sums["link_acc"] += cfg.w_link_acceleration * r_link_acc

        # joint limits
        pos    = self._robot.data.joint_pos
        lims   = self._robot.data.joint_pos_limits
        if lims.dim() == 2:
            lims = lims.unsqueeze(0).expand(self.num_envs, -1, -1)
        over_u = torch.clamp(pos - 0.95*lims[:,:,1], min=0.0)
        over_l = torch.clamp(0.95*lims[:,:,0] - pos, min=0.0)
        r_jpos = (over_u+over_l).sum(1)
        rew += cfg.w_joint_pos_limits * r_jpos
        self._ep_sums["jpos_lim"] += cfg.w_joint_pos_limits * r_jpos

        vlims  = self._robot.data.joint_vel_limits
        if vlims.dim() == 1:
            vlims = vlims.unsqueeze(0).expand(self.num_envs, -1)
        vex = torch.clamp(self._robot.data.joint_vel.abs() - 0.9*vlims, min=0.0)
        r_jvel = vex.sum(1)
        rew += cfg.w_joint_vel_limits * r_jvel
        self._ep_sums["jvel_lim"] += cfg.w_joint_vel_limits * r_jvel

        elims  = self._robot.data.joint_effort_limits
        if elims.dim() == 1:
            elims = elims.unsqueeze(0).expand(self.num_envs, -1)
        tex = torch.clamp(self._robot.data.applied_torque.abs() - 0.8*elims, min=0.0)
        r_jtau = tex.sum(1)
        rew += cfg.w_joint_torque_limits * r_jtau
        self._ep_sums["jtau_lim"] += cfg.w_joint_torque_limits * r_jtau

        return rew

    # ─────────────────────────────────────────────────────────────────────────
    # Terminations — TRON1 thigh acc threshold = 80 m/s²
    # ─────────────────────────────────────────────────────────────────────────

    def _high_thigh_acceleration(self) -> torch.Tensor:
        """TRON1: threshold 80 m/s² (lighter robot, lower threshold)."""
        thigh_vel = self._robot.data.body_lin_vel_w[:, self._thigh_rb_ids, :]
        thigh_acc = (thigh_vel - self._prev_thigh_vel) / self.step_dt
        self._prev_thigh_vel = thigh_vel.clone()
        acc_norm = torch.norm(thigh_acc, dim=-1).max(dim=1).values
        return (acc_norm > 80.0) & (self.episode_length_buf > 50)

    def _base_collision(self) -> torch.Tensor:
        """TRON1: base collision threshold = 18.508 * 9.81 N."""
        _robot_weight = 18.508 * 9.81
        base_forces = self._contact_sensor.data.net_forces_w_history[:, 0, self._base_cs_id, :]
        base_fmag = torch.norm(base_forces, dim=-1).squeeze(-1)
        return (base_fmag > _robot_weight) & (self.episode_length_buf > 50)

    # ─────────────────────────────────────────────────────────────────────────
    # Reset — 6 joints
    # ─────────────────────────────────────────────────────────────────────────

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        terminal_d_xy = torch.norm(self._get_goal_xy_body()[env_ids], dim=-1)

        self._robot.reset(env_ids)
        # Call DirectRLEnv._reset_idx (grandparent), not AME2DirectEnv._reset_idx
        super(AME2DirectEnv, self)._reset_idx(env_ids)

        self._contact_sensor.reset(env_ids)
        self._height_scanner.reset(env_ids)
        self._height_scanner_policy.reset(env_ids)

        n   = len(env_ids)
        dev = self.device

        default_root = self._robot.data.default_root_state[env_ids].clone()
        default_root[:, :3] += self._terrain.env_origins[env_ids]

        self._robot.write_root_pose_to_sim(default_root[:, :7], env_ids)
        base_xy_init = default_root[:, :2].clone()

        self._resample_goals(env_ids, base_xy=base_xy_init)

        # Fallen start
        fallen_mask = torch.rand(n, device=dev) < self.cfg.fallen_start_ratio
        roll_rand = torch.zeros(n, device=dev)
        pitch_rand = torch.zeros(n, device=dev)
        n_fallen = fallen_mask.sum().item()
        if n_fallen > 0:
            roll_rand[fallen_mask] = (
                torch.rand(int(n_fallen), device=dev)
                * (self.cfg.fallen_roll_range[1] - self.cfg.fallen_roll_range[0])
                + self.cfg.fallen_roll_range[0]
            )
            pitch_rand[fallen_mask] = (
                torch.rand(int(n_fallen), device=dev)
                * (self.cfg.fallen_pitch_range[1] - self.cfg.fallen_pitch_range[0])
                + self.cfg.fallen_pitch_range[0]
            )
            default_root[fallen_mask, 2] += 0.3

        # Initial yaw
        goal_heading_w = self._goal_heading[env_ids]
        rand_yaw       = torch.rand(n, device=dev) * 2 * math.pi - math.pi
        use_rand       = torch.rand(n, device=dev) < self.heading_curriculum_frac
        yaw_init       = torch.where(use_rand, rand_yaw, goal_heading_w)

        cr, sr = torch.cos(roll_rand / 2), torch.sin(roll_rand / 2)
        cp, sp = torch.cos(pitch_rand / 2), torch.sin(pitch_rand / 2)
        cy, sy = torch.cos(yaw_init / 2), torch.sin(yaw_init / 2)
        default_root[:, 3] = cr * cp * cy + sr * sp * sy
        default_root[:, 4] = sr * cp * cy - cr * sp * sy
        default_root[:, 5] = cr * sp * cy + sr * cp * sy
        default_root[:, 6] = cr * cp * sy - sr * sp * cy
        default_root[:, 7:10]  = (torch.rand(n, 3, device=dev) - 0.5) * 0.2
        default_root[:, 10:13] = (torch.rand(n, 3, device=dev) - 0.5) * 0.2

        self._robot.write_root_pose_to_sim(default_root[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root[:, 7:], env_ids)

        joint_pos = (
            self._robot.data.default_joint_pos[env_ids]
            + (torch.rand(n, 6, device=dev) - 0.5) * 0.1
        )
        joint_vel = self._robot.data.default_joint_vel[env_ids].clone()
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Buffer reset
        self._actions[env_ids]      = 0.0
        self._prev_actions[env_ids] = 0.0
        self._stag_pos[env_ids]     = base_xy_init.clone()
        self._stag_step[env_ids]    = self.episode_length_buf[env_ids].clone()
        self._prev_nf_contact[env_ids]   = False
        self._prev_foot_contact[env_ids] = False
        self._prev_thigh_vel[env_ids]    = 0.0
        self._prev_body_lin_vel[env_ids] = 0.0
        self._prev_joint_vel[env_ids]    = 0.0

        d_xy_all = torch.norm(self._get_goal_xy_body(), dim=-1)
        self._prev_d_xy[env_ids]    = d_xy_all[env_ids]
        self._prev_yaw_err[env_ids] = self._get_d_yaw()[env_ids]
        self._reward_d_prev[env_ids] = d_xy_all[env_ids]

        self._update_terrain_curriculum(env_ids, terminal_d_xy)
        self._log_episode_stats(env_ids, terminal_d_xy)
