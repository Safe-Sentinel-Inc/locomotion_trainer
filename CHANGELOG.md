# AME-2 Direct Training Changelog

所有版本均基于 `ame2_direct/` Direct Workflow（Isaac Lab DirectRLEnv + RSL-RL PPO）。
服务器：AutoDL RTX 5090 (`connect.westd.seetacloud.com:13550`) / BSRL 8×RTX 3090 (`fe91fae6a6756695.natapp.cc:12346`)

---

---

## v41 — 2026-03-10 (hotfix)

V40 reward ok but robot still stands still (upward=3.99 dominates, moving_to_goal=0).

**fallen start + stronger locomotion**

### config.py
- `w_upward`: 1.0 -> 0.2 (standing still no longer dominant)
- `w_vel_toward_goal`: 5.0 -> 15.0 (3x boost)
- `w_goal_coarse`: 0.3 -> 1.5 (5x boost)
- `w_moving_to_goal`: 0.1 -> 1.0 (10x boost)
- `w_goal_fine`: 2.0 -> 5.0 (2.5x boost)
- `w_action_rate_l2`: -0.001 -> -0.0005 (relax)
- New: `fallen_start_ratio=0.5`, `fallen_roll_range=(-pi,pi)`, `fallen_pitch_range=(-0.5,0.5)`

### env.py
- 50% of envs start from random fallen pose (roll [-pi,pi], pitch [-0.5,0.5])
- Fallen robots spawned 0.3m higher to avoid ground clipping
- Full RPY quaternion composition (ZYX convention) replaces yaw-only
- `bad_orientation` grace period: 1 step -> 100 steps (2s recovery time)

## v40 — 2026-03-09

**服务器**：BSRL GPUs 0/3/4（seed 42/43/44）

**根因修复：奖励权重缺少 dt 乘法**

AME-2 DirectRLEnv 的奖励计算直接用 `rew += w * r`，不乘 dt。
而 HIM (legged_robot.py:730) 和 IsaacLab (reward_manager.py:149) 都会把权重乘以 dt=0.02。

这意味着之前所有版本的权重实际上是 HIM/robot_lab 等价值的 **50倍**！
- V39 upward: `1.0 * 4.0 = 4.0/step` → 4119/episode（占总奖励 99%）
- 修复后: `1.0 * 0.02 * 4.0 = 0.08/step` → ~82/episode（合理占比）

**修复内容**：
- `env.py __init__`: 在 `_ep_sums` 初始化后，循环遍历所有 `w_` 前缀配置项，乘以 `self.step_dt`
- 与 HIM `self.reward_scales[key] *= self.dt` 完全等价
- 所有 V39 权重值保持不变（它们现在等价于 HIM/robot_lab 的配置值了）

**其他**：无配置变更，纯框架修复。

## v39 — 2026-03-09

**服务器**：BSRL GPUs 0/3/4（seed 42/43/44）

**主题**：对齐 robot_lab 奖励设计 + 全局命名清理

### 奖励修改

| 参数 | 旧值 | 新值 | 理由 |
|------|------|------|------|
| `w_upward` (原 `w_upright_bonus`) | -0.5 (HIM `gravity_xy²`) | **+1.0** robot_lab `(1-g_z)²` | HIM 式惩罚太弱（-0.07/ep），robot_lab 正奖励更强（+4.0 upright） |
| `w_feet_air_time` | 3.0 | **0.0** | 公式 `(air-0.25)*3` 在短步态返回负值，是 V38 最大惩罚项（-4.6/ep），robot_lab 也关闭 |
| `w_anti_stagnation` | 0.3→0.0 (v38) | **0.0** | 训练初期太 dominant（-15/ep），干扰学习 |
| bad_orientation roll | 0.7 (~45°) | **0.9** (~64°) | 机器狗走路时身体摇晃被误杀，放宽阈值 |

### 全局命名清理（对齐 robot_lab）

| 旧名 | 新名 | 说明 |
|------|------|------|
| `w_upright_bonus` | `w_upward` | robot_lab 函数名 |
| `w_lin_vel_tracking` | `w_lin_vel_z_l2` | 实际功能是 z 轴速度惩罚 |
| `w_base_roll_rate` | `w_ang_vel_xy_l2` | pitch+roll 双轴 |
| `w_action_smoothness` | `w_action_rate_l2` | robot_lab 命名 |
| `w_joint_regularization` | `w_joint_reg_l2` | 简洁 |
| `w_undesired_events` | `w_undesired_contacts` | robot_lab 命名 |

config.py 注释全重写：去掉 v33-v37 版本历史，按功能分组（Navigation / Stability / Gait penalties / Termination / Disabled）。

---

## v38 — 2026-03-09

**服务器**：BSRL GPUs 0/3/4

**改动**：`w_anti_stagnation`: 0.3 → 0.0（禁用）

**效果**（390 iter / 3h）：reward 5-7，`vel_toward_goal=3.6`（机器狗在走），但 `feet_air_time=-4.6`（最大惩罚项）和 `bad_orientation` 终止拖低 episode 到 11s/20s。

---

## v37 — 2026-03-09

**服务器**：BSRL GPUs 0/3/4

**改动**：`upright_bonus` 改为 HIM-style `sum(gravity_xy²)`，权重 -0.5（负惩罚）

**效果**：63 iter 后 bad_orientation 终止 50%，惩罚太弱无法引导机器狗站直。已放弃，见 v39。

---

## v36 — 2026-03-09

**服务器**：BSRL GPUs 0/3/4（seed 42/43/44），AutoDL GPU1

**改动**：

### config.py
| 参数 | 旧值 | 新值 | 理由 |
|------|------|------|------|
| `w_upright_bonus` | 0.05 | **0.5** | 10x 增强站立激励，防趴地局部最优；信号量从 ~1.8/s → ~16/s |
| `w_lin_vel_tracking` | 0.0 | **-2.0** | 重用为 lin_vel_z_l2（防弹跳），之前禁用 |
| `w_base_roll_rate` | -0.002 | **-0.05** | 25x 增强，扩展到 pitch+roll 双轴 |

### env.py
| 位置 | 改动 | 理由 |
|------|------|------|
| `r_lin`（lin_vel_tracking） | cmd_vel 追踪公式 → `vel_z.square()` | 标准 ang_vel_z_l2，v36 重用这个 key |
| `r_roll`（base_roll_rate） | `root_ang_vel_b[:, 0]` → `root_ang_vel_b[:, :2].square().sum(1)` | 同时惩罚 pitch（俯仰）和 roll（横滚） |

**效果（10 min 后首批指标）**：
- `upright_bonus` 精确 10x（16.19/s）
- `base_collision` 从 0.5 → 0.0（body 撞地终止消失）
- Mean reward 从 4.74 → 10.36

---

## v35 — 2026-03-09

**服务器**：AutoDL GPU1，BSRL GPUs 0/3/4（已被 v36 替换）

**改动**（相对 v34）：
- `w_link_contact_forces = 0.0`（禁用）：该奖励在正常行走时产生 -171/s，完全淹没目标奖励（+35/s）。`undesired_events` 已覆盖接触质量检测，无需重复。
- BSRL 同步启动三卡训练（seeds 42/43/44）

**问题**（v35 遗留，v36 修复）：
- `feet_air_time` 阈值 0.25s 虽已修正，但早期 episode 太短（~0.24s）导致持续为负
- `upright_bonus` 权重太低（0.05），机器人趴地无惩罚
- `base_roll_rate` 只惩罚 roll，忽略 pitch

---

## v34 — 2026-03-08

**服务器**：AutoDL GPU1

**改动**（相对 v33，恢复论文 Table I 奖励平衡）：

**问题诊断（v33）**：`goal_coarse(1.5)` 给站立机器人被动奖励，走路的 `action_smooth` 惩罚反而高于收益，机器人学会站着不动（"standing still"局部最优）。

**修复**：
- `w_goal_coarse`: 1.5 → 0.3（降权，仅提供梯度辅助）
- `w_goal_fine`: 5.0 → 2.0
- `w_position_tracking`: 0.0 → 2.0（**恢复**，论文 Eq.1，最后 4s 才激活，强迫机器人走到目标）
- `w_heading_tracking`: 0.0 → 1.0（**恢复**，论文 Eq.3）
- `w_moving_to_goal`: 0.0 → 0.1（**恢复**，论文 Eq.4）
- `w_standing_at_goal`: 0.0 → 0.1（**恢复**，论文 Eq.5）
- `w_anti_stagnation`: 0.1 → 0.3（加强，打破站着不动）
- `w_action_smoothness`: -0.002 → -0.001（放松，步态惩罚已由专项奖励处理）
- 6 个步态惩罚全部启用：`undesired_events(-0.02)`, `link_acceleration(-0.00002)`, `joint_vel_limits(-0.02)`, `joint_torque_limits(-0.02)`

---

## v33 — 2026-03-07

**服务器**：AutoDL GPU1，训练至 ~1450 iter

**状态**：机器人可以朝目标移动，但无正常 trot 步态（滑动/蹒跚）。

**根本原因**：论文 6 个步态惩罚均 disabled（w=0），robot 不需要"举脚"就能得到最优奖励。
步态涌现依赖 `undesired_events`（滑动/绊脚惩罚）使滑行物理上代价高——这些全部缺失。

**修复方向** → 见 v34。

---

## v25–v32 — 2026-02-xx 至 2026-03-06

- v17-v24：`runner.alg.actor_critic=ame2_net` 无效，需用 `runner.alg.policy=ame2_net`（关键经验）
- v25：首次用 `AME2ActorCritic`，2048 envs，正确调用方式
- v25 问题：noise 坍塌（iter 300 std→0.38），upright_bonus 主导，robot 站着不倒但不走路
- v32：baseline 稳定，进入 v33 步态修复阶段

---

## 关键经验总结

1. **步态不需要显式设计**：trot 是能耗最低步态，通过 `undesired_events`（滑动/绊脚惩罚）使其它方式代价更高，trot 自然涌现
2. **link_contact_forces 危险**：raw force²量级过大（正常行走 ~3.4M/step），即使 1e-6 权重也产生 -171/s，淹没目标信号
3. **upright_bonus 需要足够强**：0.05 不够，0.5 是合适量级；既防趴地，又不过度主导
4. **lin_vel_z_l2 是标配**：所有主流四足 locomotion 代码都有此项，防弹跳
5. **noise std 坍塌 = 局部最优**：std<0.5 @ iter<500 → 需要调奖励或 entropy_coef
