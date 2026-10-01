# NoMaD-PG 双路线对比计划

## 路线命名

### 路线 F：Frozen-Head Baseline（冻结动作头基线）

- 中文简称：冻结头路线
- 监督初始化：`ema_3.pth`
- 监督阶段只训练 `vision_encoder.goal_encoder`
- Transformer、视觉编码器、diffusion action head 均保持预训练权重
- PPO 输出目录：`runs/nomad_pg_ppo_v2`
- 实验目的：验证仅做 PointGoal token 对齐、随后依靠 PPO 学习动作适配的效果

### 路线 A：Action-Adapted Stage 1（动作头适配路线）

- 中文简称：动作适配路线
- 监督初始化：`stage1_data/weights/nomad_pg_zero_shot.pth`
- 监督阶段训练：
  - `vision_encoder.goal_encoder`，学习率 `1e-4`
  - `noise_pred_net`，学习率 `1e-5`
- Transformer 与视觉编码器冻结
- 最终监督 checkpoint：计划使用 `ema_19.pth`
- 零样本输出目录：`runs/nomad_pg_stage1_new_zero_shot`
- PPO 输出目录：`runs/nomad_pg_ppo_v2_stage1_new`
- 实验目的：验证先用轨迹监督适配 diffusion action head，是否能改善零样本能力、PPO 起点及最终性能

## 核心研究问题

1. 解冻 diffusion action head 是否提高监督训练后的 HM3D 零样本导航能力？
2. 动作头监督适配是否让 PPO 更快达到有效 SR/SPL？
3. 动作头适配是否牺牲视觉依赖，形成更强的 PointGoal shortcut？
4. 两条路线在相同 PPO 预算下，哪一条最终性能更高、更稳定？

## 公平对比约束

除监督初始化 checkpoint 外，下列 PPO 条件必须完全一致：

| 项目 | 固定值 |
| --- | --- |
| PPO seed | `270928` |
| 最大 epochs | 50 |
| 每轮 rollout steps | 8192 |
| PPO batch size | 256 |
| 每轮 PPO updates | 4 epochs |
| 学习率 | `1e-5` |
| clip ratio | 0.2 |
| value coefficient | 0.5 |
| entropy coefficient | 0.0 |
| 最大梯度范数 | 1.0 |
| 最大 episode steps | 500 |
| success distance | 0.2 m |
| waypoint spacing | 0.12 m |
| prediction horizon | 8 |
| controller | trajectory, lookahead index 2 |
| curriculum | `pointnav-curriculum/train_v2` |
| validation/test protocol | `protocol/validation.json.gz`、`protocol/test.json.gz` |

禁止为某一路线单独调整 reward、controller、episode 顺序、训练预算或 checkpoint 选择规则。

## 阶段一：监督 checkpoint 审计

每条路线记录：

- checkpoint 绝对路径、文件大小和 SHA-256
- 实际可训练模块
- 监督数据来源及采样权重
- batch size、epoch、学习率和 goal mask 概率
- 模型加载完整性，确认无 missing/unexpected keys

路线 F 已知关键条件：只训练 Goal MLP，`goal_mask_prob=0.0`。

路线 A 已知关键条件：训练 Goal MLP 与 diffusion head，`goal_mask_prob=0.5`，diffusion head 使用 0.1 倍学习率。

由于两条路线的监督配置不只相差一个变量，最终结论应表述为“整条训练路线的效果”，不能把差异全部归因于单独解冻 diffusion head。若要得到严格模块因果结论，需要补跑同 seed、同 batch、同 goal mask、仅改变 action head 冻结状态的监督消融。

## 阶段二：PPO 前零样本测试

在固定的100个 validation episodes 上分别测试：

1. `normal`：正常 RGB + 正常 PointGoal
2. `black`：全黑 RGB + 正常 PointGoal
3. `angle_flip`：正常 RGB + PointGoal 方向取反

每组记录：

- SR
- SPL
- 最终距离均值
- 碰撞数均值
- episode steps 均值
- 动作计数与动作分布
- 单步推理延迟

派生指标：

- 视觉依赖差值：`SR_normal - SR_black`、`SPL_normal - SPL_black`
- PointGoal 敏感度：`SR_normal - SR_angle_flip`
- 成功效率：成功 episode 的平均 steps 与 SPL

路线 A 已安排自动执行三组测试。为了严格对称，路线 F 也应使用同一脚本、同一100个 episodes 补跑三组独立零样本结果；PPO 内置 baseline normal/black 可作为交叉校验，但不能替代 angle-flip 测试。

## 阶段三：PPO 学习曲线对比

每个 epoch 对齐比较：

- rollout success、SPL、平均回报
- validation normal SR/SPL
- validation black SR/SPL
- policy loss、value loss、entropy、KL 或近似 KL
- clip fraction、梯度范数
- 碰撞率、episode 长度
- wall-clock time 和累计环境步数

重点报告：

- 首次达到 SR 0.10、0.20、0.30 所需环境步数
- 前5轮和前10轮的 AUC
- 50轮内最佳 SR/SPL
- 最终5轮均值与标准差
- 最佳 normal checkpoint 对应的 black 结果，避免只按 normal 选点掩盖视觉退化

## 阶段四：最终冻结测试

各路线只能按预先声明的规则选择一个 checkpoint：validation normal 的 `(SR, SPL)` 字典序最佳 checkpoint。

选定后在不可变 `test.json.gz` 上只运行一次：

- normal RGB
- black RGB
- angle-flipped PointGoal

最终报告同时给出 validation 选点依据和 test 结果，不允许看过 test 后重新挑 checkpoint。

## 判定规则

路线 A 被认为优于路线 F，需要同时满足：

1. 固定测试集 normal SR 或 SPL 有实质提升；
2. 提升不是仅来自更激进的碰撞或更长 episode；
3. black RGB 性能没有与 normal 同步上升到近似水平；
4. angle-flip 后性能明显下降，证明策略确实使用 PointGoal 方向；
5. PPO 样本效率或最终性能至少一项明确更好，另一项不能显著退化。

若路线 A 的 normal 指标提高、但 normal/black 差距进一步缩小，则标记为 `goal-only risk`，不能直接宣布动作适配成功。

## 当前执行状态

- 路线 F：PPO v2 已从旧 `ema_3.pth` 启动，输出到 `runs/nomad_pg_ppo_v2`。
- 路线 A：监督 Stage 1 正在运行；后台任务将在 `ema_19.pth` 生成后自动执行三组零样本测试，并启动相同 PPO v2 配置。
- 自动衔接日志：`runs/stage1_to_eval_ppo.log`

## 建议的最终结果表

| 路线 | 零样本 normal SR/SPL | normal-black 差值 | angle-flip 差值 | PPO-10 AUC | 最佳 val SR/SPL | test SR/SPL | 碰撞率 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F：Frozen-Head | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| A：Action-Adapted | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
