# ViNT-PG / RVN-Bench 实验历史

更新：2026-09-08。目标是在不使用 RVN/HM3D 轨迹作训练监督的前提下，训练能在 RVN-Bench 做 PointGoal 零样本评测的 ViNT-PG。

## 已完成的基线

| 名称 | 训练数据/初始化 | 代表结果 | 结论 |
| --- | --- | --- | --- |
| `gostandford_only` | GoStanford-only | Epoch 18：GoStanford val total loss 0.09340；distance loss 4.8854；action loss 0.13795 | 保留为单域基线。 |
| resume-from-57 | GoStanford-only，从旧 Epoch 57 接续 | 最佳 val 为 Epoch 16，total loss 约 0.1008 | 没有优于原始 Epoch 18。 |
| multidomain-init | 官方 ViNT 权重转换为 PointGoal 初始化后，仅 GoStanford 微调 | Epoch 36：val total loss 0.09568 | 视觉预训练初始化有效，但未优于原始单域基线。 |

RVN 的 2-episode gate 中，`gostandford_only`、多种 waypoint-to-discrete-controller 以及 multidomain-init 均得到 SR=0、SPL=0。官方 PPO demo 在同一环境可得到 SR=0.12，因此环境本身可运行。SR=0 对 SR=0 是地板效应，不能单独证明模型是否忽略 RGB。

## HuRoN 数据准备

- 来源文件：`sacson.h5`，已上传至服务器。
- HDF5 原始 split：train 2,067 条、test 515 条。
- 每条轨迹包含 JPEG `frames`、二维 `position`、弧度 `yaw`；字段长度已核对一致。
- 已服务器端转换为 ViNT 目录格式：2,582 条轨迹、约 1.4 GB，目录 `/root/autodl-tmp/huron_vint_20260904`。
- HuRoN 连续帧位移中位数：0.2456818 m；GoStanford 使用 0.12 m。二者分别归一化。

## 为什么停止旧路线

旧方案直接将新的 PointGoal MLP、视觉编码器和 Transformer 一起以同一学习率全量微调。PointGoal 是高信息量输入，模型可能学到“方向/距离到动作”的捷径，而不是根据 RGB 判断局部可通行性。此前黑 RGB 对照未显示出有区分力的下跌，这一风险必须被正式检测。

这不是说 PointGoal 不应存在：RVN-Bench 的 PointGoal 任务本来就会提供目标相对距离与方向。需要避免的是测试时或训练时只依赖它、完全不利用视觉。

## 新训练协议

所有阶段使用 GoStanford + HuRoN 原始轨迹，按 dataset-level 50/50 weighted sampling；不使用 RVN/HM3D 轨迹监督。

1. **Stage 1 — PointGoal token 适配**：冻结 EfficientNet 视觉编码器和 Transformer，仅训练新 PointGoal MLP 与预测头。目的是让 `[distance, cos(theta), sin(theta)]` 映射到预训练 Transformer 可用的 token 空间。
2. **Stage 2 — 小学习率融合**：解冻 Transformer，以远低于 MLP/头部的学习率训练；视觉编码器继续冻结或使用更低学习率。
3. **Stage 3 — 条件解冻视觉编码器**：只有 RGB 消融通过才允许解冻视觉编码器并进行全量微调。

## 必须记录的消融 gate

每个 checkpoint 在固定 GoStanford-val 和固定 RVN episodes 上评估：

- 正常 RGB + 正常 PointGoal；
- 全黑 RGB + 正常 PointGoal；
- 正常 RGB + 打乱 PointGoal；
- 全黑 RGB + 打乱 PointGoal。

报告 action/distance loss、waypoint 差异、动作序列、碰撞和 SR/SPL。正常 RGB 不能与黑 RGB 在这些连续指标上近似相同；否则记为 `goal-only failure`，不进入正式 RVN 评测。

## 保留与清理

保留：`gostandford_only` Epoch 18、官方多域初始化 Epoch 36、HuRoN 转换器、HuRoN 原始 HDF5 与转换结果、本文档。

清理：旧单域/续训/恢复/烟测配置和脚本；旧 run 中非上述保留 checkpoint。新阶段化训练不从旧 joint `latest.pth` 续训。
