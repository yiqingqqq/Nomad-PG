# NoMaD-PG 训练过程与数据变化汇总

更新时间：2026-10-01（Asia/Shanghai）

## 1. 一页结论

1. 新的监督 Stage-1 初始化明显优于旧初始化。PPO 的最佳点出现在 **Stage-1-new epoch 2**：固定验证集 Normal SR `38%`、SPL `21.49%`，同期旧路线只有 SR `20%`、SPL `10.29%`。
2. Stage-1-new 从 baseline 到 epoch 2，Normal SR 从 `23%` 增至 `38%`（`+15` 个百分点），SPL 从 `16.15%` 增至 `21.49%`（`+5.34` 个百分点）。
3. epoch 2 后出现非单调退化：Stage-1-new 到 epoch 6 降至 SR `21%`；旧路线在 epoch 5 直接降至 SR `0%`，epoch 6 也只有 `5%`。
4. **PPO loss 不能代表导航能力。**旧路线 epoch 6 的 loss 已降到 `0.00248`，但 rollout SR 为 `0%`、固定验证 SR 仅 `5%`，是典型策略坍塌。
5. **rollout SR 也不能代替固定验证 SR。**例如 Stage-1-new epoch 4 rollout SR 为 `41.67%`，固定验证 SR 却只有 `27%`；epoch 6 rollout SR 为 `37.5%`，固定验证 SR 只有 `21%`。
6. 黑图对照显示视觉依赖在 PPO 后增强：Stage-1-new baseline 的 Normal/Black SR 差仅 `4` 个百分点，epoch 2 扩大到 `22` 个百分点。但 Black SR 仍有 `16%`，说明 PointGoal 本身仍能支撑部分导航。
7. 最终冻结 epoch 2，并加入碰撞恢复控制器。验证 SR 保持 `38%`，SPL 从 `21.49%` 提升至 `22.13%`，平均碰撞从 `170.67` 降至 `138.91`。
8. 冻结测试集结果为：Normal SR `35%`、Black SR `22%`、PointGoal 角度翻转 SR `10%`。模型同时依赖视觉与 PointGoal，其中错误 PointGoal 的破坏显著大于去除 RGB。
9. 当前正在运行 anchor-safe 保守续训；截至本文快照只完成了续训前 Normal baseline，**尚未产生 epoch 0 更新结果**。

## 2. 阶段与数据口径

### 2.1 监督 Stage 1

- 初始化数据：GoStanford 与 HuRoN，dataset-level 50/50 weighted sampling。
- 不使用 HM3D/RVN 轨迹作为监督数据。
- 共训练 20 epochs，最终使用 `ema_19.pth`。
- 冻结 EfficientNet 和 Transformer。
- 训练 `vision_encoder.goal_encoder`（LR `1e-4`）与 `noise_pred_net`（LR `1e-5`）。
- batch size `16`，PointGoal masking probability `0.5`，AdamW，cosine scheduler，warmup 2 epochs，gradient clipping `1.0`，seed `270928`。

监督模型直接零样本闭环评测如下。这里使用 trajectory controller、无碰撞恢复；它与后面的 PPO 固定验证口径接近但并非同一执行配置，因此不应把细小差异解释成训练收益。

| 条件 | Episodes | SR | SPL | 最终距离 | 平均碰撞 | 平均步数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Normal RGB + 正常 PointGoal | 100 | 22% | 13.89% | 7.403 | 371.30 | 409.93 |
| Black RGB + 正常 PointGoal | 100 | 21% | 17.57% | 6.912 | 359.07 | 406.48 |
| Normal RGB + 角度翻转 PointGoal | 100 | 0% | 0% | 9.750 | 401.60 | 500.00 |

初步含义：Stage 1 后 PointGoal 已可用于导航，但 Normal/Black SR 仅差 1 个百分点，RGB 利用不足；角度翻转使 SR 归零，说明策略高度依赖目标方向。

### 2.2 PPO v2 的共同设置

- 两条路线唯一受控变量是初始化权重：旧 `ema_3.pth` 对比 Stage-1-new `ema_19.pth`。
- 每 epoch 收集 `8,192` transitions；PPO passes `4`；minibatch `256`；LR `1e-5`。
- clip `0.2`，GAE `gamma=0.99, lambda=0.95`，value coefficient `0.5`，gradient norm `1.0`。
- EfficientNet、PointGoal encoder、Transformer、distance head 均冻结。
- 更新 diffusion `noise_pred_net` 和新 value head。
- 每个 episode 最多 500 步；成功阈值 0.2 m。
- reward：裁剪后的 geodesic progress，步惩罚 `-0.01`，前进碰撞 `-0.10`，成功 STOP `+5.0`。
- 训练 curriculum 候选池为 direct/turn/detour 各 5,000 条。采样比例在 10 epochs 内从 `40/30/30` 逐渐变为 `20/20/60`。
- 固定验证集始终是同一批 100 episodes：10 direct、30 turn、60 detour。
- Normal RGB 用于选 checkpoint；Black RGB 仅作为视觉依赖诊断。

## 3. PPO 主结果：Stage-1-new

所有 Normal/Black 指标均来自同一固定 100-episode 验证集。Rollout 只包含完成 8,192 步所需的 21–24 个 episode，方差更大。

| 阶段 | PPO loss | Rollout ep | Rollout SR | Rollout SPL | Normal SR | Normal SPL | Normal 距离 | Black SR | Black SPL | N-B SR差 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | — | — | — | — | 23% | 16.15% | 7.625 | 19% | 16.10% | 4 pp |
| epoch 0 | 0.6284 | 21 | 28.57% | 19.95% | 29% | 17.55% | 6.709 | 10% | 6.63% | 19 pp |
| epoch 1 | 0.4362 | 21 | 23.81% | 15.48% | 33% | 18.97% | 6.477 | 17% | 9.75% | 16 pp |
| **epoch 2** | **0.4559** | **23** | **30.43%** | **18.99%** | **38%** | **21.49%** | **6.219** | **16%** | **8.53%** | **22 pp** |
| epoch 3 | 0.3486 | 23 | 34.78% | 22.03% | 32% | 20.30% | 6.365 | 17% | 10.98% | 15 pp |
| epoch 4 | 0.4752 | 24 | 41.67% | 27.35% | 27% | 17.38% | 6.623 | 16% | 10.22% | 11 pp |
| epoch 5 | 0.4240 | 24 | 29.17% | 20.99% | 31% | 22.32% | 6.671 | 17% | 15.96% | 14 pp |
| epoch 6 | 0.3532 | 24 | 37.50% | 26.86% | 21% | 15.47% | 7.159 | 14% | 12.56% | 7 pp |

关键变化：

- baseline → epoch 2：Normal SR `+15 pp`，SPL `+5.34 pp`，平均最终距离下降 `1.406 m`。
- epoch 2 → epoch 6：Normal SR `-17 pp`，SPL `-6.03 pp`，最终距离恶化 `0.940 m`。
- epoch 4 和 epoch 6 的 rollout 看起来很好，但固定验证明显下降，说明训练采样分布、episode 数量与固定验证之间存在显著偏差。
- 随 curriculum 增加 detour 比例，rollout 中 detour 数由 epoch 0 的 `7/21` 增至 epoch 6 的 `14/24`。性能回落与难例占比上升同时发生，但固定验证集没有变化，因此回落仍然代表策略参数发生了不利变化，而不只是评测变难。

## 4. PPO 对照结果：旧初始化

| 阶段 | PPO loss | Rollout ep | Rollout SR | Rollout SPL | Normal SR | Normal SPL | Normal 距离 | Black SR | Black SPL | N-B SR差 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | — | — | — | — | 20% | 11.56% | 7.521 | 24% | 18.89% | -4 pp |
| epoch 0 | 0.6231 | 26 | 38.46% | 30.24% | 23% | 11.63% | 6.737 | 14% | 9.51% | 9 pp |
| epoch 1 | 0.4927 | 20 | 25.00% | 12.63% | 22% | 11.88% | 6.676 | 14% | 9.01% | 8 pp |
| epoch 2 | 0.4881 | 22 | 31.82% | 15.49% | 20% | 10.29% | 7.194 | 10% | 6.36% | 10 pp |
| epoch 3 | 0.4566 | 20 | 15.00% | 7.45% | 19% | 11.09% | 7.142 | 13% | 8.00% | 6 pp |
| epoch 4 | 0.4845 | 24 | 37.50% | 21.21% | 20% | 10.81% | 7.051 | 8% | 3.68% | 12 pp |
| epoch 5 | 0.2458 | 18 | 11.11% | 6.47% | 0% | 0% | 8.470 | 0% | 0% | 0 pp |
| epoch 6 | 0.0025 | 17 | 0% | 0% | 5% | 1.86% | 8.768 | 0% | 0% | 5 pp |

旧路线的重要信号：

- baseline 的 Black 指标反而高于 Normal，表明旧初始化的 RGB 使用很弱或行为差异主要由随机/控制因素造成。
- epoch 5 出现明显坍塌；epoch 6 loss 几乎为零但性能没有恢复。
- 这条路线证明“优化目标收敛”不等于“导航策略改进”。

## 5. 两条路线的直接对比

| Epoch | New Normal SR | Old Normal SR | 差值 | New Normal SPL | Old Normal SPL | 差值 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 23% | 20% | +3 pp | 16.15% | 11.56% | +4.59 pp |
| 0 | 29% | 23% | +6 pp | 17.55% | 11.63% | +5.92 pp |
| 1 | 33% | 22% | +11 pp | 18.97% | 11.88% | +7.09 pp |
| 2 | 38% | 20% | +18 pp | 21.49% | 10.29% | +11.20 pp |
| 3 | 32% | 19% | +13 pp | 20.30% | 11.09% | +9.21 pp |
| 4 | 27% | 20% | +7 pp | 17.38% | 10.81% | +6.57 pp |
| 5 | 31% | 0% | +31 pp | 22.32% | 0% | +22.32 pp |
| 6 | 21% | 5% | +16 pp | 15.47% | 1.86% | +13.61 pp |

Stage-1-new 在所有已完成 epoch 上均优于旧路线，但自身仍存在 epoch 2 后的遗忘/漂移问题。

## 6. Epoch 2 checkpoint 与碰撞恢复筛选

checkpoint 只按固定 Normal validation 的 `(SR, SPL)` 字典序选择，因此冻结 Stage-1-new epoch 2。恢复策略先在固定的前 30 个 validation episodes 上筛选：

| 控制器 | SR | SPL | 平均碰撞 |
| --- | ---: | ---: | ---: |
| 无恢复 | 50.0% | 26.51% | 160.4 |
| goal-turn，1 step，threshold 1 | 16.7% | 11.25% | 156.6 |
| goal-turn，1 step，threshold 3 | 33.3% | 19.80% | 190.8 |
| goal-turn，3 steps，threshold 1 | 13.3% | 10.37% | 86.2 |
| **alternate，1 step，threshold 3** | **50.0%** | **27.93%** | **120.0** |

选定规则是：连续 3 次 forward collision 后，交替执行一次左/右恢复转向。随后在完整 100-episode validation 上确认：

| 设置 | SR | SPL | 最终距离 | 平均碰撞 |
| --- | ---: | ---: | ---: | ---: |
| epoch 2，无恢复 | 38% | 21.49% | 6.219 | 170.67 |
| epoch 2，选定恢复 | 38% | 22.13% | 6.080 | 138.91 |
| 变化 | 0 pp | +0.63 pp | -0.139 m | -31.76（-18.6%） |

恢复策略没有提高成功数，但路径效率和碰撞数均改善，因此被固定；测试结果不能反向修改这一选择。

## 7. 冻结 test 结果

| 条件 | Episodes | SR | SPL | 最终距离 | 平均步数 | 平均碰撞 | 前进率 | 最大连续转向 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Normal RGB + 正常 PointGoal | 100 | 35% | 19.48% | 6.387 | 380.25 | 157.77 | 48.64% | 183 |
| Black RGB + 正常 PointGoal | 100 | 22% | 11.04% | 7.225 | 430.18 | 195.31 | 50.22% | 202 |
| Normal RGB + 角度翻转 PointGoal | 100 | 10% | 3.48% | 7.786 | 466.82 | 182.68 | 46.40% | 495 |

相对 Normal：

- Black：SR `-13 pp`，SPL `-8.44 pp`，最终距离 `+0.838 m`，碰撞 `+37.54`。
- 角度翻转：SR `-25 pp`，SPL `-16.00 pp`，最终距离 `+1.399 m`；最大连续转向由 `183` 增到 `495`。

解释边界：Black 下降支持“模型使用了 RGB”，但 Black SR 仍有 `22%`，不能声称模型主要依赖视觉；角度翻转影响更大，说明 PointGoal 是更强的控制信号。

## 8. 当前 anchor-safe 保守续训

为了避免从退化的 epoch 6/7 继续，续训从冻结 epoch 2 权重重新开始：

- LR 从 PPO v2 的 `1e-5` 降到 `2e-6`。
- PPO passes 从 `4` 降到 `1`。
- anchor KL coefficient `1e-4`。
- anchor parameter L2 coefficient `1e-2`。
- 使用已冻结的 alternate/1-step/threshold-3 恢复控制器。
- Normal validation 前进率若低于 `0.15`，或最大连续转向超过 `250`，立即回滚并停止。
- 视觉编码器和 distance head 继续冻结，只更新 diffusion action head 与 value head。

当前已完成的续训前 baseline：Normal SR `38%`、SPL `22.13%`、最终距离 `6.073`、平均碰撞 `138.78`、前进率 `44.72%`、最大连续转向 `188`。该结果与恢复策略确认结果一致。Black baseline 尚未完成，epoch 0 尚未产生，因此当前阶段不能评价 anchor regularization 是否有效。

## 9. 建议同学重点分析的问题

1. **逐 episode 配对变化**：比较 epoch 2 与 epoch 3–6，统计哪些固定 episode 从成功变失败，按 direct/turn/detour、初始距离、场景和碰撞数分层。
2. **课程难度与遗忘**：检查 detour 占比提升是否导致对 direct/turn 能力的灾难性遗忘；固定验证集上的分类 SR 比总 SR 更重要。
3. **rollout/validation 分布偏差**：不要对 17–26 个 rollout episode 的 SR 做显著性结论；分析 rollout episode 长度、类别和成功终止比例。
4. **loss 与行为脱钩**：重点看 actor、critic、action frequency、turn streak、STOP 行为，而不是总 loss。旧 epoch 5–6 是主要反例。
5. **视觉依赖**：使用同 episode 的 Normal/Black 成败翻转表，而不只比较均值；区分“Black 下失败”和“Normal/Black 都成功”的样本。
6. **PointGoal 依赖**：角度翻转造成长转向链，建议分析左右转向计数、最大 turn streak 与 heading error 的关系。
7. **恢复控制器作用**：恢复控制器主要减少碰撞、略增 SPL，没有增加验证成功数。分析它是否只是改善已成功 episode，还是也改变失败 episode 的最终距离。
8. **续训判据**：anchor-safe 每个 epoch 必须与冻结 anchor 做配对比较；若 SR 不升、SPL 不升或视觉差距收窄，不应因为 PPO loss 下降而继续。

## 10. 数据使用注意事项

- `SR` 每 100 episodes 的最小变化单位是 1 个百分点；小于约 3–5 pp 的变化需结合逐 episode 配对结果谨慎解释。
- validation 用于选择 checkpoint；test 只能在选择冻结后使用，不能再据 test 调参。
- Normal/Black 是同一批 episode 和确定性 episode seed，适合做配对分析。
- rollout 的 episode 集合和数量每个 epoch 不同，只能视为训练诊断。
- 语义标注缺失警告与本 PointNav 任务无关，不应计为失败原因。
- 当前 v3 结果仍在生成，未完成记录不得和 v2 完整 epoch 并表做最终结论。

## 11. 原始数据位置

- Stage-1-new PPO history：`runs/nomad_pg_ppo_v2_stage1_new/history.json`
- Old PPO history：`runs/nomad_pg_ppo_v2/history.json`
- 每 epoch 原始验证 rows：对应 run 下的 `eval/epoch_NNN/validation_{normal,black}.json`
- 冻结锚点：`anchors/route_a_epoch002/policy_state.pth`
- 恢复控制器筛选：`runs/route_a_epoch2_recovery_screen/`
- 冻结测试：`runs/route_a_epoch2_frozen_test/eval/selected_epoch002/`
- 当前保守续训：`runs/nomad_pg_ppo_v3_anchor_safe/`
- 当前实时日志：`runs/nomad_pg_ppo_v3_anchor_safe.log`

