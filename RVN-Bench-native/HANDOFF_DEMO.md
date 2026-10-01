# RVN-Bench ViNT-PG Demo 交接文档

更新时间：2026-08-30（Asia/Shanghai）<br>
项目目录：`/root/autodl-tmp/RVN-Bench-native`<br>
目标文档：`Patch_Random-K_ViNT完整实验与快速Demo.md` v2.1

## 1. 最终目标与当前边界

最终目标是完成目标文档中的两级快速检测：

1. **Random Pool Probe**：在原生 ViNT-PG 的 EfficientNet 最终特征图上，用随机空间子集均值替换 Full-GAP，完成离线配对检测和固定闭环检测。
2. **Micro Information-Horizon Demo**：验证经过少量 Goal-conditioned patch token 交互后，Random-K 与 Top-K 的差距是否随插入深度缩小。

必须按门控顺序执行：

```text
原生 ViNT-PG 闭环基线可用
  -> Random Pool Probe
  -> Micro Horizon Demo
  -> 两者通过后才考虑完整 Tokenized ViNT / Dynamic-K
```

**当前仍在第一道门：修复原生 ViNT-PG 闭环基线。尚未开始 Random Pool Probe。**

## 2. 当前实时状态

截至本文写入时：

- 正在运行恢复样本混合微调，进程 PID 约为 `130526`。
- 配置：`models/gnms_levin/train/config/vint_pg_recovery_mix_ft5.yaml`
- 输出目录：`/root/autodl-tmp/RVN-Bench-stage1-runs/rvn_vint_native_stage1/vint_pg_recovery_mix20_ft5_2026_08_30_11_17_23`
- 日志：`/root/autodl-tmp/RVN-Bench-stage1-logs/train_recovery_mix_ft5.log`
- 训练规模：5 epoch，每个 epoch 约 4923 batch，batch size 64。
- 写文档时位于 epoch 1（日志编号从 1 显示）的约 42%，速度约 3.1 batch/s。
- GPU 状态正常，显存约 8.2 GB；训练无报错。
- 预计全部 5 epoch 训练约在本文写入后 2 小时左右完成。

快速查看：

```bash
cd /root/autodl-tmp/RVN-Bench-native
ps -eo pid,etimes,pcpu,pmem,state,args | grep -E 'vint_pg_recovery|train.py' | grep -v grep
nvidia-smi
tail -n 60 /root/autodl-tmp/RVN-Bench-stage1-logs/train_recovery_mix_ft5.log
find /root/autodl-tmp/RVN-Bench-stage1-runs/rvn_vint_native_stage1/vint_pg_recovery_mix20_ft5_2026_08_30_11_17_23 -maxdepth 1 -type f -name '*.pth' -ls
```

不要重复启动当前训练。先确认原进程是否仍存在、日志是否继续增长。

## 3. 已完成工作

### 3.1 原生数据与首次训练

- 已按 RVN-Bench 原生数据采集和 ViNT-PG 训练链路完成 100 个训练场景、每场景约 100 条轨迹。
- 专家数据：`/root/autodl-tmp/RVN-Bench-stage1-data/combined_100scenes`
- 原始训练/验证划分：`/root/autodl-tmp/RVN-Bench-stage1-data/splits_native_stage1`
- 首次 25 epoch 输出：
  `/root/autodl-tmp/RVN-Bench-stage1-runs/rvn_vint_native_stage1/vint_pg_native_100scene_accel_2026_08_29_19_36_01`
- 离线指标可以收敛，但固定 10-episode 闭环评测 SR=0，主要以碰撞结束。

### 3.2 闭环诊断

评测脚本：`scripts/run_vint_pg_gate.py`

已检查：

- 正常 RGB 与全黑 RGB；
- trajectory controller 与 yaw controller；
- 不同 lookahead；
- PointGoal 距离封顶；
- 动作并非恒定，能输出前进、左转、右转；
- 但上述诊断均未使 SR 脱离 0。

固定场景：

- `/root/autodl-tmp/RVN-Bench-stage1-eval/scenarios/rvn_val_fixed_10_32_stage1.yaml`
- 历史结果：`/root/autodl-tmp/RVN-Bench-stage1-eval/`

### 3.3 远目标微调

- 配置：`models/gnms_levin/train/config/vint_pg_native_fargoal_ft5.yaml`
- 输出：`/root/autodl-tmp/RVN-Bench-stage1-runs/rvn_vint_native_stage1/vint_pg_native_fargoal_ft5_2026_08_30_08_53_03`
- 共 5 个 checkpoint，闭环仍为 SR=0。
- 相对较好的 checkpoint 是 `3.pth`，平均到达 waypoint 约 0.8，但仍全部碰撞。
- 它被用作当前恢复微调的初始化权重。

### 3.4 恢复/负样本采集与混合

- 已采集 1000 条原生 `discrete_single_neg` 恢复轨迹，分布于 20 个场景、4 个 shard。
- 数据根目录：`/root/autodl-fs/RVN-Bench-stage1-recovery`
- 960 条满足核心字段对齐且长度大于 10，可用于 ViNT 训练。
- 587 条轨迹的 `discrete_action_data.yaml` 长度存在采集器裁剪遗留不一致；当前 ViNT 数据读取不使用该离散动作文件，核心位置、yaw、图像字段对齐。若后续方法需要离散动作标签，必须先修复或重新生成。
- 混合数据：`/root/autodl-fs/RVN-Bench-stage1-recovery/mixed_expert_recovery`
- 混合划分：`/root/autodl-fs/RVN-Bench-stage1-recovery/splits_recovery_mix20`
- 混合脚本：`scripts/prepare_recovery_mix.py`
- 训练条目：8000 条专家训练轨迹 + 960 条恢复轨迹重复 20 次，共 27200 个训练条目、8960 条唯一轨迹。
- 当前恢复样本约占采样量的 30%，以低学习率 `5e-5` 微调 5 epoch。

## 4. 当前训练若中断，如何恢复

先检查已有 checkpoint。训练脚本当前没有可靠的同目录断点续跑约定，最安全做法是：

1. 找到最新完整 `.pth`；
2. 复制一份新配置；
3. 将 `pretrained_checkpoint` 改为该 checkpoint；
4. 将 `epochs` 改为剩余 epoch 数；
5. 用新 run name 启动，保留旧目录和日志。

原始启动方式：

```bash
cd /root/autodl-tmp/RVN-Bench-native/models/gnms_levin/train
export PYTHONPATH=/root/autodl-tmp/RVN-Bench-native/models/gnms_levin/train:/root/autodl-tmp/RVN-Bench-native/models/diffusion_policy:/root/autodl-tmp/RVN-Bench-native
export CUDA_VISIBLE_DEVICES=0
/root/miniconda3/envs/habitat/bin/python -u train.py -c config/vint_pg_recovery_mix_ft5.yaml
```

不要覆盖现有配置、checkpoint 或日志；新实验使用新文件名和新 run name。

## 5. 训练完成后的立即动作

### 5.1 固定 10-episode checkpoint sweep

对恢复微调产生的每个 `0.pth` 到 `4.pth`，使用完全相同的固定场景、正常 RGB、trajectory controller、lookahead 2 做闭环评测。评测期间不要并行启动另一个 GPU 训练。


分别替换 checkpoint 编号和输出目录。记录每个 checkpoint 的：

- success rate / success 数；
- SPL；
- average waypoint reached；
- collision/km；
- action_counts 与 num_unique_actions。

按闭环成功率优先选择最佳 checkpoint；SR 相同时依次比较成功数、平均 waypoint、碰撞率，不能只按离线 loss 选权重。

### 5.2 基线通过标准

进入 Random Pool Probe 前至少满足：

- 固定正常 RGB 10 episodes 的 SR >= 20%（至少 2/10）；
- 动作不是恒定输出；
- RGB 遮挡后性能明显下降，说明模型确实使用视觉；
- 没有明显的坐标系、标签或控制器错配证据。

最佳 checkpoint 达标后，再跑 normal + occluded 对照，并扩展到固定 50 episodes。只有基线稳定，才能把 Full-GAP 当作 Probe 的合法参照。

### 5.3 如果恢复微调后仍为 SR=0

**停止扩数据、停止 Random-K 实现。** 优先做逐样本对齐审计：

1. 可视化输入的 6 帧 RGB、PointGoal 极坐标/笛卡尔表示、专家未来 5 个 waypoint；
2. 在同一坐标系叠加模型预测和专家轨迹；
3. 核对 `pg_rcs=True` 在采集、训练、agent 推理三处的一致性；
4. 核对 waypoint 单位、`metric_waypoint_spacing=0.25`、归一化与反归一化；
5. 对一条专家轨迹执行 teacher-forced replay，确认由轨迹转离散动作后的左右转方向正确；
6. 检查图像时间顺序、context size 5 与 agent 的 `obs_size=6`；
7. 用人工构造的直行、左转、右转预测验证 controller；
8. 检查训练目标是否因远目标/负样本采样产生与闭环不一致的短期轨迹。

若上述任一项不一致，先修正并用小数据 smoke test 验证，再重新训练；不要盲目扩大到 200–400 场景。

## 6. 基线通过后：Random Pool Probe 实施规格

### 6.1 最小实现位置

原生路径：

```text
EfficientNet.extract_features
  -> feature map [B*T, C, H', W']
  -> Full-GAP [B*T, C]
```

Probe 替换为：

```text
feature map
  -> flatten [B*T, N, C], N=H'*W'
  -> 按策略选 K 个空间位置
  -> 对 K 个位置求均值 [B*T, C]
  -> 保持 compress_obs_enc、ViNT temporal decoder、distance/waypoint heads 不变
```

不能改原生图像尺寸、PointGoal 表示、动作定义、控制器或 RVN 评测配置。建议新增独立模块和配置开关，不要直接破坏 Full-GAP 默认路径。

运行时必须保存真实形状，不能假设 `H' x W'`：

```text
input, feature_map, N, pooled_feature, context token shape
```

### 6.2 最小实验矩阵

- Full-GAP：100%；
- Random：75%、50%、25%，每个比例 5 个 mask seed；
- Stratified：50%、25%；
- Bottom：50%、25%；
- Feature-Norm Top-K：50%、25%。

必须采用均匀无放回采样。明确区分：

- 每帧独立 mask；
- 同一上下文共享 mask；
- 如测试 persistent mask，单独命名，不能混入主结果。

### 6.3 Level A：离线配对检测

先固定 500 个样本；时间允许扩到 2000。Full-GAP 与所有候选必须使用完全相同的样本顺序，输出：

- pooled feature cosine similarity；
- waypoint L1、ADE、FDE drift；
- distance prediction drift；
- 连续控制量差异；
- 离散 action agreement；
- 不同 mask seed 方差。

### 6.4 Level B：固定闭环检测

- 固定 50 个 rvn-val episodes；
- Full-GAP 与候选使用相同环境种子；
- 记录成功、碰撞、超时、轨迹分叉点；
- 最好的 2 个候选扩展到 200 episodes。

### 6.5 Probe 输出目录

```text
outputs/random_pool_probe/
├── shape_profile.json
├── offline_metrics.csv
├── episode_metrics.csv
├── mask_seed_metrics.csv
├── latency_breakdown.csv
├── waypoint_drift.png
├── performance_vs_keep_ratio.png
├── representative_success.mp4
└── representative_failure.mp4
```

报告中必须写明：

> 此实验在完整 EfficientNet 之后进行随机子集池化，只验证信息冗余，不代表正式方法的端到端加速结果。

### 6.6 Random Pool Probe 的 Go / Yellow / No-Go

Go 的主要参考：

- 50% Random 的 pooled feature cosine 中位数 >= 0.95；
- action agreement >= 90%；
- waypoint FDE 扰动不超过 Full 典型尺度的 10%；
- 50 个配对 episode 成功数不低于 Full 的 90%；
- 碰撞无明显系统性增加；
- 至少一个 Random/Stratified 接近 Feature-Norm Top-K。

Yellow：50% 略降但 75% 稳定，或 Random 方差较大、近障碍敏感。此时从 75% 起步并优先风险动态预算。

No-Go：75% 仍显著漂移、mask seed 导致大幅分叉、Top-K 稳定但 Random 持续失败，或随机遗漏障碍导致碰撞。此时转 Stratified/Safety-Anchored Random 或轻量 Top-K。

## 7. Random Pool Probe 通过后：Micro Horizon Demo

最小结构：

```text
冻结 EfficientNet 特征
  -> Spatial Tokenizer
  -> Goal 单向条件 + 3 个轻量 PreBlocks（Random 前无 Frame/全局旁路）
  -> 在 R0/R1/R2/R3 插入 Random-K 或 Top-K
  -> Random 后加入 Frame Token
  -> 2 个共享 PostBlocks
  -> 冻结或轻微微调的 ViNT 时序头
```

最小矩阵：训练集 5%–10%，1–3 epoch，插入位置 R0/R1/R2/R3，保留率 100%/50%/25%，Random/Feature-Norm Top-K/Attention Top-K，Random 至少 3 个 mask seed，固定 500–2000 离线样本和 50 配对闭环 episodes。

必须补一个 **Pre-Frame 旁路对照**，排除 Random 前已有全局 token 汇总全部图像信息。

通过标准：至少一个 keep ratio <= 50% 时 Random 与 Top-K 的性能差距随深度稳定缩小；深层差距不超过约 2–3 pp；多个 mask seed 不改变趋势；Token Information 方差下降方向一致；候选插入层之后仍有足够计算可节省。

输出：

```text
outputs/micro_horizon_probe/
├── layerwise_offline_metrics.csv
├── layerwise_closed_loop_metrics.csv
├── random_vs_topk_gap.csv
├── token_information_variance.csv
├── mask_seed_metrics.csv
├── gap_vs_depth.png
├── information_variance_vs_depth.png
└── performance_compute_pareto.png
```

## 8. 项目地图：不同文件、脚本、模型编号和数据类型是做什么的

### 8.1 `scripts/` 中各脚本的职责

| 文件 | 作用 | 本实验是否直接使用 | 注意事项 |
| --- | --- | --- | --- |
| `scripts/collect_traj_data.py` | 训练轨迹采集总入口。根据 `--model` 选择 pathfinder、普通离散专家或带负样本的采集器，根据 `--config_path` 决定场景、数量和保存方式 | 是 | 当前有为分片、续采和恢复样本采集所做的修改，不要覆盖 |
| `scripts/generate_seq_point_goal_scenario.py` | 生成闭环评测场景 YAML，包括场景、起点、朝向、32 个路径 waypoint 和 geodesic distance | 间接使用 | 这是“生成题目”，不执行模型；修改 seed 会产生另一套评测题，不能与历史结果直接配对比较 |
| `scripts/prepare_recovery_mix.py` | 检查恢复轨迹核心字段，给专家/恢复轨迹建立软链接，并生成重复 20 次的混合训练 split | 是，已执行 | 只负责组织数据，不训练；重复条目是过采样，不是复制 20 份图像 |
| `scripts/run_seq_point_goal_nav_exp.py` | RVN-Bench 通用闭环评测入口，可加载 NoMaD、ViNT-PG、PPO、DDPPO 等不同 agent | 原生参考 | 参数和输出更通用；本阶段为保证固定条件，优先使用专用 gate 脚本 |
| `scripts/run_vint_pg_gate.py` | 本实验新增的 ViNT-PG 固定闭环门控脚本；支持正常/遮挡 RGB、trajectory/yaw controller、lookahead 和 goal distance cap，并记录动作分布 | 是 | 用于判断基线能否进入 Demo，不能替代 Random Pool Probe 本身 |
| `scripts/train_depth_predicting_ppo_res_net_policy_lab.py` | 训练带深度预测辅助任务的 PPO ResNet 策略 | 否 | 属于 RVN-Bench 其他基线，与 ViNT-PG Demo 无关 |
| `scripts/train_ppo_vanilla_blind.py` | 训练不使用视觉的 blind PPO 基线 | 否 | 可作导航下界，但不是本任务模型 |

**相关实现层：**

- `agents/vint_agent.py`：把 ViNT-PG 连续 waypoint 预测包装成 RVN agent，并转成环境离散动作。
- `evaluator/seq_point_goal_nav_evaluator.py`：逐 episode 执行闭环导航并计算 SR、SPL、碰撞、超时等指标。
- `utils/habitat_action_utils.py`：把预测轨迹转为 `move_forward / turn_left / turn_right`；控制器对齐审计的重点。
- `models/gnms_levin/train/train.py`：ViNT/ViNT-PG **训练入口**，读取 `-c` 指定的 YAML。
- `models/gnms_levin/train/vint_train/training/pointgoal_train_utils.py`：PointGoal 数据批次和训练相关工具，当前有实验修改。
- `models/gnms_levin/train/vint_train/models/vint/vint_pointgoal.py`：原生 ViNT-PointGoal 网络定义；将来 Random Pool Probe 的模型侧入口之一。

### 8.2 数据采集器的 `--model` 类型

这里的 `--model` 指“用哪种规则采集轨迹”，不是选择要训练的神经网络：

| 类型 | 采集器 | 产出用途 |
| --- | --- | --- |
| `pathfinder` | `PathfinderDatasetCollector` | 使用 Habitat pathfinder 生成几何最短路径轨迹，偏理想专家数据 |
| `discrete` | `DiscreteActionDatasetCollector` | 用离散前进/左右转动作跟随规划路径，得到更接近 RVN 控制接口的专家轨迹 |
| `discrete_neg` | `DiscreteActionNegDatasetCollector` | 在离散专家轨迹中加入负样本/偏离，用于增强纠错能力 |
| `discrete_single_neg` | `DiscreteActionSingleNegDatasetCollector` | 每条轨迹注入单次偏离并恢复；本阶段 1000 条 recovery 数据使用此类型 |

离散动作固定映射：`0=move_forward`、`1=turn_left`、`2=turn_right`。

一条轨迹目录的关键内容：

- `0.jpg, 1.jpg, ...`：按时间排序的 RGB 观测；
- `traj_data.pkl`：二维 `position`（米）和 `yaw`（弧度），ViNT 由此构造相对目标和未来 waypoint；
- `discrete_action_data.yaml`：环境离散动作序列；当前 ViNT 连续 waypoint 训练不读取它，但动作分类/控制器研究会用；
- 数据配置 YAML：相机内参/外参、机器人尺寸、速度、时间步长和格式版本。

坐标转换不是原样取 Habitat 的 x/z：当前 `convert_traj_habitat_to_traj_data_2d` 使用 `[-z, -x]` 生成二维位置。这正是闭环 SR=0 时必须重点审计的坐标系环节。

### 8.3 数据目录和 split 的区别

| 路径/名称 | 内容 | 用途 |
| --- | --- | --- |
| `combined_100scenes` | 100 个训练场景采集到的专家轨迹实体 | 原生 ViNT-PG 主训练数据 |
| `splits_native_stage1/train` | 文本清单，列出主训练使用的约 8000 条轨迹名 | 决定哪些专家轨迹参与训练 |
| `splits_native_stage1/val` | 按场景隔离的约 2000 条验证轨迹名 | 离线验证；不能拿来训练 |
| `smoke_train / smoke_val` | 极小子集 | 只检查代码、数据加载、显存和保存流程，不可报告性能 |
| `RVN-Bench-stage1-recovery/recovery_shard0..3` | 1000 条带单次偏离的原始恢复轨迹 | 学习偏航、近障碍后的恢复行为 |
| `mixed_expert_recovery` | 专家和恢复轨迹的统一软链接视图 | 当前恢复微调的数据根目录 |
| `splits_recovery_mix20/train` | 8000 个专家条目 + 960 个恢复条目重复 20 次 | 当前 5-epoch 恢复微调训练 split |
| `splits_recovery_mix20/val` | 保持原专家验证集，不混入 recovery | 观察微调是否破坏原能力 |
| `data/scene_datasets/hm3d/rvn_val` | 闭环验证用 3D 场景资源 | 固定 gate 和 Demo 闭环评测 |
| `data/scene_datasets/hm3d/rvn_test` | 最终测试 3D 场景资源 | 不用于调参；方法冻结后终验 |

“数据目录”存轨迹实体；“split”通常只存 `traj_names.txt` 清单。重复 split 条目会提高采样概率，不会创建新的独立经验。训练集、离线验证集和闭环 rvn-val 是三种不同用途，指标不能混写。

### 8.4 训练配置文件命名与用途

| 配置 | 从哪里开始 | 训练设置 | 目的 |
| --- | --- | --- | --- |
| `vint_pg.yaml` | 仓库原始默认设置 | 原生参考 | 理解官方字段，非本阶段实际主配置 |
| `vint_pg_native_stage1_smoke.yaml` | zero-shot RCS 权重 | 1 epoch、batch 16、极小 split | 验证训练链路能跑通 |
| `vint_pg_native_stage1.yaml` | `vint_pg_zero_shot_rcs.pth` | 25 epoch、lr 5e-4、原始 100 场景数据 | 第一版完整原生训练配置 |
| `vint_pg_native_stage1_accel.yaml` | zero-shot 权重/早期 run 衔接 | batch 64、16 workers、缓存优化 | 实际完成的 25-epoch 加速训练分支 |
| `vint_pg_native_fargoal_ft5.yaml` | 主训练 `22.pth` | 5 epoch、lr 1e-4、distance/action max 33 | 增加更远目标覆盖；闭环仍 SR=0 |
| `vint_pg_recovery_mix_ft5.yaml` | far-goal `3.pth` | 5 epoch、lr 5e-5、专家+recovery mix20 | 当前正在运行，学习偏离后的恢复 |
| `vint_pg_nmd.yaml` | 仓库 NoMaD 风格变体 | 非本阶段 | 不要与原生 ViNT-PG 结果混用 |

名称后缀含义：

- `native`：保持 ViNT-PG 原生架构、损失、96×96 图像、动作定义和 RVN 评测接口；
- `stage1`：100 场景的第一阶段基线；
- `smoke`：仅做冒烟测试；
- `accel`：数据缓存/worker 等训练吞吐优化，不代表模型方法创新；
- `fargoal`：扩大距离和动作目标类别上限；
- `ft5`：fine-tune 5 个 epoch；
- `recovery_mix20`：恢复轨迹在 train split 中重复 20 次过采样。

### 8.5 权重来源、模型编号和 `latest.pth`

| 权重 | 含义 |
| --- | --- |
| `/root/autodl-fs/RVN-Bench-shared/weights/vint_pg_zero_shot_rcs.pth` | 初始 ViNT-PG/RCS 权重，用来迁移视觉编码器和形状兼容的 Transformer；不是本数据训练结果 |
| `.../vint_pg_native_100scene_accel.../0.pth` 到 `24.pth` | 100 场景主训练每个 epoch 结束保存的模型；数字是从 0 开始的 epoch 索引，所以 `24.pth` 是第 25 个 epoch |
| 主训练 `22.pth` | 离线比较后选来启动 far-goal 微调的权重，不表示“版本 22” |
| `.../vint_pg_native_fargoal_ft5.../0.pth` 到 `4.pth` | far-goal 的 5 个 epoch；闭环相对较好的是 `3.pth` |
| `.../vint_pg_recovery_mix20_ft5.../0.pth` 到 `4.pth` | 当前 recovery 微调的 5 个 epoch；必须全部闭环 sweep 后再决定最佳编号 |
| `latest.pth` | 每个 epoch 后覆盖，始终指向该 run 最近一次完整保存；方便恢复，不等于性能最佳 |

每个 run 目录名末尾的日期时间（如 `2026_08_30_11_17_23`）是启动时间，用来区分同名实验。模型编号只能在同一 run 内解释；不同目录的 `3.pth` 完全不是同一个模型。

选模型的规则：训练 loss/离线 waypoint 指标用于排除坏模型，最终以固定闭环的 SR、成功数、平均 waypoint 和碰撞率选择。禁止看到 `latest.pth` 或最大编号就默认它最好。

### 8.6 场景 YAML、轨迹数据和评测结果不是一回事

- 场景 YAML：定义“在哪个 3D 场景、从哪里出发、要依次到哪些 waypoint”，相当于试卷。
- 轨迹数据集：训练时监督模型的 RGB、位置、yaw 序列，相当于教材。
- checkpoint：训练得到的网络参数，相当于考生。
- result YAML：某个 checkpoint 在某套场景和控制器条件下的 SR/SPL/碰撞结果，相当于成绩单。

对比两个模型时，必须保持场景 YAML、episode 顺序、环境 seed、controller、lookahead、碰撞设置和 RGB 条件完全一致。否则结果不是配对实验。

### 8.7 Demo 阶段将新增的文件（目前尚未实现）

为避免与基线代码混淆，建议按目标文档新增：

- `models/vint_random_pool_probe.py`：只替换最终空间池化，支持 Full/Random/Stratified/Bottom/Feature-Norm；
- `scripts/run_random_pool_probe.py`：组织离线和闭环实验矩阵；
- `scripts/eval_offline_pairwise.py`：固定样本逐对比较 Full 与采样预测；
- `scripts/profile_latency.py`：记录特征提取、采样、池化和后续网络时间；
- `experiments/probe_random_pool.yaml`：所有 keep ratio、mask seed、checkpoint、样本和场景参数；
- `outputs/random_pool_probe/`：只放 Probe 结果，不混入基线 gate 结果。

Micro Demo 再新增 tokenizer、pre/post interactor、sampler 和单独配置。不要提前创建一个同时包含 Dynamic-K 的大模型，以免无法判断失败来自哪一层。

## 9. 已修改文件与数据保护

当前工作树不是干净状态。已有改动属于本实验，接手时不要 reset、checkout 或覆盖：

- `models/gnms_levin/train/train.py`
- `models/gnms_levin/train/vint_train/training/pointgoal_train_utils.py`
- `scripts/collect_traj_data.py`
- 多个 `configs/dataset_collector/*.yaml`
- `scripts/prepare_recovery_mix.py`
- `scripts/run_vint_pg_gate.py`
- 多个 `models/gnms_levin/train/config/vint_pg_*.yaml`


关键磁盘占用（写文档时）：

- recovery：约 11 GB；
- stage1-data：约 21 GB；
- stage1-runs：约 11 GB；
- eval：不足 1 MB（视频/图像输出增加后会增长）。

所有新结果使用独立目录；不删除原始数据、checkpoint、评测 YAML 或日志。

## 10. 推荐的接手执行清单

- [ ] 确认恢复微调进程正常，等待 5 epoch 完成。
- [ ] 对 5 个 checkpoint 做同一固定 10-episode sweep。
- [ ] 选择闭环最佳 checkpoint，跑 normal/occluded 对照。
- [ ] 若 SR >= 20% 且视觉依赖成立，扩展固定 50 episodes。
- [ ] 若仍 SR=0，执行坐标系/标签/controller 审计，停止扩数据和 Random-K。
- [ ] 基线通过后，新增 Random Pool Probe，不破坏 Full-GAP 默认路径。
- [ ] 先跑固定 500 样本离线配对矩阵。
- [ ] 再跑固定 50 episode 闭环矩阵，生成规定 CSV/图/视频。
- [ ] 按 Go/Yellow/No-Go 记录结论和证据边界。
- [ ] 只有 Probe 通过才实现 Micro Horizon Demo。
- [ ] Micro Demo 通过后，才进入完整 Patch Random-K / Dynamic-K。

## 11. 一句话状态

**原生 ViNT-PG 离线训练已完成，但闭环仍为 SR=0；当前正用 960 条恢复轨迹与 8000 条专家轨迹进行 5-epoch 混合微调。下一步不是直接写 Random-K，而是先用固定闭环评测证明基线可用，随后严格按 Random Pool Probe -> Micro Horizon Demo 完成 `demo.md`。**
