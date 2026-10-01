# anchor-safe 最佳检查点（Epoch 0）

本目录保存 `nomad_pg_ppo_v3_anchor_safe` 路线按检查点选择规则得到的最佳候选。

- 检查点：`best_sr_epoch000/policy_state.pth`
- 固定 Normal 验证：100 episodes，SR = **0.40**，SPL = **0.2351355**。
- 同轮 Black 验证：100 episodes，SR = **0.16**，SPL = **0.0876569**。
- Normal–Black SR 差：0.24；平均碰撞数分别为 142.40 与 195.10。

`policy_state.pth` 为可直接加载的策略 `state_dict`，不含 optimizer 或 rollout 缓存。`config.json` 记录本次训练设置，`history.json` 保存各轮训练与验证历史；最佳轮的两份固定验证原始结果位于 `best_sr_epoch000/`。
