# NoMaD-PG PointNav 迁移项目

本仓库以 `nomadpg_pointnav_migration_20260928` 为根目录，保留 PointNav NoMaD-PG 实验所使用的项目结构、代码、协议和可复现实验配置。

## 已包含内容

- `RVN-Bench-native/`：NoMaD-PG 的训练、验证与实验脚本。
- `habitat-lab/`：本项目固定使用的 Habitat-Lab 源码。
- `pointnav-curriculum/`、`protocol/`：课程配置和固定验证协议。
- `anchors/`、`checkpoints/`：实验初始化与检查点文件。
- 根目录下的训练总结、方案对比和迁移清单。

## 未上传的本地资源

为避免将大型、机器相关或可再生成的文件提交到 Git 仓库，以下内容未包含：HM3D 场景数据、Python/Conda 环境、运行时缓存、训练运行目录以及 stage-1 数据。

请单独准备 Habitat/HM3D 资源，并按照启动脚本要求放入 `habitat-data/`。当前 PPO 启动器及各方案脚本位于 `RVN-Bench-native/scripts/`。
