import glob
import json
import os
import pickle
from pathlib import Path

import numpy as np
import yaml


EXPERT_ROOT = Path("/root/autodl-tmp/RVN-Bench-stage1-data/combined_100scenes")
RECOVERY_ROOT = Path("/root/autodl-fs/RVN-Bench-stage1-recovery")
MIX_ROOT = Path("/root/autodl-fs/RVN-Bench-stage1-recovery/mixed_expert_recovery")
SPLIT_ROOT = Path("/root/autodl-fs/RVN-Bench-stage1-recovery/splits_recovery_mix20")
EXPERT_SPLIT = Path("/root/autodl-tmp/RVN-Bench-stage1-data/splits_native_stage1")
RECOVERY_REPEAT = 20


def read_names(path: Path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def safe_symlink(target: Path, link: Path):
    if link.is_symlink():
        if Path(os.path.realpath(link)) != Path(os.path.realpath(target)):
            raise RuntimeError(f"Conflicting symlink: {link}")
        return
    if link.exists():
        raise RuntimeError(f"Refusing to overwrite: {link}")
    link.symlink_to(target)


def inspect_recovery(path: Path):
    with (path / "traj_data.pkl").open("rb") as stream:
        data = pickle.load(stream)
    with (path / "discrete_action_data.yaml").open("r") as stream:
        actions = yaml.safe_load(stream)["actions"]
    positions = np.asarray(data["position"])
    yaws = np.asarray(data["yaw"])
    images = glob.glob(str(path / "[0-9]*.jpg"))
    length = len(positions)
    core_aligned = positions.shape == (length, 2) and len(yaws) == length and len(images) == length
    return {
        "length": length,
        "core_aligned": core_aligned,
        "action_aligned": len(actions) == length - 1,
        "action_count": len(actions),
        "usable": core_aligned and length > 10,
    }


def main():
    MIX_ROOT.mkdir(parents=True, exist_ok=True)
    (SPLIT_ROOT / "train").mkdir(parents=True, exist_ok=True)
    (SPLIT_ROOT / "val").mkdir(parents=True, exist_ok=True)

    expert_train = read_names(EXPERT_SPLIT / "train" / "traj_names.txt")
    expert_val = read_names(EXPERT_SPLIT / "val" / "traj_names.txt")
    for name in sorted(set(expert_train + expert_val)):
        safe_symlink(Path(os.path.realpath(EXPERT_ROOT / name)), MIX_ROOT / name)

    reports = {}
    recovery_names = []
    for shard_dir in sorted(RECOVERY_ROOT.glob("recovery_shard*")):
        if not shard_dir.is_dir():
            continue
        for path in sorted(p for p in shard_dir.iterdir() if p.is_dir()):
            report = inspect_recovery(path)
            reports[str(path)] = report
            if not report["usable"]:
                continue
            name = f"recovery_{shard_dir.name}_{path.name}"
            safe_symlink(path, MIX_ROOT / name)
            recovery_names.append(name)

    mixed_train = expert_train + recovery_names * RECOVERY_REPEAT
    (SPLIT_ROOT / "train" / "traj_names.txt").write_text("\n".join(mixed_train) + "\n")
    (SPLIT_ROOT / "val" / "traj_names.txt").write_text("\n".join(expert_val) + "\n")

    summary = {
        "expert_train_trajectories": len(expert_train),
        "expert_val_trajectories": len(expert_val),
        "recovery_total": len(reports),
        "recovery_core_aligned": sum(r["core_aligned"] for r in reports.values()),
        "recovery_action_aligned": sum(r["action_aligned"] for r in reports.values()),
        "recovery_usable": len(recovery_names),
        "recovery_repeat": RECOVERY_REPEAT,
        "mixed_train_entries": len(mixed_train),
        "mixed_unique_trajectories": len(set(mixed_train)),
    }
    (SPLIT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (SPLIT_ROOT / "recovery_validation.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
