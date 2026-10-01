#!/usr/bin/env python3
"""Select the best 30-episode mirror checkpoint screen result."""

import json
from pathlib import Path


result_dir = Path("/root/autodl-tmp/RVN-Bench-native/logs/hm3d_pointnav")
rows = []
for epoch in range(5):
    path = result_dir / f"vint_pg_{epoch}_trajectory_la2_normal_30ep.json"
    data = json.loads(path.read_text())
    rows.append(
        {
            "epoch": epoch,
            "success": data["success"],
            "spl": data["spl"],
            "mean_collisions": data["mean_collisions"],
            "source": str(path),
        }
    )

best = max(rows, key=lambda row: (row["success"], row["spl"], -row["mean_collisions"]))
selection = {"selection_rule": "success, then spl, then fewer collisions", "best": best, "candidates": rows}
output = result_dir / "mirror_checkpoint_selection.json"
output.write_text(json.dumps(selection, indent=2))
print(best["epoch"])
