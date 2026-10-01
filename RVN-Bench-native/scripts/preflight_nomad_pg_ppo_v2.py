#!/usr/bin/env python3
"""Dependency-free migration/path preflight. Does not initialize Habitat or CUDA."""
import argparse, gzip, hashlib, json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(); p.add_argument("--migration-root",type=Path,required=True); a=p.parse_args()
    root=a.migration_root.resolve(); repo=root/"RVN-Bench-native"; protocol=root/"protocol"
    checkpoint=root/"checkpoints/rvn_nomad_gostandford_huron_stage1_resume/nomad_pg_stage1_gostandford_huron_resume_2026_09_24_11_38_06/ema_3.pth"
    scenes=root/"habitat-data/versioned_data/hm3d-0.2/hm3d"
    required=[repo/"scripts/train_nomad_pg_ppo_v2.py",root/"habitat-lab/habitat-lab/habitat/config/benchmark/nav/pointnav/pointnav_hm3d.yaml",
              root/"pointnav-curriculum/train_v2/direct.json.gz",root/"pointnav-curriculum/train_v2/turn.json.gz",
              root/"pointnav-curriculum/train_v2/detour.json.gz",protocol/"manifest.json",protocol/"validation.json.gz",
              protocol/"test.json.gz",checkpoint]
    missing=[str(x) for x in required if not x.is_file()]
    gz=[]
    for path in required:
        if path.suffix==".gz" and path.is_file():
            with gzip.open(path,"rt") as f: gz.append((path.name,len(json.load(f)["episodes"])))
    glb=sum(1 for _ in scenes.rglob("*.glb")); nav=sum(1 for _ in scenes.rglob("*.navmesh"))
    sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint.is_file() else None
    result={"ok":not missing and glb>0 and nav>0,"migration_root":str(root),"missing":missing,
            "hm3d_glb":glb,"hm3d_navmesh":nav,"episode_files":gz,"checkpoint_sha256":sha,
            "cuda_note":"CUDA and Python packages are checked only when the trainer starts."}
    print(json.dumps(result,indent=2)); raise SystemExit(0 if result["ok"] else 1)

if __name__=="__main__": main()
