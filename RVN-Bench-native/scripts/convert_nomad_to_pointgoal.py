#!/usr/bin/env python3
"""Convert an ImageGoal NoMaD state dict into a NoMaD-PointGoal initializer.

Only parameters whose names and shapes match are transferred. The PointGoal
MLP is initialized deterministically and the original checkpoint is untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml

from models.model_utils.sequor_gnm_utils import get_nomad_model


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    config = yaml.safe_load(args.config.read_text())
    model, _ = get_nomad_model(
        model_type="nomad_pointgoal",
        encoding_size=config["encoding_size"],
        context_size=config["context_size"],
        mha_num_attention_heads=config["mha_num_attention_heads"],
        mha_num_attention_layers=config["mha_num_attention_layers"],
        mha_ff_dim_factor=config["mha_ff_dim_factor"],
        down_dims=config["down_dims"],
        cond_predict_scale=config["cond_predict_scale"],
        num_diffusion_iters=config["num_diffusion_iters"],
        action_dim=2,
        pg_rcs=True,
    )

    source = torch.load(args.source, map_location="cpu", weights_only=True)
    target = model.state_dict()
    compatible = {
        key: value
        for key, value in source.items()
        if key in target and tuple(value.shape) == tuple(target[key].shape)
    }
    incompatible_shapes = {
        key: {"source": list(value.shape), "target": list(target[key].shape)}
        for key, value in source.items()
        if key in target and tuple(value.shape) != tuple(target[key].shape)
    }
    result = model.load_state_dict(compatible, strict=False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)

    report = {
        "source": str(args.source),
        "source_sha256": sha256(args.source),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "seed": args.seed,
        "source_keys": len(source),
        "target_keys": len(target),
        "compatible_keys": len(compatible),
        "compatible_target_numel_percent": round(
            100
            * sum(value.numel() for value in compatible.values())
            / sum(value.numel() for value in target.values()),
            6,
        ),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
        "incompatible_shapes": incompatible_shapes,
        "source_only_keys": sorted(set(source) - set(target)),
    }
    report_path = args.output.with_suffix(args.output.suffix + ".conversion.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
