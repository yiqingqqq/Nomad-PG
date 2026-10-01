"""Initialize ViNT-PointGoal from the official ViNT visual/navigation weights."""

import argparse
from pathlib import Path

import torch

from vint_train.models.vint.vint_pointgoal import ViNTPointGoal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    source = checkpoint["model"]
    if hasattr(source, "module"):
        source = source.module
    source_state = source.state_dict()

    model = ViNTPointGoal(
        context_size=5,
        len_traj_pred=5,
        learn_angle=True,
        obs_encoder="efficientnet-b0",
        obs_encoding_size=512,
        late_fusion=False,
        mha_num_attention_heads=4,
        mha_num_attention_layers=4,
        mha_ff_dim_factor=4,
        pg_rcs=True,
    )
    target_state = model.state_dict()
    transferred = {
        key: value
        for key, value in source_state.items()
        if key in target_state and target_state[key].shape == value.shape
    }
    missing, unexpected = model.load_state_dict(transferred, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected transferred keys: {unexpected}")

    required_prefixes = (
        "obs_encoder.",
        "compress_obs_enc.",
        "decoder.",
        "dist_predictor.",
        "action_predictor.",
    )
    absent_required = [
        key for key in target_state
        if key.startswith(required_prefixes) and key not in transferred
    ]
    if absent_required:
        raise RuntimeError(f"Missing required official weights: {absent_required[:10]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model,
            "source_checkpoint": str(args.input),
            "transferred_keys": len(transferred),
            "randomly_initialized_keys": missing,
        },
        args.output,
    )
    print(
        f"saved {args.output}: transferred={len(transferred)}, "
        f"random_init={len(missing)}"
    )


if __name__ == "__main__":
    main()
