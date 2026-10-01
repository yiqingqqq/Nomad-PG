"""Evaluate the official ViNT checkpoint on Habitat InstanceImageNav.

This is a diagnostic zero-shot adapter.  It deliberately uses only the RGB
history and the task-provided instance goal image when choosing actions.
Ground-truth distances are recorded after each step for diagnosis only and
never enter the controller.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, deque
from pathlib import Path

import habitat
import numpy as np
import torch
import torchvision.transforms.functional as tf
from PIL import Image
from torchvision import transforms

from vint_train.models.vint.vint import ViNT


NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)
IMAGE_SIZE = (85, 64)  # Official ViNT config: width, height.


def preprocess(rgb: np.ndarray) -> torch.Tensor:
    """Official deployment preprocessing: resize, then ImageNet normalization."""
    return NORMALIZE(tf.to_tensor(Image.fromarray(rgb).resize(IMAGE_SIZE)))


def load_official_model(checkpoint_path: Path, device: torch.device) -> ViNT:
    """Recreate the official architecture and strictly load its state dict."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_model = checkpoint["model"]
    model = ViNT(
        context_size=5,
        len_traj_pred=5,
        learn_angle=True,
        obs_encoder="efficientnet-b0",
        obs_encoding_size=512,
        late_fusion=False,
        mha_num_attention_heads=4,
        mha_num_attention_layers=4,
        mha_ff_dim_factor=4,
    )
    model.load_state_dict(saved_model.state_dict(), strict=True)
    return model.to(device).eval()


def choose_action(
    trajectory: np.ndarray,
    distance_prediction: float,
    stop_threshold: float | None,
    turn_threshold_deg: float,
) -> str:
    """Map the third predicted waypoint to Habitat's discrete action space."""
    if stop_threshold is not None and distance_prediction <= stop_threshold:
        return "stop"
    point = trajectory[2]
    heading = math.atan2(float(point[1]), float(point[0]))
    threshold = math.radians(turn_threshold_deg)
    if heading > threshold:
        return "turn_left"
    if heading < -threshold:
        return "turn_right"
    return "move_forward"


def vint_pd_command(
    trajectory: np.ndarray,
    distance_prediction: float,
    stop_threshold: float | None,
    time_step: float,
    max_linear_speed: float,
    max_angular_speed_deg: float,
) -> tuple[str, dict[str, object]]:
    """Port ViNT's official waypoint PD law to Habitat VelocityAction."""
    if stop_threshold is not None and distance_prediction <= stop_threshold:
        return "stop", {"action": "stop"}

    # Official navigate.py rescales normalized waypoint coordinates by MAX_V/RATE
    # before passing them to pd_controller.py.
    dx, dy = (float(value) * max_linear_speed / 4.0 for value in trajectory[2, :2])
    if abs(dx) < 1e-8:
        linear_speed = 0.0
        angular_speed = math.copysign(math.pi / (2 * time_step), dy)
    else:
        linear_speed = dx / time_step
        angular_speed = math.atan(dy / dx) / time_step

    linear_speed = float(np.clip(linear_speed, 0.0, max_linear_speed))
    max_angular_speed = math.radians(max_angular_speed_deg)
    angular_speed = float(
        np.clip(angular_speed, -max_angular_speed, max_angular_speed)
    )

    # Habitat VelocityAction expects normalized [-1, 1] inputs and maps them
    # onto the physical speed ranges configured below.
    normalized_linear = 2.0 * linear_speed / max_linear_speed - 1.0
    normalized_angular = angular_speed / max_angular_speed
    return "velocity_control", {
        "action": "velocity_control",
        "action_args": {
            "linear_velocity": float(normalized_linear),
            "angular_velocity": float(normalized_angular),
        },
    }


def agent_position(environment: habitat.Env) -> np.ndarray:
    """Return the simulator position without exposing it to the controller."""
    return np.asarray(environment.sim.get_agent_state().position, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="minival", choices=["train", "val", "minival"])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--stop-threshold", type=float, default=None)
    parser.add_argument("--turn-threshold-deg", type=float, default=15.0)
    parser.add_argument(
        "--controller", choices=["discrete", "vint_pd"], default="discrete"
    )
    parser.add_argument("--pd-time-step", type=float, default=0.25)
    parser.add_argument("--pd-max-linear-speed", type=float, default=0.2)
    parser.add_argument("--pd-max-angular-speed-deg", type=float, default=22.9183118052)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostic-images", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(
        "cpu" if args.device == "cpu" else
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    model = load_official_model(args.checkpoint, device)
    overrides = [
        f"habitat.dataset.split={args.split}",
        f"habitat.environment.max_episode_steps={args.max_steps}",
    ]
    if args.controller == "vint_pd":
        overrides.extend(
            [
                "+habitat/task/actions@habitat.task.actions.velocity_control=velocity_control",
                f"habitat.task.actions.velocity_control.lin_vel_range=[0.0,{args.pd_max_linear_speed}]",
                f"habitat.task.actions.velocity_control.ang_vel_range=[-{args.pd_max_angular_speed_deg},{args.pd_max_angular_speed_deg}]",
                f"habitat.task.actions.velocity_control.time_step={args.pd_time_step}",
            ]
        )
    config = habitat.get_config(
        "benchmark/nav/instance_imagenav/instance_imagenav_hm3d_v2.yaml",
        overrides=overrides,
    )
    environment = habitat.Env(config=config)
    context_length = model.context_size + 1
    action_counts: Counter[str] = Counter()
    episode_results = []

    if args.diagnostic_images:
        args.diagnostic_images.mkdir(parents=True, exist_ok=True)

    try:
        for episode_number in range(args.episodes):
            observation = environment.reset()
            history: deque[np.ndarray] = deque(
                [observation["rgb"]] * context_length, maxlen=context_length
            )
            goal_image = observation["instance_imagegoal"]
            initial_metrics = environment.get_metrics()
            initial_distance = float(initial_metrics["distance_to_goal"])
            min_distance = initial_distance
            predictions: list[float] = []
            waypoint_samples: list[np.ndarray] = []
            heading_samples: list[float] = []
            step_displacements: list[float] = []
            forward_displacements: list[float] = []
            collisions = 0
            trace = []

            if args.diagnostic_images and episode_number == 0:
                Image.fromarray(observation["rgb"]).save(args.diagnostic_images / "rgb_initial.png")
                Image.fromarray(goal_image).save(args.diagnostic_images / "instance_goal.png")

            for step in range(args.max_steps):
                observation_input = torch.cat(
                    [preprocess(frame).unsqueeze(0) for frame in history], dim=1
                ).to(device)
                goal_input = preprocess(goal_image).unsqueeze(0).to(device)
                with torch.inference_mode():
                    distance, trajectory = model(observation_input, goal_input)

                distance_prediction = float(distance.item())
                trajectory_prediction = trajectory[0].detach().cpu().numpy()
                selected_waypoint = trajectory_prediction[2]
                selected_heading = math.atan2(
                    float(selected_waypoint[1]), float(selected_waypoint[0])
                )
                if args.controller == "vint_pd":
                    action, habitat_action = vint_pd_command(
                        trajectory_prediction,
                        distance_prediction,
                        args.stop_threshold,
                        args.pd_time_step,
                        args.pd_max_linear_speed,
                        args.pd_max_angular_speed_deg,
                    )
                else:
                    action = choose_action(
                        trajectory_prediction,
                        distance_prediction,
                        args.stop_threshold,
                        args.turn_threshold_deg,
                    )
                    habitat_action = {"action": action}
                action_counts[action] += 1
                predictions.append(distance_prediction)
                waypoint_samples.append(trajectory_prediction)
                heading_samples.append(selected_heading)
                position_before = agent_position(environment)
                observation = environment.step(habitat_action)
                position_after = agent_position(environment)
                displacement = float(np.linalg.norm(position_after - position_before))
                step_displacements.append(displacement)
                if action == "move_forward":
                    forward_displacements.append(displacement)
                velocity_args = habitat_action.get("action_args", {})
                commanded_translation = (
                    action == "velocity_control"
                    and float(velocity_args.get("linear_velocity", -1.0)) > -0.999
                )
                collided = bool(environment.sim.previous_step_collided) or (
                    commanded_translation and displacement < 1e-4
                )
                collisions += int(collided)
                history.append(observation["rgb"])
                metrics = environment.get_metrics()
                min_distance = min(min_distance, float(metrics["distance_to_goal"]))
                if step < 10 or (step + 1) % 25 == 0:
                    trace.append(
                        {
                            "step": step + 1,
                            "action": action,
                            "collided": collided,
                            "displacement_m": displacement,
                            "distance_to_goal_m": float(metrics["distance_to_goal"]),
                            "distance_prediction": distance_prediction,
                            "selected_waypoint": selected_waypoint.tolist(),
                            "selected_heading_deg": math.degrees(selected_heading),
                        }
                    )
                if environment.episode_over:
                    break

            metrics = environment.get_metrics()
            waypoint_array = np.asarray(waypoint_samples)
            heading_array = np.asarray(heading_samples)
            forward_array = np.asarray(forward_displacements)
            episode_results.append(
                {
                    "episode_id": environment.current_episode.episode_id,
                    "scene_id": environment.current_episode.scene_id,
                    "steps": step + 1,
                    "initial_distance_m": initial_distance,
                    "final_distance_m": float(metrics["distance_to_goal"]),
                    "minimum_distance_m_diagnostic": min_distance,
                    "success": float(metrics["success"]),
                    "spl": float(metrics["spl"]),
                    "soft_spl": float(metrics["soft_spl"]),
                    "distance_prediction_min": min(predictions),
                    "distance_prediction_max": max(predictions),
                    "distance_prediction_mean": float(np.mean(predictions)),
                    "collision_count": collisions,
                    "collision_rate": collisions / (step + 1),
                    "total_translation_m": float(np.sum(step_displacements)),
                    "forward_zero_motion_count": int(np.sum(forward_array < 1e-4)),
                    "forward_zero_motion_rate": (
                        float(np.mean(forward_array < 1e-4))
                        if len(forward_array)
                        else None
                    ),
                    "forward_displacement_mean_m": (
                        float(np.mean(forward_array)) if len(forward_array) else None
                    ),
                    "selected_heading_deg_min": float(np.degrees(heading_array).min()),
                    "selected_heading_deg_max": float(np.degrees(heading_array).max()),
                    "selected_heading_deg_mean": float(np.degrees(heading_array).mean()),
                    "waypoint_xy_min": waypoint_array[..., :2].min(axis=(0, 1)).tolist(),
                    "waypoint_xy_max": waypoint_array[..., :2].max(axis=(0, 1)).tolist(),
                    "waypoint_xy_mean": waypoint_array[..., :2].mean(axis=(0, 1)).tolist(),
                    "trace": trace,
                }
            )
    finally:
        environment.close()

    result = {
        "checkpoint": str(args.checkpoint),
        "model": "official_vint",
        "split": args.split,
        "episode_count": len(episode_results),
        "max_steps": args.max_steps,
        "stop_threshold": args.stop_threshold,
        "controller": args.controller,
        "turn_threshold_deg": args.turn_threshold_deg,
        "pd_time_step": args.pd_time_step,
        "pd_max_linear_speed": args.pd_max_linear_speed,
        "pd_max_angular_speed_deg": args.pd_max_angular_speed_deg,
        "action_counts": dict(action_counts),
        "mean_success": float(np.mean([item["success"] for item in episode_results])),
        "mean_spl": float(np.mean([item["spl"] for item in episode_results])),
        "mean_soft_spl": float(np.mean([item["soft_spl"] for item in episode_results])),
        "episodes": episode_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
