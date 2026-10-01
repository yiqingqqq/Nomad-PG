import os
import sys
import argparse
from datetime import datetime


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from evaluator.evaluation_scenario_generator import EvaluationScenarioGenerator


def main(args):
    curr_datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    randomize_start_rotation = True
    save_debug_plot = True
    allow_stairs = False

    num_ep_to_collect_per_scene = 20
    num_waypoints_per_ep = 32

    margin_to_obst = 0.1
    robot_radius = 0.18
    robot_height = 1.0

    min_path_length = 4.0
    max_path_length = 8.0

    path_finder_seed = 2508311909
    scenario_name = f"rvn_test_{num_ep_to_collect_per_scene}_{num_waypoints_per_ep}_{path_finder_seed}"

    # =============================== Set Paths ===============================
    curr_path = os.path.dirname(__file__)

    data_dir = os.path.join(curr_path, "../data/")
    dataset_type = "hm3d"
    # targert_scene_dir = "scene_datasets/hm3d/train/"
    targert_scene_dir = "scene_datasets/hm3d/rvn_test/"
    # scene_dataset_config_file = (
    #     "scene_datasets/hm3d/train/hm3d_annotated_train_basis.scene_dataset_config.json"
    # )
    scene_dataset_config_file = "scene_datasets/hm3d/rvn_test/hm3d_annotated_rvn_test_basis.scene_dataset_config.json"

    scenario_save_dir = os.path.join(
        curr_path, "../scenarios/seq_point_goal_nav_eval_scenarios/"
    )
    os.makedirs(scenario_save_dir, exist_ok=True)
    scenario_save_path = os.path.join(scenario_save_dir, f"{scenario_name}.yaml")

    if save_debug_plot:
        debug_img_save_dir = os.path.join(
            curr_path, f"../logs/scenario_generator/debug_imgs/{curr_datetime_str}"
        )
        os.makedirs(debug_img_save_dir, exist_ok=True)

    evaluation_scenario_generator = EvaluationScenarioGenerator()
    evaluation_scenario_generator.generate_scenario(
        scenario_save_path=scenario_save_path,
        data_dir=data_dir,
        randomize_start_rotation=randomize_start_rotation,
        allow_stairs=allow_stairs,
        num_ep_to_collect_per_scene=num_ep_to_collect_per_scene,
        num_waypoints_per_ep=num_waypoints_per_ep,
        margin_to_obst=margin_to_obst,
        robot_radius=robot_radius,
        robot_height=robot_height,
        min_path_length=min_path_length,
        max_path_length=max_path_length,
        target_scene_dir=targert_scene_dir,
        scene_dataset_config_file=scene_dataset_config_file,
        dataset_type=dataset_type,
        path_finder_seed=path_finder_seed,
        save_debug_plot=save_debug_plot,
        debug_img_save_dir=debug_img_save_dir,
    )


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
