import os
import sys
import argparse
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dataset_collector.pathfinder_dataset_collector import PathfinderDatasetCollector
from dataset_collector.discrete_action_dataset_collector import (
    DiscreteActionDatasetCollector,
)
from dataset_collector.discrete_action_neg_dataset_collector import (
    DiscreteActionNegDatasetCollector,
)
from dataset_collector.discrete_action_single_neg_dataset_collector import (
    DiscreteActionSingleNegDatasetCollector,
)

def main(args):
    curr_datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("args", args)

    dataset_name = (
        args.name if args.name is not None else f"{args.model}_{curr_datetime_str}"
    )
    save_dir = (
        os.path.join(os.path.dirname(__file__), "../data/trajectory_datasets")
        if args.save_dir is None
        else args.save_dir
    )

    dataset_save_dir = os.path.join(save_dir, dataset_name)
    debug_path_img_save_dir = os.path.join(save_dir, f"{dataset_name}_debug_path_img")
    os.makedirs(dataset_save_dir, exist_ok=True)

    collector_args = {
        "dataset_save_dir": dataset_save_dir,
        "seed": args.seed,
        "save_debug_path_img": args.save_debug_path_img,
        "debug_path_img_save_dir": debug_path_img_save_dir,
        "num_trials_traj_planning_per_scene": int(1e6),
    }
    if args.config_path is not None:
        collector_args["config_path"] = args.config_path

    if args.model == "pathfinder":
        dataset_collector = PathfinderDatasetCollector(**collector_args)
        dataset_collector.collect_data()
    elif args.model == "discrete":
        dataset_collector = DiscreteActionDatasetCollector(**collector_args)
        dataset_collector.collect_data()
    elif args.model == "discrete_neg":
        dataset_collector = DiscreteActionNegDatasetCollector(**collector_args)
        dataset_collector.collect_data()
    elif args.model == "discrete_single_neg":
        dataset_collector = DiscreteActionSingleNegDatasetCollector(**collector_args)
        dataset_collector.collect_data()
    else:
        raise ValueError(f"Invalid model: {args.model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect trajectory data")
    parser.add_argument(
        "--model",
        "-m",
        default="discrete_single_neg",
        type=str,
        help="Select the target model [pathfinder, discrete, discrete_neg, discrete_single_neg].",
    )
    parser.add_argument(
        "--config_path",
        "-cp",
        default="configs/dataset_collector/train_dataset_collector_d1.yaml",
        type=str,
        help="Path to config file.",
    )
    parser.add_argument(
        "--name",
        "-n",
        default=None,
        type=str,
        help="Name of the dataset to create. If not provided, a timestamp will be used.",
    )
    parser.add_argument(
        "--seed",
        "-s",
        default=0,
        type=int,
        help="Seed for random number generators.",
    )
    parser.add_argument(
        "--save_dir",
        "-sd",
        default=None,
        type=str,
        help="Path to config file.",
    )
    parser.add_argument(
        "--save_debug_path_img",
        action="store_true",
        help="Save per-trajectory debug path plots (disabled by default for faster collection).",
    )

    args = parser.parse_args()
    main(args)
