import os
from datetime import datetime
from typing import Optional
import threading

import yaml
import numpy as np
import torch
import quaternion as qt
import matplotlib
import pyzed.sl as sl
import cv2
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

matplotlib.use("Agg")

from agents.agent import Agent, ObsType, InfoType

import utils.metric_utils as MetricUtils
import utils.plot_utils as PlotUtils
import utils.camera_utils as CameraUtils

SEED = 1

ROTATE_AMOUNT = np.pi / 6  # 30 degree


class CmdVelPublisher(Node):
    def __init__(self):
        super().__init__("cmd_vel_publisher")
        self.publisher_ = self.create_publisher(
            Twist, "/jackal_velocity_controller/cmd_vel", 10
        )

        # Internal state for publishing
        self.timer = None
        self.count = 0
        self.max_count = 0
        self.msg = None
        self.stop_msg = Twist()

    def publish_cmd_vel(self, linear_x=1.0, angular_z=0.0, rate=10, times=20):
        """Publish a Twist at given rate and count."""
        self.count = 0
        self.max_count = times

        # Build message
        self.msg = Twist()
        self.msg.linear.x = linear_x
        self.msg.angular.z = angular_z

        # Create timer at given rate (seconds per tick)
        period = 1.0 / rate
        if self.timer is not None:
            self.timer.cancel()
        self.timer = self.create_timer(period, self.timer_callback)

    def timer_callback(self):
        if self.count < self.max_count:
            self.publisher_.publish(self.msg)
            self.get_logger().info(
                f"Publishing {self.count+1}/{self.max_count}: {self.msg}"
            )
            self.count += 1
        else:
            self.publisher_.publish(self.stop_msg)
            self.timer.cancel()
            self.timer = None
            self.get_logger().info("Finished publishing sequence.")


class RobotState:
    """
    Representation of a real robot.
    have position and yaw.
    yaw is in radian.
    """

    def __init__(self, init_position: np.ndarray, init_yaw: float, args=None):
        self.position = init_position
        self.yaw = init_yaw

        rclpy.init(args=args)
        self._node = CmdVelPublisher()
        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(self._node)
        executor_thread = threading.Thread(target=executor.spin, daemon=True)
        executor_thread.start()

    def get_state(self) -> tuple[np.ndarray, float]:
        return self.position, self.yaw

    def set_state(self, position: np.ndarray, yaw: float):
        self.position = position
        self.yaw = yaw

    def move_forward(self, amount: float) -> bool:
        collision = False
        # TODO:
        # MOVE ROBOT
        user_input = input("Press Enter to move the robot forward...")
        if user_input.lower() == "c":
            collision = True
            print("Simulating collision.")
            return collision

        self._node.publish_cmd_vel(linear_x=0.25, angular_z=0.0, rate=25, times=27)

        dx = amount * np.cos(self.yaw)
        dy = amount * np.sin(self.yaw)
        self.position += np.array([dx, dy])

        return collision

    def rotate(self, angle: float):
        collision = False

        # TODO:
        # ROTATE ROBOT
        user_input = input(f"Press Enter to rotate the robot by {angle}...")
        if user_input.lower() == "c":
            collision = True
            print("Simulating collision.")
            return collision

        self._node.publish_cmd_vel(
            linear_x=0.0, angular_z=angle * 1.13, rate=50, times=50
        )

        self.yaw += angle
        self.yaw = (self.yaw + np.pi) % (2 * np.pi) - np.pi  # normalize to [-pi, pi]

        return collision


class RealWorldEvaluator:
    """
    Instead of using a sim, get real-world obs.
    """

    def __init__(
        self,
        scenario_dir: str = "scenarios/real_world_scenario/",
        scenario_name: str = "guro_debug_0",
        rvn_agent: Agent = None,
    ):
        self._scenario_name = scenario_name
        self._rvn_agent = rvn_agent
        # set scenario
        scenario_path = os.path.join(scenario_dir, f"{scenario_name}.yaml")
        with open(scenario_path, "r") as file:
            self._eval_scenario = yaml.safe_load(file)

    def evaluate(
        self,
        target_model: str = "random",  # name of target model
        max_ep_steps: int = int(1e7),
        max_wp_steps: int = 100,
        dist_to_goal_threshold: float = 0.36,
        move_amount: float = 0.25,
        allow_collision: bool = False,
        result_save_dir: str = "logs/eval_results/real_world_exp/results",
        save_plot: bool = False,
        plot_save_dir: str = "logs/eval_results/real_world_exp/debug_images",
        seed: int = SEED,
        weight_path: Optional[str] = None,  # path of the loaded weight of model
        obs_size: int = 1,
    ):
        assert (
            self._rvn_agent is not None
        ), "Agent not set. Set agent using set_rvn_model() method."
        assert (
            self._eval_scenario is not None
        ), "Scenario not set. Set scenario using set_scenario() method."

        self._init_camera(fy_dst=155.0)

        self._set_seed(seed)

        eval_start_datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(result_save_dir, exist_ok=True)

        self._target_model = target_model
        self._weight_path = weight_path
        self._obs_size = obs_size

        self._save_plot = save_plot
        if self._save_plot:
            self._plot_save_dir = os.path.join(plot_save_dir, eval_start_datetime_str)
            self._context_plot_save_dir = os.path.join(self._plot_save_dir, "context")
            self._raw_plot_save_dir = os.path.join(self._plot_save_dir, "raw")

            os.makedirs(self._plot_save_dir, exist_ok=True)
            os.makedirs(self._context_plot_save_dir, exist_ok=True)
            os.makedirs(self._raw_plot_save_dir, exist_ok=True)

        self._max_ep_steps = max_ep_steps
        self._max_wp_steps = max_wp_steps

        self._dist_to_goal_threshold = dist_to_goal_threshold
        self._move_amount = move_amount
        self._allow_collision = allow_collision

        self._init_eval_results(result_save_dir, eval_start_datetime_str)

        self._loaded_scene_id: str = None
        for ep_idx, episode in enumerate(self._eval_scenario["episodes"]):
            self._eval_episode(ep_idx, episode)

        self._calcualte_metrics()
        self._save_eval_results()

        self._close_camera()

    def _init_camera(
        self,
        w_dst=256,
        h_dst=256,
        fx_dst=128.0,
        fy_dst=128.0,
        cx_dst=128.0,
        cy_dst=128.0,
    ):
        self._zed_camera = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD1200
        init_params.camera_fps = 30
        init_params.depth_mode = sl.DEPTH_MODE.NONE
        init_params.sdk_verbose = 1

        err = self._zed_camera.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"Camera Open error: {repr(err)}. Exit.")
            exit()

        # source intrinsics from ZED
        cam_info = self._zed_camera.get_camera_information()
        calib = cam_info.camera_configuration.calibration_parameters
        src_params = calib.left_cam
        src_fx, src_fy = src_params.fx, src_params.fy
        src_cx, src_cy = src_params.cx, src_params.cy
        w_src = cam_info.camera_configuration.resolution.width
        h_src = cam_info.camera_configuration.resolution.height

        print(f"Zed:{w_src}x{h_src} fx:{src_fx} fy:{src_fy} cx:{src_cx} cy:{src_cy}")

        k_src = CameraUtils.build_intrinsics_matrix(src_fx, src_fy, src_cx, src_cy)
        rgb_img_size_src = (w_src, h_src)

        k_dst = CameraUtils.build_intrinsics_matrix(fx_dst, fy_dst, cx_dst, cy_dst)
        rgb_img_size_dst = (w_dst, h_dst)

        self._map_x, self._map_y = CameraUtils.make_remap(
            k_src, rgb_img_size_src, k_dst, rgb_img_size_dst
        )

        self._zed_image = sl.Mat()
        self._zed_runtime = sl.RuntimeParameters()

    def _close_camera(self):
        self._zed_camera.close()

    def _set_seed(self, seed):
        # set seed
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _init_eval_results(
        self,
        result_save_dir,
        eval_start_datetime_str,
    ):
        self._eval_results = {}
        self._eval_results["scenario"] = self._scenario_name
        self._eval_results["model"] = self._target_model
        self._eval_results["weight_path"] = self._weight_path
        self._eval_results["episode_results"] = []
        self._eval_results["max_ep_steps"] = self._max_ep_steps
        self._eval_results["max_wp_steps"] = self._max_wp_steps
        self._eval_results["dist_to_goal_threshold"] = self._dist_to_goal_threshold
        self._eval_results["move_amount"] = self._move_amount
        self._eval_results["allow_collision"] = self._allow_collision
        self._eval_results["num_waypoints_per_ep"] = self._eval_scenario[
            "num_waypoints_per_ep"
        ]

        self._total_steps = 0
        self._total_distance_traveled = 0

        self._num_successes = 0
        self._num_collisions = 0
        self._num_ep_evaluated = 0
        self._num_ep_timeout = 0
        self._num_wp_timeout = 0

        self._result_file_path = os.path.join(
            result_save_dir,
            f"results_{eval_start_datetime_str}_{self._scenario_name}_{self._target_model}.yaml",
        )

    def _get_rgb_sensor_observation(self):
        print("Getting RGB observation from ZED X 2.2mm")
        while True:
            if self._zed_camera.grab(self._zed_runtime) == sl.ERROR_CODE.SUCCESS:
                self._zed_camera.retrieve_image(self._zed_image, sl.VIEW.LEFT)
                ts = self._zed_camera.get_timestamp(sl.TIME_REFERENCE.IMAGE)
                print(f"Image fetched at {ts.get_milliseconds()} ms")

                bgra = self._zed_image.get_data()
                if bgra is None or bgra.size == 0:
                    continue
                bgr_src = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
                bgr_dst = CameraUtils.convert_to_intrinsics_bgr(
                    bgr_src, self._map_x, self._map_y
                )
                rgb_dst = cv2.cvtColor(bgr_dst, cv2.COLOR_BGR2RGB)
                return rgb_dst
            else:
                time.sleep(0.001)

    def _save_raw_rgb(self):
        while True:
            if self._zed_camera.grab(self._zed_runtime) == sl.ERROR_CODE.SUCCESS:
                self._zed_camera.retrieve_image(self._zed_image, sl.VIEW.LEFT)
                ts = self._zed_camera.get_timestamp(sl.TIME_REFERENCE.IMAGE)
                print(f"Image fetched at {ts.get_milliseconds()} ms")

                bgra = self._zed_image.get_data()
                if bgra is None or bgra.size == 0:
                    continue
                rgb_src = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)
                # save rgb_src
                cv2.imwrite(
                    os.path.join(
                        self._raw_plot_save_dir, f"{ts.get_milliseconds()}.png"
                    ), rgb_src[:, :, ::-1]
                )
                return
            else:
                time.sleep(0.001)

        pass

    def _get_pointgoal_with_gps_compass(self, goal_position, goal_yaw) -> np.ndarray:
        """
        Returns: radius, yaw
        """
        robot_position, robot_yaw = self._robot_state.get_state()

        vec_to_goal = goal_position - robot_position
        radius = np.linalg.norm(vec_to_goal)
        angle_to_goal = np.arctan2(vec_to_goal[1], vec_to_goal[0])
        yaw = angle_to_goal - robot_yaw
        yaw = (yaw + np.pi) % (2 * np.pi) - np.pi  # normalize to [-pi, pi]

        return np.array([radius, yaw], dtype=np.float32)

    def _get_obs_info(
        self,
        rgb_obs_history: list[np.ndarray],
        goal_position: np.ndarray,
        goal_yaw: float,
    ) -> tuple[ObsType, InfoType]:
        pointgoal_with_gps_compass = self._get_pointgoal_with_gps_compass(
            goal_position, goal_yaw
        )
        obs: ObsType = {
            "rgb": np.stack(rgb_obs_history),
            "pointgoal_with_gps_compass": pointgoal_with_gps_compass,
        }
        info: InfoType = {"goal_rgb": np.zeros((256, 256, 3), dtype=np.uint8)}

        return obs, info

    def _step_real_robot_and_get_rgb(self, action: str, obs: ObsType) -> np.ndarray:
        # TODO:
        print(f"Stepping real robot with action {action}")
        wp_reached = False

        if action == "move_forward":
            distance_moved = self._move_amount
            collision_occured = self._robot_state.move_forward(self._move_amount)
        elif action == "turn_left":
            distance_moved = 0.0
            collision_occured = self._robot_state.rotate(ROTATE_AMOUNT)
        elif action == "turn_right":
            distance_moved = 0.0
            collision_occured = self._robot_state.rotate(-ROTATE_AMOUNT)

        # wait for robot to stabilize and save rgb
        for _ in range(6):
            self._save_raw_rgb()
            time.sleep(0.1)
        time.sleep(0.4)

        rgb_obs = self._get_rgb_sensor_observation()

        robot_position, robot_yaw = self._robot_state.get_state()

        print(f"robot: {robot_position}, {robot_yaw}")
        print(f"goal : {self._current_goal_position}, {self._current_goal_yaw}")
        print(f"gps_compass: {obs['pointgoal_with_gps_compass']}")
        user_input = input(
            "g: wp_reached, c: collision, r: reset robot pose, Enter: no collision"
        )
        if user_input.lower() == "g":
            wp_reached = True
            print("Simulating waypoint reached.")
        elif user_input.lower() == "c":
            collision_occured = True
            print("Simulating collision.")
        elif user_input.lower() == "r":
            while True:
                # Taking three float inputs in one line
                try:
                    new_robot_x, new_robot_y, new_robot_yaw = map(
                        float,
                        input("New robot x y yaw: ").split(),
                    )
                    if (
                        input(
                            f"Resetting robot to {new_robot_x}, {new_robot_y}, {new_robot_yaw} (y/n)? "
                        ).lower()
                        == "y"
                    ):
                        break
                except ValueError:
                    print(
                        "Invalid input. Please enter three float values separated by space."
                    )
                    continue
            self._robot_state.set_state(
                np.array([new_robot_x, new_robot_y]), new_robot_yaw
            )
            robot_position, robot_yaw = self._robot_state.get_state()

        dist_to_goal = np.linalg.norm(
            np.array(self._current_goal_position) - robot_position
        )
        if dist_to_goal < self._dist_to_goal_threshold:
            wp_reached = True
            print("Waypoint reached!")

        return rgb_obs, collision_occured, wp_reached, distance_moved

    def _eval_episode(self, ep_idx: int, episode):
        time_ep_start = datetime.now()
        self._rvn_agent.reset()

        start_position = np.array(episode["start_position"])
        start_yaw = episode["start_yaw"]

        distance_traveled_in_ep = 0
        ep_steps = 0
        num_wp_reached = 0
        num_ep_collisions = 0
        ep_timeout = False
        wp_timeout = False
        collision_occured = False
        rgb_obs_history = []

        self._robot_state = RobotState(start_position, start_yaw)

        rgb_obs = self._get_rgb_sensor_observation()

        for _ in range(self._obs_size):
            rgb_obs_history.append(rgb_obs)

        # iterate through goals in the episode
        for idx_goal, goal_yaw in enumerate(episode["waypoint_yaws"]):
            self._current_goal_position = np.array(
                episode["waypoint_positions"][idx_goal]
            )
            self._current_goal_yaw = goal_yaw

            for wp_step in range(self._max_wp_steps):
                ep_steps += 1
                self._total_steps += 1

                # ==================== get obs ====================

                obs, info = self._get_obs_info(
                    rgb_obs_history,
                    self._current_goal_position,
                    self._current_goal_yaw,
                )

                print(
                    f"pointgoal_with_gps_compass: {obs['pointgoal_with_gps_compass']}"
                )

                if self._save_plot and False:  # save context images for debugging
                    PlotUtils.save_context_images(
                        self._target_model, ep_idx, ep_steps, obs, self._context_plot_save_dir
                    )

                # ==================== get action ====================
                action: str = self._rvn_agent.act(obs, info)

                # ==================== step ====================
                (
                    rgb_obs,
                    collision_occured,
                    wp_reached,
                    distance_moved,
                ) = self._step_real_robot_and_get_rgb(action, obs)

                # add obs to history
                if len(rgb_obs_history) == self._obs_size:
                    rgb_obs_history.pop(0)
                rgb_obs_history.append(rgb_obs)

                # ==================== calculate metrics ====================

                distance_traveled_in_ep += distance_moved
                self._total_distance_traveled += distance_moved

                if self._save_plot:
                    PlotUtils.save_debug_images(
                        self._target_model,
                        ep_idx,
                        ep_steps,
                        obs,
                        info,
                        action,
                        self._plot_save_dir,
                        collision_occured,
                        wp_reached,
                        wp_step == self._max_wp_steps - 1,
                        save_goal_rgb=False,
                    )
                if collision_occured:
                    self._num_collisions += 1
                    num_ep_collisions += 1
                    if not self._allow_collision:
                        wp_reached = False
                        break

                if wp_reached:
                    num_wp_reached += 1
                    break

                if wp_step == self._max_wp_steps - 1:
                    wp_timeout = True
                    self._num_wp_timeout += 1
                    break

                if ep_steps >= self._max_ep_steps:
                    ep_timeout = True
                    self._num_ep_timeout += 1
                    break

            if not wp_reached:
                break

        if wp_reached:
            self._num_successes += 1

        total_geodesic_distance = sum(episode["geodesic_distances"])
        num_wp = len(episode["waypoint_positions"])

        ep_result = {
            "ep_idx": ep_idx,
            "num_steps": ep_steps,
            "distance_traveled": distance_traveled_in_ep,
            "geodesic_distance": total_geodesic_distance,
            "goal_reached": wp_reached,
            "num_wp_reached": num_wp_reached,
            "num_wp": num_wp,
            "ep_timeout": ep_timeout,
            "wp_timeout": wp_timeout,
            "collision": collision_occured,
            "num_collisions": num_ep_collisions,
        }
        self._eval_results["episode_results"].append(ep_result)
        self._num_ep_evaluated += 1
        time_ep_end = datetime.now()

        print(
            f"Episode {ep_idx}\tcompleted in {ep_steps} steps ({(time_ep_end - time_ep_start).microseconds/1e6:.2f}s)",
            f" success: {wp_reached} ({num_wp_reached}/{num_wp}), timeout: (ep:{ep_timeout} wp: {wp_timeout}), collision: {collision_occured}, dist_travel: {distance_traveled_in_ep:.2f}",
        )

    def _calcualte_metrics(self):
        print("\nCalculating metrics...")
        print("\tnum_successes:\t", self._num_successes)
        print("\tnum_collisions:\t", self._num_collisions)
        print("\tnum_timeout:\t", self._num_ep_timeout + self._num_wp_timeout)
        print("\t\tnum_ep_timeout:\t", self._num_ep_timeout)
        print("\t\tnum_wp_timeout:\t", self._num_wp_timeout)

        success_rate = self._num_successes / self._num_ep_evaluated
        success_rate_by_num_wp = MetricUtils.get_success_rate_by_num_waypoints(
            self._eval_results["episode_results"]
        )
        average_wp_reached = MetricUtils.get_average_wp_reached(
            self._eval_results["episode_results"]
        )

        spl = MetricUtils.get_success_weighted_by_path_length(
            self._eval_results["episode_results"]
        )
        collisions_per_km_traveled = (
            self._num_collisions / self._total_distance_traveled * 1000
        )

        self._eval_results["success_rate"] = success_rate
        self._eval_results["success_rate_by_num_wp"] = success_rate_by_num_wp
        self._eval_results["average_wp_reached"] = average_wp_reached
        self._eval_results["spl"] = spl
        self._eval_results["num_successes"] = self._num_successes
        self._eval_results["num_collisions"] = self._num_collisions
        self._eval_results["num_timeout"] = self._num_ep_timeout
        self._eval_results["num_ep_evaluated"] = self._num_ep_evaluated
        self._eval_results["total_steps"] = self._total_steps
        self._eval_results["total_distance_traveled"] = self._total_distance_traveled
        self._eval_results["collisions_per_km_traveled"] = collisions_per_km_traveled

    def _save_eval_results(self):
        with open(self._result_file_path, "w") as file:
            yaml.dump(self._eval_results, file)

        print(
            f"\nEvaluation finished.",
            f"\n\tsr by num wp: \t{self._eval_results['success_rate_by_num_wp']}",
            f"\n\tsr: \t{self._eval_results['success_rate']:.2f}, ",
            f"\n\taverage wp reached: \t{self._eval_results['average_wp_reached']:.3f}, ",
            f"\n\tspl: \t{self._eval_results['spl']:.2f}, ",
            f"\n\tcollisions/km: \t{self._eval_results['collisions_per_km_traveled']:.1f}",
            f"\nresult saved to: \t{self._result_file_path}",
        )
