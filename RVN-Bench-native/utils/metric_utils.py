def get_success_weighted_by_path_length(episode_results: list[dict]):
    if len(episode_results) == 0:
        return 0.0
    success_by_path_length = 0
    for ep_result in episode_results:
        if not ep_result["goal_reached"]:
            continue
        l_ep = ep_result["geodesic_distance"]
        p_ep = ep_result["distance_traveled"]
        success_by_path_length += l_ep / max(p_ep, l_ep)

    spl = success_by_path_length / len(episode_results)
    return spl


def get_success_rate_by_num_waypoints(episode_results: list[dict]) -> list[float]:
    if len(episode_results) == 0:
        return 0.0

    # get max num_wp in all episodes
    max_num_wp = max([ep_result["num_wp"] for ep_result in episode_results])
    success_by_num_waypoints = [0] * (max_num_wp)

    for ep_result in episode_results:
        if ep_result["num_wp_reached"] == 0:
            continue
        success_by_num_waypoints[ep_result["num_wp_reached"] - 1] += 1

    # make success_by_num_waypoints cumultaive from the end
    for i in range(max_num_wp - 2, -1, -1):
        success_by_num_waypoints[i] += success_by_num_waypoints[i + 1]

    success_rate_by_num_waypoints = [
        success_by_num_waypoints[i] / len(episode_results) for i in range(max_num_wp)
    ]
    return success_rate_by_num_waypoints


def get_average_wp_reached(episode_results: list[dict]) -> list[float]:
    if len(episode_results) == 0:
        return 0.0

    n_episodes = len(episode_results)
    n_waypoints_reached = sum(
        [ep_result["num_wp_reached"] for ep_result in episode_results]
    )

    return n_waypoints_reached / n_episodes


if __name__ == "__main__":
    # TODO: add pytest
    episode_results = [
        {
            "goal_reached": False,
            "geodesic_distance": 10,
            "distance_traveled": 10,
            "num_wp_reached": 1,
            "num_wp": 4,
        },
        {
            "goal_reached": True,
            "geodesic_distance": 10,
            "distance_traveled": 20,
            "num_wp_reached": 8,
            "num_wp": 8,
        },
    ]
    print(get_success_weighted_by_path_length(episode_results))  # 0.25
    print(get_success_rate_by_num_waypoints(episode_results=episode_results))
