import os
import sys

import numpy as np
import torch

from gym.spaces import Box, Discrete
from gym.spaces import Dict as SpaceDict

from habitat_baselines.rl.ddppo.policy import PointNavResNetPolicy
from habitat_baselines.common.tensor_dict import TensorDict

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from agents.agent import Agent, ObsType, InfoType, DIST_TO_GOAL_THRESHOLD


def preproc_rvn_obs_res_net_policy(obs: ObsType, device: torch.device) -> TensorDict:
    rvn_obs = TensorDict()
    rvn_obs["rgb"] = torch.tensor(obs["rgb"], dtype=torch.uint8).unsqueeze(0).to(device)
    rvn_obs["pointgoal_with_gps_compass"] = torch.tensor(
        [obs["pointgoal_with_gps_compass"]], dtype=torch.float32
    ).to(device)

    return rvn_obs


def postproc_rvn_action(action: np.ndarray, obs: ObsType) -> str:
    dist_to_goal = obs["pointgoal_with_gps_compass"][0]
    if dist_to_goal < DIST_TO_GOAL_THRESHOLD:
        return "stop"

    rvn_command = None
    if action == 0:
        rvn_command = "move_forward"
    elif action == 1:
        rvn_command = "turn_left"
    elif action == 2:
        rvn_command = "turn_right"
    elif action == 3:
        rvn_command = "stop"
    else:
        raise ValueError(f"Invalid action: {action}")
    return rvn_command


class DdppoResNetLstmAgent(Agent):
    def __init__(
        self,
        # weight_path,
        obs_dim=2,
        action_dim=4,
        hidden_size=512,
        weight_path="agents/weights/resnetlstm_ddppo_ckpt.3.pth",
        device: torch.device = torch.device("cpu"),
    ):
        self._device = device
        self._hidden_size = hidden_size

        observation_spaces = SpaceDict(
            {
                "pointgoal_with_gps_compass": Box(
                    low=-np.inf,
                    high=-np.inf,
                    shape=(obs_dim,),
                    dtype=np.float32,
                ),
                "rgb": Box(
                    low=0,
                    high=255,
                    shape=(256, 256, 3),
                    dtype=np.uint8,
                ),
            }
        )
        action_spaces = Discrete(action_dim)

        # policy_config = {
        #     "name": "PointNavResNetPolicy",
        #     "action_distribution_type": "categorical",
        #     "action_dist": {
        #         "use_log_std": True,
        #         "use_softplus": False,
        #         "std_init": "???",
        #         "log_std_init": 0.0,
        #         "use_std_param": False,
        #         "clamp_std": True,
        #         "min_std": 1e-06,
        #         "max_std": 1,
        #         "min_log_std": -5,
        #         "max_log_std": 2,
        #         "action_activation": "tanh",
        #         "scheduled_std": False,
        #     },
        #     "obs_transforms": {},
        #     "hierarchical_policy": "???",
        # }

        self._policy = PointNavResNetPolicy(
            observation_space=observation_spaces,
            action_space=action_spaces,
            hidden_size=self._hidden_size,
            num_recurrent_layers=2,
            rnn_type="LSTM",
            backbone="resnet50",
            # policy_config=policy_config,
            aux_loss_config={},
            normalize_visual_inputs=True,
        ).to(self._device)

        ckpt_dict = torch.load(weight_path, weights_only=False)
        # print(ckpt_dict.keys())
        # print(ckpt_dict["state_dict"].keys())

        self._policy.load_state_dict(ckpt_dict["state_dict"])
        self._policy.eval()

    def reset(self):
        self._test_recurrent_hidden_states = torch.zeros(
            1,
            self._policy.net.num_recurrent_layers,
            self._hidden_size,
            device=self._device,
        )
        self._not_done_masks = torch.zeros(1, 1, device=self._device, dtype=torch.bool)
        self._prev_actions = torch.zeros(1, 1, dtype=torch.long, device=self._device)
        pass

    def act(self, obs: ObsType, info: InfoType):
        rvn_obs = preproc_rvn_obs_res_net_policy(obs, self._device)
        batch_input = {
            "observations": rvn_obs,
            "rnn_hidden_states": self._test_recurrent_hidden_states,
            "prev_actions": self._prev_actions,
            "masks": self._not_done_masks,
        }

        with torch.no_grad():
            action_data = self._policy.act(
                **batch_input,
                deterministic=False,
            )
            self._test_recurrent_hidden_states = action_data.rnn_hidden_states
            self._prev_actions.copy_(action_data.actions)  # type: ignore
            #  Make masks not done till reset (end of episode) will be called
            self._not_done_masks.fill_(True)

        action_idx = action_data.env_actions[0][0].item()
        # action_idx = 0
        rvn_command = postproc_rvn_action(action_idx, obs)

        return rvn_command


def main():
    print("DdppoResNetLstmAgent main")
    agent = DdppoResNetLstmAgent(
        weight_path="agents/weights/resnetlstm_ddppo_ckpt.3.pth",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )


if __name__ == "__main__":
    main()
