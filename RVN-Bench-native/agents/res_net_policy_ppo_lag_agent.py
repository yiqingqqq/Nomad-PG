import os
import sys

import numpy as np
import torch

from gym.spaces import Box, Discrete
from gym.spaces import Dict as SpaceDict

from habitat_baselines.rl.ddppo.policy import PointNavResNetPolicy

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from agents.res_net_policy_agent import ResNetPolicyAgent

class CostHead(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = torch.nn.Linear(input_size, 1)
        torch.nn.init.orthogonal_(self.fc.weight)
        torch.nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


class ResNetPolicyPpoLagAgent(ResNetPolicyAgent):
    def __init__(
        self,
        # weight_path,
        obs_dim=2,
        action_dim=3,
        hidden_size=512,
        weight_path="agents/weights/ppo_actor_critic_best_2242162.pth",
        device: torch.device = torch.device("cpu"),
    ):
        self._device = device
        self._hidden_size = hidden_size

        observation_spaces = SpaceDict(
            {
                "pointgoal_with_gps_compass": Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
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

        self._policy = PointNavResNetPolicy(
            observation_space=observation_spaces,
            action_space=action_spaces,
            hidden_size=self._hidden_size,
            normalize_visual_inputs=True,
            num_recurrent_layers=2,
            rnn_type="LSTM",
            backbone="resnet50",
        )
        self._policy.critic_c = CostHead(input_size=self._policy.net.output_size)
        self._policy.to(self._device)
        self._policy.load_state_dict(torch.load(weight_path))
        self._policy.eval()


def main():
    print("Hello, World!")


if __name__ == "__main__":
    main()
