import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


from agents.agent import Agent, ObsType, InfoType


class RandomAgent(Agent):
    def __init__(
        self,
        possible_actions=["move_forward", "turn_left", "turn_right"],
        action_probabilities=[0.5, 0.25, 0.25],
    ):
        self._possibe_actions = possible_actions
        self._action_probability = action_probabilities

    def reset(self):
        pass

    def act(self, obs: ObsType, info: InfoType):
        return np.random.choice(self._possibe_actions, p=self._action_probability)


def main():
    agent = RandomAgent()
    agent.reset()
    for _ in range(10):
        print(agent.act(None))


if __name__ == "__main__":
    main()
