import os
import sys
import time
from typing import Optional, Union, Tuple

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))


def compute_gae_returns(
    batch_rewards: torch.Tensor,
    batch_value_preds: torch.Tensor,
    batch_not_done_masks: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> torch.Tensor:
    """
    Compute the Generalized Advantage Estimation (GAE) returns.

    Args:
        batch_rewards: (n_steps+1, n_env, 1)
        batch_value_preds: (n_steps+1, n_env, 1)
        batch_not_done_masks: (n_steps+1, n_env, 1)
        gamma: Discount factor.
        lam: Lambda for GAE.
    Returns:
        batch_returns: (n_steps+1, n_env, 1)
    """

    batch_returns = torch.zeros_like(batch_rewards)

    gae = 0.0
    num_steps = batch_rewards.shape[0] - 1

    for step in reversed(range(num_steps)):
        delta = (
            batch_rewards[step]
            + gamma * batch_value_preds[step + 1] * batch_not_done_masks[step + 1]
            - batch_value_preds[step]
        )

        gae = delta + gamma * lam * gae * batch_not_done_masks[step + 1]
        batch_returns[step] = gae + batch_value_preds[step]

    return batch_returns


def get_advantages_from_returns(
    batch_returns: torch.Tensor,
    batch_value_preds: torch.Tensor,
    normalize_advantage: bool = False,
) -> torch.Tensor:
    """
    Compute the advantages from the returns and value predictions.

    Args:
        batch_returns: (n_steps+1, n_env, 1)
        batch_value_preds: (n_steps+1, n_env, 1)
        batch_not_done_masks: (n_steps+1, n_env, 1)

    Returns:
        advantages: (n_steps+1, n_env, 1)

    """

    assert normalize_advantage is False, "Normalizing advantage is not supported yet."

    return batch_returns - batch_value_preds
