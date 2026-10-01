import os
import sys
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Optional, List, Tuple, cast, Dict


import numpy as np
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch import Tensor

from gym.spaces import Box, Discrete
from gym.spaces import Dict as SpaceDict
from gym import spaces
from habitat_baselines.rl.ppo import Net, NetPolicy
from habitat_baselines.rl.ppo.policy import PolicyActionData
from habitat_baselines.rl.ddppo.policy.resnet_policy import PointNavResNetNet
from habitat_baselines.common.baseline_registry import baseline_registry

from habitat_baselines.rl.ddppo.policy import PointNavResNetPolicy
from habitat import VectorEnv
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_pack_info_from_dones,
    build_rnn_build_seq_info,
    build_rnn_state_encoder,
)
from habitat_baselines.rl.ddppo.policy.running_mean_and_var import (
    RunningMeanAndVar,
)
from habitat.tasks.nav.nav import (
    IntegratedPointGoalGPSAndCompassSensor,
    ImageGoalSensor
)

if TYPE_CHECKING:
    from omegaconf import DictConfig

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from algo.ppo_vanilla import PpoTrainerDiscrete
import algo.utils.ppo_utils as ppo_utils
from agents.res_net_policy_agent import (
    preproc_rvn_obs_res_net_policy,
    postproc_rvn_action,
)

from habitat_baselines.rl.ddppo.policy.resnet import ResNeXtBottleneck, ResNet, Block
from .utils.basic_block import BasicBlock
from torchvision.transforms import Resize


class DepthPredictingResNet(ResNet):
    def __init__(
        self,
        in_channels: int,
        base_planes: int,
        ngroups: int,
        block: Block,
        layers: List[int],
        cardinality: int = 1,
        depth_conv_channels: int = 128
    ):
        super().__init__(
            in_channels,
            base_planes,
            ngroups,
            block,
            layers,
            cardinality,
        )
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(128, depth_conv_channels, kernel_size=3, padding=1),
            nn.Conv2d(256, depth_conv_channels, kernel_size=3, padding=1),
            nn.Conv2d(512, depth_conv_channels, kernel_size=3, padding=1),
        ])

        self.fpn_conv = nn.Conv2d(depth_conv_channels, depth_conv_channels, kernel_size=3, padding=1)

        self.depth_conv = nn.Sequential(
            BasicBlock(depth_conv_channels, depth_conv_channels),
            BasicBlock(depth_conv_channels, depth_conv_channels),
            BasicBlock(depth_conv_channels, depth_conv_channels)
        )

        self.depth_pred = nn.Conv2d(
            depth_conv_channels, 1, kernel_size=1, stride=1, padding=0
        )


        #TODO: define depth prediction layers
    
    def forward_with_depth_prediction(self, x) -> Tensor:

        # feature extraction
        x = self.conv1(x)
        x = self.maxpool(x)
        x = cast(Tensor, x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        # depth prediction
        x1_lateral_conv_output = self.lateral_convs[0](x1)
        x2_lateral_conv_output = self.lateral_convs[1](x2)
        x3_lateral_conv_output = self.lateral_convs[2](x3)
        fpn_conv_input = x3_lateral_conv_output
        fpn_conv_input = x2_lateral_conv_output + F.interpolate(fpn_conv_input, size=16, mode='bilinear', align_corners=True)
        fpn_conv_input = x1_lateral_conv_output + F.interpolate(fpn_conv_input, size=32, mode='bilinear', align_corners=True)
        fpn_conv_input = F.interpolate(fpn_conv_input, size=64, mode='bilinear', align_corners=True)
        fpn_conv_output = self.fpn_conv(fpn_conv_input)
        depth_conv_output = self.depth_conv(fpn_conv_output)
        depth_pred = self.depth_pred(depth_conv_output)
        depth_pred = depth_pred.squeeze(0)

        return x4, depth_pred


def resnet50_depth(in_channels, base_planes, ngroups):
    model = DepthPredictingResNet(
        in_channels,
        base_planes,
        ngroups,
        ResNeXtBottleneck,
        [3, 4, 6, 3],
        cardinality=int(base_planes / 2),
    )

    return model


class DepthPredictingResNetEncoder(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Dict,
        baseplanes: int = 32,
        ngroups: int = 32,
        spatial_size: int = 128,
        make_backbone=None,
        normalize_visual_inputs: bool = False,
    ):
        super().__init__()

        # Determine which visual observations are present
        self.visual_keys = [
            k
            for k, v in observation_space.spaces.items()
            if len(v.shape) > 1 and k != ImageGoalSensor.cls_uuid
        ]
        self.key_needs_rescaling = {k: None for k in self.visual_keys}
        for k, v in observation_space.spaces.items():
            if v.dtype == np.uint8:
                self.key_needs_rescaling[k] = 1.0 / v.high.max()

        # Count total # of channels
        self._n_input_channels = sum(
            observation_space.spaces[k].shape[2] for k in self.visual_keys
        )

        if normalize_visual_inputs:
            self.running_mean_and_var: nn.Module = RunningMeanAndVar(
                self._n_input_channels
            )
        else:
            self.running_mean_and_var = nn.Sequential()

        if not self.is_blind:
            spatial_size_h = (
                observation_space.spaces[self.visual_keys[0]].shape[0] // 2
            )
            spatial_size_w = (
                observation_space.spaces[self.visual_keys[0]].shape[1] // 2
            )
            self.backbone = make_backbone(
                self._n_input_channels, baseplanes, ngroups
            )

            final_spatial_h = int(
                np.ceil(spatial_size_h * self.backbone.final_spatial_compress)
            )
            final_spatial_w = int(
                np.ceil(spatial_size_w * self.backbone.final_spatial_compress)
            )
            after_compression_flat_size = 2048
            num_compression_channels = int(
                round(
                    after_compression_flat_size
                    / (final_spatial_h * final_spatial_w)
                )
            )
            self.compression = nn.Sequential(
                nn.Conv2d(
                    self.backbone.final_channels,
                    num_compression_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(1, num_compression_channels),
                nn.ReLU(True),
            )

            self.output_shape = (
                num_compression_channels,
                final_spatial_h,
                final_spatial_w,
            )

    @property
    def is_blind(self):
        return self._n_input_channels == 0

    def layer_init(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:  # type: ignore
        if self.is_blind:
            return None

        cnn_input = []
        for k in self.visual_keys:
            obs_k = observations[k]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            obs_k = obs_k.permute(0, 3, 1, 2)
            if self.key_needs_rescaling[k] is not None:
                obs_k = (
                    obs_k.float() * self.key_needs_rescaling[k]
                )  # normalize
            cnn_input.append(obs_k)

        x = torch.cat(cnn_input, dim=1)
        x = F.avg_pool2d(x, 2)

        x = self.running_mean_and_var(x)
        x = self.backbone(x)
        x = self.compression(x)
        return x
    
    def forward_with_depth_prediction(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.is_blind:
            return None

        cnn_input = []
        for k in self.visual_keys:
            obs_k = observations[k]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            obs_k = obs_k.permute(0, 3, 1, 2)
            if self.key_needs_rescaling[k] is not None:
                obs_k = (
                    obs_k.float() * self.key_needs_rescaling[k]
                )  # normalize
            cnn_input.append(obs_k)

        x = torch.cat(cnn_input, dim=1)
        x = F.avg_pool2d(x, 2)

        x = self.running_mean_and_var(x)
        x, depth_prediction = self.backbone.forward_with_depth_prediction(x)
        x = self.compression(x)
        return x, depth_prediction


class PointNavDepthPredictingResNetNet(Net):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int,
        num_recurrent_layers: int,
        rnn_type: str,
        backbone,
        resnet_baseplanes,
        normalize_visual_inputs: bool,
        fuse_keys: Optional[List[str]],
        force_blind_policy: bool = False,
        discrete_actions: bool = True,
    ):
        super().__init__()
        self.prev_action_embedding: nn.Module
        self.discrete_actions = discrete_actions
        self._n_prev_action = 32
        if discrete_actions:
            self.prev_action_embedding = nn.Embedding(
                action_space.n + 1, self._n_prev_action
            )
        else:
            raise NotImplementedError
        self._n_prev_action = 32
        rnn_input_size = self._n_prev_action  # test

        # Only fuse the 1D state inputs. Other inputs are processed by the
        # visual encoder
        if fuse_keys is None:
            fuse_keys = observation_space.spaces.keys()
            # removing keys that correspond to goal sensors
            goal_sensor_keys = {
                IntegratedPointGoalGPSAndCompassSensor.cls_uuid,
            }
            fuse_keys = [k for k in fuse_keys if k not in goal_sensor_keys]
        self._fuse_keys_1d: List[str] = [
            k for k in fuse_keys if len(observation_space.spaces[k].shape) == 1
        ]
        if len(self._fuse_keys_1d) != 0:
            rnn_input_size += sum(
                observation_space.spaces[k].shape[0]
                for k in self._fuse_keys_1d
            )

        if (
            IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            in observation_space.spaces
        ):
            n_input_goal = (
                observation_space.spaces[
                    IntegratedPointGoalGPSAndCompassSensor.cls_uuid
                ].shape[0]
                + 1
            )
            self.tgt_embeding = nn.Linear(n_input_goal, 32)
            rnn_input_size += 32
        self._hidden_size = hidden_size

        if force_blind_policy:
            use_obs_space = spaces.Dict({})
        else:
            use_obs_space = spaces.Dict(
                {
                    k: observation_space.spaces[k]
                    for k in fuse_keys
                    if len(observation_space.spaces[k].shape) == 3
                }
            )

        if backbone.startswith("resnet50_clip"):
            raise NotImplementedError
        else:
            self.visual_encoder = DepthPredictingResNetEncoder(
                use_obs_space,
                baseplanes=resnet_baseplanes,
                ngroups=resnet_baseplanes // 2,
                # make_backbone=getattr(resnet, backbone),
                make_backbone = resnet50_depth,
                normalize_visual_inputs=normalize_visual_inputs,
            )

            if not self.visual_encoder.is_blind:
                self.visual_fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(
                        np.prod(self.visual_encoder.output_shape), hidden_size
                    ),
                    nn.ReLU(True),
                )

        self.state_encoder = build_rnn_state_encoder(
            (0 if self.is_blind else self._hidden_size) + rnn_input_size,
            self._hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    @property
    def recurrent_hidden_size(self):
        return self._hidden_size

    @property
    def perception_embedding_size(self):
        return self._hidden_size


    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
        rnn_build_seq_info: Optional[Dict[str, torch.Tensor]] = None,
        depth_prediction = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        x = []
        aux_loss_state = {}
        if not self.is_blind:
            # We CANNOT use observations.get() here because self.visual_encoder(observations)
            # is an expensive operation. Therefore, we need `# noqa: SIM401`
            if (  # noqa: SIM401
                PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY
                in observations
            ):
                raise NotImplementedError
            else:
                if not depth_prediction:
                    visual_feats = self.visual_encoder(observations)
                else:
                    visual_feats, depth_preds = self.visual_encoder.forward_with_depth_prediction(observations)

            visual_feats = self.visual_fc(visual_feats)
            aux_loss_state["perception_embed"] = visual_feats
            x.append(visual_feats)

        if len(self._fuse_keys_1d) != 0:
            fuse_states = torch.cat(
                [observations[k] for k in self._fuse_keys_1d], dim=-1
            )
            x.append(fuse_states.float())

        if IntegratedPointGoalGPSAndCompassSensor.cls_uuid in observations:
            goal_observations = observations[
                IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            ]
            if goal_observations.shape[1] == 2:
                # Polar Dimensionality 2
                # 2D polar transform
                goal_observations = torch.stack(
                    [
                        goal_observations[:, 0],
                        torch.cos(-goal_observations[:, 1]),
                        torch.sin(-goal_observations[:, 1]),
                    ],
                    -1,
                )
            else:
                assert (
                    goal_observations.shape[1] == 3
                ), "Unsupported dimensionality"
                vertical_angle_sin = torch.sin(goal_observations[:, 2])
                # Polar Dimensionality 3
                # 3D Polar transformation
                goal_observations = torch.stack(
                    [
                        goal_observations[:, 0],
                        torch.cos(-goal_observations[:, 1])
                        * vertical_angle_sin,
                        torch.sin(-goal_observations[:, 1])
                        * vertical_angle_sin,
                        torch.cos(goal_observations[:, 2]),
                    ],
                    -1,
                )

            x.append(self.tgt_embeding(goal_observations))

        if self.discrete_actions:
            prev_actions = prev_actions.squeeze(-1)
            start_token = torch.zeros_like(prev_actions)
            # The mask means the previous action will be zero, an extra dummy action
            prev_actions = self.prev_action_embedding(
                torch.where(masks.view(-1), prev_actions + 1, start_token)
            )
        else:
            prev_actions = self.prev_action_embedding(
                masks * prev_actions.float()
            )

        x.append(prev_actions)

        out = torch.cat(x, dim=1)
        out, rnn_hidden_states = self.state_encoder(
            out, rnn_hidden_states, masks, rnn_build_seq_info
        )
        aux_loss_state["rnn_output"] = out
        if depth_prediction:
            return out, rnn_hidden_states, aux_loss_state, depth_preds
        else:
            return out, rnn_hidden_states, aux_loss_state

        


@baseline_registry.register_policy
class PointNavDepthPredictingResNetPolicy(NetPolicy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int = 512,
        num_recurrent_layers: int = 1,
        rnn_type: str = "GRU",
        resnet_baseplanes: int = 32,
        backbone: str = "resnet18",
        normalize_visual_inputs: bool = False,
        force_blind_policy: bool = False,
        policy_config: "DictConfig" = None,
        aux_loss_config: Optional["DictConfig"] = None,
        fuse_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Keyword arguments:
        rnn_type: RNN layer type; one of ["GRU", "LSTM"]
        backbone: Visual encoder backbone; one of ["resnet18", "resnet50", "resneXt50", "se_resnet50", "se_resneXt50", "se_resneXt101", "resnet50_clip_avgpool", "resnet50_clip_attnpool"]
        """

        assert backbone in [
            "resnet18",
            "resnet50",
            "resnet50_depth",
            "resneXt50",
            "se_resnet50",
            "se_resneXt50",
            "se_resneXt101",
            "resnet50_clip_avgpool",
            "resnet50_clip_attnpool",
        ], f"{backbone} backbone is not recognized."

        if policy_config is not None:
            discrete_actions = (
                policy_config.action_distribution_type == "categorical"
            )
            self.action_distribution_type = (
                policy_config.action_distribution_type
            )
        else:
            discrete_actions = True
            self.action_distribution_type = "categorical"
        super().__init__(
            PointNavDepthPredictingResNetNet(
                observation_space=observation_space,
                action_space=action_space,  # for previous action
                hidden_size=hidden_size,
                num_recurrent_layers=num_recurrent_layers,
                rnn_type=rnn_type,
                backbone=backbone,
                resnet_baseplanes=resnet_baseplanes,
                normalize_visual_inputs=normalize_visual_inputs,
                fuse_keys=fuse_keys,
                force_blind_policy=force_blind_policy,
                discrete_actions=discrete_actions,
            ),
            action_space=action_space,
            policy_config=policy_config,
            aux_loss_config=aux_loss_config,
        )

    @classmethod
    def from_config(
        cls,
        config: "DictConfig",
        observation_space: spaces.Dict,
        action_space,
        **kwargs,
    ):
        # Exclude cameras for rendering from the observation space.
        ignore_names = [
            sensor.uuid
            for sensor in config.habitat_baselines.eval.extra_sim_sensors.values()
        ]
        filtered_obs = spaces.Dict(
            OrderedDict(
                (
                    (k, v)
                    for k, v in observation_space.items()
                    if k not in ignore_names
                )
            )
        )

        agent_name = None
        if "agent_name" in kwargs:
            agent_name = kwargs["agent_name"]

        if agent_name is None:
            if len(config.habitat.simulator.agents_order) > 1:
                raise ValueError(
                    "If there is more than an agent, you need to specify the agent name"
                )
            else:
                agent_name = config.habitat.simulator.agents_order[0]

        return cls(
            observation_space=filtered_obs,
            action_space=action_space,
            hidden_size=config.habitat_baselines.rl.ppo.hidden_size,
            rnn_type=config.habitat_baselines.rl.ddppo.rnn_type,
            num_recurrent_layers=config.habitat_baselines.rl.ddppo.num_recurrent_layers,
            backbone=config.habitat_baselines.rl.ddppo.backbone,
            normalize_visual_inputs="rgb" in observation_space.spaces,
            force_blind_policy=config.habitat_baselines.force_blind_policy,
            policy_config=config.habitat_baselines.rl.policy[agent_name],
            aux_loss_config=config.habitat_baselines.rl.auxiliary_losses,
            fuse_keys=None,
        )


    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
        return_depth_prediction=False,
    ):
        
        if return_depth_prediction:
            features, rnn_hidden_states, depth_predictions, _ = self.net.forward_with_depth(
                observations, rnn_hidden_states, prev_actions, masks
            )
        else:
            features, rnn_hidden_states, _ = self.net(
                observations, rnn_hidden_states, prev_actions, masks
            )

        
        distribution = self.action_distribution(features)
        value = self.critic(features)

        if deterministic:
            if self.action_distribution_type == "categorical":
                action = distribution.mode()
            elif self.action_distribution_type == "gaussian":
                action = distribution.mean
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        if return_depth_prediction:
            return PolicyActionData(
                values=value,
                actions=action,
                action_log_probs=action_log_probs,
                rnn_hidden_states=rnn_hidden_states,
            ), depth_predictions
        return PolicyActionData(
            values=value,
            actions=action,
            action_log_probs=action_log_probs,
            rnn_hidden_states=rnn_hidden_states,
        )

class DepthPredictingPpoTrainerResNetPolicyLab(PpoTrainerDiscrete):
    def __init__(
        self,
        depth_pseudo_labeling_model,
        envs: VectorEnv,
        hyper_parameters: dict = {},
        depth_loss_coeff: Optional[float] = None,
        obs_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
        hidden_size: Optional[int] = None,
        weight_save_dir: str = "weights",
        render_save_dir: str = "images",
        weight_load_path: Optional[str] = None,
        device: torch.device = torch.device("cpu"),
        
    ):
        self._render = True
        print(f"Init DepthPredictingPpoTrainerResNetPolicyLab device: {device} n_env: {envs.num_envs}")
        self._envs = envs

        self._preproc_obs = preproc_rvn_obs_res_net_policy
        self._postproc_action = postproc_rvn_action
        self._depth_loss_coeff = depth_loss_coeff
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._hidden_size = hidden_size
        self._weight_save_dir = weight_save_dir
        self._render_save_dir = render_save_dir
        self._device = device
        self._n_env = envs.num_envs

        os.makedirs(weight_save_dir, exist_ok=True)
        os.makedirs(render_save_dir, exist_ok=True)

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
        self._depth_pseudo_labeling_model = depth_pseudo_labeling_model
        self._init_hyperparmeters(**hyper_parameters, hidden_dim=hidden_size)

        assert self._n_env > self._n_mini_batch
        if self._n_env % self._n_mini_batch != 0:
            print(
                f"Warning: n_env ({self._n_env}) % n_mini_batch ({self._n_mini_batch}) != 0"
            )

        # ALG STEP 1
        # Initialize actor and critic networks
        self._policy = PointNavDepthPredictingResNetPolicy(
            observation_space=observation_spaces,
            action_space=action_spaces,
            hidden_size=hidden_size,
            normalize_visual_inputs=True,
            num_recurrent_layers=2,
            rnn_type="LSTM",
            backbone="resnet50",
        ).to(self._device)

        self._optim = Adam(self._policy.parameters(), lr=self._lr, eps=self._eps)

        # print(self._policy)

        # This logger will help us with printing out summaries of each iteration
        self._logger = {
            "delta_t": time.time_ns(),
            "t_so_far": 0,  # timesteps so far
            "i_so_far": 0,  # iterations so far
            "rewards_by_env": [0.0 for _ in range(self._n_env)],  # rewards by env
            "ep_lens_by_env": [0 for _ in range(self._n_env)],  # episode lengths by env
            "batch_rews": [],  # rewards of episodes terminated in current batch
            "batch_ep_lens": [],  # lengths of episodes terminated in batch
            "batch_first_wp_reached": [],  # number of first waypoints reached for terminated ep in batch
            "batch_num_wp_reached": [],  # number of waypoints reached for terminated ep in batch
            "batch_travel_distances": [],  # travel distances for terminated ep in batch
            "batch_l2_dist_to_1st_wp": [],  # l2 distance to first waypoint for terminated ep in batch
            "batch_l2_dist_to_goal": [],  # l2 distance to first waypoint for terminated ep in batch
            "batch_collisions": [],
            "batch_timeouts": [],  # timeouts for terminated ep in batch
            "batch_count": 0,  # number of episodes terminated in batch
            "actor_losses": [],  # losses of actor network in current iteration
            "value_losses": [],  # losses of value network in current iteration
            "dist_entropies": [],  # entropy of action distribution in current iteration
            "depth_losses": [], # losses of depth prediction in current iteration
        }

    def learn(
        self,
        total_timesteps: int,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        run_name: Optional[str] = None,
    ):
        print(
            f"Learning... Running {self._max_timesteps_per_episode} timesteps per episode, ",
            end="",
        )
        print(
            f"{self._timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps"
        )

        print(f"Using device           : {self._device}")
        print(f"batch_size             : {self._timesteps_per_batch}")
        print(f"gamma                  : {self._gamma}")
        print(f"lam                    : {self._lam}")
        print(f"lr                     : {self._lr}")
        print(f"eps                    : {self._eps}")
        print(f"clip_ratio             : {self._clip_ratio}")
        print(f"n_updates_per_iteration: {self._n_updates_per_iteration}")
        print(f"n_mini_batch           : {self._n_mini_batch}")
        print(f"value_loss_coeff       : {self._value_loss_coeff}")
        print(f"entropy_loss_coeff     : {self._entropy_loss_coeff}")
        print(f"depth_loss_coeff       : {self._depth_loss_coeff}")
        print(f"normalize_advantage    : {self._normalize_advantage}")
        print(f"use_clipped_value_loss : {self._use_clipped_value_loss}")
        print(f"hidden_dim             : {self._hidden_dim}")
        print(f"num_envs               : {self._n_env}")

        self._num_ep = 0
        self._use_wandb = use_wandb
        self._best_train_avg_ep_rew = -np.inf

        if wandb_project is None or run_name is None:
            print("wandb_project not specified, set use_wandb to False")
            use_wandb = False
        if use_wandb:
            wandb.init(
                project=wandb_project,
                config={
                    "batch_size": self._timesteps_per_batch,
                    "gamma": self._gamma,
                    "lam": self._lam,
                    "lr": self._lr,
                    "eps": self._eps,
                    "clip_ratio": self._clip_ratio,
                    "n_updates_per_iteration": self._n_updates_per_iteration,
                    "n_mini_batch": self._n_mini_batch,
                    "value_loss_coeff": self._value_loss_coeff,
                    "entropy_loss_coeff": self._entropy_loss_coeff,
                    "depth_loss_coeff": self._depth_loss_coeff,
                    "normalize_advantage": self._normalize_advantage,
                    "hidden_dim": self._hidden_dim,
                },
            )
            wandb.run.name = run_name
            wandb.run.save()

        t_so_far = 0  # Timesteps simulated so far
        i_so_far = 0  # Iterations ran so far

        while t_so_far < total_timesteps:  # ALG STEP 2
            # ALG STEP 3
            # Collect set of trajectories D_k = {τ_i}
            # by running policy π_k = π(θ_k) in the environment.
            (
                batch_obs,
                batch_recurrent_hidden_states,
                batch_rewards,
                batch_value_preds,
                batch_action_log_probs,
                batch_actions,
                batch_prev_actions,
                batch_not_done_masks,
            ) = self._multi_env_rollout()
            print("batch obs gathered from _multi_env_rollout")

            # print("batch_actions:", batch_actions)
            # print("batch_prev_actions:", batch_prev_actions)

            t_so_far += np.sum(self._timesteps_per_batch * self._n_env)
            i_so_far += 1
            # Logging timesteps so far and iterations so far
            self._logger["t_so_far"] = t_so_far
            self._logger["i_so_far"] = i_so_far

            # ALG STEP 5
            # Calculate the adventage
            batch_returns = ppo_utils.compute_gae_returns(
                batch_rewards,
                batch_value_preds,
                batch_not_done_masks,
                gamma=self._gamma,
                lam=self._lam,
            ).detach()
            batch_advantages = (batch_returns - batch_value_preds).detach()

            for idx_update_itr in range(self._n_updates_per_iteration):
                rand_perm = torch.randperm(self._n_env).chunk(self._n_mini_batch)
                dones_cpu = (
                    torch.logical_not(batch_not_done_masks)
                    .cpu()
                    .view(-1, self._n_env)
                    .numpy()
                )

                for inds in rand_perm:
                    curr_slice = (slice(0, self._timesteps_per_batch), inds)
                    mini_batch_obs = {
                        "pointgoal_with_gps_compass": batch_obs[
                            "pointgoal_with_gps_compass"
                        ][curr_slice],
                        "rgb": batch_obs["rgb"][curr_slice],
                    }
                    mini_batch_recurrent_hidden_states = batch_recurrent_hidden_states[
                        curr_slice
                    ]
                    mini_batch_rewards = batch_rewards[curr_slice]
                    mini_batch_value_preds = batch_value_preds[curr_slice]
                    mini_batch_action_log_probs = batch_action_log_probs[curr_slice]
                    mini_batch_actions = batch_actions[curr_slice]
                    mini_batch_prev_actions = batch_prev_actions[curr_slice]
                    mini_batch_not_done_masks = batch_not_done_masks[curr_slice]
                    mini_batch_returns = batch_returns[curr_slice]
                    mini_batch_advantages = batch_advantages[curr_slice]
                    mini_batch_rnn_build_seq_info = build_rnn_build_seq_info(
                        device=self._device,
                        build_fn_result=build_pack_info_from_dones(
                            dones_cpu[
                                0 : self._timesteps_per_batch, inds.numpy()
                            ].reshape(-1, len(inds)),
                        ),
                    )

                    mini_batch_obs["rgb"] = mini_batch_obs["rgb"].flatten(0, 1)
                    mini_batch_obs["pointgoal_with_gps_compass"] = mini_batch_obs[
                        "pointgoal_with_gps_compass"
                    ].flatten(0, 1)
                    # ???? Why use [0:1] here?
                    # mini_batch_recurrent_hidden_states = (
                    #     mini_batch_recurrent_hidden_states[0:1].flatten(0, 1)
                    # )
                    mini_batch_recurrent_hidden_states = (
                        mini_batch_recurrent_hidden_states.flatten(0, 1)
                    )
                    mini_batch_rewards = mini_batch_rewards.flatten(0, 1)
                    mini_batch_value_preds = mini_batch_value_preds.flatten(0, 1)
                    mini_batch_action_log_probs = mini_batch_action_log_probs.flatten(
                        0, 1
                    )
                    mini_batch_actions = mini_batch_actions.flatten(0, 1)
                    mini_batch_prev_actions = mini_batch_prev_actions.flatten(0, 1)
                    mini_batch_not_done_masks = mini_batch_not_done_masks.flatten(0, 1)
                    mini_batch_returns = mini_batch_returns.flatten(0, 1)
                    mini_batch_advantages = mini_batch_advantages.flatten(0, 1)

                    self._update_w_mini_batch(
                        mini_batch_obs,
                        mini_batch_recurrent_hidden_states,
                        mini_batch_value_preds,
                        mini_batch_action_log_probs,
                        mini_batch_actions,
                        mini_batch_prev_actions,
                        mini_batch_not_done_masks,
                        mini_batch_returns,
                        mini_batch_advantages,
                        mini_batch_rnn_build_seq_info,
                    )

            # Print a summary of our training so far
            # Save our model if it's time
            if i_so_far % self._save_freq == 0:
                self._save_model_weights()
            self._log_summary()

    def _update_w_mini_batch(
        self,
        mini_batch_obs,
        mini_batch_recurrent_hidden_states,
        mini_batch_value_preds,
        mini_batch_action_log_probs,
        mini_batch_actions,
        mini_batch_prev_actions,
        mini_batch_not_done_masks,
        mini_batch_returns,
        mini_batch_advantages,
        mini_batch_rnn_build_seq_info,
    ):

        # print(f"mini_batch_recurrent_hidden_states: {mini_batch_recurrent_hidden_states.shape}")
        # print(f"mini_batch_value_preds     : {mini_batch_value_preds.shape}")
        # print(f"mini_batch_action_log_probs: {mini_batch_action_log_probs.shape}")
        # print(f"mini_batch_actions         : {mini_batch_actions.shape}")
        # print(f"mini_batch_prev_actions    : {mini_batch_prev_actions.shape}")
        # print(f"mini_batch_not_done_masks  : {mini_batch_not_done_masks.shape}")
        # exit()
        # TODO: set_grads_to_none ??

        # print(f"rnn_build_seq_info:", mini_batch_rnn_build_seq_info)
        # (
        #     value_theta,
        #     action_log_probs_theta,
        #     _,
        #     dist_entropy_theta,
        #     aux_loss_res_theta,
        # ) = self._policy.evaluate_actions(
        #     mini_batch_obs,
        #     mini_batch_recurrent_hidden_states,
        #     mini_batch_prev_actions,
        #     mini_batch_not_done_masks,
        #     mini_batch_actions,
        #     mini_batch_rnn_build_seq_info,
        # )
        value_theta, action_log_probs_theta, dist_entropy_theta, depth_preds = (
            self._evaluate_actions(
                mini_batch_obs,
                mini_batch_recurrent_hidden_states,
                mini_batch_prev_actions,
                mini_batch_not_done_masks,
                mini_batch_actions,
            )
        )
        import ipdb;ipdb.set_trace()

        # generate depth pseudo-labels
        with torch.no_grad():
            generated_pseudo_labeled_depths = []
            for batch_idx in range(mini_batch_obs['rgb'].shape[0]):
                generated_pseudo_labeled_depth = self._depth_pseudo_labeling_model.infer_image(mini_batch_obs['rgb'][batch_idx].clone().detach().cpu().numpy().astype(np.uint8))
                generated_pseudo_labeled_depths.append(Resize((64, 64))(torch.from_numpy(generated_pseudo_labeled_depth).unsqueeze(0)))
            generated_pseudo_labeled_depths = torch.cat(generated_pseudo_labeled_depths).to(depth_preds.device)

        ratio = torch.exp(action_log_probs_theta - mini_batch_action_log_probs)

        surr_1 = mini_batch_advantages * ratio
        surr_2 = mini_batch_advantages * (
            torch.clamp(ratio, 1.0 - self._clip_ratio, 1.0 + self._clip_ratio)
        )
        action_loss = -torch.min(surr_1, surr_2)

        value_theta = value_theta.float()

        if self._use_clipped_value_loss:
            delta = value_theta.detach() - mini_batch_value_preds

            value_pred_clipped = mini_batch_value_preds + delta.clamp(
                -self._clip_ratio, self._clip_ratio
            )
            value_theta = torch.where(
                delta.abs() < self._clip_ratio,
                value_theta,
                value_pred_clipped,
            )

        value_loss = 0.5 * F.mse_loss(value_theta, mini_batch_returns, reduction="none")


        """
        TODO: MSE Loss of depth estimation and pseudo-labeled depth
        depth_loss = ???
        """


        action_loss, value_loss, dist_entropy_theta = map(
            torch.mean,
            (action_loss, value_loss, dist_entropy_theta),
        )

        depth_loss = self._compute_depth_loss(depth_preds, generated_pseudo_labeled_depths)

        all_losses = [
            self._value_loss_coeff * value_loss,
            action_loss,
            -self._entropy_loss_coeff * dist_entropy_theta,
            self._depth_loss_coeff * depth_loss
        ]
        total_loss = torch.stack(all_losses).sum()

        self._optim.zero_grad()
        total_loss.backward()
        self._optim.step()

        self._logger["actor_losses"].append(action_loss.detach())
        self._logger["value_losses"].append(value_loss.detach())
        self._logger["dist_entropies"].append(dist_entropy_theta.detach())
        self._logger["depth_losses"].append(depth_loss.detach())

    def _compute_depth_loss(
            self,
            depth_pred,
            generated_pseudo_labeled_depth
    ) -> torch.Tensor:
        depth_pred = depth_pred.flatten()
        generated_pseudo_labeled_depth = generated_pseudo_labeled_depth.flatten()

        mask = (generated_pseudo_labeled_depth < 20) & (generated_pseudo_labeled_depth > 0)
        generated_pseudo_labeled_depth = generated_pseudo_labeled_depth[mask]
        generated_pseudo_labeled_depth = generated_pseudo_labeled_depth / 20
        depth_pred = depth_pred[mask]
        l2_loss = (depth_pred - generated_pseudo_labeled_depth).pow(2).mean()
        return l2_loss

    def _multi_env_rollout(
        self,
    ) -> Tuple[
        dict,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        batch_obs = {
            "pointgoal_with_gps_compass": [],  # (n_steps+1, n_env, 2)
            "rgb": [],  # (n_steps+1, n_env, 256, 256, 3)
        }
        batch_recurrent_hidden_states = []  # (n_steps+1, n_env, n_recurrent_layer, 512)
        batch_rewards = []  # (n_steps+1, n_env, 1)
        batch_value_preds = []  # (n_steps+1, n_env, 1)
        # batch_returns = []  # (n_steps+1, n_env, 1)
        batch_action_log_probs = []  # (n_steps+1, n_env, 1)
        batch_actions = []  # (n_steps+1, n_env, 1)
        batch_prev_actions = []  # (n_steps+1, n_env, 1)
        batch_not_done_masks = []  # (n_steps+1, n_env, 1)

        prev_recurrent_hidden_states = torch.zeros(
            self._n_env,
            self._policy.net.num_recurrent_layers,
            self._hidden_size,
            device=self._device,
        )
        prev_actions = torch.zeros(
            self._n_env, 1, dtype=torch.long, device=self._device
        )
        prev_not_done_masks = torch.zeros(
            self._n_env, 1, dtype=torch.bool, device=self._device
        )

        # ---------- Reset envs ----------
        observations = self._envs.reset()
        prev_obs = self._batch_step_obs(observations)

        for batch_step in range(self._timesteps_per_batch):
            # print(f"batch_step {batch_step} / {self._timesteps_per_batch}")
            # ---------- Get action and step envs ----------
            (
                step_actions,  # (n_env, 1)
                step_recurrent_hidden_states,  # (n_env, n_recurrent_layer, 512)
                step_value_preds,  # (n_env, 1)
                step_action_log_probs,  # (n_env, 1)
            ) = self._get_actions_from_obs(
                prev_obs,
                prev_recurrent_hidden_states,
                prev_actions,
                prev_not_done_masks,
            )

            (
                step_obs,  # {"pointgoal_with_gps_compass": (n_env, 2), "rgb": (n_env, 256, 256, 3)}
                step_rewards,  # (n_env, 1)
                step_not_done_masks,  # (n_env, 1)
            ) = self._step_venvs(
                step_actions,  # (n_env, 1)
            )

            batch_obs["pointgoal_with_gps_compass"].append(
                prev_obs["pointgoal_with_gps_compass"].clone()
            )
            batch_obs["rgb"].append(prev_obs["rgb"].clone())
            batch_recurrent_hidden_states.append(
                prev_recurrent_hidden_states.detach().clone()
            )
            batch_rewards.append(step_rewards.detach().clone())
            batch_value_preds.append(step_value_preds.detach().clone())
            batch_action_log_probs.append(step_action_log_probs.detach().clone())
            batch_actions.append(step_actions.detach().clone())
            batch_prev_actions.append(prev_actions.detach().clone())
            batch_not_done_masks.append(prev_not_done_masks.detach().clone())

            prev_obs = step_obs
            prev_actions = step_actions
            prev_recurrent_hidden_states = step_recurrent_hidden_states
            prev_not_done_masks = step_not_done_masks

            # If the environment is done, init prev_actions and hidden states.
            for idx_env in range(self._n_env):
                if not step_not_done_masks[idx_env]:
                    prev_actions[idx_env][0] = 0
                    prev_recurrent_hidden_states[idx_env] = torch.zeros(
                        self._policy.net.num_recurrent_layers,
                        self._hidden_size,
                        device=self._device,
                    )

        # ---------- Get the last value prediction ----------
        (
            step_actions,  # (n_env, 1)
            step_recurrent_hidden_states,  # (n_env, n_recurrent_layer, 512)
            step_value_preds,  # (n_env, 1)
            step_action_log_probs,  # (n_env, 1)
        ) = self._get_actions_from_obs(
            prev_obs,
            prev_recurrent_hidden_states,
            prev_actions,
            prev_not_done_masks,
        )

        batch_obs["pointgoal_with_gps_compass"].append(
            prev_obs["pointgoal_with_gps_compass"].clone()
        )
        batch_obs["rgb"].append(prev_obs["rgb"].clone())
        batch_recurrent_hidden_states.append(
            prev_recurrent_hidden_states.detach().clone()
        )
        batch_rewards.append(
            torch.zeros(self._n_env, 1, dtype=torch.long, device=self._device)
        )
        batch_value_preds.append(step_value_preds.detach().clone())
        batch_action_log_probs.append(
            torch.zeros(self._n_env, 1, dtype=torch.long, device=self._device)
        )
        batch_actions.append(
            torch.zeros(self._n_env, 1, dtype=torch.long, device=self._device)
        )
        batch_prev_actions.append(prev_actions.detach().clone())
        batch_not_done_masks.append(prev_not_done_masks)

        # ---------- Convert to tensors ----------
        for k in batch_obs:
            batch_obs[k] = torch.stack(batch_obs[k], dim=0)
        batch_recurrent_hidden_states = torch.stack(
            batch_recurrent_hidden_states, dim=0
        )
        batch_rewards = torch.stack(batch_rewards, dim=0)
        batch_value_preds = torch.stack(batch_value_preds, dim=0)
        batch_action_log_probs = torch.stack(batch_action_log_probs, dim=0)
        batch_actions = torch.stack(batch_actions, dim=0)
        batch_prev_actions = torch.stack(batch_prev_actions, dim=0)
        batch_not_done_masks = torch.stack(batch_not_done_masks, dim=0)

        return (
            batch_obs,
            batch_recurrent_hidden_states,
            batch_rewards,
            batch_value_preds,
            batch_action_log_probs,
            batch_actions,
            batch_prev_actions,
            batch_not_done_masks,
        )

    def _batch_step_obs(self, observations):
        step_obs = {
            "pointgoal_with_gps_compass": [
                obs["pointgoal_with_gps_compass"] for obs in observations
            ],  # (n_env, 2)
            "rgb": [obs["rgb"] for obs in observations],  # (n_env, 256, 256)
        }

        step_obs["pointgoal_with_gps_compass"] = torch.tensor(
            np.array(step_obs["pointgoal_with_gps_compass"]), dtype=torch.float32
        ).to(self._device)
        step_obs["rgb"] = torch.tensor(
            np.array(step_obs["rgb"]), dtype=torch.float32
        ).to(self._device)

        return step_obs

    def _get_actions_from_obs(
        self,
        step_obs,
        prev_recurrent_hidden_states,
        prev_actions,
        not_done_masks,
        depth_prediction = False
    ) -> Tuple[
        torch.Tensor,  # step_actions (n_env, 1)
        torch.Tensor,  # step_recurrent_hidden_states (n_env,  n_recurrent_layer, 512)
        torch.Tensor,  # step_value_preds (n_env, 1)
        torch.Tensor,  # step_action_log_probs (n_env, 1)
        torch.Tensor,  # step_depth_predictions (n_env, 256, 256)
    ]:
        """
        Parameters
        ----------
        step_obs: dict
            {
                "pointgoal_with_gps_compass": (n_env, 2),
                "rgb": (n_env, 256, 256, 3)
            }
        """

        batch_input = {
            "observations": step_obs,
            "rnn_hidden_states": prev_recurrent_hidden_states,
            "prev_actions": prev_actions,
            "masks": not_done_masks,
        }
        if depth_prediction:
            action_data, depth_prediction = self._policy.act(**batch_input, deterministic=False, depth_prediction=True)
        else:
            action_data = self._policy.act(**batch_input, deterministic=False)
        

        # print("action_data:", action_data)
        # print("actions shape:", action_data.actions.shape)  # [4, 1]
        # print("values  shape:", action_data.values.shape)  # [4, 1]
        # print("action_log_probs shape:", action_data.action_log_probs.shape)  # [4, 1]
        # print("rnn_hidden_states s:", action_data.rnn_hidden_states.shape)  # [4 4,512]

        if depth_prediction:
            return (
                action_data.actions,
                action_data.rnn_hidden_states,
                action_data.values,
                action_data.action_log_probs,
                depth_prediction
            )
        else:
            return (
                action_data.actions,
                action_data.rnn_hidden_states,
                action_data.values,
                action_data.action_log_probs,   
            )

    def _evaluate_actions(
        self,
        mini_batch_obs,
        mini_batch_recurrent_hidden_states,
        mini_batch_prev_actions,
        mini_batch_not_done_masks,
        mini_batch_actions,
    ):
        features, rnn_hidden_states, aux_loss_state, depth_preds = self._policy.net.forward(
            observations=mini_batch_obs,
            rnn_hidden_states=mini_batch_recurrent_hidden_states,
            prev_actions=mini_batch_prev_actions,
            masks=mini_batch_not_done_masks,
            depth_prediction=True
        )

        value = self._policy.critic(features)

        distribution = self._policy.action_distribution(features)
        action_log_probs = distribution.log_probs(mini_batch_actions)

        dist_entropy = distribution.entropy()

        return value, action_log_probs, dist_entropy, depth_preds

    def _step_venvs(
        self,
        actions: torch.Tensor,  # (n_env, 1)
    ) -> Tuple[
        dict,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        If the environment is done, reset it too.

        Returns
        -------
        s_t+1: dict
        r_t+1: torch.Tensor (n_env, 1)
        notdone_t+1: torch.Tensor (n_env, 1)
        """
        # step
        for index_env in range(self._n_env):
            action = actions[index_env].item()
            self._envs.async_step_at(index_env, action)

        # wait result

        outputs = [
            self._envs.wait_step_at(index_env) for index_env in range(self._n_env)
        ]

        observations, rewards_l, dones, infos = [list(x) for x in zip(*outputs)]

        step_obs = self._batch_step_obs(observations)
        step_rewards = torch.tensor(
            rewards_l,
            dtype=torch.float,
            device=self._device,
        )
        step_rewards = step_rewards.unsqueeze(1)

        step_not_done_masks = torch.tensor(
            [[not done] for done in dones],
            dtype=torch.bool,
            device=self._device,
        )

        # Add logs
        for idx_env in range(self._n_env):
            self._logger["rewards_by_env"][idx_env] += step_rewards[idx_env].item()
            self._logger["ep_lens_by_env"][idx_env] += 1

            if not step_not_done_masks[idx_env]:
                self._num_ep += 1
                self._logger["batch_rews"].append(
                    self._logger["rewards_by_env"][idx_env]
                )
                self._logger["batch_num_wp_reached"].append(
                    infos[idx_env]["num_wp_reached"]
                )
                self._logger["batch_first_wp_reached"].append(
                    infos[idx_env]["first_wp_reached"]
                )
                self._logger["batch_l2_dist_to_goal"].append(
                    infos[idx_env]["l2_dist_to_goal"]
                )

                if infos[idx_env]["first_wp_reached"]:
                    self._logger["batch_l2_dist_to_1st_wp"].append(0.0)
                else:
                    self._logger["batch_l2_dist_to_1st_wp"].append(
                        infos[idx_env]["l2_dist_to_goal"]
                    )
                self._logger["batch_collisions"].append(infos[idx_env]["collision"])
                self._logger["batch_travel_distances"].append(
                    infos[idx_env]["travel_distance"]
                )
                self._logger["batch_timeouts"].append(infos[idx_env]["timeout"])
                self._logger["batch_ep_lens"].append(
                    self._logger["ep_lens_by_env"][idx_env]
                )
                self._logger["batch_count"] += 1

                # Reset logs
                self._logger["rewards_by_env"][idx_env] = 0.0
                self._logger["ep_lens_by_env"][idx_env] = 0

        return step_obs, step_rewards, step_not_done_masks

    def _log_summary(self):
        # Calculate logging values. I use a few python shortcuts to calculate each value
        # without explaining since it's not too important to PPO; feel free to look it over,
        # and if you have any questions you can email me (look at bottom of README)
        delta_t = self._logger["delta_t"]
        self._logger["delta_t"] = time.time_ns()
        delta_t = (self._logger["delta_t"] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        num_terminated_episodes = self._logger["batch_count"]

        t_so_far = self._logger["t_so_far"]
        i_so_far = self._logger["i_so_far"]
        avg_ep_rews = np.mean(self._logger["batch_rews"])
        avg_ep_lens = np.mean(self._logger["batch_ep_lens"])
        sr_1 = np.mean(self._logger["batch_first_wp_reached"])
        avg_wp_reached = np.mean(self._logger["batch_num_wp_reached"])
        avg_dist_traveled = np.mean(self._logger["batch_travel_distances"])
        avg_dist_to_1st_wp = np.mean(self._logger["batch_l2_dist_to_1st_wp"])
        avg_dist_to_goal = np.mean(self._logger["batch_l2_dist_to_goal"])
        avg_collisions = np.mean(self._logger["batch_collisions"])
        avg_timeouts = np.mean(self._logger["batch_timeouts"])

        avg_actor_loss = np.mean(
            [losses.cpu().float().mean() for losses in self._logger["actor_losses"]]
        )
        avg_value_loss = np.mean(
            [losses.cpu().float().mean() for losses in self._logger["value_losses"]]
        )
        avg_dist_entropy = np.mean(
            [losses.cpu().float().mean() for losses in self._logger["dist_entropies"]]
        )
        avg_depth_loss = np.mean(
            [losses.cpu().float().mean() for losses in self._logger["depth_losses"]]
        )

        if self._use_wandb:
            wandb.log(
                step=t_so_far,
                data={
                    "avg_ep_len": avg_ep_lens,
                    "reward": avg_ep_rews,
                    "avg_actor_loss": avg_actor_loss,
                    # habitat-lab metrics
                    "metrics/first_wp_reached": sr_1,
                    "metrics/num_wp_reached": avg_wp_reached,
                    "metrics/travel_distance": avg_dist_traveled,
                    "metrics/l2_dist_to_1st_wp": avg_dist_to_1st_wp,
                    "metrics/l2_dist_to_goal": avg_dist_to_goal,
                    "metrics/collision": avg_collisions,
                    "metrics/timeout": avg_timeouts,
                    "learner/action_loss": avg_actor_loss,
                    "learner/value_loss": avg_value_loss,
                    "learner/dist_entropy": avg_dist_entropy,
                },
            )

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))
        avg_value_loss = str(round(avg_value_loss, 5))
        avg_dist_entropy = str(round(avg_dist_entropy, 5))

        # Print logging statements
        print(flush=True)
        print(
            f"-------------------- Iteration #{i_so_far} --------------------",
            flush=True,
        )
        print(f"Iteration took  : {delta_t} secs", flush=True)
        print(f"Success Rate 1  : {sr_1}", flush=True)
        print(f"avg wp reached  : {avg_wp_reached}", flush=True)
        print(f"")
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Episodes So Far : {self._num_ep}", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Reward: {avg_ep_rews}", flush=True)
        print(f"Average Action Loss : {avg_actor_loss}", flush=True)
        print(f"Average Value Loss  : {avg_value_loss}", flush=True)
        print(f"Average Dist Entropy: {avg_dist_entropy}", flush=True)
        print(f"")
        print(f"avg dist traveled : {avg_dist_traveled}", flush=True)
        print(f"avg dist to 1st wp: {avg_dist_to_1st_wp}", flush=True)
        print(f"collisions       : {avg_collisions}", flush=True)
        print(f"timeouts         : {avg_timeouts}", flush=True)
        print(f"num terminated eps: {num_terminated_episodes}", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        # Reset batch-specific logging data
        self._logger["rewards_by_env"] = [0.0 for _ in range(self._n_env)]
        self._logger["ep_lens_by_env"] = [0 for _ in range(self._n_env)]

        self._logger["batch_rews"] = []
        self._logger["batch_ep_lens"] = []
        self._logger["batch_first_wp_reached"] = []
        self._logger["batch_num_wp_reached"] = []
        self._logger["batch_travel_distances"] = []
        self._logger["batch_l2_dist_to_1st_wp"] = []
        self._logger["batch_l2_dist_to_goal"] = []
        self._logger["batch_collisions"] = []
        self._logger["batch_timeouts"] = []
        self._logger["batch_count"] = 0
        self._logger["actor_losses"] = []
        self._logger["value_losses"] = []
        self._logger["dist_entropies"] = []
        self._logger["depth_losses"] = []
