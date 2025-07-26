# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

from abc import abstractmethod
from isaaclab.envs.direct_rl_env import DirectRLEnv
from isaaclab.envs.utils.spaces import sample_space, spec_to_gym_space
from isaaclab.envs.common import VecEnvObs, VecEnvStepReturn
from .direct_diol_env_cfg import DirectDIOLCfg


class DirectDIOL(DirectRLEnv):

    def __init__(self, cfg: DirectDIOLCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)


    def _configure_gym_env_spaces(self):
        # set up spaces
        if type(self.cfg.high_level_action_space["where"]) == int:
            self.cfg.high_level_action_space["where"] = {self.cfg.high_level_action_space["where"]}
        self.single_observation_space_h = gym.spaces.Dict()
        self.single_observation_space_h["policy"] = spec_to_gym_space(self.cfg.high_level_observation_space)
        self.single_action_space_h = spec_to_gym_space(self.cfg.high_level_action_space)

        self.single_observation_space_l = gym.spaces.Dict()
        self.single_observation_space_l["policy"] = spec_to_gym_space(self.cfg.low_level_observation_space)
        self.single_action_space_l = spec_to_gym_space(self.cfg.low_level_action_space)

        # batch the spaces for vectorized environments
        self.observation_space_h = gym.vector.utils.batch_space(self.single_observation_space_h["policy"], self.num_envs)
        self.action_space_h = gym.vector.utils.batch_space(self.single_action_space_h, self.num_envs)

        self.observation_space_l = gym.vector.utils.batch_space(self.single_observation_space_l["policy"], self.num_envs)
        self.action_space_l = gym.vector.utils.batch_space(self.single_action_space_l, self.num_envs)

        # optional state space for asymmetric actor-critic architectures
        self.state_space = None
        self.state_space_h = None
        self.state_space_l = None
        if self.cfg.state_space:
            self.single_observation_space_h["value"] = spec_to_gym_space(self.cfg.high_level_state_space)
            self.state_space_h = gym.vector.utils.batch_space(self.single_observation_space_h["value"], self.num_envs)
            self.single_observation_space_l["value"] = spec_to_gym_space(self.cfg.low_level_state_space)
            self.state_space_l = gym.vector.utils.batch_space(self.single_observation_space_l["value"], self.num_envs)

        # instantiate actions (needed for tasks for which the observations computation is dependent on the actions)
        self.actions_h = sample_space(self.single_action_space_h, self.sim.device, batch_size=self.num_envs, fill_value=0)
        self.actions_l = sample_space(self.single_action_space_l, self.sim.device, batch_size=self.num_envs, fill_value=0)


        # 환경과 직접 상호작용 하는 에이전트는 Low-Level Agent
        self.single_observation_space = self.single_observation_space_l
        self.single_action_space = self.single_action_space_l
        self.observation_space = self.observation_space_l
        self.action_space = self.action_space_l
        self.state_space = self.state_space_l

    @abstractmethod
    def _get_high_reward(self) -> torch.Tensor:
        raise NotImplementedError(f"Please implement the '_get_high_reward' method for {self.__class__.__name__}.")

    @abstractmethod
    def _apply_high_action(self) -> None:
        raise NotImplementedError(f"Please implement the '_apply_high_action' method for {self.__class__.__name__}.")
    
    @abstractmethod
    def _set_extra_infos(self) -> None:
        raise NotImplementedError(f"Please implement the '_set_extra_infos' method for {self.__class__.__name__}.")