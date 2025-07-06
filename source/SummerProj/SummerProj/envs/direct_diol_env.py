# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import builtins
import omni.log
from dataclasses import MISSING
import gymnasium as gym
import numpy as np
import torch
from abc import abstractmethod

from isaaclab.envs.direct_rl_env import DirectRLEnv
from isaaclab.envs.utils.spaces import sample_space, spec_to_gym_space
from .direct_diol_env_cfg import DirectDIOLCfg

class DirectDIOL(DirectRLEnv):
    """
        DirectDIOL is a class that extends the DirectRLEnv class to provide hierarchical reinforcement learning capabilities.
    """
    cfg: DirectDIOLCfg
    def __init__(self, cfg: DirectDIOLCfg, render_mode: str | None = None, **kwargs):
        """
        Initialize the DirectDIOL environment with the given configuration.

        Args:
            cfg (DirectDIOLCfg): Configuration for the environment.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(cfg, render_mode, **kwargs)

        # HRL 관련 버퍼 초기화 (저 수준 목표 및 고 수준 최종 목표)
        self.reward_buf = -1.0 * torch.ones(self.num_envs, dtype=torch.float, device=self.device)

        # extras 딕셔너리에 HRL 관련 키들을 미리 초기화합니다.
        self.extras["high_level_reward"] = torch.zeros(self.num_envs, device=self.device)
        self.extras["option_terminated"] = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        print("[INFO] DirectDIOL: Hierarchical RL Environment Manger Initialized.")

    def _configure_gym_env_spaces(self):

        # ============== Environment Agent (Low-Level) spaces ===============
        # set up spaces
        self.single_action_space = gym.spaces.Dict({
            "policy": gym.spaces.Box(low=-torch.inf, high=torch.inf, shape=(self.cfg.action_space,)),
            "critic": gym.spaces.Box(low=-torch.inf, high=torch.inf, shape=(1,))
        })
        self.single_observation_space = gym.spaces.Dict({
            "policy": gym.spaces.Dict({
                "observation": gym.spaces.Box(low=-torch.inf, high=torch.inf, shape=(self.cfg.observation_space,)),
                "desired_goal": gym.spaces.Box(low=-torch.inf, high=torch.inf, shape=(self.cfg.low_level_goal_dim,))
            }),
            "critic": gym.spaces.Dict({
                "observation": gym.spaces.Box(low=-torch.inf, high=torch.inf, shape=(self.cfg.observation_space,)),
                "desired_goal": gym.spaces.Box(low=-torch.inf, high=torch.inf, shape=(self.cfg.low_level_goal_dim,)),
                "taken_action": gym.spaces.Box(low=-torch.inf, high=torch.inf, shape=(self.cfg.action_space,))
            })
        })

        # set up batched spaces
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

        self.actions = sample_space(self.single_action_space, self.sim.device, batch_size=self.num_envs, fill_value=0)


    def _reset_idx(self, env_ids) -> None:
        super()._reset_idx(env_ids)
        self.reward_buf[env_ids] = -1.0