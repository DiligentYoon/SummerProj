from typing import Any, Mapping, Optional, Tuple, Union

import copy
import gymnasium
from packaging import version

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config, logger
from skrl.agents.torch.ddpg import DDPG
from skrl.memories.torch import Memory
from .replay_buffer import LowLevelHindsightReplayBuffer
from skrl.models.torch import Model

# fmt: off
# [start-config-dict-torch]
DDPG_DEFAULT_CONFIG = {
    "gradient_steps": 1,            # gradient steps
    "batch_size": 64,               # training batch size

    "discount_factor": 0.99,        # discount factor (gamma)
    "polyak": 0.005,                # soft update hyperparameter (tau)

    "actor_learning_rate": 1e-3,    # actor learning rate
    "critic_learning_rate": 1e-3,   # critic learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.observation_space})

    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

    "grad_norm_clip": 0,            # clipping coefficient for the norm of the gradients

    "exploration": {
        "noise": None,              # exploration noise
        "initial_scale": 1.0,       # initial scale for the noise
        "final_scale": 1e-3,        # final scale for the noise
        "timesteps": None,          # timesteps for the noise decay
    },

    "rewards_shaper": None,         # rewards shaping function: Callable(reward, timestep, timesteps) -> reward

    "mixed_precision": False,       # enable automatic mixed precision for higher performance

    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": "auto",   # TensorBoard writing interval (timesteps)

        "checkpoint_interval": "auto",      # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately

        "wandb": False,             # whether to use Weights & Biases
        "wandb_kwargs": {}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    }
}
# [end-config-dict-torch]
# fmt: on


class DDPGAgent(DDPG):
    def __init__(self,
                 models: Mapping[str, Model],
                 memory: Optional[Union[LowLevelHindsightReplayBuffer, Tuple[LowLevelHindsightReplayBuffer]]] = None,
                 observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
                 action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 cfg: Optional[dict] = None) -> None:
        
        # DDPG의 __init__ 내용을 거의 그대로 가져옵니다.
        _cfg = copy.deepcopy(DDPG_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})

        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=_cfg,
        )

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        super().__init__(trainer_cfg=trainer_cfg)

    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        # states는 반드시 train 코드쪽에서 Dict -> Tensor 타입으로 변환해서 줘야 한다.
        return super().act(states, timestep, timesteps)
    
    def record_transition(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        
        super().record_transition(states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps)

    
    def _update(self, timestep, timesteps):
        
        # gradient steps
        for gradient_step in range(self._gradient_steps):

            # sample a batch from memory
            (
                sampled_states,
                sampled_actions,
                sampled_rewards,
                sampled_next_states,
                sampled_terminated,
                sampled_truncated,
            ) = self.memory.sample(names=self._tensors_names, batch_size=self._batch_size)[0]

            with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                pass