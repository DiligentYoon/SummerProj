from typing import Any, Mapping, Optional, Tuple, Union

import copy
import gymnasium
import numpy as np
from packaging import version

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config, logger
from skrl.agents.torch import Agent
from skrl.models.torch import Model

from .replay_buffer import LowLevelHindSightReplayBuffer

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


class DDPGAgent(Agent):
    def __init__(self,
                 models: Mapping[str, Model],
                 memory: Optional[Union[LowLevelHindSightReplayBuffer, Tuple[LowLevelHindSightReplayBuffer]]] = None,
                 observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
                 action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 cfg: Optional[dict] = None) -> None:

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

        # pre-trained
        self.use_pre_trained_model = self.cfg.get("use_pre_trained", False)
        self.checkpoint_path      = self.cfg.get("checkpoint_path", "")

        # models
        self.policy = self.models.get("policy", None)
        self.critic = {
            "critic_1": self.models.get("critic_1"),
            "critic_2": self.models.get("critic_2")
        }
        self.target_critic = {
            "target_critic_1": self.models.get("target_critic_1"),
            "target_critic_2": self.models.get("target_critic_2")
        }

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["critic"] = self.critic
        self.checkpoint_modules["target_critic"] = self.target_critic

        # freeze target networks with respect to optimizers (update via .update_parameters())
        for critic_model, target_critic_model in zip(self.critic.values(), self.target_critic.values()):
            if target_critic_model is not None:
                target_critic_model.freeze_parameters(True)
                target_critic_model.update_parameters(critic_model)
        
        # configuration
        self._gradient_steps = self.cfg["gradient_steps"]
        self._batch_size = self.cfg["batch_size"]

        self._discount_factor = self.cfg["discount_factor"]
        self._polyak = self.cfg["polyak"]

        self._actor_learning_rate = self.cfg["actor_learning_rate"]
        self._critic_learning_rate = self.cfg["critic_learning_rate"]
        self._learning_rate_scheduler = self.cfg["learning_rate_scheduler"]

        self._state_preprocessor = self.cfg["state_preprocessor"]

        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self._grad_norm_clip = self.cfg["grad_norm_clip"]

        self._exploration_noise = self.cfg["exploration"]["noise"]
        self._exploration_initial_scale = self.cfg["exploration"]["initial_scale"]
        self._exploration_final_scale = self.cfg["exploration"]["final_scale"]
        self._exploration_timesteps = self.cfg["exploration"]["timesteps"]

        self._rewards_shaper = self.cfg["rewards_shaper"]

        self._mixed_precision = self.cfg["mixed_precision"]

        if config.torch.is_distributed:
            logger.info(f"Broadcasting models' parameters")
            if self.policy is not None:
                self.policy.broadcast_parameters()
            for k, v in self.critic.items():
                if v is not None:
                    v.broadcast_parameters()

        # set up automatic mixed precision
        self._device_type = torch.device(device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self._mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self._mixed_precision)

        # set up optimizers and learning rate schedulers
        if self.policy is not None and self.critic is not None:
            self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self._actor_learning_rate)
            if self._learning_rate_scheduler is not None:
                self.policy_scheduler = self._learning_rate_scheduler(
                    self.policy_optimizer, **self.cfg["learning_rate_scheduler_kwargs"]
                )
        
        self.critic_optimizer = {}
        self.critic_scheduler = {}
        if not self.use_pre_trained_model:
            for k, v in self.critic.items():
                if v is not None:
                    self.critic_optimizer[k] = torch.optim.Adam(v.parameters(), lr=self._critic_learning_rate)
                    if self._learning_rate_scheduler is not None:
                        self.critic_scheduler[k] = self._learning_rate_scheduler(
                            self.critic_optimizer[k], **self.cfg["learning_rate_scheduler_kwargs"]
                        )

            self.checkpoint_modules["policy_optimizer"] = self.policy_optimizer
            self.checkpoint_modules["critic_optimizer"] = self.critic_optimizer
            

        # set up preprocessors
        if self._state_preprocessor:
            _policy_state_preprocessor = self._state_preprocessor(**self.cfg["state_preprocessor_kwargs"]["policy"])
            self.checkpoint_modules["policy_state_preprocessor"] = _policy_state_preprocessor

            _critic_state_preprocessor = self._state_preprocessor(**self.cfg["state_preprocessor_kwargs"]["critic"])
            self.checkpoint_modules["critic_state_preprocessor"] = _critic_state_preprocessor
        else:
            _policy_state_preprocessor = self._empty_preprocessor
            _critic_state_preprocessor = self._empty_preprocessor
        
        self._state_preprocessor = {
            "policy": _policy_state_preprocessor,
            "critic": _critic_state_preprocessor}

        self._tensors_names = []


    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the agent"""
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")

        if not self.use_pre_trained_model:
            # create tensors in memory
            if self.memory is not None:
                self.memory.create_tensor(name="states", size=self.observation_space["policy"], dtype=torch.float32)
                self.memory.create_tensor(name="next_states", size=self.observation_space["policy"], dtype=torch.float32)
                self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
                self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
                self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
                self.memory.create_tensor(name="truncated", size=1, dtype=torch.bool)

            # clip noise bounds
            if self.action_space is not None:
                self.clip_actions_min = torch.tensor(self.action_space.low, device=self.device)
                self.clip_actions_max = torch.tensor(self.action_space.high, device=self.device)
        else:
            # 가중치 로드
            state = torch.load(self.checkpoint_path, map_location=self.device)
            self.policy.load_state_dict(state["policy"])
            for k in ["critic_1", "critic_2"]:
                if k in state:
                    self.critic[k].load_state_dict(state[k])

            # 파라미터 동결
            for p in self.policy.parameters():
                p.requires_grad_(False)
            for c in self.critic.values():
                for p in c.parameters():
                    p.requires_grad_(False)
    

    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        """Process the environment's states to make a decision (actions) using the main policy

        :param states: Environment's states
        :type states: torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :return: Actions
        :rtype: torch.Tensor
        """

        # sample deterministic actions
        with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            actions, _, = self.policy.compute({"states": self._state_preprocessor(states)}, role="policy")
        
        return actions, None, {}


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
        
        if self.use_pre_trained_model:
            return
        
        super().record_transition(
            states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps
        )

        if self.memory is not None:
            # storage transition in memory
            self.memory.add_samples(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
            )
    
    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        timestep += 1

        # update best models and write checkpoints
        if timestep > 1 and self.checkpoint_interval > 0 and not timestep % self.checkpoint_interval:
            # update best models
            reward = np.mean(self.tracking_data.get("Reward / Total reward (mean)", -(2**31)))
            if reward > self.checkpoint_best_modules["reward"]:
                self.checkpoint_best_modules["timestep"] = timestep
                self.checkpoint_best_modules["reward"] = reward
                self.checkpoint_best_modules["saved"] = False
                self.checkpoint_best_modules["modules"] = {
                    k: copy.deepcopy(self._get_internal_value(v)) for k, v in self.checkpoint_modules.items()
                }
            # write checkpoints
            self.write_checkpoint(timestep, timesteps)

        # write to tensorboard
        if timestep > 1 and self.write_interval > 0 and not timestep % self.write_interval:
            self.write_tracking_data(timestep, timesteps)


    
    def _update(self, timestep, timesteps):

        if self.use_pre_trained_model:
            return {}
        
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

