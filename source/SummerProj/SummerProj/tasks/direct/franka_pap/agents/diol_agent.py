# File: source/SummerProj/SummerProj/tasks/direct/franka_pap/agents/diol_agent.py
from typing import Any, Mapping, Optional, Tuple, Union, Dict

import os
import copy
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from packaging import version

from skrl import config, logger
from skrl.agents.torch import Agent
from skrl.models.torch import Model

from .replay_buffer import HighLevelHindSightReplayBuffer

# DIOLAgent를 위한 기본 설정값
DIOL_DEFAULT_CONFIG = {
    "gradient_steps": 1,            # gradient steps
    "batch_size": 64,               # training batch size

    "discount_factor": 0.99,        # discount factor (gamma)
    "polyak": 0.005,                # soft update hyperparameter (tau)

    "learning_rate": 1e-4,                  # learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "random_timesteps": 0,          # 랜덤 탐험 스텝 수
    "learning_starts": 0,           # 학습 시작 전 최소 경험 수

    "update_interval": 1,           # 네트워크 학습 주기
    "target_update_interval": 10,   # 타겟 네트워크 학습 주기

    "exploration": {
        "initial_epsilon": 1.0,     # 초기 탐험율 (epsilon-greedy)
        "final_epsilon": 0.01,      # 최종 탐험율
        "timesteps": 100000,        # 엡실론이 최종값까지 감소하는 데 걸리는 스텝 수
    },

    "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.observation_space})

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

# DIOL Agent 클래스를 정의.
# Reference: 
class DIOLAgent(Agent):
    def __init__(self,
                 models: Dict[str, Model],
                 memory: HighLevelHindSightReplayBuffer,
                 observation_space: gym.Space,
                 action_space: gym.Space,
                 device: Union[str, torch.device],
                 cfg: Mapping[str, Any]) -> None:
        
        _cfg = copy.deepcopy(DIOL_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})

        super().__init__(models=models,
                         memory=memory,
                         observation_space=observation_space,
                         action_space=action_space,
                         device=device,
                         cfg=_cfg)

        self.q_network = self.models["q_network"]
        self.target_q_network = self.models["target_q_network"]

        # 타겟 네트워크는 학습되지 않도록 설정 : Soft 업데이트만 수행함.
        if config.torch.is_distributed:
            logger.info(f"Broadcasting models' parameters")
            if self.q_network is not None:
                self.q_network.broadcast_parameters()

        if self.target_q_network is not None:
            # freeze target networks with respect to optimizers (update via .update_parameters())
            self.target_q_network.freeze_parameters(True)

            # update target networks (hard update)
            self.target_q_network.update_parameters(self.q_network, polyak=1)

        # config 기반 컴포넌트 설정
        self.checkpoint_modules["q_network"] = self.q_network
        self.checkpoint_modules["target_q_network"] = self.target_q_network
        

        # configuration
        self._gradient_steps = self.cfg["gradient_steps"]
        self._batch_size = self.cfg["batch_size"]

        self._discount_factor = self.cfg["discount_factor"]
        self._polyak = self.cfg["polyak"]

        self._learning_rate = self.cfg["learning_rate"]
        self._learning_rate_scheduler = self.cfg["learning_rate_scheduler"]

        self._state_preprocessor = self.cfg["state_preprocessor"]

        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self._update_interval = self.cfg["update_interval"]
        self._target_update_interval = self.cfg["target_update_interval"]

        self._exploration_initial_epsilon = self.cfg["exploration"]["initial_epsilon"]
        self._exploration_final_epsilon = self.cfg["exploration"]["final_epsilon"]
        self._exploration_timesteps = self.cfg["exploration"]["timesteps"]

        self._rewards_shaper = self.cfg["rewards_shaper"]

        self._mixed_precision = self.cfg["mixed_precision"]

        # set up automatic mixed precision
        self._device_type = torch.device(device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self._mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self._mixed_precision)

        # set up optimizer and learning rate scheduler
        if self.q_network is not None:
            self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self._learning_rate)
            if self._learning_rate_scheduler is not None:
                self.scheduler = self._learning_rate_scheduler(
                    self.optimizer, **self.cfg["learning_rate_scheduler_kwargs"]
                )

            self.checkpoint_modules["optimizer"] = self.optimizer

        # set up preprocessors
        if self._state_preprocessor:
            self._state_preprocessor = self._state_preprocessor(**self.cfg["state_preprocessor_kwargs"])
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor
        
        self._tensors_names = []
    
    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the agent"""
        super().init(trainer_cfg=trainer_cfg)

        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(name="states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="next_states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.int64)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="truncated", size=1, dtype=torch.bool)


    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> tuple[torch.Tensor, None, dict]:
        """Epsilon-greedy 전략에 따라 행동을 결정"""

        # 엡실론 값 업데이트
        self._exploration_epsilon = max(self.cfg["exploration"]["final_epsilon"], self._exploration_epsilon - self._epsilon_decay)
        
        # 무작위 행동 선택 (Exploration)
        if torch.rand(1).item() < self._exploration_epsilon:
            actions = self.action_space.sample(states.shape[0])
            return actions, None, {}
        
        # Q-value가 가장 높은 행동 선택 (Exploitation)
        q_values, _ = self.q_network.compute({"states": states})
        actions = torch.argmax(q_values, dim=1, keepdim=True)
        return actions, None, {}

    def record_transition(self,
                          states: torch.Tensor,
                          actions: torch.Tensor,
                          rewards: torch.Tensor,
                          next_states: torch.Tensor,
                          terminated: torch.Tensor,
                          truncated: torch.Tensor,
                          infos: dict,
                          **kwargs) -> None:
        """
            HRL을 위한 특별한 전환(transition)을 메모리에 기록
        """
        if self.write_interval > 0:
            # compute the cumulative sum of the rewards and timesteps
            if self._cumulative_rewards is None:
                self._cumulative_rewards = torch.zeros_like(rewards, dtype=torch.float32)
                self._cumulative_timesteps = torch.zeros_like(rewards, dtype=torch.int32)

            self._cumulative_rewards.add_(rewards)
            self._cumulative_timesteps.add_(1)

            # check ended episodes : 반드시 High-Level 기준으로 측정
            finished_episodes = (terminated + truncated).nonzero(as_tuple=False)
            if finished_episodes.numel():

                # storage cumulative rewards and timesteps
                self._track_rewards.extend(self._cumulative_rewards[finished_episodes][:, 0].reshape(-1).tolist())
                self._track_timesteps.extend(self._cumulative_timesteps[finished_episodes][:, 0].reshape(-1).tolist())

                # reset the cumulative rewards and timesteps
                self._cumulative_rewards[finished_episodes] = 0
                self._cumulative_timesteps[finished_episodes] = 0

                # record reward data
                self.tracking_data["[High] Reward / Instantaneous reward (max)"].append(torch.max(rewards).item())
                self.tracking_data["[High] Reward / Instantaneous reward (min)"].append(torch.min(rewards).item())
                self.tracking_data["[High] Reward / Instantaneous reward (mean)"].append(torch.mean(rewards).item())

                if len(self._track_rewards):
                    track_rewards = np.array(self._track_rewards)

                    self.tracking_data["[High] Reward / Total reward (max)"].append(np.max(track_rewards))
                    self.tracking_data["[High] Reward / Total reward (min)"].append(np.min(track_rewards))
                    self.tracking_data["[High] Reward / Total reward (mean)"].append(np.mean(track_rewards))


    def _update(self, timestep: int, timesteps: int) -> None:
        """DIOL 업데이트 규칙에 따라 Q-네트워크를 학습"""
        # 학습 시작 조건 확인
        if timestep < self.cfg["learning_starts"]:
            return
        
        # N 스텝마다 한 번씩 학습
        if not timestep % self.cfg["update_interval"] == 0:
            return

        # --- 미니배치 샘플링 ---
        sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones, sampled_infos = \
            self.memory.sample(self.cfg["batch_size"])
        
        # 옵션 종료 여부 플래그 추출
        option_terminated_mask = sampled_infos["option_terminated"]

        # --- Target Q-value 계산 ---
        with torch.no_grad():
            # 다음 상태에 대한 모든 행동의 Q-value를 타겟 네트워크에서 계산
            next_q_values_target, _ = self.target_q_network.compute({"states": sampled_next_states})

            # 경우 1 (옵션 종료 상황): 다음 상태에서 가장 높은 Q-value를 선택 (max Q)
            next_max_q_values = torch.max(next_q_values_target, dim=1, keepdim=True).values
            
            # 경우 2 (옵션 진행): 다음 상태에서 현재와 '동일한 행동'을 계속했을 때의 Q-value를 선택
            # 논문 Equation (3) 구현:
            # 옵션이 종료되지 않았을 때의 가치: 동일한 액션(sampled_actions)을 계속했을 때의 가치
            # 현재 state만 입력받아 모든 action의 q-value를 출력하므로,
            # "동일한 액션을 했을 때의 가치"를 그대로 가져오기 위해 gather를 사용.
            q_values_for_same_action = next_q_values_target.gather(dim=1, index=sampled_actions.long())
            
            # 옵션 종료 여부에 따라 두 경우의 가치를 선택
            next_q_values = torch.where(option_terminated_mask, next_max_q_values, q_values_for_same_action)

            # 최종 Target Q-value 계산: y = r + gamma * Q_target(s', a')
            target_q_values = sampled_rewards + self.cfg["discount_factor"] * next_q_values * (1 - sampled_dones)

        # --- Current Q-value 계산 ---
        # 현재 Q-네트워크에서 (상태, 행동) 쌍에 대한 Q-value를 가져옴
        q_values, _ = self.q_network.compute({"states": sampled_states})
        current_q_values = q_values.gather(dim=1, index=sampled_actions.long())

        # --- Loss 계산 및 Back Propagation ---
        loss = nn.functional.mse_loss(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # --- Target Network의 Soft Update ---
        for target_param, param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.cfg["polyak"] * param.data + (1.0 - self.cfg["polyak"]) * target_param.data)


    
    def write_tracking_data(self, timestep: int, timesteps: int) -> None:
        """Write tracking data to TensorBoard

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        for k, v in self.tracking_data.items():
            if k.endswith("(min)"):
                self.writer.add_scalar(k, np.min(v), timestep)
            elif k.endswith("(max)"):
                self.writer.add_scalar(k, np.max(v), timestep)
            else:
                self.writer.add_scalar(k, np.mean(v), timestep)
        # reset data containers for next iteration
        self._track_rewards.clear()
        self._track_timesteps.clear()
        self.tracking_data.clear()
    
    def write_checkpoint(self, timestep: int, timesteps: int) -> None:
        """Write checkpoint (modules) to disk

        The checkpoints are saved in the directory 'checkpoints' in the experiment directory.
        The name of the checkpoint is the current timestep if timestep is not None, otherwise it is the current time.

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        tag = str(timestep if timestep is not None else datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f"))
        # separated modules
        if self.checkpoint_store_separately:
            for name, module in self.checkpoint_modules.items():
                torch.save(
                    self._get_internal_value(module),
                    os.path.join(self.experiment_dir, "high_checkpoints", f"{name}_{tag}.pt"),
                )
        # whole agent
        else:
            modules = {}
            for name, module in self.checkpoint_modules.items():
                modules[name] = self._get_internal_value(module)
            torch.save(modules, os.path.join(self.experiment_dir, "high_checkpoints", f"agent_{tag}.pt"))

        # best modules
        if self.checkpoint_best_modules["modules"] and not self.checkpoint_best_modules["saved"]:
            # separated modules
            if self.checkpoint_store_separately:
                for name, module in self.checkpoint_modules.items():
                    torch.save(
                        self.checkpoint_best_modules["modules"][name],
                        os.path.join(self.experiment_dir, "high_checkpoints", f"best_{name}.pt"),
                    )
            # whole agent
            else:
                modules = {}
                for name, module in self.checkpoint_modules.items():
                    modules[name] = self.checkpoint_best_modules["modules"][name]
                torch.save(modules, os.path.join(self.experiment_dir, "high_checkpoints", "best_agent.pt"))
            self.checkpoint_best_modules["saved"] = True
        