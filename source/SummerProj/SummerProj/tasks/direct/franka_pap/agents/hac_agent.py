# File: source/SummerProj/SummerProj/tasks/direct/franka_pap/agents/diol_agent.py
from typing import Any, Mapping, Optional, Tuple, Union, Dict

import os
import copy
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces

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
    "temperature": 1.0,

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

class HybridActorCriticAgent(Agent):
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

        # models
        self.policy = self.models.get("policy", None)
        self.value = {
            "value_1": self.models.get("value_1"),
            "value_2": self.models.get("value_2")
        }
        self.target_value = {
            "target_value_1": self.models.get("target_value_1"),
            "target_value_2": self.models.get("target_value_2")
        }

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["value_1"] = self.value["value_1"]
        self.checkpoint_modules["value_2"] = self.value["value_2"]
        self.checkpoint_modules["target_value_1"] = self.target_value["target_value_1"]
        self.checkpoint_modules["target_value_2"] = self.target_value["target_value_2"]

        # 타겟 네트워크는 학습되지 않도록 설정 : Soft 업데이트만 수행함.
        for value_model, target_value_model in zip(self.value.values(), self.target_value.values()):
            if target_value_model is not None:
                target_value_model.freeze_parameters(True)
                target_value_model.update_parameters(value_model)
        
        # pre-trained
        self.use_pre_trained_model = self.cfg.get("use_pre_trained", False)
        self.checkpoint_path      = self.cfg.get("checkpoint_path", "")

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
        self._temperature = self.cfg["temperature"]

        self._update_interval = self.cfg["update_interval"]
        self._target_update_interval = self.cfg["target_update_interval"]

        self._exploration_initial_epsilon = self.cfg["exploration"]["initial_epsilon"]
        self._exploration_final_epsilon = self.cfg["exploration"]["final_epsilon"]
        self._exploration_timesteps = self.cfg["exploration"]["timesteps"]

        self._rewards_shaper = self.cfg["rewards_shaper"]

        self._mixed_precision = self.cfg["mixed_precision"]

        # set up optimizers and learning rate schedulers
        if self.policy is not None and self.value is not None:
            self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self._learning_rate)
            if self._learning_rate_scheduler is not None:
                self.policy_scheduler = self._learning_rate_scheduler(
                    self.policy_optimizer, **self.cfg["learning_rate_scheduler_kwargs"]
                )
        
        self.value_optimizer = {}
        self.value_scheduler = {}
        if not self.use_pre_trained_model:
            for k, v in self.value.items():
                if v is not None:
                    self.value_optimizer[k] = torch.optim.Adam(v.parameters(), lr=self._learning_rate)
                    if self._learning_rate_scheduler is not None:
                        self.value_scheduler[k] = self._learning_rate_scheduler(
                            self.value_optimizer[k], **self.cfg["learning_rate_scheduler_kwargs"]
                        )

            self.checkpoint_modules["policy_optimizer"] = self.policy_optimizer
            self.checkpoint_modules["value_optimizer"] = self.value_optimizer

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
        self.set_mode("eval")

        if not self.use_pre_trained_model:
            # create tensors in memory
            if self.memory is not None:
                self.memory.create_tensor(name="states", size=self.observation_space, dtype=torch.float32)
                self.memory.create_tensor(name="next_states", size=self.observation_space, dtype=torch.float32)
                self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
                self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
                self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
                self.memory.create_tensor(name="truncated", size=1, dtype=torch.bool)
                self.memory.create_tensor(name="desired_goal", size=1, dtype=torch.bool)

            # clip noise bounds
            self.clip_actions_min = {}
            self.clip_actions_max = {}
            if self.action_space is not None:
                if isinstance(self.action_space, spaces.Dict):
                    for key, value in self.action_space.items():
                        if isinstance(self.action_space, spaces.Box):
                            self.clip_actions_min[key] = torch.tensor(value.low, device=self.device)
                            self.clip_actions_max[key] = torch.tensor(value.high, device=self.device)
                        else:
                            self.clip_actions_min[key] = None
                            self.clip_actions_max[key] = None
        else:
            self.set_running_mode("eval")


    def act(self, states: torch.Tensor, timestep: int, timesteps: int, deterministic=False) -> tuple[torch.Tensor, None, dict]:
        
        decay_ratio = min(1.0, timestep / (timesteps + 1e-6))
        current_epsilon = self._exploration_initial_epsilon - \
                         (self._exploration_initial_epsilon - self._exploration_final_epsilon) * decay_ratio
        inputs = {"states": states, "taken_actions": None}

        # 모델을 평가 모드로 설정 (베이스라인 코드의 set_training_mode(False)와 동일)
        self.policy.eval()
        if self.value is not None:
            self.value["value_1"].eval()
        
        with torch.no_grad():
            # 2. Actor 계산 -> Per-Point 모션 파라미터 얻기
            per_point_motions, _ = self.policy.compute(inputs)
            
            # 3. Critic 계산 -> Q-Map 얻기
            inputs["taken_actions"] = per_point_motions
            # TD3 구조이므로 첫 번째 Critic(critic_1)을 사용
            q_map, _ = self.value["value_1"].compute(inputs)

        # 모델을 다시 학습 모드로 돌려놓음
        self.policy.train()
        if self.value is not None:
            self.value["value_1"].train()

        # 4. q_map 형태: (B, N, 1), q_values 형태: (B, N)
        q_values = q_map.squeeze(-1)
        
        # Softmax를 적용하여 확률로 변환
        # batch 단위 처리를 위해 dim=1 기준으로 Softmax 적용
        action_scores = torch.nn.functional.softmax(q_values / self._temperature, dim=1)
        
        # 배치(batch)의 각 환경에 대해 독립적으로 인덱스 선택
        batch_size = q_values.shape[0]
        best_point_indices = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        
        if deterministic:
            # Deterministic: 전체 배치에 대해 argmax를 한 번에 적용
            best_point_indices = torch.argmax(action_scores, dim=1)
        else:
            # Stochastic: Epsilon-greedy

            # 5-1. 정책에 따른 행동
            policy_indices = torch.multinomial(action_scores, num_samples=1).squeeze(-1)
            
            # 5-2. 완전 무작위 행동 (탐험)
            random_indices = torch.randint(0, action_scores.shape[1], (action_scores.shape[0],), device=self.device)
            
            # 5-3. 각 환경이 탐험할지 여부를 결정하는 boolean mask 생성
            if timestep < self.cfg.get("learning_starts", 0):
                # 최초 학습 전까진 무조건 탐험
                explore_mask = torch.ones(action_scores.shape[0], dtype=torch.bool, device=self.device)
            else:
                # 학습 이후엔 Epsilon-Greedy 적용
                explore_mask = (torch.rand(action_scores.shape[0], device=self.device) < current_epsilon)

            # 5-4. torch.where를 사용해 if/else 로직 대체
            # explore_mask가 True인 위치에는 random_indices를, False인 위치에는 policy_indices를 선택
            best_point_indices = torch.where(explore_mask, random_indices, policy_indices)

        # 5. 최종 행동 구성
        # 선택된 인덱스를 사용해 최종 모션 파라미터 추출
        batch_indices = torch.arange(batch_size).to(self.device)
        final_motion_params = per_point_motions[batch_indices, best_point_indices]

        # gym.spaces.Dict 형식에 맞춰 최종 행동 반환
        final_action = {
            "where": best_point_indices.unsqueeze(-1), # (B, 1)
            "how": final_motion_params                 # (B, k)
        }

        infos = {"current_epsilon": current_epsilon,
                 "action_score": action_scores}
        

        return final_action, None, infos


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
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
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

                sampled_states = self._state_preprocessor(sampled_states, train=True)
                sampled_next_states = self._state_preprocessor(sampled_next_states, train=True)

                with torch.no_grad():
                    # target policy smoothing
                    next_actions, _, _ = self.target_policy.act({"states": sampled_next_states}, role="target_policy")
                    if self._smooth_regularization_noise is not None:
                        noises = torch.clamp(
                            self._smooth_regularization_noise.sample(next_actions.shape),
                            min=-self._smooth_regularization_clip,
                            max=self._smooth_regularization_clip,
                        )
                        next_actions.add_(noises)
                        next_actions.clamp_(min=self.clip_actions_min, max=self.clip_actions_max)

                    # compute target values
                    target_q1_values, _, _ = self.target_critic_1.act(
                        {"states": sampled_next_states, "taken_actions": next_actions}, role="target_critic_1"
                    )
                    target_q2_values, _, _ = self.target_critic_2.act(
                        {"states": sampled_next_states, "taken_actions": next_actions}, role="target_critic_2"
                    )
                    target_q_values = torch.min(target_q1_values, target_q2_values)
                    target_values = (
                        sampled_rewards
                        + self._discount_factor
                        * (sampled_terminated | sampled_truncated).logical_not()
                        * target_q_values
                    )

                # compute critic loss
                critic_1_values, _, _ = self.critic_1.act(
                    {"states": sampled_states, "taken_actions": sampled_actions}, role="critic_1"
                )
                critic_2_values, _, _ = self.critic_2.act(
                    {"states": sampled_states, "taken_actions": sampled_actions}, role="critic_2"
                )

                critic_loss = F.mse_loss(critic_1_values, target_values) + F.mse_loss(critic_2_values, target_values)

            # optimization step (critic)
            self.critic_optimizer.zero_grad()
            self.scaler.scale(critic_loss).backward()

            if config.torch.is_distributed:
                self.critic_1.reduce_parameters()
                self.critic_2.reduce_parameters()

            if self._grad_norm_clip > 0:
                self.scaler.unscale_(self.critic_optimizer)
                nn.utils.clip_grad_norm_(
                    itertools.chain(self.critic_1.parameters(), self.critic_2.parameters()), self._grad_norm_clip
                )

            self.scaler.step(self.critic_optimizer)

            # delayed update
            self._critic_update_counter += 1
            if not self._critic_update_counter % self._policy_delay:

                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                    # compute policy (actor) loss
                    actions, _, _ = self.policy.act({"states": sampled_states}, role="policy")
                    critic_values, _, _ = self.critic_1.act(
                        {"states": sampled_states, "taken_actions": actions}, role="critic_1"
                    )

                    policy_loss = -critic_values.mean()

                # optimization step (policy)
                self.policy_optimizer.zero_grad()
                self.scaler.scale(policy_loss).backward()

                if config.torch.is_distributed:
                    self.policy.reduce_parameters()

                if self._grad_norm_clip > 0:
                    self.scaler.unscale_(self.policy_optimizer)
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)

                self.scaler.step(self.policy_optimizer)

                # update target networks
                self.target_critic_1.update_parameters(self.critic_1, polyak=self._polyak)
                self.target_critic_2.update_parameters(self.critic_2, polyak=self._polyak)
                self.target_policy.update_parameters(self.policy, polyak=self._polyak)

            self.scaler.update()  # called once, after optimizers have been stepped

            # update learning rate
            if self._learning_rate_scheduler:
                self.policy_scheduler.step()
                self.critic_scheduler.step()

            # record data
            if not self._critic_update_counter % self._policy_delay:
                self.track_data("Loss / Policy loss", policy_loss.item())
            self.track_data("Loss / Critic loss", critic_loss.item())

            self.track_data("Q-network / Q1 (max)", torch.max(critic_1_values).item())
            self.track_data("Q-network / Q1 (min)", torch.min(critic_1_values).item())
            self.track_data("Q-network / Q1 (mean)", torch.mean(critic_1_values).item())

            self.track_data("Q-network / Q2 (max)", torch.max(critic_2_values).item())
            self.track_data("Q-network / Q2 (min)", torch.min(critic_2_values).item())
            self.track_data("Q-network / Q2 (mean)", torch.mean(critic_2_values).item())

            self.track_data("Target / Target (max)", torch.max(target_values).item())
            self.track_data("Target / Target (min)", torch.min(target_values).item())
            self.track_data("Target / Target (mean)", torch.mean(target_values).item())

            if self._learning_rate_scheduler:
                self.track_data("Learning / Policy learning rate", self.policy_scheduler.get_last_lr()[0])
                self.track_data("Learning / Critic learning rate", self.critic_scheduler.get_last_lr()[0])


    
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
        