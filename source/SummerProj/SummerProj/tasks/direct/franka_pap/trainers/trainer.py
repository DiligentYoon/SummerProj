
import numpy as np
import copy
from typing import List, Optional, Union, Dict, Any
import tqdm
import torch
import sys

from skrl.agents.torch import Agent
from skrl.envs.wrappers.torch import Wrapper
from skrl.trainers.torch import Trainer
from isaaclab.utils.math import quat_error_magnitude

from ..agents.replay_buffer import EpisodeWiseReplayBuffer

HRL_TRAINER_DEFAULT_CONFIG = {
    "timesteps": 100000,          # 총 학습 타임스텝
    "headless": False,            # 렌더링 없는 헤드리스 모드 사용 여부
    "disable_progressbar": False, # 진행률 표시바 비활성화 여부
    "close_environment_at_exit": True, # 종료 시 환경 자동 닫기 여부
    
    # HRL에 특화된 설정 추가
    "cycle_interval": 50,         # 학습을 수행할 cycle 간격         
    "epoch_interval": 1,          # 평가를 수행할 Epoch 간격
    "episode_interval": 16,        # 하나의 Cycle을 구성하는 에피소드 간격

    # AAES (Auto-Adjusting Exploration Strategy) 설정
    "aaes_kwargs": {
        "min_std": 0.05,
        "max_std": 1.0,
        "min_uniform": 0.05,
        "max_uniform": 0.95,
        "reduction_factor_noise": 0.9,
        "reduction_factor_uniform": 0.6,
    }
}

class HRLTrainer(Trainer):
    def __init__(self, 
                 env: Wrapper, 
                 agents: Dict[str, Agent],
                 **kwargs) -> None:
        
        # kwargs를 통해 따로 정의하지 않은 모든 config 관련 설정값들을 하나의 딕셔너리로 묶어 처리
        _cfg = copy.deepcopy(HRL_TRAINER_DEFAULT_CONFIG)
        _cfg.update(kwargs if kwargs is not None else {})
        if _cfg["timesteps"] is None:
            _cfg["timesteps"] = (env._unwrapped.max_episode_length * _cfg["epoch_interval"]   * \
                                                                     _cfg["episode_interval"] * \
                                                                     _cfg["cycle_interval"])

        # skrl의 Trainer를 상속받아 초기화 : Low Level Agent만 전달.
        # Superclass의 유효성 검사 통과하여 기본 기능 상속받기 위함.
        super().__init__(env=env, agents=agents["low_level"], cfg=_cfg)

        # 에이전트 할당
        self.high_level_agent = agents["high_level"]
        self.low_level_agent = agents["low_level"]

        if self.high_level_agent is None or self.low_level_agent is None:
            raise ValueError("The 'agents' dictionary must contain 'high_level' and 'low_level' keys.")

        # AAES 파라미터
        self.use_pre_trained_low = self.low_level_agent.cfg["use_pre_trained"]
        if self.use_pre_trained_low:
            self.checkpoint_path_low = self.low_level_agent.cfg["checkpoint_path"]
        else:
            self.checkpoint_path_low = None
        self.aaes_params = self.cfg.get("aaes_kwargs", {})
        self.current_noise_std = self.aaes_params.get("max_std")
        self.current_random_action_ratio = self.aaes_params.get("max_uniform")
        self.action_space = self.low_level_agent.action_space

        # Training 파라미터 (train 루프에서 쉽게 사용하기 위함)
        self.epoch_interval = _cfg["epoch_interval"]
        self.episode_interval = _cfg["episode_interval"]
        self.cycle_interval = _cfg["cycle_interval"]
        self.timesteps = _cfg["timesteps"]

        # 학습 로직 관련 변수
        self.current_high_level_action = {}
        for name, val in self.env.cfg.high_level_action_space.items():
            if type(val) == set:
                val = list(val)[0]
            self.current_high_level_action[name] = torch.zeros((self.env.num_envs, val), device=self.env.device)

        self.h_start_states = {
            "observation": torch.zeros((self.env.num_envs, self.high_level_agent.observation_space.shape[-1]), 
                                       device=self.env.device, dtype=torch.float32),
            "desired_goal": torch.zeros((self.env.num_envs, self.high_level_agent.observation_space.shape[-1]), 
                                        device=self.env.device, dtype=torch.float32)
        }
        self.h_actions = torch.zeros((self.env.num_envs, 1), device=self.env.device)

        # HACMAN 전용 HER을 구현하기 위한 에피소드 버퍼 초기화
        self.max_episode_buffer_length = env._unwrapped.max_episode_length
        
        self.episode_buffer = EpisodeWiseReplayBuffer(self.max_episode_buffer_length, self.env.num_envs, self.env.device)
        # self.episode_buffer.create_tensor("states", size=1, dtype=torch.float32)
        # self.episode_buffer.create_tensor("action", size=1, dtype=torch.float32)
        # self.episode_buffer.create_tensor("reward", size=1, dtype=torch.float32)
        # self.episode_buffer.create_tensor("next_states", size=1, dtype=torch.float32)
        # self.episode_buffer.create_tensor("terminated", size=1, dtype=torch.bool)
        # self.episode_buffer.create_tensor("truncated", size=1, dtype=torch.bool)
        # self.episode_buffer.create_tensor("achieved_goal", size=1, dtype=torch.float32)
        # self.episode_buffer.create_tensor("desired_goal", size=1, dtype=torch.float32)

        # 에이전트 초기화 & 리플레이 버퍼에 HRL 전용 Term 추가
        self.high_level_agent.init(trainer_cfg = self.cfg)
        self.low_level_agent.init(trainer_cfg = self.cfg)
        
        for k_high, k_low in zip(self.high_level_agent.memory.tensors.keys(), self.low_level_agent.memory.tensors.keys()):
            self.high_level_agent._tensors_names.append(k_high)
            self.low_level_agent._tensors_names.append(k_low)
        print("[HRLTrainer] Initialize agents...")
    

    # ========== HRL Train 및 Evaluation ===========
    def train(self) -> None:
        print("[HRLTrainer] Starting HRL Training...")
        self.high_level_agent.set_running_mode("train")
        self.low_level_agent.set_running_mode("eval" if self.use_pre_trained_low else "train")

        obs, info = self.env.reset()

        # High-Level Logic
        self.high_level_obs_start = info["high_level_obs"]
        self.needs_new_high_level_action = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)

        progress_bar = tqdm.tqdm(range(self.initial_timestep, self.timesteps), 
                                 disable=self.disable_progressbar, 
                                 file=sys.stdout)
        
        # Main Learning Loop
        for timestep in progress_bar:
            # self.high_level_agent.pre_interaction(timestep=timestep, timesteps=self.timesteps)
            # self.low_level_agent.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            goal_request_indices = torch.where(self.needs_new_high_level_action)[0]
            # ====== Hierarchical Control Logic =======
            if len(goal_request_indices) > 0:
                with torch.no_grad():
                    self.high_level_obs_start[goal_request_indices] = info["high_level_obs"][goal_request_indices]
                    high_level_action = self._sample_new_high_level_goal(self.high_level_obs_start, timestep)
                    # Mapping from high level abstraction to low level goal
                    self.needs_new_high_level_action[goal_request_indices] = False
                    for name, tensor in high_level_action.items():
                        self.current_high_level_action[name][goal_request_indices] = tensor
                    self.env._unwrapped._apply_high_action(self.current_high_level_action)

            # ====== Low-Level action with AAES Logic ======
            with torch.no_grad():
                low_level_obs = obs
                low_level_action = self.low_level_agent.act(low_level_obs, timestep=timestep, timesteps=self.timesteps)[0]
                # AAES 우선 보류
                # action_to_env = self._apply_aaes(low_level_action)
                # Environment step (E, k)차원의 데이터
                next_obs, reward, terminated, truncated, info = self.env.step(low_level_action)

            # ====== Recording Experience in Episode Buffer for HER ======
            # self._store_to_episode_buffer(low_level_obs, 
            #                               low_level_action, 
            #                               reward, 
            #                               next_obs, 
            #                               terminated, 
            #                               truncated,
            #                               info)

            # Low-Level Agent Replay Buffer
            if not self.use_pre_trained_low:
                self.low_level_agent.record_transition(
                    states=low_level_obs,
                    actions=low_level_action,
                    rewards=reward,
                    next_states=next_obs,
                    terminated=terminated,
                    truncated=truncated,
                    infos=info,
                    timestep=timestep,
                    timesteps=self.timesteps
                )

            # ===== Parameter Update & Data Logging =====
            self.high_level_agent.post_interaction(timestep, self.timesteps)
            if not self.use_pre_trained_low:
                self.low_level_agent.post_interaction(timestep, self.timesteps)


            # =========== Update Observation to Next =============
            is_episode_done = terminated | truncated
            if is_episode_done.any():
                done_indices = torch.where(is_episode_done)[0]
                # Data 저장 in High-Level Replay Buffer
                # (E, k) Dimension
                # self._apply_her_and_record_transition()
                actions = {}
                for name, tensor in self.current_high_level_action:
                    actions[name] = tensor[done_indices]
                self.high_level_agent.record_transition(
                    states=self.high_level_obs_start[done_indices],
                    actions=actions,
                    rewards=info["high_level_reward"][done_indices],
                    next_states=info["high_level_obs"][done_indices],
                    truncated=truncated[done_indices],
                    terminated=terminated[done_indices],
                    infos=info,
                    timestep=timestep,
                    timesteps=self.timesteps
                )
                self.needs_new_high_level_action[done_indices] = True
                self.episode_counter += 1
                obs, info = self.env.reset()
            else:
                obs = next_obs

    
    # def eval(self, eval_episodes: int = 100):
    #     print(f"[HRLTrainer] Starting HRL Evaluation at [Epoch {self.epoch_count}]...")
    #     self.high_level_agent.set_running_mode("eval")
    #     self.low_level_agent.set_running_mode("eval")

    #     successes = []
    #     for _ in range(eval_episodes):
    #         current_demo_step = 0
    #         needs_new_high_level_action = True
    #         obs, info = self.env.reset()
    #         high_level_goal = self._sample_new_high_level_goal()
    #         self.env._unwrapped.set_high_level_goal(high_level_goal)
            
    #         # High-Level (=Episode) Loop
    #         for _ in range(self.env._unwrapped.max_episode_length):
    #             # 새로운 고수준 행동이 필요할 때만 결정
    #             if needs_new_high_level_action:
    #                 high_level_action = self._get_demonstration_action(current_demo_step, high_level_demo)
    #                 current_demo_step += 1
                    
    #                 # 매핑 및 저수준 목표 설정
    #                 low_level_goal = self._map_goal(high_level_action, obs)
    #                 self.env._unwrapped.set_low_level_goals(low_level_goal)
                    
    #                 needs_new_high_level_action = False

    #             # 저수준 행동 결정
    #             low_level_state = torch.cat((obs["policy"]["observation"], obs["policy"]["sub_goal"]), dim=1)
    #             low_level_action, _, _ = self.low_level_agent.act(low_level_state, timestep=0, timesteps=0)
                
    #             # 환경 스텝
    #             obs, _, terminated, truncated, info = self.env.step(low_level_action)
                
    #             # 단일 환경을 기준으로 종료 여부 판단
    #             if terminated[0].item() or truncated[0].item():
    #                 break

    #             # 단일 환경이 옵션을 완료했다면, 다음 스텝에 새로운 고수준 행동이 필요함
    #             if info.get("option_terminated")[0].item():
    #                 needs_new_high_level_action = True

    #         # 최종적으로 0번 환경이 성공적으로 끝났는지(terminated)로 성공 여부 판단
    #         successes.append(1.0 if terminated[0].item() else 0.0)

    #     self.high_level_agent.set_running_mode("train")
    #     self.low_level_agent.set_running_mode("train")

    #     avg_success_rate = np.mean(successes)
    #     print(f"[HRLTrainer] Evaluation finished at [Epoch {self.epoch_count}]. Average Success Rate: {avg_success_rate:.2f}")
    
    #     return avg_success_rate

    # ====================================================================== #
    # ======================= Auxillary Function =========================== #
    # ====================================================================== #
    def _sample_new_high_level_goal(self, obs: torch.Tensor, timestep: int) -> torch.Tensor:
        action, _ = self.high_level_agent.act(obs, timestep=timestep, timesteps=self.timesteps)
        return action
    
    def _store_to_episode_buffer(self, 
                                 obs: Dict,
                                 action: torch.Tensor,
                                 reward: torch.Tensor,
                                 next_obs: Dict,
                                 terminated: torch.Tensor,
                                 truncated: torch.Tensor) -> None:
    
        """
            한 타임스텝의 저수준 경험을 에피소드 임시 버퍼에 저장.
            HER을 적용하기 전에 한 에피소드의 모든 데이터를 수집하는 역할.
        """
        self.episode_buffer.add_samples(states=obs["policy"]["observation"],
                                        sub_goal=obs["policy"]["sub_goal"],
                                        action=action,
                                        reward=reward,
                                        next_states=next_obs["policy"]["observation"],
                                        terminated=terminated,
                                        truncated=truncated)


    def _recompute_reward(self, achieved_goal: torch.Tensor, desired_goal: torch.Tensor) -> torch.Tensor:
        loc_dist = torch.norm(achieved_goal[:, :3] - desired_goal[:, :3], dim=-1)
        rot_dist = quat_error_magnitude(achieved_goal[:, 3:7], desired_goal[:, 3:7])

        reward = torch.where(torch.logical_and(
            loc_dist < self.env._unwrapped.cfg.low_level_loc_threshold,
            rot_dist < self.env._unwrapped.cfg.low_level_rot_threshold,
        ), 0.0, -1.0)

        return reward


    def _map_goal(self, high_level_action: torch.Tensor, current_obs: dict) -> torch.Tensor:
        """
            실제 매핑 로직을 환경 객체에 위임하고, 자신은 호출만 수행.
            ._unwrapped를 사용하여 래퍼를 우회하고 환경의 원본 메소드에 접근.
        """
        return self.env._unwrapped._map_high_level_action_to_low_level_goal(high_level_action, current_obs)


    def _apply_aaes(self, action: torch.Tensor) -> torch.Tensor:
        """
            결정론적 액션에 AAES 노이즈를 적용.
        """
        # Action From policy vs Action From Uniform Disctribution 결정
        # We use tanh activation in policy network. So, action range is [-1, 1]
        if torch.rand(1).item() < self.current_random_action_ratio:
            low = torch.tensor(-1, device=self.env.device)
            high = torch.tensor(1, device=self.env.device)
            action = torch.rand_like(action) * (high - low) + low

        # 가우시안 노이즈 생성
        noise = torch.normal(mean=0, std=self.current_noise_std, size=action.shape, device=self.env.device)
        
        # 액션에 노이즈 추가
        noisy_action = action + noise
        
        return noisy_action


    def _update_aaes_params(self, success_rate: float) -> None:
        """
            태스크 성공률에 기반하여 AAES 파라미터를 업데이트.
        """
        self.current_random_action_ratio = self.aaes_params.get("reduction_factor_uniform") * (1-success_rate)
        self.current_noise_std = self.aaes_params.get("reduction_factor_noise") * (1-success_rate)

        # 정의된 범위를 벗어나지 않도록 각 파라미터를 클리핑
        self.current_random_action_ratio = max(self.aaes_params.get("min_uniform"),
                                               min(self.current_random_action_ratio, self.aaes_params.get("max_uniform")))
        self.current_noise_std = max(self.aaes_params.get("min_std"), 
                                   min(self.current_noise_std, self.aaes_params.get("max_std")))