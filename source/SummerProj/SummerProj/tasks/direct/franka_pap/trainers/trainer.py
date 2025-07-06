
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

        # skrl의 Trainer를 상속받아 초기화 : Low Level Agent만 전달.
        # Superclass의 유효성 검사 통과하여 기본 기능 상속받기 위함.
        super().__init__(env=env, agents=agents["low_level"], cfg=_cfg)

        # 에이전트 할당
        self.high_level_agent = agents["high_level"]
        self.low_level_agent = agents["low_level"]

        if self.high_level_agent is None or self.low_level_agent is None:
            raise ValueError("The 'agents' dictionary must contain 'high_level' and 'low_level' keys.")

        # Training 파라미터 (train 루프에서 쉽게 사용하기 위함)
        self.timesteps = self.cfg["timesteps"]
        self.epoch_interval = self.cfg["epoch_interval"]
        self.episode_interval = self.cfg["episode_interval"]
        self.cycle_interval = self.cfg["cycle_interval"]

        # AAES 파라미터
        self.aaes_params = self.cfg.get("aaes_kwargs", {})
        self.current_noise_std = self.aaes_params.get("max_std")
        self.current_random_action_ratio = self.aaes_params.get("max_uniform")
        self.action_space = self.low_level_agent.action_space

        # 학습 로직 관련 변수
        self.current_high_level_action = torch.zeros((self.env.num_envs, 1), dtype=torch.long, device=self.env.device)
        self.needs_new_high_level_action = True
        self.h_start_states = {
            "observation": torch.zeros((self.env.num_envs, self.high_level_agent.observation_space["observation"].shape[-1]), 
                                       device=self.env.device, dtype=torch.float32),
            "desired_goal": torch.zeros((self.env.num_envs, self.high_level_agent.observation_space["desired_goal"].shape[-1]), 
                                        device=self.env.device, dtype=torch.float32)
        }
        self.h_actions = torch.zeros((self.env.num_envs, 1), device=self.env.device)

        # HER을 구현하기 위한 에피소드 버퍼
        high_level_obs_space = self.high_level_agent.observation_space
        low_level_obs_space = self.low_level_agent.observation_space
        low_level_action_space = self.low_level_agent.action_space
        self.max_episode_buffer_length = env._unwrapped.max_episode_length

        states_buffer = {}
        next_states_buffer = {}
        for key, space in low_level_obs_space["policy"].items():
            states_buffer[key] = torch.zeros((self.max_episode_buffer_length, self.env.num_envs, space.shape[-1]),
                                             device=self.env.device, dtype=torch.float32)
            next_states_buffer[key] = torch.zeros((self.max_episode_buffer_length, self.env.num_envs, space.shape[-1]),
                                                  device=self.env.device, dtype=torch.float32)
        
        self.episode_buffer = EpisodeWiseReplayBuffer(self.max_episode_buffer_length, self.env.num_envs, self.env.device)
        self.episode_buffer.create_tensor("states", size=low_level_obs_space["policy"]["observation"], dtype=torch.float32)
        self.episode_buffer.create_tensor("sub_goal", size=self.env._unwrapped.cfg.low_level_goal_dim, dtype=torch.float32)
        self.episode_buffer.create_tensor("final_goal", size=self.env._unwrapped.cfg.high_level_goal_dim,dtype=torch.float32)
        self.episode_buffer.create_tensor("action", size=low_level_action_space["policy"], dtype=torch.float32)
        self.episode_buffer.create_tensor("reward", size=1, dtype=torch.float32)
        self.episode_buffer.create_tensor("achieved_goal", size=self.env._unwrapped.cfg.achieved_goal_dim, dtype=torch.float32)
        self.episode_buffer.create_tensor("next_states", size=low_level_obs_space["policy"]["observation"], dtype=torch.float32)
        self.episode_buffer.create_tensor("terminated", size=1, dtype=torch.bool)
        self.episode_buffer.create_tensor("truncated", size=1, dtype=torch.bool)

        # 에이전트 초기화 & 리플레이 버퍼에 HRL 전용 Term 추가
        self.high_level_agent.init(trainer_cfg = self.cfg)
        self.low_level_agent.init(trainer_cfg = self.cfg)

        self.high_level_agent.memory.create_tensor("desired_goal", size=self.env._unwrapped.cfg.high_level_goal_dim, dtype=torch.float32)
        self.high_level_agent.memory.create_tensor("achieved_goal", size=self.env._unwrapped.cfg.high_level_goal_dim, dtype=torch.float32)
        self.low_level_agent.memory.create_tensor("desired_goal", size=self.env._unwrapped.cfg.low_level_goal_dim, dtype=torch.float32)
        self.low_level_agent.memory.create_tensor("achieved_goal", size=self.env._unwrapped.cfg.achieved_goal_dim, dtype=torch.float32)
        
        for k_high, k_low in zip(self.high_level_agent.memory.tensors.keys(), self.low_level_agent.memory.tensors.keys()):
            self.high_level_agent._tensors_names.append(k_high)
            self.low_level_agent._tensors_names.append(k_low)
        
        print("[HRLTrainer] Initialize agents...")
    

    # ========== HRL Train 및 Evaluation ===========
    # Reference : 


    def train(self) -> None:
        print("[HRLTrainer] Starting HRL Training...")
        self.high_level_agent.set_running_mode("train")
        self.low_level_agent.set_running_mode("train")

        obs, info = self.env.reset()

        # High-Level Goal
        self.high_level_demo = self._sample_new_high_level_demo()
        self.high_level_goal = self._sample_new_high_level_goal()
        self.env._unwrapped.set_high_level_goal(self.high_level_goal)

        # Initialization Count for Loop
        self.episode_count = 0
        self.cycle_count = 0
        self.epoch_count = 0

        progress_bar = tqdm.tqdm(range(self.initial_timestep, self.timesteps), 
                                 disable=self.disable_progressbar, 
                                 file=sys.stdout)
        
        # Main Learning Loop
        for timestep in progress_bar:
            self.high_level_agent.pre_interaction(timestep=timestep, timesteps=self.timesteps)
            self.low_level_agent.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            # ====== Hierarchical Control Logic =======
            if self.needs_new_high_level_action:
                # High-Level state : (s, g^H)
                high_level_state = torch.cat((obs["policy"]["observation"], obs["policy"]["final_goal"]), dim=1)
                # High-Level Agent action : Discrete Action from DIOL Agent
                high_level_action, _, _ = self.high_level_agent.act(high_level_state, timestep=timestep, timesteps=self.timesteps)

            # [Mapping Function] : From High-Level Action to Low-Level Goal State
            low_level_goal = self._map_goal(high_level_action, obs)
            self.env._unwrapped.set_low_level_goals(low_level_goal)

            self.needs_new_high_level_action = False
            # Recording High-Level Action at Start Point
            self._start_high_level_transition(obs, high_level_action)


            # ====== Low-Level action with AAES Logic ======
            low_level_state = torch.cat((obs["policy"]["observation"], obs["policy"]["sub_goal"]), dim=1)
            low_level_action = self.low_level_agent.act(low_level_state, timestep=timestep, timesteps=self.timesteps)[0]
            action_to_env = self._apply_aaes(low_level_action)

            # Environment step
            next_obs, reward, terminated, truncated, info = self.env.step(action_to_env)
            next_obs_dict = info["obs"]

            # ====== Recording Experience in Episode Buffer ======
            self._store_to_episode_buffer(obs, 
                                          action_to_env, 
                                          reward, 
                                          next_obs_dict, 
                                          terminated, 
                                          truncated)
            
            self._record_high_level_transition(obs, 
                                               high_level_action, 
                                               info.get("high_level_reward"), 
                                               next_obs_dict,
                                               terminated,
                                               truncated,
                                               info,
                                               timestep,
                                               self.timesteps)

            is_option_done = info.get("option_terminated", torch.zeros_like(terminated)).any()
            is_episode_done = terminated.any() or truncated.any()
            if is_option_done or is_episode_done:               
                self.needs_new_high_level_action = True
            
            if is_episode_done:
                # HER apply for Low-Level Agent : Recording in Low-Level Agent's Replay Buffer
                self._apply_her_and_record_transition()
                self.episode_counter += 1
                obs, info = self.env.reset()
                self.needs_new_high_level_action = True

            # ===== 주기적인 학습 및 평가 =====
            # [Per Cycle] 두 에이전트 모두 학습
            if self.episode_count > 0 and self.episode_count % self.cfg["episode_interval"] == 0:
                self.cycle_count += 1
                self.high_level_agent._update()
                self.low_level_agent._update()
                self.episode_count = 0 # 사이클 내 에피소드 카운터 리셋
            
            # [Per Epoch] 성능 평가 및 AAES 업데이트
            if self.cycle_count > 0 and self.cycle_count % self.cfg["cycle_interval"] == 0:
                self.epoch_count += 1
                success_rate = self.eval()
                self._update_aaes_params(success_rate)
                self.cycle_count = 0


            # ===== Checkpoint Generation & Logging : Post Interaction =====
            self.high_level_agent.post_interaction(timestep, self.timesteps)
            self.low_level_agent.post_interaction(timestep, self.timesteps)


            # =========== Update Observation to Next =============
            obs = next_obs_dict

    
    def eval(self, eval_episodes: int = 100):
        print(f"[HRLTrainer] Starting HRL Evaluation at [Epoch {self.epoch_count}]...")
        self.high_level_agent.set_running_mode("eval")
        self.low_level_agent.set_running_mode("eval")

        successes = []
        for _ in range(eval_episodes):
            current_demo_step = 0
            needs_new_high_level_action = True
            obs, info = self.env.reset()
            high_level_demo = self._sample_new_high_level_demo()
            high_level_goal = self._sample_new_high_level_goal()
            self.env._unwrapped.set_high_level_goal(high_level_goal)
            
            # High-Level (=Episode) Loop
            for _ in range(self.env._unwrapped.max_episode_length):
                # 새로운 고수준 행동이 필요할 때만 결정
                if needs_new_high_level_action:
                    high_level_action = self._get_demonstration_action(current_demo_step, high_level_demo)
                    current_demo_step += 1
                    
                    # 매핑 및 저수준 목표 설정
                    low_level_goal = self._map_goal(high_level_action, obs)
                    self.env._unwrapped.set_low_level_goals(low_level_goal)
                    
                    needs_new_high_level_action = False

                # 저수준 행동 결정
                low_level_state = torch.cat((obs["policy"]["observation"], obs["policy"]["sub_goal"]), dim=1)
                low_level_action, _, _ = self.low_level_agent.act(low_level_state, timestep=0, timesteps=0)
                
                # 환경 스텝
                obs, _, terminated, truncated, info = self.env.step(low_level_action)
                
                # 단일 환경을 기준으로 종료 여부 판단
                if terminated[0].item() or truncated[0].item():
                    break

                # 단일 환경이 옵션을 완료했다면, 다음 스텝에 새로운 고수준 행동이 필요함
                if info.get("option_terminated")[0].item():
                    needs_new_high_level_action = True

            # 최종적으로 0번 환경이 성공적으로 끝났는지(terminated)로 성공 여부 판단
            successes.append(1.0 if terminated[0].item() else 0.0)

        self.high_level_agent.set_running_mode("train")
        self.low_level_agent.set_running_mode("train")

        avg_success_rate = np.mean(successes)
        print(f"[HRLTrainer] Evaluation finished at [Epoch {self.epoch_count}]. Average Success Rate: {avg_success_rate:.2f}")
    
        return avg_success_rate


    # ====================================================================== #
    # ======================= Auxillary Function =========================== #
    # ====================================================================== #
    def _sample_new_high_level_demo(self) -> torch.Tensor:
        """
            사전에 정의된 최종 목표 목록에서 새로운 목표를 샘플링.
        """
        # Pre-defined된 Abstract Demonstration을 Task-Specific Configuration에서 호출
        # 규칙 : 0부터 수행해야 하는 Ground Truth Sequence에 따라 +1 간격으로 정의되어 있음.
        _demo = torch.linspace(0, self.env._unwrapped.cfg.high_level_goal_dim, 1)
        
        # num_envs 차원으로 확장하여 반환
        return _demo.repeat(self.env.num_envs, 1)
    

    def _sample_new_high_level_goal(self) -> torch.Tensor:

        return self.env._unwrapped.set_high_level_goal()


    def _start_high_level_transition(self, obs: Dict, high_level_action: torch.Tensor) -> None:
        """
            High-Level Policy가 action을 추출한 시점을 기록
            추후, Replay Buffer 구성을 위함.
        """
        self.h_start_states.copy_(obs["policy"]["observation"])
        self.h_actions.copy_(high_level_action)


    def _record_high_level_transition(self,
                                      obs: Dict,
                                      actions: torch.Tensor,
                                      rewards: torch.Tensor,
                                      next_obs: Dict,
                                      terminated: torch.Tensor,
                                      truncated: torch.Tensor,
                                      infos: Any,
                                      **kwargs) -> None:
        """
            DIOL 학습을 위해 매 타임스텝의 고수준 경험을 리플레이 버퍼에 기록
        """
        states_to_store = {
            "observation": obs["policy"]["observation"],
            "desired_goal": obs["policy"]["final_goal"]
        }
        next_states_to_store = {
            "observation": next_obs["policy"]["observation"],
            "desired_goal": obs["policy"]["final_goal"]
        }

        self.high_level_agent.record_transition(states=obs["policy"]["observation"],
                                                actions=actions,
                                                rewards=rewards,
                                                next_states=next_obs["policy"]["observation"],
                                                terminated=terminated,
                                                truncated=truncated,
                                                infos=infos,
                                                **kwargs)

        self.high_level_agent.memory.add_samples(states=states_to_store,
                                                 actions=actions,
                                                 rewards=rewards,
                                                 next_states=next_states_to_store,
                                                 terminated=infos.get("option_terminated"),
                                                 truncated=torch.zeros_like(infos.get("option_terminated")))

    
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

    
    def _apply_her_and_record_transition(self, env_ids: torch.Tensor) -> None:
        
        """
            에피소드가 종료된 환경들에 대해 HER을 적용하고, 
            원본 및 증강된 경험을 저수준 리플레이 버퍼에 저장.
            여기서, 종료된 환경들에 대한 개별 처리를 통해 각자의 길이를 고려한 HER을 적용.
        """
        # HER 관련 파라미터 가져오기
        her_cfg = self.cfg.get("her_kwargs", {})
        k_ratio = her_cfg.get("k_ratio")
        strategy = her_cfg.get("strategy")
        batch_to_store = {
            "states": [], "actions": [], "rewards": [], "next_states": [],
            "terminated": [], "truncated": []
        }

        # ======== 종료된 환경들의 Per Episode Trajectory 추출 =========
        episode_lengths = self.episode_step_ptr[env_ids]
        for i, env_id in enumerate(env_ids):
            real_length = episode_lengths[i].item()
            if real_length == 0:
                continue
            trajectory = {k: v[:real_length, env_id] for k, v in self.episode_buffer.items()}
            
            batch_to_store["states"].append(trajectory["states"])
            batch_to_store["actions"].append(trajectory["actions"])
            batch_to_store["rewards"].append(trajectory["rewards"])
            batch_to_store["next_states"].append(trajectory["next_states"])
            batch_to_store["terminated"].append(trajectory["terminated"])
            batch_to_store["truncated"].append(trajectory["truncated"])
            
            # ====== Augmented Data 저장 by HER ======
            for t in range(real_length):
                # t 시점의 데이터 세팅
                obs = trajectory["states"][t]
                action = trajectory["actions"][t]
                reward = trajectory["rewards"][t]
                next_obs = trajectory["next_states"][t]
                terminated = trajectory["terminated"][t]
                truncated = trajectory["truncated"][t]

                for _ in range(k_ratio):
                    if strategy == "future":
                        future_step_idx = torch.randint(t, real_length)
                    elif strategy == "episode":
                        future_step_idx = torch.randint(0, real_length)
                    else:
                        ValueError("Not Supported HER Strategy.")
                    
                    # Original achieved goal : t시점의 achieved goal
                    original_achieved_goal = trajectory["next_states"]["policy"]["achieved_goal"][t]
                    # New desired goal : future_index 시점의 achieved goal
                    new_desired_goal = trajectory["next_states"]["policy"]["achieved_goal"][future_step_idx]
                    # Data Augmentation
                    new_reward = self._recompute_reward(original_achieved_goal, new_desired_goal)
                    synthetic_obs = copy.deepcopy(obs)
                    synthetic_next_obs = copy.deepcopy(next_obs)
                    synthetic_obs["policy"]["desired_goal"] = new_desired_goal
                    synthetic_next_obs["policy"]["desired_goal"] = new_desired_goal
                    
                    batch_to_store["states"].append(synthetic_obs.unsqueeze(0))
                    batch_to_store["actions"].append(action.unsqueeze(0))
                    batch_to_store["rewards"].append(new_reward.unsqueeze(0))
                    batch_to_store["next_states"].append(synthetic_next_obs.unsqueeze(0))
                    batch_to_store["terminated"].append(terminated.unsqueeze(0))
                    batch_to_store["truncated"].append(truncated.unsqueeze(0))
            
            if not batch_to_store["states"]:
                return
            
            batch = {
                "states": {k: torch.cat([s[k] for s in batch_to_store["states"]], dim=0) for k in batch_to_store["states"][0].keys()},
                "actions": torch.cat(batch_to_store["actions"], dim=0),
                "rewards": torch.cat(batch_to_store["rewards"], dim=0),
                "next_states": {k: torch.cat([s[k] for s in batch_to_store["next_states"]], dim=0) for k in batch_to_store["next_states"][0].keys()},
                "terminated": torch.cat(batch_to_store["terminated"], dim=0),
                "truncated": torch.cat(batch_to_store["truncated"], dim=0),
            }

            # ======= 데이터 증강으로 인해, num_envs보다 차원이 커지는 경우, 배치 슬라이스 =======
            total_samples_to_add = batch["actions"].shape[0]
            slice_size = self.env.num_envs
            if total_samples_to_add > 0:
                for i in range(0, total_samples_to_add, slice_size):
                    end_index = min(i + slice_size, total_samples_to_add)

                    slice_batch = {}
                    for name, value in batch.items():
                        if isinstance(value, dict):
                            slice_batch[name] = {k: v[i:end_index] for k, v in value.items()}
                        else:
                            slice_batch[name] = value[i:end_index]

                    self.low_level_agent.memory.add_samples(**slice_batch)
            
            # Episode Buffer Clear
            self.episode_buffer.clear()


    
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