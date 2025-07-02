
import copy
from typing import List, Optional, Union, Dict, Any
import tqdm
import torch
import sys

from skrl.agents.torch import Agent
from skrl.envs.wrappers.torch import Wrapper
from skrl.trainers.torch import Trainer
from isaaclab.utils.math import quat_error_magnitude

HRL_TRAINER_DEFAULT_CONFIG = {
    "timesteps": 100000,          # 총 학습 타임스텝
    "headless": False,            # 렌더링 없는 헤드리스 모드 사용 여부
    "disable_progressbar": False, # 진행률 표시바 비활성화 여부
    "close_environment_at_exit": True, # 종료 시 환경 자동 닫기 여부
    
    # HRL에 특화된 설정 추가
    "cycle_interval": 50,         # 학습을 수행할 cycle 간격         
    "epoch_interval": 1,          # 평가를 수행할 Epoch 간격
    "episode_interval": 16        # 하나의 Cycle을 구성하는 에피소드 간격
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

        # 인스턴스 변수로 파라미터 저장 (train 루프에서 쉽게 사용하기 위함)
        self.timesteps = self.cfg["timesteps"]
        self.epoch_interval = self.cfg["epoch_interval"]
        self.episode_interval = self.cfg["episode_interval"]
        self.cycle_interval = self.cfg["cycle_interval"]

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
        low_level_obs_space = self.low_level_agent.observation_space
        low_level_action_space = self.low_level_agent.action_space
        self.max_episode_buffer_length = env.max_episode_lengths

        states_buffer = {}
        next_states_buffer = {}
        for key, space in low_level_obs_space["policy"].items():
            states_buffer[key] = torch.zeros((self.max_episode_buffer_length, self.env.num_envs, space.shape[-1]),
                                             device=self.env.device, dtype=torch.float32)
            next_states_buffer[key] = torch.zeros((self.max_episode_buffer_length, self.env.num_envs, space.shape[-1]),
                                                  device=self.env.device, dtype=torch.float32)

        self.episode_buffer = {
            "states": {"policy": states_buffer},
            "actions": torch.zeros((self.max_episode_buffer_length, self.env.num_envs, low_level_action_space.shape[0]), device=self.env.device),
            "rewards": torch.zeros((self.max_episode_buffer_length, self.env.num_envs, 1), device=self.env.device),
            "next_states": {"policy": next_states_buffer},
            "terminated": torch.zeros((self.max_episode_buffer_length, self.env.num_envs, 1), dtype=torch.bool, device=self.env.device),
            "truncated": torch.zeros((self.max_episode_buffer_length, self.env.num_envs, 1), dtype=torch.bool, device=self.env.device),
        }
        self.episode_step_ptr = torch.zeros(self.env.num_envs, dtype=torch.long, device=self.env.device)

        
        self.high_level_agent.init(trainer_cfg = self.cfg)
        self.low_level_agent.init(trainer_cfg = self.cfg)
        print("[HRLTrainer] Initialize agents...")
    

    # ========== HRL Train 및 Evaluation ===========
    # Reference : 


    def train(self) -> None:
        print("[HRLTrainer] Starting HRL Training...")

        self.high_level_agent.set_running_mode("train")
        self.low_level_agent.set_running_mode("train")

        obs, info = self.env.reset()

        # High-Level Goal
        self.high_level_goal = self._sample_new_high_level_goal()
        self.env.set_high_level_goal(self.high_level_goal)

        # Initialization Count for Loop
        self.episode_count = 0
        self.cycle_count = 0

        progress_bar = tqdm.tqdm(range(self.initial_timestep, self.timesteps), 
                                 disable=self.disable_progressbar, 
                                 file=sys.stdout)
        
        # Main Learning Loop
        for timestep in progress_bar:
            self.high_level_agent.pre_interaction(timestep=timestep, timesteps=self.timesteps)
            self.low_level_agent.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            # ====== Hierarchical Control Logic =======
            if self.needs_new_high_level_action:
                current_state = obs["policy"]["observation"]

                # High-Level Agent action : Discrete Action from DIOL Agent
                high_level_action = self.high_level_agent.act(current_state, timestep=timestep, timesteps=self.timesteps)[0]

            # [Mapping Function] : From High-Level Action to Low-Level Goal State
            low_level_goal = self._map_goal(high_level_action, current_state)
            self.env.set_low_level_goals(low_level_goal)

            self.needs_new_high_level_action = False
            # Recording High-Level Action at Start Point
            self._start_high_level_transition(obs, high_level_action)


            # ====== Low-Level action with AAES Logic ======
            low_level_action = self.low_level_agent.act(obs, timestep=timestep, timesteps=self.timesteps)[0]
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
            # [Cycle 종료] 일정 수의 에피소드가 끝나면(Cycle), 두 에이전트 모두 학습
            if self.episode_counter > 0 and self.episode_counter % self.cfg["episode_interval"] == 0:
                self.cycle_counter += 1
                self.high_level_agent.train()
                self.low_level_agent.train()
                self.episode_counter = 0 # 사이클 내 에피소드 카운터 리셋
            
            # [Epoch 종료] 일정 수의 사이클이 끝나면, 성능 평가 및 AAES 업데이트
            if self.cycle_counter > 0 and self.cycle_counter % self.cfg["cycle_interval"] == 0:
                success_rate = self.eval()
                self._update_aaes_params(success_rate)
                self.cycle_counter = 0


            # ===== 주기적인 로깅 및 체크포인트 =====
            if timestep > 1 and self.high_level_agent.checkpoint_interval > 0 and not timestep % self.high_level_agent.checkpoint_interval:
                pass



            # =========== Update Observation to Next =============
            obs = next_obs_dict


    
    def eval(self):
        return super().eval()
    





    # ====================================================================== #
    # ======================= Auxillary Function =========================== #
    # ====================================================================== #
    def _sample_new_high_level_goal(self) -> torch.Tensor:
        """
            사전에 정의된 최종 목표 목록에서 새로운 목표를 샘플링합니다.
        """
        ####### 실제로는 이 Pre-defined된 목표값들을 cfg 등에서 불러와야 합니다. #####
        possible_goals = [
            torch.tensor([1, 1, 0, 0, ...], device=self.env.device),
            torch.tensor([0, 0, 1, 1, ...], device=self.env.device) 
        ]
        
        # 모든 환경에 동일한 목표를 무작위로 샘플링하여 할당합니다.
        goal_index = torch.randint(0, len(possible_goals), (1,)).item()
        selected_goal = possible_goals[goal_index]
        
        # num_envs 차원으로 확장하여 반환
        return selected_goal.repeat(self.env.num_envs, 1)


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
            "desired_goal": self.high_level_goal
        }
        next_states_to_store = {
            "observation": next_obs["policy"]["observation"],
            "desired_goal": self.high_level_goal
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
                                                 terminated=terminated,
                                                 truncated=truncated)

    
    def _store_to_episode_buffer(self, 
                                 obs: Dict,
                                 action: torch.Tensor,
                                 reward: torch.Tensor,
                                 next_obs: Dict,
                                 terminated: torch.Tensor,
                                 truncated: torch.Tensor) -> None:
    
        """
            한 타임스텝의 저수준 경험을 에피소드 임시 버퍼에 저장합니다.
            HER을 적용하기 전에 한 에피소드의 모든 데이터를 수집하는 역할을 합니다.
        """
        step_idx = self.episode_step_ptr[0]

        self.episode_buffer["states"]["policy"]["observation"][step_idx] = obs["policy"]["observation"]
        self.episode_buffer["states"]["policy"]["achieved_goal"][step_idx] = obs["policy"]["achieved_goal"]
        self.episode_buffer["states"]["policy"]["desired_goal"][step_idx] = obs["policy"]["desired_goal"]
        self.episode_buffer["actions"][step_idx] = action
        self.episode_buffer["rewards"][step_idx] = reward.unsqueeze(-1)
        self.episode_buffer["next_states"]["policy"]["observation"][step_idx] = next_obs["policy"]["observation"]
        self.episode_buffer["next_states"]["policy"]["achieved_goal"][step_idx] = next_obs["policy"]["achieved_goal"]
        self.episode_buffer["next_states"]["policy"]["desired_goal"][step_idx] = next_obs["policy"]["desired_goal"]
        self.episode_buffer["terminated"][step_idx] = terminated
        self.episode_buffer["truncated"][step_idx] = truncated

        self.episode_step_ptr += 1

    
    def _apply_her_and_record_transition(self, env_ids: torch.Tensor) -> None:
        
        """
            에피소드가 종료된 환경들에 대해 HER을 적용하고, 
            원본 및 증강된 경험을 저수준 리플레이 버퍼에 저장합니다.
            여기서, 종료된 환경들에 대한 개별 처리를 통해 각자의 길이를 고려한 HER을 적용합니다.
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


    
    def _recompute_reward(self, achieved_goal: torch.Tensor, desired_goal: torch.Tensor) -> torch.Tensor:
        loc_dist = torch.norm(achieved_goal[:, :3] - desired_goal[:, :3], dim=-1)
        rot_dist = quat_error_magnitude(achieved_goal[:, 3:7], desired_goal[:, 3:7])

        reward = torch.where(torch.logical_and(
            loc_dist < self.env.cfg.low_level_loc_threshold,
            rot_dist < self.env.cfg.low_level_rot_threshold,
        ), 0.0, -1.0)

        return reward
    

    def _apply_aaes(self):
        pass

    def _map_goal(self):
        pass

    def _update_aaes_params(self):
        pass