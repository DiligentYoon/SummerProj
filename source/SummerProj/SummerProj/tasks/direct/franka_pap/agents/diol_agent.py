# File: source/SummerProj/SummerProj/tasks/direct/franka_pap/agents/diol_agent.py
from typing import Union, Dict, Any, Mapping

import copy
import torch
import torch.nn as nn
import gymnasium as gym
from torch.optim import Adam

from skrl.agents.torch import Agent
from skrl.memories.torch import Memory
from skrl.models.torch import Model

# DIOLAgent를 위한 기본 설정값
DIOL_DEFAULT_CONFIG = {
    "discount_factor": 0.99,        # 감가율 (gamma)
    "learning_rate": 1e-4,          # 학습률
    "batch_size": 64,               # 미니배치 크기
    "polyak": 0.005,                # 타겟 네트워크 소프트 업데이트 계수 (tau)
    "learning_starts": 1000,        # 학습 시작 전 최소 경험 수
    "update_interval": 1,           # 몇 스텝마다 학습할지
    "exploration": {
        "initial_epsilon": 1.0,     # 초기 탐험율 (epsilon-greedy)
        "final_epsilon": 0.01,      # 최종 탐험율
        "timesteps": 100000,        # 엡실론이 최종값까지 감소하는 데 걸리는 스텝 수
    },
}

# DIOL Agent 클래스를 정의.
# Reference: 
class DIOLAgent(Agent):
    def __init__(self,
                 models: Dict[str, Model],
                 memory: Memory,
                 observation_space: gym.Space,
                 action_space: gym.Space,
                 device: Union[str, torch.device],
                 cfg: Mapping[str, Any]) -> None:

        super().__init__(models=models,
                         memory=memory,
                         observation_space=observation_space,
                         action_space=action_space,
                         device=device,
                         cfg=DIOL_DEFAULT_CONFIG)
        

        self.cfg.update(cfg)

        self.q_network = self.models["q_network"]
        self.target_q_network = self.models["target_q_network"]

        # 타겟 네트워크의 파라미터를 Q-네트워크와 동일하게 초기화
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        # 타겟 네트워크는 학습되지 않도록 설정 : Soft 업데이트만 수행함.
        for param in self.target_q_network.parameters():
            param.requires_grad = False

        # 옵티마이저 설정
        self.optimizer = Adam(self.q_network.parameters(), lr=self.cfg["learning_rate"])

        # Epsilon-greedy Exploration 설정
        self._exploration_epsilon = self.cfg["exploration"]["initial_epsilon"]
        self._epsilon_decay = (self.cfg["exploration"]["initial_epsilon"] - self.cfg["exploration"]["final_epsilon"]) / self.cfg["exploration"]["timesteps"]

    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        """Epsilon-greedy 전략에 따라 행동을 결정"""
        # 엡실론 값 업데이트
        self._exploration_epsilon = max(self.cfg["exploration"]["final_epsilon"], self._exploration_epsilon - self._epsilon_decay)
        
        # 무작위 행동 선택 (Exploration)
        if torch.rand(1).item() < self._exploration_epsilon:
            return self.action_space.sample(states.shape[0])
        
        # Q-value가 가장 높은 행동 선택 (Exploitation)
        q_values, _ = self.q_network.compute({"states": states})
        return torch.argmax(q_values, dim=1, keepdim=True)

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
            'option_terminated' 플래그를 infos에서 가져와 버퍼에 함께 저장
        """
        # 부모 클래스의 record_transition을 호출하여 기본적인 정보를 메모리에 저장
        super().record_transition(states=states,
                                  actions=actions,
                                  rewards=rewards,
                                  next_states=next_states,
                                  terminated=terminated,
                                  truncated=truncated,
                                  infos=infos,
                                  **kwargs)

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