import torch
import gymnasium
from typing import List, Optional, Tuple, Union
from skrl.memories.torch.base import Memory
from copy import deepcopy as dcp


class EpisodeWiseReplayBuffer(Memory):

    def __init__(
        self,
        memory_size: int,
        num_envs: int = 1,
        device: Optional[Union[str, torch.device]] = None,
        export: bool = False,
        export_format: str = "pt",
        export_directory: str = "",
    ) -> None:
        super().__init__(memory_size, num_envs, device, export, export_format, export_directory)

        self.memory_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_lengths = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)


    def add_samples(self, 
                    states: torch.Tensor, 
                    sub_goal: torch.Tensor, 
                    action: torch.Tensor, 
                    reward: torch.Tensor, 
                    achieved_goal: torch.Tensor, 
                    next_states: torch.Tensor, 
                    terminated: torch.Tensor, 
                    truncated: torch.Tensor) -> None:
        """
            입력 텐서들의 shape: (num_envs, feature_dim)
        """
        # shape: (num_envs,)
        current_indices = self.memory_index
        
        # 병렬 환경의 인덱스를 나타내는 텐서를 생성합니다 (0, 1, 2, ..., num_envs-1)
        env_indices = torch.arange(self.num_envs, device=self.device)

        # 각 텐서의 (쓰기_인덱스, 환경_인덱스) 위치에 데이터를 저장합니다.
        # 예: self.tensors["states"]의 [0, 0] 위치에 0번 환경의 0번째 스텝 데이터 저장
        self.tensors["states"][current_indices, env_indices] = states
        self.tensors["sub_goal"][current_indices, env_indices] = sub_goal
        self.tensors["action"][current_indices, env_indices] = action
        self.tensors["reward"][current_indices, env_indices] = reward
        self.tensors["achieved_goal"][current_indices, env_indices] = achieved_goal
        self.tensors["next_states"][current_indices, env_indices] = next_states
        self.tensors["terminated"][current_indices, env_indices] = terminated
        self.tensors["truncated"][current_indices, env_indices] = truncated

        # 다음 데이터를 저장하기 위해 메모리 인덱스를 1 증가시킵니다.
        self.memory_index += 1

        # 에피소드가 종료(terminated or truncated)된 환경이 있는지 확인합니다.
        # dones의 shape: (num_envs,)
        dones = (terminated | truncated).squeeze(-1)
        
        # 종료된 환경의 인덱스는 0으로 리셋하여 다음 에피소드를 처음부터 저장하도록 합니다.
        if dones.any():
            self.episode_lengths[dones] = self.memory_index[dones]
        
        # 버퍼의 최대 크기를 넘지 않도록 나머지 연산을 수행합니다 (안전장치).
        self.memory_index[dones] = 0
        self.memory_index %= self.memory_size

class HighLevelHindSightReplayBuffer(Memory):

    def __init__(
        self,
        memory_size: int,
        device: Optional[Union[str, torch.device]] = None,
        export: bool = False,
        export_format: str = "pt",
        export_directory: str = "",
        k_num: int = 4,
        strategy: str = "future"
    ) -> None:
        super().__init__(memory_size=memory_size, 
                         num_envs=1, 
                         device=device, 
                         export=export, 
                         export_format=export_format, 
                         export_directory=export_directory)

        self.k = k_num
        self.strategy = strategy
    
    def add_samples(self, **tensors):
        # 현재 배치 크기 확인
        batch_size = next(iter(tensors.values())).shape[0]
        
        # 남은 공간 계산
        space_left = self.memory_size - self.memory_index
        
        # 한 번에 추가할 수 있는 양과, 순환하여 추가할 양으로 나눔
        fit_size = min(batch_size, space_left)
        overflow_size = batch_size - fit_size

        # ======= 버퍼에 데이터 복사 ========
        for name, tensor in tensors.items():
            if name in self.tensors:
                # 버퍼의 남은 공간에 데이터 채우기
                #    (batch_size, feat) -> (batch_size, 1, feat)로 unsqueeze하여 저장
                self.tensors[name][self.memory_index : self.memory_index + fit_size] = tensor[:fit_size].unsqueeze(1)

                # 버퍼 용량을 넘는 데이터는 처음부터 덮어쓰기 (Queue형 버퍼)
                if overflow_size > 0:
                    self.tensors[name][0:overflow_size] = tensor[fit_size:].unsqueeze(1)
        
        # 포인터 업데이트
        self.memory_index = (self.memory_index + batch_size) % self.memory_size
        if not self.filled and (self.memory_index + batch_size >= self.memory_size):
            self.filled = True
    
    def sample(self, batch_size: int, names: list[str]) -> tuple[list[torch.Tensor], torch.Tensor]:
        """
        버퍼에서 무작위로 transition 배치를 샘플링
        
        :param batch_size: 샘플링할 배치의 크기
        :param names: 샘플링할 텐서의 이름 리스트
        :return: 텐서 리스트와 샘플링된 인덱스
        """
        # 샘플링 가능한 최대 인덱스
        max_index = self.memory_size if self.filled else self.memory_index
        if max_index == 0:
            return [torch.empty(0) for _ in names], torch.empty(0)
            
        # batch_size 만큼의 랜덤 인덱스 생성
        indexes = torch.randint(0, max_index, (batch_size,), device=self.device)
        
        # 해당 인덱스의 데이터를 가져와서 리스트에 담음
        sampled_tensors = []
        for name in names:
            # (batch_size, 1, feat) -> (batch_size, feat)로 squeeze하여 반환
            sampled_tensors.append(self.tensors[name][indexes].squeeze(1))
            
        return sampled_tensors

