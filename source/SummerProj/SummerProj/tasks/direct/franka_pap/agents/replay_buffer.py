import torch
import gymnasium
from typing import List, Optional, Tuple, Union
from skrl.memories.torch.base import Memory
from skrl.utils.spaces.torch import compute_space_size


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
                    action: torch.Tensor, 
                    reward: torch.Tensor, 
                    next_states: torch.Tensor, 
                    terminated: torch.Tensor, 
                    truncated: torch.Tensor,
                    achieved_goal: torch.Tensor,
                    desired_goal: torch.Tensor) -> None:
        """
            입력 텐서들의 shape: (num_envs, feature_dim)
        """
        # (num_envs,)
        current_indices = self.memory_index
        
        # 병렬 환경의 인덱스를 나타내는 텐서를 생성합니다 (0, 1, 2, ..., num_envs-1)
        env_indices = torch.arange(self.num_envs, device=self.device)

        # 각 텐서의 (쓰기_인덱스, 환경_인덱스) 위치에 데이터를 저장합니다.
        # 예: self.tensors["states"]의 [0, 0] 위치에 0번 환경의 0번째 스텝 데이터 저장
        self.tensors["states"][current_indices, env_indices] = states
        self.tensors["action"][current_indices, env_indices] = action
        self.tensors["reward"][current_indices, env_indices] = reward
        self.tensors["next_states"][current_indices, env_indices] = next_states
        self.tensors["terminated"][current_indices, env_indices] = terminated
        self.tensors["truncated"][current_indices, env_indices] = truncated
        self.tensors["desired_goal"][current_indices, env_indices] = desired_goal

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

        self.replacement = True
        self.k = k_num
        self.strategy = strategy


    def create_tensor(
        self,
        name: str,
        size: Union[int, Tuple[int], gymnasium.Space],
        dtype: Optional[torch.dtype] = None,
        keep_dimensions: bool = False,
    ) -> bool:
        
        if keep_dimensions and isinstance(size, gymnasium.spaces.Dict):
            # Dict 스페이스의 경우, 각 하위 스페이스에 대해 재귀적으로 create_tensor 호출
            for key, space in size.spaces.items():
                # 텐서 이름을 조합하여 생성 (예: "states" -> "states_pos", "states_x")
                self.create_tensor(f"{name}_{key}", space, dtype, keep_dimensions)
            # Dict 자체에 대한 텐서는 생성하지 않으므로 여기서 함수 종료
            return True

        # skrl의 기본 로직과 거의 동일하지만, tensor_shape를 만드는 부분을 수정  
        # compute_space_size는 그대로 사용하여 gym.space를 숫자 형태로 변환
        # 단, keep_dimensions=True일 때는 이 결과값을 사용하지 않음
        flat_size = compute_space_size(size, occupied_size=True)
        
        # check dtype and size if the tensor exists
        if name in self.tensors:
            expected_shape = None
            if self.tensors_keep_dimensions[name] and isinstance(size, (tuple, list, gymnasium.spaces.Box)):
                shape_tuple = size.shape if hasattr(size, 'shape') else size
                expected_shape = tuple(shape_tuple)
            else:
                flat_size = compute_space_size(size, occupied_size=True)
                expected_shape = (flat_size,)

            #    (memory_size, num_envs, *data_shape) 이므로 앞의 2개 차원을 제외.
            existing_shape = self.tensors[name].shape[2:]
            
            if existing_shape != expected_shape:
                raise ValueError(f"Shape of tensor '{name}' ({expected_shape}) doesn't match the existing one ({existing_shape})")
            
            if dtype is not None and self.tensors[name].dtype != dtype:
                raise ValueError(f"Dtype of tensor '{name}' ({dtype}) doesn't match the existing one ({self.tensors[name].dtype})")
            
            return False
        
        if keep_dimensions and isinstance(size, (tuple, list, gymnasium.spaces.Box)):
            # size가 tuple, list, 또는 Box space일 경우, 그 shape을 그대로 사용
            shape = size.shape if hasattr(size, 'shape') else size
            tensor_shape = (self.memory_size, self.num_envs, *shape)
            view_shape = (-1, *shape)
        else:
            # 그 외의 경우 (또는 keep_dimensions=False), 기존 로직대로 flatten
            tensor_shape = (self.memory_size, self.num_envs, flat_size)
            view_shape = (-1, flat_size)
        
        # create tensor (_tensor_<name>) and add it to the internal storage
        setattr(self, f"_tensor_{name}", torch.zeros(tensor_shape, device=self.device, dtype=dtype))
        # update internal variables
        self.tensors[name] = getattr(self, f"_tensor_{name}")
        self.tensors_view[name] = self.tensors[name].view(*view_shape)
        self.tensors_keep_dimensions[name] = keep_dimensions
        # fill the tensors (float tensors) with NaN
        for tensor in self.tensors.values():
            if torch.is_floating_point(tensor):
                tensor.fill_(float("nan"))
        return True


    def add_samples(self, **tensors):
        """
            **tensors:
                "states": high-level 관측 벡터
                "actions_how" : high-level 액션 벡터 (motion parameters)
                "actions_where": high-level 액션 벡터 (target position)
                "next_states": high-level 관측 벡터
                "rewards": high-level 원본 보상
                "truncated": 에피소드 종료 여부 1
                "terminated": 에피소드 종료 여부 2
                "desired_goal_obj_state": 물체의 목표 위치
                "desired_goal_tcp_state": 로봇 TCP의 목표 위치
            
            Procedure:
                1. 원본 데이터 먼저 저장 (에피소드 버퍼로부터 넘겨받은 데이터)
                2. HER 데이터 증강 후, 저장

        """
        # (E, k) -> we need to know Env Dimension
        batch_size = next(iter(tensors.values())).shape[0]
        
        # Compute residual sapce
        space_left = self.memory_size - self.memory_index
        
        # 한 번에 추가할 수 있는 양과, 순환하여 추가할 양으로 나눔
        fit_size = min(batch_size, space_left)
        overflow_size = batch_size - fit_size

        # ======= 버퍼에 데이터 복사 ========
        for name, tensor in tensors.items():
            if name in self.tensors:
                # (batch_size, feat) -> (batch_size, 1, feat)로 unsqueeze하여 저장
                if len(tensor.shape) == 1:
                    tensor = tensor.unsqueeze(-1)

                self.tensors[name][self.memory_index : self.memory_index + fit_size] = tensor[:fit_size].unsqueeze(1)

                # 버퍼 용량을 넘는 데이터는 처음부터 덮어쓰기 (Queue형 버퍼)
                if overflow_size > 0:
                    self.tensors[name][0:overflow_size] = tensor[fit_size:].unsqueeze(1)
        
        # 포인터 업데이트
        self.memory_index = (self.memory_index + batch_size) % self.memory_size
        if not self.filled and (self.memory_index + batch_size >= self.memory_size):
            self.filled = True

    
    # def her_augment(self, **tensors):
    #     """
    #         HER(Hindsight Experience Replay) Augmentation

    #         Procedure:
    #             1. 입력받은 tensor의 인덱스에서 k개의 샘플을 랜덤하게 선택
    #             2. 각 샘플에 대해 목표 상태를 변경
    #             3. 변경된 목표 상태를 바탕으로 리워드 계산
    #             4. 새로운 리워드, 새로운 목표 상태를 가진 증강 데이터 버퍼에 저장
    #     """
    #     # 에피소드 데이터에서 k개의 샘플을 랜덤하게 선택
    #     batch_size = next(iter(tensors.values())).shape[0]
    #     for t in batch_size:
    #         if t > self.memory_index:
    #             raise ValueError(f"Batch size {t} exceeds current memory index {self.memory_index}. Cannot sample k elements.")

    #         future_indices = torch.randint(t, batch_size, (self.k,), device=self.device)
    #         next_states = tensors["achieved_goal"][t]
    #         fake_goal = tensors["achieved_goal"][future_indices]

    #         if type(fake_goal) == dict and type(next_states) == dict:
    #             fake_done = torch.logical_and(torch.equal(next_states["obj_state"],
    #                                                       fake_goal["obj_state"]),
    #                                           torch.equal(next_states["tcp_state"],
    #                                                       ))
                

    #         fake_done = torch.equal(fake_goal, next_states)
    #         fake_rewards = torch.int(fake_done) - 1

    #         tensors["terminated"] = fake_done
    #         tensors["truncated"] = fake_done
    #         tensors["desired_goal"] = fake_goal
    #         tensors["reward"] = fake_rewards

    #         for name, tensor in tensors.items():
    #             # 메모리에 저장
    #             # {states, actions, next_states, fake_rewards, truncated, terminated, fake_goals}
    #             if name in self.tensors:
    #                 if len(tensor.shape) == 1:
    #                     tensor = tensor.unsqueeze(-1)
    #                 if name == "rewards":
    #                      self.tensors[name]
                        
                # done -> fake_done
                # goal -> fake_goal
                # rewards -> fake_rewards

            # {states, actions, next_states, fake_rewards, truncated, terminated, fake_goals}

            


        # indices = torch.randint(0, self.memory_index, (self.k,), device=self.device)
        

        # state = self.tensors["states"][indices]


        # # 각 샘플에 대해 목표 상태를 변경
        # for name in ["desired_goal_obj_state", "desired_goal_tcp_state"]:
        #     if name in self.tensors:
        #         # 선택된 인덱스의 목표 상태를 변경
        #         state = self.tensors[name][indices]

        #         self.tensors[name][indices] = self.tensors["achieved_goal"][indices]


    
    def sample(self, batch_size: int, names: list[str]) -> tuple[list[torch.Tensor], torch.Tensor]:
        """
        버퍼에서 Random으로 transition 배치를 샘플링
        
        :param batch_size: 샘플링할 배치의 크기
        :param names: 샘플링할 텐서의 이름 리스트
        :return: 텐서 리스트와 샘플링된 인덱스
        """
        # 샘플링 가능한 최대 인덱스
        max_index = self.memory_size if self.filled else self.memory_index
        if max_index == 0:
            return [torch.empty(0) for _ in names], torch.empty(0)
            
        # batch_size 만큼의 랜덤 인덱스 생성
        if self.replacement:
            indexes = torch.randint(0, max_index, (batch_size,), device=self.device)
        else:
            indexes = torch.randperm(max_index, dtype=torch.long, device=self.device)[:batch_size]
        
        # 해당 인덱스의 데이터를 가져와서 리스트에 담음
        sampled_tensors = []
        for name in names:
            # (batch_size, 1, feat) -> (batch_size, feat)로 squeeze하여 반환
            sampled_tensors.append(self.tensors[name][indexes].squeeze(1))
            
        return sampled_tensors

