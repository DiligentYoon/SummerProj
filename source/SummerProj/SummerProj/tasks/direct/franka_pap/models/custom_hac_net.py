
from typing import List, Dict

import torch
import torch.nn as nn
from skrl.models.torch import Model, DeterministicMixin
from source.feature_extractor.model.pointnet2_hac import PointNet2Segmentation

class HybridActorNet(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device,
                 mlp_features: List[int] = [256, 128],
                 feature_dim = 128):
        """
        Args:
            observation_space (gym.Space): 관측 공간. pre-processed per-point feature를 포함.
            action_space (gym.Space): 행동 공간.
            device (torch.device): 텐서가 위치할 장치.
            feature_dim (int): 환경에서 처리되어 넘어오는 Per-Point 피처 벡터의 차원 (F).
            mlp_features (List[int]): Actor 헤드를 구성하는 MLP의 중간 레이어 크기.
        """
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=True) # 행동 값을 -1~1로 클리핑

        in_channel_dim = self.observation_space.shape[-1] - 3 # pos dimension
        action_dim = self.action_space.shape[-1]

        # Feature Extractor
        self.feature_extractor = PointNet2Segmentation(in_channels=in_channel_dim, 
                                                       out_channels=[],
                                                       normalize_pos=True,
                                                       pos_in_feature=False).to(device)

        action_dim = self.action_space.shape[-1]
        
        in_features = feature_dim
        layers = []
        for out_features in mlp_features:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            in_features = out_features
        layers.append(nn.Linear(in_features, action_dim))
        self.mlp_head = nn.Sequential(*layers)

        self.apply(self._initialize_weights)

    
    def _initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.414)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def compute(self, inputs: Dict[str, torch.Tensor], role: str = "") -> tuple[torch.Tensor, ...]:
            # 입력 관측값에서 pos(좌표)와 x(추가 특징)를 분리
            # skrl은 관측값을 inputs["states"]로 전달합니다.
            # 텐서 형태: (B, N, Dim) - B:배치, N:점 개수, Dim:차원
            obs = inputs["states"]
            pos = obs[:, :, :3]
            x = obs[:, :, 3:]
            
            batch_size, num_points, _ = pos.shape
            
            # PointNet++ 입력 형식에 맞게 텐서 형태 변경: (B, N, D) -> (B*N, D)
            pos_flat = pos.reshape(-1, 3)
            x_flat = x.reshape(-1, x.shape[-1])

            # torch_geometric을 위한 배치 인덱스 텐서 생성
            batch_indices = torch.arange(batch_size, device=self.device).repeat_interleave(num_points)

            # 특징 추출기를 통과시켜 Per-Point 피처 계산
            # 출력 형태: (B*N, 128)
            per_point_features_flat = self.feature_extractor(x=x_flat, pos=pos_flat, batch=batch_indices)
            
            # 원래 형태로 복원: (B, N, 128)
            per_point_features = per_point_features_flat.reshape(batch_size, num_points, -1)

            # MLP 헤드를 통과시켜 Per-Point 모션 파라미터 계산
            # DeterministicMixin이 tanh를 적용하여 출력을 -1~1로 제한합니다.
            # 출력 형태: (B, N, how_dim)
            per_point_motions = self.mlp_head(per_point_features)
            
            # Model은 계산된 '모든' 잠재적 행동을 반환하는 역할만 수행
            return per_point_motions, {}
    

class HybridCriticNet(Model):
    def __init__(self, observation_space, action_space, device,
                 feature_dim = 128,
                 mlp_features: List[int] = [256, 128],
                 motion_params_featrues = 3):
        """
        Args:
            observation_space (gym.Space): 관측 공간.
            action_space (gym.Space): 행동 공간.
            device (torch.device): 텐서가 위치할 장치.
            feature_dim (int): Per-Point 피처 벡터의 차원 (F).
            mlp_features (List[int]): Critic을 구성하는 MLP의 중간 레이어 크기.
        """
        Model.__init__(self, observation_space, action_space, device)

        in_channel_dim = self.observation_space.shape[-1] - 3 # pos dimension

        # Feature Extractor
        self.feature_extractor = PointNet2Segmentation(in_channels=in_channel_dim, 
                                                       out_channels=[],
                                                       normalize_pos=True,
                                                       pos_in_feature=False).to(device)
        
        self.feature_normalizer = nn.LayerNorm(feature_dim)

        # MLP 네트워크 구성
        in_features = feature_dim + motion_params_featrues
        layers = []
        for out_features in mlp_features:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            in_features = out_features
        # 최종 출력 레이어: 각 점에 대한 스칼라 Q-값
        layers.append(nn.Linear(in_features, 1))
        self.mlp_head = nn.Sequential(*layers)
    
    def _initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.414)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def compute(self, inputs: Dict[str, torch.Tensor], role: str = "") -> tuple[torch.Tensor, ...]:
        # skrl Critic의 입력: 상태(states)와 행동(taken_actions)
        obs = inputs["states"]
        per_point_motions = inputs["taken_actions"] # Actor가 계산한 (B, N, action_dim) 텐서

        # 1. 입력으로 Pcd 정보 받기 (obs)
        pos = obs[:, :, :3]
        x = obs[:, :, 3:]
        
        batch_size, num_points, _ = pos.shape
        
        # PointNet++ 입력 형식에 맞게 텐서 형태 변경: (B, N, D) -> (B*N, D)
        pos_flat = pos.reshape(-1, 3)
        x_flat = x.reshape(-1, x.shape[-1])
        batch_indices = torch.arange(batch_size, device=self.device).repeat_interleave(num_points)

        # 2. Pcd를 FE에 통과시켜 피처 벡터 얻기
        # 출력 형태: (B*N, 128)
        per_point_features_flat = self.feature_extractor(x=x_flat, pos=pos_flat, batch=batch_indices)
        # 원래 형태로 복원: (B, N, 128)
        per_point_features = per_point_features_flat.reshape(batch_size, num_points, -1)

        normalized_features = self.feature_normalizer(per_point_features)
        
        # 3. 모션 파라미터 정보와 피처 벡터를 concat
        # 입력 텐서 형태: (B, N, 128 + action_dim)
        critic_input = torch.cat([normalized_features, per_point_motions], dim=-1)

        # 4. 최종적으로 MLP를 통과시켜 per-point Q-value 얻기
        # 출력 텐서 형태: (B, N, 1) -> 이것이 바로 'Critic Map'
        critic_map = self.mlp_head(critic_input)
        
        return critic_map, {}