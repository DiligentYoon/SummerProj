from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.models.torch import Model, DeterministicMixin

# --- 정책함수 클래스 ---
class FrankaDeterministicPolicy(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device,
                 encoder_features: List[int] = [256, 128],
                 policy_features: List[int] = [64],
                 clip_actions: bool = False):
        
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions, device)

        obs_dim = observation_space["observation"].shape[0]
        goal_dim = observation_space["desired_goal"].shape[0]
        in_features = obs_dim + goal_dim
        
        # Backbone
        encoder_layers = []
        for out_features in encoder_features:
            encoder_layers.append(nn.Linear(in_features, out_features))
            encoder_layers.append(nn.ReLU())
            in_features = out_features
        self.encoder = nn.Sequential(*encoder_layers)

        # Policy Head
        policy_layers = []
        in_features_policy = in_features
        for out_features in policy_features:
            policy_layers.append(nn.Linear(in_features_policy, out_features))
            policy_layers.append(nn.ReLU())
            in_features_policy = out_features
        self.policy_branch = nn.Sequential(*policy_layers)
        
        self.action_head = nn.Linear(in_features_policy, self.num_actions)

    def compute(self, inputs: dict, role: str = "") -> tuple[torch.Tensor, ...]:
        obs = inputs["states"]["policy"]["observation"]
        goal = inputs["states"]["policy"]["desired_goal"]
        x = self.encoder(torch.cat((obs, goal), dim=-1))
        p = self.policy_branch(x)
        
        actions = F.tanh(self.action_head(p))
        
        return actions, {}

# --- 가치함수 클래스 ---
class FrankaValue(Model):
    def __init__(self, observation_space, action_space, device,
                 encoder_features: List[int] = [256, 128],
                 value_features: List[int] = [64]):
        
        Model.__init__(self, observation_space, action_space, device)
        
        obs_dim = observation_space["observation"].shape[0]
        goal_dim = observation_space["desired_goal"].shape[0]
        action_dim = observation_space["taken_action"].shape[0]

        # Backbone
        in_features_encoder = obs_dim + goal_dim + action_dim
        encoder_layers = []
        for out_features in encoder_features:
            encoder_layers.append(nn.Linear(in_features_encoder, out_features))
            encoder_layers.append(nn.ReLU())
            in_features_encoder = out_features
        self.encoder = nn.Sequential(*encoder_layers)

        # Value Head
        in_features_value = encoder_features[-1] + action_dim
        value_layers = []
        for out_features in value_features:
            value_layers.append(nn.Linear(in_features_value, out_features))
            value_layers.append(nn.ReLU())
            in_features_value = out_features
        self.value_branch = nn.Sequential(*value_layers)
        self.value_head = nn.Linear(in_features_value, 1)

    def compute(self, inputs: dict, role: str = "") -> tuple[torch.Tensor, ...]:
        obs = inputs["states"]["critic"]["observation"]
        goal = inputs["states"]["critic"]["desired_goal"]
        actions = inputs["taken_actions"]
        x = self.encoder(torch.cat((obs, goal), dim=-1))
        v = self.value_branch(torch.cat((x, actions), dim=-1))
        return self.value_head(v), {}
    

# --- Q Network 클래스 ---
class FrankaQNetwork(Model):
    def __init__(self, observation_space, action_space, device,
                 features: List[int] = [256, 128, 64]):
        
        Model.__init__(self, observation_space, action_space, device)

        obs_dim = observation_space["observation"].shape[0]
        goal_dim = observation_space["desired_goal"].shape[0]
        in_features = obs_dim + goal_dim
        num_actions = action_space.nvec[0]

        # MLP 네트워크
        layers = []
        for out_features in features:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            in_features = out_features
        # 출력 레이어: 각 이산 행동에 대한 Q-value를 출력
        layers.append(nn.Linear(in_features, num_actions))
        self.net = nn.Sequential(*layers)
    
    def compute(self, inputs: dict, role: str = "") -> tuple[torch.Tensor, ...]:
        # DIOL 에이전트의 관측: Dict 타입에서 추출
        input = inputs["states"]
        # 네트워크를 통과시켜 모든 행동에 대한 Q-value들을 반환합니다.
        return self.net(input), {}