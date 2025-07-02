
from dataclasses import MISSING
from isaaclab.envs.direct_rl_env_cfg import DirectRLEnvCfg
from isaaclab.utils.configclass import configclass

@configclass
class GoalsCfg:
    """Goal-conditioned 학습에 필요한 목표 공간 설정을 정의"""
    # 저수준 목표(g^L)의 차원.
    low_level_dim: int = 7
    # 고수준 목표(g^H)의 차원.
    high_level_dim: int = 10
    # 달성된 목표(achieved_goal)의 차원. 
    achieved_goal_dim: int = 7

@configclass
class AAESCfg:
    """Auto-Adjusting Exploration Strategy (AAES) 설정을 정의"""
    # 무작위 행동 확률(alpha)의 최대 상한값 (c_a)
    ca_upper_bound: float = 0.2
    # 행동 노이즈 표준편차(sigma)의 최대 상한값 (c_sigma)
    csigma_upper_bound: float = 0.01
    # 성공률의 부드러운 업데이트를 위한 지수 이동 평균(EMA) 계수
    smoothing_factor: float = 0.05

@configclass
class DemonstrationsCfg:
    """추상적 시연(Abstract Demonstrations) 설정을 정의"""
    # 전체 학습 에피소드 중 시연을 사용할 비율 (0.0 ~ 1.0)
    demo_ratio: float = 0.7



@configclass
class DirectDIOLCfg(DirectRLEnvCfg):
    """Configuration for a DirectDIOL environment.

    This class extends the DirectRLEnvCfg to include additional configurations specific to DirectDIOL.
    """

    # Hierarchical Reinforcement Learning settings
    goals: GoalsCfg = GoalsCfg()
    """Dimension of the goal space. This is required for the low and high level policy."""

    # Auto-Adjusting Exploration Strategy settings
    aaes: AAESCfg = AAESCfg()

    # Abstract Demonstrations settings
    demonstrations: DemonstrationsCfg = DemonstrationsCfg()

    # Reward shaping settings
    low_level_loc_threshold: float = 0.02
    low_level_rot_threshold: float = 0.1

    high_level_loc_threshold: float = 0.02
    high_level_loc_rot_threshold: float = 0.1
    