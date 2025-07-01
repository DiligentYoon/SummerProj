
import copy
from typing import List, Optional, Union, Dict

from skrl.agents.torch import Agent
from skrl.envs.wrappers.torch import Wrapper
from skrl.trainers.torch import Trainer

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

        self.high_level_agent = agents["high_level"]
        self.low_level_agent = agents["low_level"]

        # 인스턴스 변수로 파라미터 저장 (train 루프에서 쉽게 사용하기 위함)
        self.timesteps = self.cfg["timesteps"]
        self.epoch_interval = self.cfg["epoch_interval"]
        self.episode_interval = self.cfg["episode_interval"]
        self.cycle_interval = self.cfg["cycle_interval"]

        if self.high_level_agent is None or self.low_level_agent is None:
            raise ValueError("The 'agents' dictionary must contain 'high_level' and 'low_level' keys.")
        
        self.high_level_agent.init(trainer_cfg = self.cfg)
        self.low_level_agent.init(trainer_cfg = self.cfg)
        print("[HRLTrainer] Initialize agents...")

