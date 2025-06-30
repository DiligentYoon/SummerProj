from typing import Any, Mapping
import copy

# hydra.utils.instantiate를 사용하기 위해 임포트
import hydra

# skrl의 원래 Runner와 필요한 클래스들을 Import
from skrl import logger
import gymnasium as gym
from skrl.utils.runner.torch import Runner
from skrl.models.torch import Model

from ....SummerProj.source.SummerProj.SummerProj.tasks.direct.franka_pap.agents.diol_agent import DIOLAgent

class AISLDIOLRunner(Runner):
    # 기존 SKRL에서 제공하는 Runner Class를 오버라이딩. Custom Model에 대한 처리만 따로 수행
    def _generate_models(self, env, cfg: Mapping[str, Any]) -> Mapping[str, Mapping[str, Model]]:
        device = env.device
        models_cfg = copy.deepcopy(cfg.get("models", {}))


        # ==== 각 Level에 맞는 Action/Observation Space 정의 ====
        # High-Level 정책(DIOL)을 위한 공간
        high_level_observation_space = env.single_observation_space["observation"]
        high_level_action_space = gym.spaces.Discrete(cfg.get("high_level_num_actions"))

        # Low-Level 정책(DDPG)을 위한 공간
        low_level_observation_space = env.observation_space 
        low_level_action_space = env.action_space


        # ==== 모델을 저장할 Dictionary 초기화 ====
        models = {
            "high_level": {},
            "low_level": {}
        }


        # ======== High Level 모델 생성 (DIOL Q Network) ========
        print("[AISLRunner] Instantiating high-level models (for DIOL Agent)...")
        high_level_models_cfg = models_cfg.get("high_level", {})
        if not high_level_models_cfg:
            raise ValueError("Configuration 'models.high_level' not found.")
        
        for role, model_config in high_level_models_cfg.items():
            # Hydra를 사용하여 커스텀 모델 인스턴스화
            if "_target_" in model_config:
                model_config["observation_space"] = high_level_observation_space
                model_config["action_space"] = high_level_action_space
                model_config["device"] = device
                models["high_level"][role] = hydra.utils.instantiate(model_config)
                print(f"  - Instantiated high-level model for role: '{role}'")


        # ========== Low-Level 모델 (DDPG Networks) 생성 ==========
        print("[AISLRunner] Instantiating low-level models (for DDPG Agent)...")
        low_level_models_cfg = models_cfg.get("low_level", {})
        if not low_level_models_cfg:
            raise ValueError("Configuration 'models.low_level' not found.")
        
        # non-shared models
        if "separate" in low_level_models_cfg:
            del low_level_models_cfg["separate"]

        for role, model_config in low_level_models_cfg.items():
            # Hydra를 사용하여 커스텀 모델 인스턴스화
            if "_target_" in model_config:
                model_config["observation_space"] = low_level_observation_space
                model_config["action_space"] = low_level_action_space
                model_config["device"] = device
                models["low_level"][role] = hydra.utils.instantiate(model_config)
                print(f"  - Instantiated low-level model for role: '{role}'")
            else:
                ValueError(f"No '_target_' field defined")

        return models


    def _generate_agent(self, env, cfg: Mapping[str, Any], models: Mapping[str, Any]) -> None:
        """
        DIOL/UOF HRL 프레임워크를 위해 고수준/저수준 에이전트를 각각 생성하고,
        'self.high_level_agent'와 'self.low_level_agent'에 저장
        """
        agents = {
            "high_level": None,
            "low_level": None
        }
        self.high_level_agent = None
        self.low_level_agent = None
        
        agent_cfg = copy.deepcopy(cfg.get("agent", {}))
        memory_cfg = copy.deepcopy(cfg.get("memory", {}))
        
        # --- 1. 고수준 에이전트 (DIOLAgent) 생성 ---
        print("[AISLRunner] Instantiating high-level agent (DIOLAgent)...")
        agent_cfg_high = agent_cfg.get("high_level", {})
        memory_cfg_high = memory_cfg.get("high_level", {})
        models_high = models.get("high_level", {})

        if agent_cfg_high and memory_cfg_high and models_high:
            # DIOLAgent를 위한 공간 정보 정의
            observation_space_high = env.single_observation_space["observation"]
            action_space_high = gym.spaces.Discrete(cfg["runner"]["high_level_num_actions"])
            
            # 고수준 메모리(리플레이 버퍼) 생성
            memory_class = self._component(memory_cfg_high.get("class", "RandomMemory"))
            memory_high = memory_class(memory_size=memory_cfg_high.get("memory_size", 10000), 
                                       num_envs=env.num_envs, 
                                       device=env.device)
            
            # Preprocessor 설정
            agent_cfg_high.get("state_preprocessor_kwargs", {}).update(
            {"size": observation_space_high, "device": self.device})

            # 커스텀 DIOLAgent 클래스 인스턴스화
            self.high_level_agent = DIOLAgent(models=models_high,
                                              memory=memory_high,
                                              observation_space=observation_space_high,
                                              action_space=action_space_high,
                                              device=env.device,
                                              cfg=agent_cfg_high)
            print("  - Instantiated high-level agent: DIOLAgent")
        
        else:
            raise ValueError("Configuration for high-level agent is incomplete or missing. Please check 'high_level' flag in the configuration.")

        # --- 2. 저수준 에이전트 (DDPG) 생성 ---
        print("[AISLRunner] Instantiating low-level agent (DDPG Agent)...")
        agent_cfg_low = agent_cfg.get("low_level", {})
        memory_cfg_low = memory_cfg.get("low_level", {})
        models_low = models.get("low_level", {})

        if agent_cfg_low and memory_cfg_low and models_low:
            # 저수준 DDPG 에이전트는 skrl의 표준 클래스를 사용
            ddpg_agent_class = self._component(agent_cfg_low.get("class"))
            
            # 저수준 메모리(리플레이 버퍼) 생성
            memory_class = self._component(memory_cfg_low.get("class", "RandomMemory"))
            memory_low = memory_class(memory_size=memory_cfg_low.get("memory_size", 100000),
                                      num_envs=env.num_envs,
                                      device=env.device)
            
            # preprocessor 설정
            agent_cfg_low.get("state_preprocessor_kwargs", {}).update(
                {"size": env.observation_space, "device": self.device})
            agent_cfg_low.get("value_preprocessor_kwargs", {}).update(
                {"size": 1, "device": self.device})

            # skrl의 DDPG 에이전트 인스턴스화
            self.low_level_agent = ddpg_agent_class(models=models_low,
                                                    memory=memory_low,
                                                    observation_space=env.observation_space,
                                                    action_space=env.action_space,
                                                    device=env.device,
                                                    cfg=agent_cfg_low)
            
            print("  - Instantiated low-level agent: DDPG")
        
        else:
            raise ValueError("Configuration for low-level agent is incomplete or missing. Please check 'low_level' flag in the configuration.")
        
        agents["high_level"] = self.high_level_agent
        agents["low_level"] = self.low_level_agent

        return agents
    
    
    def _generate_trainer(self, env, cfg, agent):
        return self
    

    def run(self, mode: str = "train") -> None:
        """Run the training/evaluation

        :param mode: Running mode: ``"train"`` for training or ``"eval"`` for evaluation (default: ``"train"``)

        :raises ValueError: The specified running mode is not valid
        """

        if mode == "train":
            print("[AISLRunner] Starting HRL training...")
            obs_dict, info = self._env.reset()


        elif mode == "eval":
            print("[AISLRunner] Starting HRL evaluation...")
            obs_dict, info = self._env.reset()
        else:
            raise ValueError(f"Unknown running mode: {mode}")