import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from typing import Any, Mapping, Dict
import copy

import torch
import hydra

# skrl의 원래 Runner와 필요한 클래스들을 Import
import gymnasium as gym
from skrl.utils.runner.torch import Runner
from skrl.models.torch import Model
from skrl.agents.torch import Agent

from source.SummerProj.SummerProj.tasks.direct.franka_pap.agents.replay_buffer import HighLevelHindSightReplayBuffer
from source.SummerProj.SummerProj.tasks.direct.franka_pap.agents.hac_agent import HybridActorCriticAgent
from source.SummerProj.SummerProj.tasks.direct.franka_pap.trainers.trainer import HRLTrainer


class AISLDIOLRunner(Runner):
    # 기존 SKRL에서 제공하는 Runner Class를 오버라이딩. Customizing 요소에 대한 처리만 따로 수행
    def _generate_models(self, env, cfg: Mapping[str, Any]) -> Mapping[str, Mapping[str, Model]]:
        device = env.device
        models_cfg = copy.deepcopy(cfg.get("models", {}))

        # ==== 모델을 저장할 Dictionary 초기화 ====
        models = {
            "high_level": {},
            "low_level": {}
        }

        # ======== High Level 모델 생성 ========
        print("[AISLRunner] Instantiating high-level models ...")
        high_level_models_cfg = models_cfg.get("high_level", {})
        if not high_level_models_cfg:
            raise ValueError("Configuration 'models.high_level' not found.")
        
        for role, model_config in high_level_models_cfg.items():
            # Hydra를 사용하여 커스텀 모델 인스턴스화
            if "_target_" in model_config:
                model_config["observation_space"] = env._unwrapped.single_observation_space_h["policy"]
                model_config["action_space"] = env._unwrapped.single_action_space_h
                model_config["device"] = device
                models["high_level"][role] = hydra.utils.instantiate(model_config)
                print(f"  - Instantiated high-level model for role: '{role}'")


        # ========== Low-Level 모델 생성 ==========
        print("[AISLRunner] Instantiating low-level models ...")
        low_level_models_cfg = models_cfg.get("low_level", {})
        if not low_level_models_cfg:
            raise ValueError("Configuration 'models.low_level' not found.")
        
        # non-shared models
        if "separate" in low_level_models_cfg:
            del low_level_models_cfg["separate"]

        for role, model_config in low_level_models_cfg.items():
            # Hydra를 사용하여 커스텀 모델 인스턴스화
            if "_target_" in model_config:
                model_config["observation_space"] = env._unwrapped.single_observation_space["policy"]
                model_config["action_space"] = env.action_space
                model_config["device"] = device
                models["low_level"][role] = hydra.utils.instantiate(model_config)
                print(f"  - Instantiated low-level model for role: '{role}'")
            else:
                ValueError(f"No '_target_' field defined")
        
        print("✅ [INFO] Models are instatiated successfully !")

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
        
        # --- High-Level 에이전트 생성 ---
        print("[AISLRunner] Instantiating high-level agent ...")
        high_level_observation_space = env._unwrapped.single_observation_space_h["policy"]
        high_level_action_space = env._unwrapped.single_action_space_h
        high_level_goal_space = env._unwrapped.single_goal_space

        agent_cfg_high = self._process_cfg(cfg["agent"]["high_level"])
        memory_cfg_high = memory_cfg.get("high_level", {})
        models_high = models.get("high_level", {})

        if agent_cfg_high and models_high:
            # 고수준 메모리(리플레이 버퍼) 생성
            memory_high = HighLevelHindSightReplayBuffer(memory_size=memory_cfg_high.get("memory_size"),  
                                                         device=env.device)
            
            # Preprocessor 설정
            if agent_cfg_high.get("state_preprocessor_kwargs") is None:
                agent_cfg_high["state_preprocessor_kwargs"] = {}
            agent_cfg_high["state_preprocessor_kwargs"].update(
            {"size": high_level_observation_space, "device": env.device})

            # 커스텀 DIOLAgent 클래스 인스턴스화
            self.high_level_agent = HybridActorCriticAgent(models=models_high,
                                                           memory=memory_high,
                                                           observation_space=high_level_observation_space,
                                                           action_space=high_level_action_space,
                                                           goal_sapce=high_level_goal_space,
                                                           device=env.device,
                                                           cfg=agent_cfg_high)
            print("  - Instantiated high-level agent: DIOLAgent")
        
        else:
            raise ValueError("Configuration for high-level agent is incomplete or missing. Please check 'high_level' flag in the configuration.")

        # --- Low-Level 에이전트 생성 ---
        print("[AISLRunner] Instantiating low-level agent...")
        low_level_observation_space = env._unwrapped.single_observation_space["policy"]
        low_level_action_space = env._unwrapped.single_action_space

        agent_cfg_low = agent_cfg.get("low_level", {})
        memory_cfg_low = memory_cfg.get("low_level", {})
        models_low = models["low_level"]
        
        agent_class = agent_cfg_low.get("class", "").lower()
        if not agent_class:
            raise ValueError(f"No 'class' field defined in 'agent' cfg")
        del agent_cfg_low["class"]
        # check for memory configuration (backward compatibility)
        if memory_cfg_low:
            memory_cfg_low = {"class": "RandomMemory", "memory_size": -1}
        # get memory class and remove 'class' field
        try:
            memory_class = self._component(memory_cfg_low["class"])
            del memory_cfg_low["class"]
        except KeyError:
            memory_class = self._component("RandomMemory")
            print("No 'class' field defined in 'memory' cfg. 'RandomMemory' will be used as default")
        # instantiate memory
        if memory_cfg_low["memory_size"] < 0:
            memory_cfg_low["memory_size"] = agent_cfg_low["rollouts"]  # memory_size is the agent's number of rollouts
        memory_low = memory_class(num_envs=env.num_envs, device=env.device, **self._process_cfg(memory_cfg_low))

        # agent 생성
        agent_cfg_low = self._component(f"{agent_class}_DEFAULT_CONFIG").copy()
        agent_cfg_low["use_pre_trained"] = False
        agent_cfg_low["checkpoint_path"] = ''
        agent_cfg_low.update(self._process_cfg(agent_cfg_low))
        agent_cfg_low.get("state_preprocessor_kwargs", {}).update(
            {"size": low_level_observation_space, "device": env.device}
        )
        agent_cfg_low.get("value_preprocessor_kwargs", {}).update({"size": 1, "device": env.device})
        if agent_cfg_low.get("exploration", {}).get("noise", None):
            agent_cfg_low["exploration"].get("noise_kwargs", {}).update({"device": env.device})
            agent_cfg_low["exploration"]["noise"] = agent_cfg_low["exploration"]["noise"](
                **agent_cfg_low["exploration"].get("noise_kwargs", {})
            )
        if agent_cfg_low.get("smooth_regularization_noise", None):
            agent_cfg_low.get("smooth_regularization_noise_kwargs", {}).update({"device": env.device})
            agent_cfg_low["smooth_regularization_noise"] = agent_cfg_low["smooth_regularization_noise"](
                **agent_cfg_low.get("smooth_regularization_noise_kwargs", {})
            )
        agent_kwargs = {
            "models": models_low,
            "memory": memory_low,
            "observation_space": low_level_observation_space,
            "action_space": low_level_action_space,
        }

        self.low_level_agent = self._component(agent_class)(cfg=agent_cfg_low, device=env.device, **agent_kwargs)
        
        agents["high_level"] = self.high_level_agent
        agents["low_level"] = self.low_level_agent

        print("✅ [INFO] Agents are instatiated successfully !")

        return agents
    
    
    def _generate_trainer(self, env, cfg:Mapping[str, Any], agents: Dict[str, Agent]) -> HRLTrainer:
        """
            Generate a custom HRL trainer instance for the AISL DIOL framework.
            
            Args:
                env (Wrapper): 학습에 사용될 환경.
                cfg (Mapping[str, Any]): 전체 설정 딕셔너리.
                agents (Dict[str, Agent]): _generate_agent에서 반환한 
                                        {"high_level": ..., "low_level": ...} 형태의 딕셔너리.

            Returns:
                HRLTrainer: 인스턴스화된 커스텀 HRL 트레이너 객체.
        """
        print("[AISLRunner] Instantiating custom HRLTrainer...")
        trainer_cfg = cfg.get("trainer", {})

        # Hydra의 _target_기능 사용하여 커스텀 HRLTrainer 인스턴스화
        if "_target_" in trainer_cfg:
            # HRLTrainer의 __init__에 필요한 모든 인자를 전달합니다.
            return hydra.utils.instantiate(config=trainer_cfg, env=env, agents=agents)
        
        # 또는 skrl의 기본 방식을 따를 경우 (class 키 사용): 이 경우에는 SequentialTrainer와 같이 SKRL에서 구현된 Trainer사용하는 경우에만 사용
        else:
            try:
                # yaml 파일에서 trainer 클래스 경로를 가져옵니다.
                trainer_class_str = trainer_cfg["class"]
                del trainer_cfg["class"]
                trainer_class = self._component(trainer_class_str)
            except KeyError:
                # _target_이나 class 키가 없으면 에러를 발생시킵니다.
                raise ValueError("A 'class' or '_target_' must be defined for the trainer in the configuration.")
            
            # HRLTrainer에 환경과 두 에이전트를 모두 전달하여 생성합니다.
            return trainer_class(env=env, agents=agents, cfg=trainer_cfg)

        
        
    
    # ============== 보조 함수 ===============
    def _recompute_reward(self, achieved_goal, desired_goal):
        """
            Compute the reward based on the achieved goal for the HER strategy.
        """
        dist = torch.norm(achieved_goal - desired_goal, dim=-1)

        return torch.where(dist < self.her_cfg["low_threshold"], 0.0, -1.0)