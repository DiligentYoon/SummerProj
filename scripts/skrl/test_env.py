# debug_diol_runner.py

"""
Script to debug the instantiation of the HRL framework components (Runner, Agents, Models).
"""

import argparse
import sys
import hydra
import torch

from isaaclab.app import AppLauncher

# --- (1) 기존 학습 스크립트의 Argument Parser와 AppLauncher 부분은 그대로 사용 ---
parser = argparse.ArgumentParser(description="Debug HRL agent setup with skrl.")
# task 인자는 우리가 만든 환경을 지정해야 합니다.
parser.add_argument("--task", type=str, default="Franka-PickandPlace-Direct-v0", help="Name of the task.") 
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)

# ... (seed 등 다른 필요한 인자들 추가)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.skrl import SkrlVecEnvWrapper

# --- (2) 우리가 만든 커스텀 Runner와 Task를 임포트 ---
# 이 경로들은 실제 파일 위치에 맞게 수정해야 합니다.
import SummerProj.tasks # noqa: F401
from runner_diol import AISLDIOLRunner

algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"

@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: dict, agent_cfg: dict):
    """
        Runner와 그 내부 컴포넌트들의 생성 및 초기 상태를 확인합니다.
    """
    # 환경 설정 오버라이드
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Isaac Lab 환경 생성
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env)

    # ---  AISLDIOLRunner 인스턴스 ---
    # __init__, _generate_models, _generate_agent, _generate_trainer가 모두 호출됩니다.
    print("\n[DEBUG] Instantiating AISLDIOLRunner...")
    runner = AISLDIOLRunner(env, agent_cfg)
    print("✅ AISLDIOLRunner and its components instantiated successfully!")

    # --- 생성된 객체 확인 ---
    print("\n--- Verifying generated components ---")
    print(f"High-Level Agent: {type(runner.high_level_agent)}")
    print(f"Low-Level Agent: {type(runner.low_level_agent)}")
    assert runner.high_level_agent is not None and runner.low_level_agent is not None, "Agents were not created!"
    print("✅ Both agents are created.")

    # --- 단일 스텝 연동 테스트 ---
    print("\n--- Performing a single-step test ---")
    obs, info = env.reset()
    
    # 임의의 저수준 행동 생성
    action = torch.from_numpy(env.unwrapped.action_space.sample())
    if action.ndim == 1:
        action = action.unsqueeze(0) # num_envs=1 일 경우를 대비

    next_obs, reward, terminated, truncated, extras = env.step(action)
    
    assert "high_level_reward" in extras and "option_terminated" in extras, "HRL signals are missing!"
    print("✅ Single step successful. HRL signals are present in 'extras'.")

    # 시뮬레이터 종료
    env.close()


if __name__ == "__main__":
    # 메인 함수 실행
    main()
    # 시뮬레이션 앱 종료
    simulation_app.close()