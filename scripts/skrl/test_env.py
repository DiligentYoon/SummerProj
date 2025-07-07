# debug_diol_runner.py

"""
Script to debug the instantiation of the HRL framework components (Runner, Agents, Models).
"""

import argparse
import sys
import hydra
import torch

import gymnasium as gym
import os
import random
from datetime import datetime

import skrl

from isaaclab.app import AppLauncher

# --- (1) 기존 학습 스크립트의 Argument Parser와 AppLauncher 부분은 그대로 사용 ---
parser = argparse.ArgumentParser(description="Debug HRL agent setup with skrl.")
# task 인자는 우리가 만든 환경을 지정해야 합니다.
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Franka-PickandPlace-Direct-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
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
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

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

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["high_level"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging HRL experiment in base directory: {log_root_path}")

    experiment_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"

    if agent_cfg["agent"]["high_level"]["experiment"]["experiment_name"]:
        experiment_name += f'_{agent_cfg["agent"]["high_level"]["experiment"]["experiment_name"]}'
    print(f"Generated unique experiment name: {experiment_name}")

    for level in ["high_level", "low_level"]:
        if level in agent_cfg["agent"]:
            print(f"Updating configuration for {level} agent...")
            agent_cfg["agent"][level]["experiment"]["directory"] = log_root_path
            agent_cfg["agent"][level]["experiment"]["experiment_name"] = experiment_name

    log_dir = os.path.join(log_root_path, experiment_name)

    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)

    print(f"Dumping configuration files to: {os.path.join(log_dir, 'params')}")
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)

    # get checkpoint path (to resume training)
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    # Isaac Lab 환경 생성
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env)

    # ---  AISLDIOLRunner 인스턴스 ---
    # __init__, _generate_models, _generate_agent, _generate_trainer가 모두 호출됩니다.
    print("\n[DEBUG] Instantiating AISLDIOLRunner...")
    runner = AISLDIOLRunner(env, agent_cfg)
    print("✅ [AISLRunner] Runner Class and its components are instantiated successfully!")

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