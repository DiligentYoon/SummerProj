# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of an RL agent from skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Franka-Grasp-Direct-v0", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
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
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import copy
import numpy as np
from datetime import datetime

import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.2"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg

import SummerProj.tasks  # noqa: F401
from runner import AISLRunner

from logger import PerEpisodeExcelLogger


# config shortcuts
algorithm = args_cli.algorithm.lower()

def _row_flat(prefix, x):
    """1D로 펴서 {f"{prefix}_{i}": val} dict로 반환"""
    if hasattr(x, "detach"):
        x = x.detach().cpu().view(-1).tolist()
    elif isinstance(x, np.ndarray):
        x = x.reshape(-1).tolist()
    elif isinstance(x, (list, tuple)):
        x = list(x)
    else:
        x = [float(x)]
    return {f"{prefix}_{i}": float(v) for i, v in enumerate(x)}



def main():
    """Play with skrl agent."""
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    try:
        experiment_cfg = load_cfg_from_registry(args_cli.task, f"skrl_{algorithm}_cfg_entry_point")
    except ValueError:
        experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints
    runner = AISLRunner(env, experiment_cfg)
    # runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    runner.agent.set_running_mode("eval")
    # runner.agent.load(resume_path)
    # # set agent to evaluation mode
    # runner.agent.set_running_mode("eval")

    # reset environment
    obs, _ = env.reset()
    timestep = 0

    # ---- set once ----
    gamma = 0.99
    eps_d_place = 0.05   # 거리 경계 (원하는 값으로)
    eps_a_place = 0.10   # 각도 경계 (라디안 기준)
    eps_d_retract = 0.05   # 거리 경계 (원하는 값으로)
    eps_a_retract = 0.10   # 각도 경계 (라디안 기준)
    eps_d_approach = 0.05   # 거리 경계 (원하는 값으로)
    eps_a_approach = 0.10   # 각도 경계 (라디안 기준)

    # 누적 통계 (전이/잔류)
    A_T_sum_p = A_T_cnt_p = 0.0
    A_S_sum_p = A_S_cnt_p = 0.0
    A_T_sum_r = A_T_cnt_r = 0.0
    A_S_sum_r = A_S_cnt_r = 0.0
    A_T_sum_g = A_T_cnt_g = 0.0
    A_S_sum_g = A_S_cnt_g = 0.0

    # 이전 단계의 phase id와 V(s_t)를 저장
    prev_phase_id = None
    prev_V = None
    prev_probe = None  # 이전 스텝의 probe (phase 판정t에 필요)

    # ---- 상수 인코딩 ----
    APPR, GRASP, RETRACT, PLACE = 0, 1, 2, 3

    timestep = 0
    metric_path = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_metrics.xlsx"

    logger = PerEpisodeExcelLogger(path=os.path.join(log_dir, metric_path))
    num_envs = env.num_envs  # 또는 env.num_envs
    ep_id    = np.zeros(num_envs, dtype=np.int64)
    step_in_ep = np.zeros(num_envs, dtype=np.int64)
    while simulation_app.is_running():
        start_time = time.time()

        with torch.inference_mode():
            # 1) 현재 상태에서 V(s_t)
            V_t, _, _ = runner.agent.value.act({"states": runner.agent._state_preprocessor(obs)}, role="value")
            V_t = V_t.squeeze(-1)  # [N] 가정

            # 2) 정책 행동
            outs = runner.agent.act(obs, timestep=0, timesteps=0)
            actions = outs[-1].get("mean_actions", outs[0])  # single-agent 가정

            # 3) 환경 전개 → s_{t+1}, r_t, d_t, info_{t+1}
            obs_next, reward, terminated, truncated, info = env.step(actions)
            done = terminated | truncated

            # 4) V(s_{t+1})
            V_tp1, _, _ = runner.agent.value.act({"states": runner.agent._state_preprocessor(obs_next)}, role="value")
            V_tp1 = V_tp1.squeeze(-1)

            # 5) TD(1) advantage: A_t
            # (배치 [N] 환경 기준, 텐서 연산)
            A_t = reward + gamma * (1.0 - done.float()) * V_tp1 - V_t   # shape [N]

            # 6) 정보 가져오기
            probe_tp1 = copy.deepcopy(info["probe"])
            assert probe_tp1 is not None, "env.extras['probe']가 info로 전달되도록 env를 보강하세요."
            robot_info = info["robot"]
            assert robot_info is not None, "env.extras['robot']가 info로 전달되도록 env를 보강하세요."


            # phase id 인코딩: Approach=0, Grasp=1, Retract=2, Place=3
            # (Place는 is_success 전/후 모두 3으로 취급)
            def encode_phase(p) ->torch.Tensor:
                # 텐서 bool -> long
                is_g = p["is_grasp"].long()
                is_r = p["is_retract"].long()
                is_s = p["is_success"].long()
                # 우선순위: success>retract>grasp>approach
                return torch.where(is_s==1, torch.full_like(is_g, 3),
                    torch.where(is_r==1, torch.full_like(is_g, 2),
                    torch.where(is_g==1, torch.full_like(is_g, 1),
                                torch.zeros_like(is_g))))

            # t시점 phase는 이전 스텝의 probe로부터
            if prev_probe is None:
                # 첫 스텝은 비교 불가 → 다음 스텝부터 집계
                prev_probe = probe_tp1
                obs = obs_next
                # real-time sleep
                if args_cli.real_time:
                    sleep_time = dt - (time.time() - start_time)
                    if sleep_time > 0: time.sleep(sleep_time)
                continue

            # t 시점으로부터 (prev_probe 사용!)
            probe_t  = prev_probe
            phase_t  = encode_phase(probe_t)      # [N]
            phase_tp1= encode_phase(probe_tp1)    # [N]

            # 7) 경계 근처(t 기준)
            near_grasp   = (probe_t["approach_loc"] <= eps_d_approach*2) & (probe_t["approach_rot"] <= eps_a_approach*2)
            near_retract = (probe_t["retract_loc"]  <= eps_d_retract*2)  & (probe_t["retract_rot"]  <= eps_a_retract*2)
            near_place   = (probe_t["place_loc"]    <= eps_d_place*2)    & (probe_t["place_rot"]    <= eps_a_place*2)

            # 8) 전이/잔류 마스크 (방향별로 분해)
            to_grasp      = (phase_t == APPR)    & (phase_tp1 == GRASP)      # APPR -> GRASP (정방향)
            back_from_g   = (phase_t == GRASP)   & (phase_tp1 == APPR)       # GRASP -> APPR (역방향)
            stay_appr     = (phase_t == APPR)    & (phase_tp1 == APPR)

            to_retract    = (phase_t == GRASP)   & (phase_tp1 == RETRACT)    # GRASP -> RETRACT (정방향)
            back_from_r   = (phase_t == RETRACT) & (phase_tp1 == GRASP)      # RETRACT -> GRASP (역방향)
            stay_grasp    = (phase_t == GRASP)   & (phase_tp1 == GRASP)

            to_place      = (phase_t == RETRACT) & (phase_tp1 == PLACE)      # RETRACT -> PLACE (정방향)
            back_from_p   = (phase_t == PLACE)   & (phase_tp1 == RETRACT)    # PLACE -> RETRACT (역방향)
            stay_retract  = (phase_t == RETRACT) & (phase_tp1 == RETRACT)

            print(f"Phase : {phase_tp1}")

            # 9) 경계별로 ΔA 집계
            # Grasp 경계: APPR 근처에서 APPR에 머무름 vs GRASP로 전이
            mask_T_g = near_grasp & to_grasp
            mask_S_g = near_grasp & stay_appr
            if mask_T_g.any():
                A_T_sum_g += A_t[mask_T_g].sum().item()
                A_T_cnt_g += mask_T_g.sum().item()
            if mask_S_g.any():
                A_S_sum_g += A_t[mask_S_g].sum().item()
                A_S_cnt_g += mask_S_g.sum().item()

            # Retract 경계: GRASP 근처에서 GRASP에 머무름 vs RETRACT로 전이
            mask_T_r = near_retract & to_retract
            mask_S_r = near_retract & stay_grasp
            if mask_T_r.any():
                A_T_sum_r += A_t[mask_T_r].sum().item()
                A_T_cnt_r += mask_T_r.sum().item()
            if mask_S_r.any():
                A_S_sum_r += A_t[mask_S_r].sum().item()
                A_S_cnt_r += mask_S_r.sum().item()

            # Place 경계: RETRACT 근처에서 RETRACT에 머무름 vs PLACE로 전이
            mask_T_p = near_place & to_place
            mask_S_p = near_place & stay_retract
            if mask_T_p.any():
                A_T_sum_p += A_t[mask_T_p].sum().item()
                A_T_cnt_p += mask_T_p.sum().item()
            if mask_S_p.any():
                A_S_sum_p += A_t[mask_S_p].sum().item()
                A_S_cnt_p += mask_S_p.sum().item()


            # --- 각 env i에 대해 한 줄씩 로깅 ---
            for i in range(num_envs):
                row = {}

                # --- t+1 시점 ---
                row.update(_row_flat("des_q",   robot_info["impedance_desired_joint_pos"][i]))
                row.update(_row_flat("kp",      robot_info["impedance_stiffness"][i]))
                row.update(_row_flat("zeta",    robot_info["impedance_damping"][i]))
                row.update(_row_flat("q",       robot_info["joint_pos"][i]))
                row.update(_row_flat("dq",      robot_info["joint_vel"][i]))
                row.update(_row_flat("tcpvel",  robot_info["hand_vel"][i]))
                row.update(_row_flat("rewards", robot_info["total_reward"][i]))

                # 로깅
                logger.log_step(env_id=i, ep_id=int(ep_id[i]), step_idx=int(step_in_ep[i]), row_dict=row)

                # done이면 시트로 Flush
                if bool(done[i]):
                    logger.end_episode(env_id=i, ep_id=int(ep_id[i]))
                    ep_id[i] += 1
                    step_in_ep[i] = 0
                else:
                    step_in_ep[i] += 1



            # Next Loops
            prev_probe = probe_tp1
            prev_phase_id = phase_tp1
            prev_V = V_tp1
            obs = obs_next

        # 주기적으로 화면에 ΔA 출력
        timestep += 1
        if timestep % 50 == 0:
            A_T_mean_g = A_T_sum_g / max(1, A_T_cnt_g)
            A_S_mean_g = A_S_sum_g / max(1, A_S_cnt_g)
            print(f"[Grasp boundary] E[A|transition]={A_T_mean_g:.4f}  E[A|stay]={A_S_mean_g:.4f}  Δ={A_T_mean_g - A_S_mean_g:.4f}")

            A_T_mean_r = A_T_sum_r / max(1, A_T_cnt_r)
            A_S_mean_r = A_S_sum_r / max(1, A_S_cnt_r)
            print(f"[Retract boundary] E[A|transition]={A_T_mean_r:.4f}  E[A|stay]={A_S_mean_r:.4f}  Δ={A_T_mean_r - A_S_mean_r:.4f}")

            A_T_mean_p = A_T_sum_p / max(1, A_T_cnt_p)
            A_S_mean_p = A_S_sum_p / max(1, A_S_cnt_p)
            print(f"[Place boundary] E[A|transition]={A_T_mean_p:.4f}  E[A|stay]={A_S_mean_p:.4f}  Δ={A_T_mean_p - A_S_mean_p:.4f}")


        # 비디오/리얼타임 처리
        if args_cli.video:
            if timestep == args_cli.video_length:
                break
        if args_cli.real_time:
            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    # 최종 ΔA 출력
    A_T_mean_p = A_T_sum_p / max(1, A_T_cnt_p)
    A_S_mean_p = A_S_sum_p / max(1, A_S_cnt_p)
    print(f"[FINAL] ΔA (Place boundary) = {A_T_mean_p - A_S_mean_p:.4f}  "
        f"(transition={A_T_mean_p:.4f}, stay={A_S_mean_p:.4f}")
    
    A_T_mean_r = A_T_sum_r / max(1, A_T_cnt_r)
    A_S_mean_r = A_S_sum_r / max(1, A_S_cnt_r)
    print(f"[FINAL] ΔA (Retract boundary) = {A_T_mean_r - A_S_mean_r:.4f}  "
        f"(transition={A_T_mean_r:.4f}, stay={A_S_mean_r:.4f}")
    
    A_T_mean_g = A_T_sum_g / max(1, A_T_cnt_g)
    A_S_mean_g = A_S_sum_g / max(1, A_S_cnt_g)
    print(f"[FINAL] ΔA (Grasp boundary) = {A_T_mean_g - A_S_mean_g:.4f}  "
        f"(transition={A_T_mean_g:.4f}, stay={A_S_mean_g:.4f}")
    

    # close the simulator
    env.close()
    logger.close()
    print("Excel 저장 완료:", logger.path)

    # # simulate environment
    # while simulation_app.is_running():
    #     start_time = time.time()

    #     # run everything in inference mode
    #     with torch.inference_mode():
    #         # agent stepping
    #         outputs = runner.agent.act(obs, timestep=0, timesteps=0)
    #         values, _, _ = runner.agent.value.act({"states": runner.agent._state_preprocessor(obs)}, role="value")
    #         v_t = values.squeeze(-1)
    #         # - multi-agent (deterministic) actions
    #         if hasattr(env, "possible_agents"):
    #             actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
    #         # - single-agent (deterministic) actions
    #         else:
    #             actions = outputs[-1].get("mean_actions", outputs[0])
    #         # env stepping
    #         obs, _, _, _, info = env.step(actions)
    #     if args_cli.video:
    #         timestep += 1
    #         # exit the play loop after recording one video
    #         if timestep == args_cli.video_length:
    #             break

    #     # time delay for real-time evaluation
    #     sleep_time = dt - (time.time() - start_time)
    #     if args_cli.real_time and sleep_time > 0:
    #         time.sleep(sleep_time)

    # # close the simulator
    # env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
