# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to spawn multiple rigid objects from a local directory
into the scene using the Isaac Lab framework.
It finds all USD files in a structured directory and arranges them in a grid.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/aisl/spawn_google_objects.py

"""

import os
import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Spawns multiple USD objects from a local directory.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext


def design_scene(loop_index):
    """Designs the scene by spawning objects from a local directory."""
    # ----------------- 주요 설정 -----------------
    # 1. USD 파일들이 있는 상위 폴더 경로를 지정하세요.
    BASE_PATH = "C:/Users/AISL/SummerProj/Dataset/google_objects_usd"

    # 2. 물체를 배치할 격자(grid) 설정을 조정하세요.
    SPACING = 1.0  # 물체 사이의 간격
    NUM_COLUMNS = 6   # 한 줄에 배치할 물체의 개수
    # -------------------------------------------

    # Ground-plane
    # if loop_index == 0:
    plane_cfg = sim_utils.GroundPlaneCfg()
    plane_cfg.func("/World/defaultGroundPlane", plane_cfg)
    
    # Lights
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9))
    light_cfg.func("/World/Light", light_cfg)
    

    # 생성된 객체들을 저장할 딕셔너리
    spawned_objects = {}
    
    # 격자 배치를 위한 위치 인덱스
    x_idx = 0
    y_idx = 0
    per_loop_obj = 30

    forder_name = sorted(os.listdir(BASE_PATH))

    # 지정된 경로가 존재하는지 확인
    if not os.path.exists(BASE_PATH):
        print(f"[ERROR] The specified path does not exist: {BASE_PATH}")
        return spawned_objects

    print(f"[INFO] Searching for USD files in: {BASE_PATH}")
    # BASE_PATH 내의 모든 하위 폴더를 순회
    loop_forder_name = forder_name[loop_index * per_loop_obj : (loop_index+1) * per_loop_obj]
    for folder_name in loop_forder_name:
        folder_path = os.path.join(BASE_PATH, folder_name)

        if os.path.isdir(folder_path):
            usd_file_path = os.path.join(folder_path, f"{folder_name}.usd")

            if os.path.exists(usd_file_path):
                # 격자 내에서 현재 물체의 위치를 계산 (cm 단위)
                pos_x = x_idx * SPACING
                pos_y = y_idx * SPACING
                pos_z = 0.1 # 바닥에서 살짝 띄워서 스폰
                
                # 객체 스폰을 위한 설정(Configuration) 생성
                object_cfg = RigidObjectCfg(
                    prim_path=f"/World/Objects/item_{folder_name}", # 씬 내의 고유 경로
                    spawn=sim_utils.UsdFileCfg(
                        usd_path=usd_file_path,
                        scale=(1.0, 1.0, 1.0), # 스케일이 작을 경우 100배 (cm단위로 변환)
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=(pos_x, pos_y, pos_z) # 초기 위치 설정
                    ),
                )
                # 설정으로부터 RigidObject 객체 생성
                spawned_objects[folder_name] = RigidObject(cfg=object_cfg)
                print(f"success to spawn : {folder_name}")
                

                # 다음 물체 위치를 위해 인덱스를 업데이트
                x_idx += 1
                if x_idx >= NUM_COLUMNS:
                    x_idx = 0
                    y_idx += 1
    
    return spawned_objects


def main():
    """Main function."""
    # Isaac Lab의 시뮬레이션 컨텍스트 초기화
    loop_index = 8
    sim = SimulationContext(
        sim_utils.SimulationCfg(
            dt=1.0 / 120.0, # 시뮬레이션 스텝 시간
            device=args_cli.device,
            )
        )
    
    # 메인 카메라 위치 설정
    sim.set_camera_view(eye=[1.5, 0.0, 2.0], target=[0.0, 0.0, 0.0])
    
    # 씬 디자인 함수 호출
    design_scene(loop_index)
    
    # 시뮬레이션 시작
    sim.reset()
    print("[INFO] Scene setup complete. Running simulation...")

    # 사용자가 창을 닫을 때까지 시뮬레이션 루프 실행
    step = 0
    while simulation_app.is_running():
        # 시뮬레이션 한 스텝 진행
        # if step % 1000 == 0:
        #     design_scene(loop_index)
        #     sim.reset()
        #     print("[INFO] Scene setup complete. Running simulation...")
        #     loop_index += 1

        # step += 1
        sim.step()


if __name__ == "__main__":
    # 메인 함수 실행
    main()
    # 시뮬레이션 앱 종료
    simulation_app.close()