# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script spawns a predefined list of rigid objects from the Isaac Sim Nucleus server.
The objects are spawned as dynamic rigid bodies with physics properties.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/aisl/spawn_nucleus_rigid_bodies.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on spawning rigid bodies from Nucleus into the scene.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
# -- 수정된 부분: RigidObject 관련 클래스 임포트 --
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


def design_scene():
    """Designs the scene by spawning rigid bodies from a predefined list."""
    # Ground-plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # spawn distant light
    cfg_light_distant = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.8, 0.8, 0.8))
    cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(1, 0, 10))

    # create a new xform prim for all objects to be spawned under
    prim_utils.create_prim("/World/Objects", "Xform")

    # --- 스폰할 객체 목록 정의 ---
    BASE_PATH = "C:/Users/AISL/SummerProj/Dataset/google_objects_usd"

    ASSET_LIST = {
        "Mug": [
            "025_mug/025_mug.usd",
            "Room_Essentials_Mug_White_Yellow/Room_Essentials_Mug_White_Yellow.usd",
        ],
        "Cube": [
            "010_potted_meat_can/010_potted_meat_can.usd",
            "003_cracker_box/003_cracker_box.usd",
            "004_sugar_box/004_sugar_box.usd",
            "008_pudding_box/008_pudding_box.usd",
        ],
        "Cylinder": [
            "006_mustard_bottle/006_mustard_bottle.usd",
            "021_bleach_cleanser/021_bleach_cleanser.usd",
            "005_tomato_soup_can/005_tomato_soup_can.usd",
            "Saccharomyces_Boulardii_MOS_Value_Size/Saccharomyces_Boulardii_MOS_Value_Size.usd",
        ],
    }

    # --- 격자 배치 설정 ---
    SPACING = 0.5
    NUM_COLUMNS = 5
    total_counter = 0

    # 생성된 객체들을 저장할 딕셔너리
    spawned_objects = {}

    # 모든 카테고리와 경로를 순회하며 객체를 스폰
    for category, paths in ASSET_LIST.items():
        prim_utils.create_prim(f"/World/Objects/{category}", "Xform")
        for asset_path in paths:
            pos_x = (total_counter % NUM_COLUMNS) * SPACING
            pos_y = (total_counter // NUM_COLUMNS) * SPACING
            pos_z = 2.0  # 중력을 받아 떨어질 수 있도록 더 높이 스폰

            prim_name = f"item_{os.path.splitext(os.path.basename(asset_path))[0]}"
            prim_path = f"/World/Objects/{category}/{prim_name}"
            
            # -- 수정된 부분: RigidObjectCfg를 사용하여 객체 설정 --
            # 객체 스폰을 위한 설정(Configuration) 생성
            object_cfg = RigidObjectCfg(
                prim_path=prim_path,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(pos_x, pos_y, pos_z)),
                # 'spawn' 객체 내부에 형상과 관련된 물리 속성을 정의합니다.
                spawn=sim_utils.UsdFileCfg(
                    usd_path=f"{BASE_PATH}/{asset_path}",
                    # 충돌 활성화를 위한 설정
                    # mass_props= sim_utils.MassPropertiesCfg(mass=0.1),
                    # collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                    # 물리 재질 설정 (마찰 등)
                    rigid_props=RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=1,
                        max_angular_velocity=1000.0,
                        max_linear_velocity=1000.0,
                        max_depenetration_velocity=5.0,
                        disable_gravity=False,
                    ),
                ),
            )

            # 설정으로부터 RigidObject 객체 생성 및 딕셔너리에 저장
            spawned_objects[prim_name] = RigidObject(cfg=object_cfg)
            print(f"success to spawn : {prim_name}")
            # ---------------------------------------------

            total_counter += 1

    print(f"[INFO] Spawned a total of {total_counter} rigid bodies from Nucleus.")
    # 생성된 객체들을 반환하여 main 함수 등에서 사용할 수 있도록 함
    return spawned_objects


def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.5, 0.5, 0.0])

    # 씬 디자인 함수를 호출하여 객체들을 씬에 추가
    design_scene()

    sim.reset()
    print("[INFO]: Setup complete...")

    while simulation_app.is_running():
        sim.step()


if __name__ == "__main__":
    main()
    simulation_app.close()