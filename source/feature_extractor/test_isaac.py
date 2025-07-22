# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to add and simulate on-board sensors for a robot.

We add the following sensors on the quadruped robot, ANYmal-C (ANYbotics):

* USD-Camera: This is a camera sensor that is attached to the robot's base.
* Height Scanner: This is a height scanner sensor that is attached to the robot's base.
* Contact Sensor: This is a contact sensor that is attached to the robot's feet.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/04_sensors/add_sensors_on_robot.py --enable_cameras

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on adding sensors on a robot.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# parser.add_argument("--enable_cameras", type=bool, default=True)
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
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, CollisionPropertiesCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors import CameraCfg, Camera
from isaaclab.utils import configclass
from isaaclab.sensors.camera.utils import create_pointcloud_from_depth, convert_to_torch
from isaaclab.utils.math import quat_mul, unproject_depth, transform_points

FRONT_ROT = (0.61237, 0.35355, 0.35355, 0.61237)
LEFT_ROT = (0.81013, 0.53848, -0.13613, -0.18761)
RIGHT_ROT = (0.18761, 0.13613, -0.53848, -0.81013)

# Experimental known (Configuration -> Simulation Transformation)
QUAT_CON_TO_SIM = (0.5, 0.5, -0.5, -0.5)
# Inverse Quaternion (w, -x, -y, -z)
QUAT_SIM_TO_CON = (0.5, -0.5, 0.5, 0.5)

# Tranformation from setted simulation quat to offset configuration thing
FRONT_ROT_CON = tuple(quat_mul(torch.tensor(FRONT_ROT), torch.tensor(QUAT_SIM_TO_CON)).numpy())
LEFT_ROT_CON = tuple(quat_mul(torch.tensor(LEFT_ROT), torch.tensor(QUAT_SIM_TO_CON)).numpy())
RIGHT_ROT_CON = tuple(quat_mul(torch.tensor(RIGHT_ROT), torch.tensor(QUAT_SIM_TO_CON)).numpy())

BACKGROUND_RGBA = (0, 0, 0, 255)
TABLE_RGBA = (140, 255, 25, 255)
OBJECT_RGBA = (140, 25, 255, 255)
RGBA_MAP = [
    BACKGROUND_RGBA,
    TABLE_RGBA,
    OBJECT_RGBA
]

OBJECT_DIR = {
    "mug_1": {
        "url": "/025_mug.usd",
        "class": "mug"
    },

    "mug_2": {
        "url": "/SM_Mug_B1.usd",
        "class": "mug"
    },

    "mug_3": {
        "url": "/SM_Mug_C1.usd",
        "class": "mug"
    },

    "mug_4": {
        "url": "/SM_Mug_D1.usd",
        "class": "mug"
    },

    "cube_1": {
        "url": "/010_potted_meat_can.usd",
        "class": "cube"
    },

    "cube_2": {
        "url": "/003_cracker_box.usd",
        "class": "cube"
    },

    "cylinder_1": {
        "url": "/002_master_chef_can.usd",
        "class": "cylinder"
    },

    "cylinder_2": {
        "url": "/005_tomato_soup_can.usd",
        "class": "cylinder"
    },
}

@configclass
class SensorsSceneCfg(InteractiveSceneCfg):
    """Design the scene with sensors on the robot."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.7405)),
    )
    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    # Table
    table = AssetBaseCfg(
        prim_path="/World/envs/env/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0], rot=[1.0, 0, 0, 0.0]),
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd",
                                   scale=(1.0, 1.0, 1.0),
                                   semantic_tags=[("class", "table")]
                                   ),
    )

    # object
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0.0, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(os.getcwd(), "Dataset", "mydata") + OBJECT_DIR["mug_2"]["url"],
                collision_props=CollisionPropertiesCfg(collision_enabled=True),
                scale=(0.01, 0.01, 0.01),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                semantic_tags=[("class", "object")]
            ),
        )


    # sensor
    front_camera = CameraCfg(
        prim_path=f"/World/envs/env/FrontCam",
        update_period=0.1,
        height=1024,
        width=1024,
        data_types=["distance_to_image_plane", "normals", "semantic_segmentation"],
        spawn=sim_utils.PinholeCameraCfg(
            # 데이터 수집 시 파라미터와 동일하게 설정
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.954999923706055,
            vertical_aperture=15.290800094604492,  
            clipping_range=(1.0, 1000000.0)
        ),
        offset=CameraCfg.OffsetCfg(pos=(1.6, 0.0, 0.7), rot=FRONT_ROT_CON, convention="world"),
        semantic_segmentation_mapping ={
            "class:table": (140, 255, 25, 255),
            "class:object": (140, 25, 255, 255),
        }
    )

    # # sensor
    left_behind_camera = CameraCfg(
        prim_path="/World/envs/env/Leftcam",
        update_period=0.1,
        height=1024,
        width=1024,
        data_types=["pointcloud"],
        spawn=sim_utils.PinholeCameraCfg(
            # 데이터 수집 시 파라미터와 동일하게 설정
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.954999923706055,
            vertical_aperture=15.290800094604492,  
            clipping_range=(1.0, 1000000.0)   
        ),
        offset=CameraCfg.OffsetCfg(pos=(-0.13, -1.2, 0.5), rot=LEFT_ROT_CON, convention="world"),
        semantic_segmentation_mapping ={
            "class:table": (140, 255, 25, 255),
            "class:object": (140, 25, 255, 255),
        }
    )

    # sensor
    right_behind_camera = CameraCfg(
        prim_path="/World/envs/env/Rightcam",
        update_period=0.1,
        height=1024,
        width=1024,
        data_types=["pointcloud"],
        spawn=sim_utils.PinholeCameraCfg(
            # 데이터 수집 시 파라미터와 동일하게 설정
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.954999923706055,
            vertical_aperture=15.290800094604492,  # 이 값을 명시적으로 추가합니다.
            clipping_range=(1.0, 1000000.0)      # Near/Far 값을 정확히 맞춰줍니다.
        ),
        offset=CameraCfg.OffsetCfg(pos=(-0.13, 1.2, 0.5), rot=RIGHT_ROT_CON, convention="world"),
        semantic_segmentation_mapping ={
            "class:table": (140, 255, 25, 255),
            "class:object": (140, 25, 255, 255),
        }
    )
    



def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the simulator."""
    # Define simulation stepping
    cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/CameraPointCloud")
    cfg.markers["hit"].radius = 0.002
    pc_markers = VisualizationMarkers(cfg)
    front_camera: Camera = scene["front_camera"]
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Simulate physics
    while simulation_app.is_running():
        # Reset
        if count % 500 == 0:
            # reset counter
            count = 0
            default_state = scene["object"].data.default_root_state
            scene["object"].write_root_pose_to_sim(default_state[:, :7])
            scene["object"].write_root_velocity_to_sim(default_state[:, 7:])
            scene.write_data_to_sim()
            scene.reset()
            print("[INFO]: Resetting robot state...")
        
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        scene.update(sim_dt)


        # 각 카메라에 담긴 정보 : (Env, Height, Width, k)
        # depth : (Env, Height, Width, 1)
        # semantic_label : (Env, Height, Width, 4)

        # - valid_mask => depth 기준으로 추출 (1024, 1024, 1) : (1024, 1024, 1) -> (vx, vy, 1)
        # - valid_labels => 
        #   1. RGBA -> Seg label로 변환 : (1024, 1024, 4) -> (1024, 1024, 1)
        #   2. valid_mask 적용하여 valid label 추출 : (1024, 1024, 1) -> (vx, vy, 1)
        #   3. label <---> pointcloud 매핑을 위해 valid_label flatten : (vx, vy, 1) -> (vx * vy, 1)
        # - pointcloud => valid depth 이미지에 대해서 수행 : (vx, vy, 1) -> (vx * vy, 3)
        total_pcd = []
        total_label = []
        print("-" * 40)
        for i in range(3):
            semantic_labels = convert_rgba_to_id(convert_to_torch(front_camera.data.output["semantic_segmentation"][0], 
                                                                    dtype=torch.long, 
                                                                    device=sim.device))
            normal_directions = convert_to_torch(front_camera.data.output["normals"])[0].reshape(-1, 3)

            pointcloud = create_pointcloud_from_depth(
                intrinsic_matrix=front_camera.data.intrinsic_matrices[i],
                depth=front_camera.data.output["distance_to_image_plane"][i],
                position=front_camera.data.pos_w[i],
                orientation=front_camera.data.quat_w_ros[i],
                keep_invalid=True,
                device=sim.device,
            )
            pts_idx_to_keep = torch.all(torch.logical_and(~torch.isnan(pointcloud), ~torch.isinf(pointcloud)), dim=1)
            valid_cloud = pointcloud[pts_idx_to_keep, ...]
            valid_label = semantic_labels[pts_idx_to_keep, ...]
            valid_normal = normal_directions[pts_idx_to_keep, ...]

            if valid_cloud.size()[0] > 0:
                # pc_markers.visualize(translations=valid_cloud)
                print(f"Cam_{i} Valid points 개수 : {valid_cloud.shape[0]}")
        
        print("-" * 40)
        print("\n\n")


def convert_rgba_to_id(rgba_image: torch.Tensor, color_map: list[tuple[int, int, int]]=RGBA_MAP) -> torch.Tensor:
    """
    (H, W, 4) RGBA 시맨틱 이미지를 (H * W) ID 텐서로 효율적으로 변환합니다.

    Args:
        rgba_image (torch.Tensor): (H, W, 4) 형상의 RGBA 이미지. torch.uint8 타입.
        color_map (list[tuple[int, int, int]]): 클래스 ID를 인덱스로 하는 색상(RGB) 리스트.
            예: [(0,0,0), (255,0,0), (0,255,0)] -> ID 0, 1, 2에 해당.

    Returns:
        torch.Tensor: (H * W) 형상의 1D 정수 ID 텐서.
    """

    # 입력 이미지의 크기와 디바이스 정보 가져오기
    height, width, _ = rgba_image.shape
    device = rgba_image.device

    # 최종 ID 맵을 0 (배경 ID)으로 초기화
    # 각 픽셀에 어떤 클래스 ID가 들어갈지 저장할 텐서
    id_map = torch.zeros((height, width), dtype=torch.long, device=device)

    # color_map을 텐서로 변환하여 GPU로 이동
    colors_tensor = torch.tensor(color_map, dtype=torch.uint8, device=device)

    # 클래스 ID 1부터 순회 (0은 배경이므로 이미 처리됨)
    for class_id, color in enumerate(colors_tensor):
        mask = torch.all(rgba_image == color, dim=-1)
        # 배경(ID 0)은 건너뛰어 불필요한 연산 방지
        if class_id == 0:
            print(f"class : None, # of points : {torch.sum(mask)}")
            continue
        
        # 현재 클래스의 색상과 일치하는 모든 픽셀에 대한 마스크 생성
        mask = torch.all(rgba_image == color, dim=-1)
        print(f"class : {class_id}, # of points : {torch.sum(mask)}")

        # 마스크가 True인 위치에 현재 클래스 ID를 할당
        id_map[mask] = class_id

    # (H, W) -> (H * W)로 평탄화하여 반환
    return id_map.flatten()


def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])
    # design scene
    scene_cfg = SensorsSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()