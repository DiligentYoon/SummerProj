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
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.sensors import CameraCfg, Camera, Camera, CameraCfg
from isaaclab.utils import configclass
from isaaclab.sensors.camera.utils import create_pointcloud_from_depth, convert_to_torch
from isaaclab.utils.math import quat_mul, unproject_depth, transform_points

FRONT_ROT = (0.65004, 0.26831, 0.26532, 0.65959)
LEFT_ROT = (0.8966, 0.3504, -0.0986, -0.2523)
RIGHT_ROT = (-0.27194, -0.11369, 0.3686, 0.88163)

# Experimental known (Configuration -> Simulation Transformation)
QUAT_CON_TO_SIM = (0.5, 0.5, -0.5, -0.5)
# Inverse Quaternion (w, -x, -y, -z)
QUAT_SIM_TO_CON = (0.5, -0.5, 0.5, 0.5)

# Tranformation from setted simulation quat to offset configuration thing
FRONT_ROT_CON = tuple(quat_mul(torch.tensor(FRONT_ROT), torch.tensor(QUAT_SIM_TO_CON)).numpy())
LEFT_ROT_CON = tuple(quat_mul(torch.tensor(LEFT_ROT), torch.tensor(QUAT_SIM_TO_CON)).numpy())
RIGHT_ROT_CON = tuple(quat_mul(torch.tensor(RIGHT_ROT), torch.tensor(QUAT_SIM_TO_CON)).numpy())

BACKGROUND_RGBA = (0, 0, 0, 255)
TABLE_RGBA = (0, 255, 0, 255)    # 순수한 녹색
OBJECT_RGBA = (0, 0, 255, 255)   # 순수한 파란색
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

    "table":{
        "url": "/table.usd"
    },

    "stand": {
        "url": "/stand.usd"
    }
}

@configclass
class SensorsSceneCfg(InteractiveSceneCfg):
    """Design the scene with sensors on the robot."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(semantic_tags=[("class", "ground")]),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.612)),
    )
    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    # Table
    table = AssetBaseCfg(
        prim_path="/World/envs/env/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.67, 0.0, -0.3], rot=[1.0, 0, 0, 0.0]),
        spawn=sim_utils.UsdFileCfg(usd_path=os.path.join(os.getcwd(), "Dataset", "mydata") + OBJECT_DIR["table"]["url"],
                                   collision_props=CollisionPropertiesCfg(collision_enabled=True),
                                   scale=(0.7, 1.0, 0.6),
                                   semantic_tags=[("class", "table")]
                                   ),
    )

    # object
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.67, 0.0, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
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
    
    # stand
    stand = AssetBaseCfg(
        prim_path="/World/envs/env/stand",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0], rot=[1.0, 0, 0, 0.0]),
        spawn=sim_utils.UsdFileCfg(usd_path=os.path.join(os.getcwd(), "Dataset", "mydata") + OBJECT_DIR["stand"]["url"],
                                   scale=(1.2, 1.2, 1.2),
                                   ),
    )


    # sensor
    front_camera = CameraCfg(
        prim_path="/World/envs/env/Cam",
        update_period=0,
        height=360,
        width=480,
        data_types=["distance_to_image_plane", "normals", "semantic_segmentation"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1000000.0)
        ),
        offset=CameraCfg.OffsetCfg(pos=(1.5, 0.0, 0.75), rot=FRONT_ROT, convention="opengl"),
        colorize_semantic_segmentation=False
    )

    # sensor
    left_behind_camera = CameraCfg(
        prim_path="/World/envs/env/Leftcam",
        update_period=0,
        height=360,
        width=480,
        data_types=["distance_to_image_plane", "normals", "semantic_segmentation"],
        # data_types = ["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            # 데이터 수집 시 파라미터와 동일하게 설정
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.954999923706055,
            clipping_range=(0.1, 1000000.0)  
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.26, -0.76, 0.848), rot=LEFT_ROT, convention="opengl"),
        colorize_semantic_segmentation=False
    )

    # sensor
    right_behind_camera = CameraCfg(
        prim_path="/World/envs/env/Rightcam",
        update_period=0.1,
        height=360,
        width=480,
        data_types=["distance_to_image_plane", "normals", "semantic_segmentation"],
        # data_types = ["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.954999923706055,
            clipping_range=(1.0, 1000000.0)      
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.22, 0.76, 0.9), rot=RIGHT_ROT, convention="opengl"),
        colorize_semantic_segmentation=False
    )
    



def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the simulator."""
    # Define simulation stepping
    cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/CameraPointCloud")
    cfg.markers["hit"].radius = 0.002
    pc_markers = VisualizationMarkers(cfg)
    cam_list: list[Camera, Camera, Camera] = [scene["front_camera"], scene["left_behind_camera"], scene["right_behind_camera"]]
    # cam_list: list[Camera] = [scene["right_behind_camera"]]
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Simulate physics
    first = True
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

            object_id = None
            semantic_info = cam_list[0].data.info[0]["semantic_segmentation"]["idToLabels"]
            for key, value in semantic_info.items():
                # 내부 딕셔너리의 'class' 값이 'object'와 일치하는지 확인합니다.
                if value.get('class') == 'object':
                    object_id = key  # 일치하면 해당 키를 저장하고
                    break            # 반복을 중단합니다.

            print("[INFO]: Resetting robot state...")

        
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        for cam in cam_list:
            cam.update(sim_dt)
        scene.update(sim_dt)
        
        # 각 카메라에 담긴 정보 : (Env, Height, Width, k)
        # depth : (Env, Height, Width, 1)
        # semantic_label : (Env, Height, Width, 4)
        # normal : (Env, Height, Width, 3)
        total_clouds = None
        total_labels = None
        total_normals = None
        # print("-" * 40)
        for i, cam in enumerate(cam_list):
            #   1.RGBA -> Seg label로 변환 : (H, W, 4) -> (H * W, 1)
            semantic_info = cam.data.info[0]["semantic_segmentation"]["idToLabels"]
            semantic_labels = convert_to_torch(cam.data.output["semantic_segmentation"][0])
            transposed_labels = semantic_labels.transpose(0, 1)
            semantic_labels = transposed_labels.flatten()
                                                                  
            #   2. Normal Vector : (H * W, 3)
            normal_directions = convert_to_torch(cam.data.output["normals"])[0].reshape(-1, 3)

            #   3. Point Cloud : (H * W, 3)
            pointcloud = create_pointcloud_from_depth(
                intrinsic_matrix=cam.data.intrinsic_matrices[0],
                depth=cam.data.output["distance_to_image_plane"][0],
                position=cam.data.pos_w[0],
                orientation=cam.data.quat_w_ros[0],
                keep_invalid=True,
                device=sim.device,
            )

            #  4. Validation Mask : (H * W, 1)
            valid_mask = extract_valid_mask(pointcloud, semantic_labels, object_id)
            if valid_mask is not None:
                valid_cloud = pointcloud[valid_mask, ...]
                valid_label = semantic_labels[valid_mask, ...]
                valid_normal = normal_directions[valid_mask, ...]
                
                if total_clouds is None:
                    total_clouds = valid_cloud
                    total_labels = valid_label
                    total_normals = valid_normal
                else:
                    total_clouds = torch.concat((total_clouds, valid_cloud), dim=0)
                    total_labels = torch.concat((total_labels, valid_label), dim=0)
                    total_normals = torch.concat((total_normals, valid_normal), dim=0)
                
                print(f"Cam_{i} --> Valid points 개수 : {valid_cloud.shape[0]}")
        
        # # print(f"총 Valid Points 개수 : {total_clouds.shape[0]}")
        if total_clouds.shape[0] > 0:
            pc_markers.visualize(translations=total_clouds)
        # pc_markers.visualize(translations=pointcloud)
        # print("-" * 40)
        # print("\n\n")


def convert_rgba_to_id(rgba_image: torch.Tensor, color_map: list[tuple[int, int, int, int]]) -> torch.Tensor:
    """
    (H, W, 4) RGBA 시맨틱 이미지를 (H * W) ID 텐서로 효율적으로 변환합니다.
    이 함수는 for 루프 대신 벡터화된 연산을 사용하여 매우 빠릅니다.

    Args:
        rgba_image (torch.Tensor): (H, W, 4) 형상의 RGBA 이미지. torch.uint8 타입.
        color_map (list[tuple[int, int, int, int]]): 클래스 ID를 인덱스로 하는 색상(RGBA) 리스트.

    Returns:
        torch.Tensor: (H * W) 형상의 1D 정수 ID 텐서.
    """
    height, width, _ = rgba_image.shape
    device = rgba_image.device

    # 컬러맵을 (C, 4) 텐서로 변환 (C는 클래스 수)
    colors_tensor = torch.tensor(color_map, dtype=torch.uint8, device=device)

    # (H, W, 4) 이미지를 (H, W, 1, 4)로 차원 확장
    # (H, W, 1, 4)와 (C, 4)를 브로드캐스팅하여 비교 -> 결과는 (H, W, C, 4)
    comparison = (rgba_image.unsqueeze(2) == colors_tensor)

    # 마지막 채널(RGBA) 차원에 대해 모두 일치하는지 확인 -> 결과는 (H, W, C)
    mask = torch.all(comparison, dim=-1)

    # 각 픽셀에 대해 True인 첫 번째 클래스 ID를 찾음 (C 차원에서 argmax)
    # .long()을 통해 boolean 텐서를 0과 1로 변환
    id_map = torch.argmax(mask.long(), dim=-1)

    # (H, W) -> (H * W)로 평탄화하여 반환
    return id_map.flatten()

# 사용 예시
# semantic_labels = convert_rgba_to_id_vectorized(rgba_image_tensor, RGBA_MAP)


def extract_valid_mask(pointcloud: torch.Tensor, label: torch.Tensor, object_id: str) -> torch.Tensor:
    """
    유효한 포인트 클라우드 마스크를 추출합니다.
    1. 포인트 좌표에 NaN이나 Inf가 없어야 합니다.
    2. 레이블이 1(Table) 또는 2(Object)여야 합니다.
    """
    # 1. torch.isfinite()를 사용하여 NaN과 Inf를 한 번에 효율적으로 확인합니다.
    #    pointcloud의 모든 좌표(x, y, z)가 유한한 수인지 검사합니다.
    # pcd_valid = torch.all(torch.isfinite(pointcloud), dim=-1)

    # 2. 불필요한 텐서 생성을 제거하고 간결하게 비교합니다.
    #    '|' 연산자는 tensor에서 torch.logical_or와 동일하게 작동합니다.
    # label_valid = (flat_labels == 2) | (flat_labels == 3)
    if object_id is not None:
        label_valid = label == int(object_id)
    else:
        label_valid = None
    
    # 3. 두 마스크를 AND 연산하여 최종 결과를 반환합니다.
    return label_valid




def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.1, device=args_cli.device)
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