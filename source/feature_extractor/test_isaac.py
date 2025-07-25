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

import torch.random
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
import importlib

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, CollisionPropertiesCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.sensors import CameraCfg, Camera
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
from isaaclab.utils import configclass
from isaaclab.sensors.camera.utils import create_pointcloud_from_depth, convert_to_torch
from isaaclab.utils.math import quat_mul, unproject_depth, transform_points

from model.pointnet2 import get_model

FRONT_ROT = (0.65004, 0.26831, 0.26532, 0.65959)
LEFT_ROT = (0.8966, 0.3504, -0.0986, -0.2523)
RIGHT_ROT = (-0.27194, -0.11369, 0.3686, 0.88163)

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


# --- 인수 파싱 (기존과 동일) ---
def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet2', help='model name')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during testing')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
    parser.add_argument('--npoint', type=int, default=2400, help='Point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='Experiment root directory')
    parser.add_argument('--model_path', type=str, required=True, help='Path to a pre-trained model (.pth file)')
    parser.add_argument('--gif_dir', type=str, default='gif_results', help='Directory to save visualization gifs')
    return parser.parse_args()



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

    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=12, solver_velocity_iteration_count=0),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,
            "panda_finger_joint.*": 0.04,
        }
        ))
    # Impedance Controller를 사용하는 경우, 액추에이터 PD제어 모델 사용 X (중복 토크 계산)
    # 액추에이터에 Impedance Controller가 붙음으로써 최하단 제어기의 역할을 하게 되는 개념.
    robot.actuators["panda_shoulder"].stiffness = 0.0
    robot.actuators["panda_shoulder"].damping = 0.0
    robot.actuators["panda_forearm"].stiffness = 0.0
    robot.actuators["panda_forearm"].damping = 0.0


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
    



def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, args):
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

    # ID 뽑아두기
    table_id = None
    object_id = None
    semantic_info = cam_list[0].data.info[0]["semantic_segmentation"]["idToLabels"]
    for key, value in semantic_info.items():
        if value.get('class') == 'object':
            object_id = key  
        elif value.get('class') == 'table':
            table_id = key

        if object_id is not None and table_id is not None:
            break

    # Model
    '''MODEL LOADING'''
    classifier = get_model(2).cuda()
    checkpoint = torch.load(args.model_path)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.eval()

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
        all_clouds = []
        all_labels = []
        all_normals = []
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
            valid_mask, mapped_labels = extract_valid_mask(pointcloud, semantic_labels, object_id, table_id)
            if valid_mask is not None:
                valid_cloud = pointcloud[valid_mask, ...]
                valid_label = mapped_labels[valid_mask, ...]
                valid_normal = normal_directions[valid_mask, ...]
                all_clouds.append(valid_cloud)
                all_labels.append(valid_label)
                all_normals.append(valid_normal)

                if torch.sum(valid_label) != torch.sum(mapped_labels):
                      print(f"분류 오류")
                
        if all_clouds:
            total_clouds = torch.cat(all_clouds, dim=0)
            total_labels = torch.cat(all_labels, dim=0)
            total_normals = torch.cat(all_normals, dim=0)
        else:
            total_clouds = torch.empty((0, 3), device=sim.device)
            total_labels = torch.empty((0,), dtype=torch.long, device=sim.device)
            total_normals = torch.empty((0, 3), device=sim.device)
            
                
        sampled_points, sampled_labels, sampled_normals = uniform_sampling_for_pointnet(total_clouds, total_labels, total_normals)

        if total_clouds.shape[0] > 0:
            pc_markers.visualize(translations=sampled_points)
        
        with torch.no_grda():
            # (N, 4+3=7)
            inputs = torch.cat([sampled_points, sampled_normals, sampled_labels], dim=-1)
            # (N, 7) -> (1, N, 7) -> (1, 7, N)
            inputs = inputs.squeeze(0).transpose(2, 1)
            outputs, _ = classifier(inputs)
            pred_labels = torch.argmax(outputs, 2).squeeze()


def extract_valid_mask(pointcloud: torch.Tensor, label: torch.Tensor, object_id: str, table_id: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    유효한 포인트 클라우드 마스크를 추출합니다.
    1. 포인트 좌표에 NaN이나 Inf가 없어야 합니다.
    2. 레이블이 Table 또는 Object여야 합니다.
    """
    mapped_labels = torch.zeros_like(label, dtype=label.dtype, device=label.device)
    if object_id is not None and table_id is not None:
        object_label_mask = (label == int(object_id))
        mapped_labels[object_label_mask] = 1
        valid_mask = (label == int(object_id)) | (label == int(table_id))
    else:
        valid_mask = None
    
    return valid_mask, mapped_labels


def uniform_sampling_for_pointnet(pointcloud: torch.Tensor,
                                  label: torch.Tensor,
                                  normals: torch.Tensor,
                                  num_obj=800, num_bg=1600) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    # 포인트가 전혀 없는 경우, 빈 텐서를 즉시 반환
    if pointcloud.shape[0] == 0:
        return (torch.empty((0, 3), device=pointcloud.device),
                torch.empty((0,), dtype=label.dtype, device=label.device),
                torch.empty((0, 3), device=normals.device))

    obj_ids = torch.where(label == 1)[0]
    bg_ids = torch.where(label != 1)[0]
    
    sampled_obj_ids = torch.empty((0,), dtype=torch.long, device=pointcloud.device)
    sampled_bg_ids = torch.empty((0,), dtype=torch.long, device=pointcloud.device)

    # --- Object 샘플링 (안정성 강화) ---
    if len(obj_ids) > 0:
        if len(obj_ids) >= num_obj: # 포인트가 충분하면 비복원 추출
            sampled_obj_ids = obj_ids[torch.randperm(len(obj_ids))[:num_obj]]
        else: # 포인트가 부족하면 복원 추출로 개수를 맞춤
            indices_to_sample = torch.randint(0, len(obj_ids), (num_obj,), device=pointcloud.device)
            sampled_obj_ids = obj_ids[indices_to_sample]

    # --- Background 샘플링 (안정성 강화) ---
    if len(bg_ids) > 0:
        if len(bg_ids) >= num_bg: # 포인트가 충분하면 비복원 추출
            sampled_bg_ids = bg_ids[torch.randperm(len(bg_ids))[:num_bg]]
        else: # 포인트가 부족하면 복원 추출
            indices_to_sample = torch.randint(0, len(bg_ids), (num_bg,), device=pointcloud.device)
            sampled_bg_ids = bg_ids[indices_to_sample]

    # 최종 셔플링 로직은 동일
    cat_sampled_ids = torch.cat((sampled_obj_ids, sampled_bg_ids))
    
    # 합쳐진 인덱스가 없는 경우 처리
    if len(cat_sampled_ids) == 0:
        return (torch.empty((0, 3), device=pointcloud.device),
                torch.empty((0,), dtype=label.dtype, device=label.device),
                torch.empty((0, 3), device=normals.device))

    sampled_ids = cat_sampled_ids[torch.randperm(len(cat_sampled_ids))]

    return pointcloud[sampled_ids, ...], label[sampled_ids, ...], normals[sampled_ids, ...]


# def uniform_sampling_for_pointnet(pointcloud: torch.Tensor, 
#                                   label: torch.Tensor, 
#                                   normals: torch.Tensor,
#                                   num_obj=800, num_bg = 1600) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

#     # label 기반으로 obj, bg index 분류 -> index 자체 샘플링후 매핑
#     obj_ids = torch.where(label==1)[0]
#     bg_ids = torch.where(label!=1)[0]

#     sampled_obj_ids = obj_ids[torch.randperm(len(obj_ids))[:num_obj]]
#     sampled_bg_ids = bg_ids[torch.randperm(len(bg_ids))[:num_bg]]

#     cat_sampled_ids = torch.cat((sampled_obj_ids, sampled_bg_ids))

#     sampled_ids = cat_sampled_ids[torch.randperm(len(cat_sampled_ids))]

#     return pointcloud[sampled_ids, ...], label[sampled_ids, ...], normals[sampled_ids, ...]




def main():
    """Main function."""

    # Initialize the simulation context
    args = parse_args()
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
    run_simulator(sim, scene, args)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()