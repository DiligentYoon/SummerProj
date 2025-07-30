from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from .franka_base_env_diol_cfg import FrankaBaseDIOLEnvCfg


# Pre-defined Rotation by OpenGL convention
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
}


@configclass
class FrankaVisionBaseCfg(FrankaBaseDIOLEnvCfg):
    
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16, env_spacing=3.0, replicate_physics=False)

    # Multi object (Total 6 Objects)
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.67, 0.0, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                sim_utils.UsdFileCfg(
                    usd_path=os.path.join(os.getcwd(), "Dataset", "mydata") + OBJECT_DIR["mug_1"]["url"],
                    scale=(0.01, 0.01, 0.01)
                ),
                sim_utils.UsdFileCfg(
                    usd_path=os.path.join(os.getcwd(), "Dataset", "mydata") + OBJECT_DIR["mug_2"]["url"],
                    scale=(0.01, 0.01, 0.01)
                ),
                sim_utils.UsdFileCfg(
                    usd_path=os.path.join(os.getcwd(), "Dataset", "mydata") + OBJECT_DIR["cube_1"]["url"],
                    scale=(0.01, 0.01, 0.01)
                ),
                sim_utils.UsdFileCfg(
                    usd_path=os.path.join(os.getcwd(), "Dataset", "mydata") + OBJECT_DIR["cube_2"]["url"],
                    scale=(0.01, 0.01, 0.01)
                ),
                sim_utils.UsdFileCfg(
                    usd_path=os.path.join(os.getcwd(), "Dataset", "mydata") + OBJECT_DIR["cylinder_1"]["url"],
                    scale=(0.01, 0.01, 0.01)
                ),
                sim_utils.UsdFileCfg(
                    usd_path=os.path.join(os.getcwd(), "Dataset", "mydata") + OBJECT_DIR["cylinder_2"]["url"],
                    scale=(0.01, 0.01, 0.01)
                ),
            ],
            random_choice=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                    ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            semantic_tags=[("class", "object")]
            ),
        )


    # sensor
    front_camera = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Cam",
        update_period=0,
        height=240,
        width=320,
        data_types=["distance_to_image_plane", "semantic_segmentation"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1000000.0)
        ),
        offset=TiledCameraCfg.OffsetCfg(pos=(1.5, 0.0, 0.75), rot=FRONT_ROT, convention="opengl"),
        colorize_semantic_segmentation=False
    )

    # sensor
    left_behind_camera = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Leftcam",
        update_period=0,
        height=240,
        width=320,
        data_types=["distance_to_image_plane", "semantic_segmentation"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.954999923706055,
            clipping_range=(0.1, 1000000.0)  
        ),
        offset=TiledCameraCfg.OffsetCfg(pos=(0.26, -0.76, 0.848), rot=LEFT_ROT, convention="opengl"),
        colorize_semantic_segmentation=False
    )

    # sensor
    right_behind_camera = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Rightcam",
        update_period=0.1,
        height=240,
        width=320,
        data_types=["distance_to_image_plane", "semantic_segmentation"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.954999923706055,
            clipping_range=(1.0, 1000000.0)      
        ),
        offset=TiledCameraCfg.OffsetCfg(pos=(0.22, 0.76, 0.9), rot=RIGHT_ROT, convention="opengl"),
        colorize_semantic_segmentation=False
    )

    num_cam: int = 3
