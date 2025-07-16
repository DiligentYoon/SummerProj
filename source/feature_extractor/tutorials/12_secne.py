config = {
    "launch_config": {
        "renderer": "RayTracedLighting",
        "headless": False,
    },
    "resolution": [512, 512],
    "rt_subframes": 16,
    "num_frames": 20,
    "env_url": "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd",
    "writer": "BasicWriter",
    "writer_config": {
        "output_dir": "_out_scene_based_sdg",
        "rgb": True,
        "bounding_box_2d_tight": True,
        "semantic_segmentation": True,
        "distance_to_image_plane": True,
        "bounding_box_3d": True,
        "occlusion": True,
    },
    "clear_previous_semantics": True,
    "forklift": {
        "url": "/Isaac/Props/Forklift/forklift.usd",
        "class": "Forklift",
    },
    "cone": {
        "url": "/Isaac/Environments/Simple_Warehouse/Props/S_TrafficCone.usd",
        "class": "TrafficCone",
    },
    "pallet": {
        "url": "/Isaac/Environments/Simple_Warehouse/Props/SM_PaletteA_01.usd",
        "class": "Pallet",
    },
    "cardbox": {
        "url": "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_04.usd",
        "class": "Cardbox",
    },
    "close_app_after_run": True,
}


from isaacsim import SimulationApp


# Create the simulation app with the given launch_config
simulation_app = SimulationApp(launch_config=config["launch_config"])

# Late import of runtime modules (the SimulationApp needs to be created before loading the modules)
import omni.replicator.core as rep
import omni.usd

# Custom util functions for the example
from isaacsim.core.utils import prims
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.stage import get_current_stage, open_stage
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf