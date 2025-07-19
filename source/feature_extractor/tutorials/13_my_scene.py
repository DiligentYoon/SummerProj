
import argparse
import json
import math
import os
import random
import numpy as np

import yaml
from isaacsim import SimulationApp

config = {
    "launch_config": {
        "renderer": "RayTracedLighting",
        "headless": False,
    },
    "resolution": [512, 512],
    "rt_subframes": 24,
    "num_frames": 10,
    "env_url": "/Isaac/Environments/Terrains/flat_plane.usd",
    "writer": "BasicWriter",
    "writer_config": {
        "output_dir": "Dataset",
        "rgb": True,
        "semantic_segmentation": True,
        "pointcloud": True,                         
    },
    "clear_previous_semantics": True,

    "mug_1": {
        "url": "/Isaac/Props/YCB/Axis_Aligned/025_mug.usd",
        "class": "mug"
    },

    "mug_2": {
        "url": "/Isaac/Props/Mugs/SM_Mug_B1.usd",
        "class": "mug"
    },

    "mug_3": {
        "url": "/Isaac/Props/Mugs/SM_Mug_C1.usd",
        "class": "mug"
    },

    "mug_4": {
        "url": "/Isaac/Props/Mugs/SM_Mug_D1.usd",
        "class": "mug"
    },

    "cube_1": {
        "url": "/Isaac/Props/YCB/Axis_Aligned/010_potted_meat_can.usd",
        "class": "cube"
    },

    "cube_2": {
        "url": "/Isaac/Props/YCB/Axis_Aligned/003_cracker_box.usd",
        "class": "cube"
    },

    "cylinder_1": {
        "url": "/Isaac/Props/YCB/Axis_Aligned/002_master_chef_can.usd",
        "class": "cylinder"
    },

    "cyliner_2": {
        "url": "/Isaac/Props/YCB/Axis_Aligned/005_tomato_soop_can.usd",
        "class": "cylinder"
    },

    "table": {
        "url": "/Isaac/Props/Mounts/ThorlabsTable/table_instanceable.usd",
        "class": "Table"
    },

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


# Check if there are any config files (yaml or json) are passed as arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=False, help="Include specific config parameters (json or yaml))")
args, unknown = parser.parse_known_args()
args_config = {}
if args.config and os.path.isfile(args.config):
    print("File exist")
    with open(args.config, "r") as f:
        if args.config.endswith(".json"):
            args_config = json.load(f)
        elif args.config.endswith(".yaml"):
            args_config = yaml.safe_load(f)
        else:
            print(f"File {args.config} is not json or yaml, will use default config")
else:
    print(f"File {args.config} does not exist, will use default config")

# If there are specific writer parameters in the input config file make sure they are not mixed with the default ones
if "writer_config" in args_config:
    config["writer_config"].clear()

# Update the default config dictionay with any new parameters or values from the config file
config.update(args_config)

# Create the simulation app with the given launch_config
simulation_app = SimulationApp(launch_config=config["launch_config"])

# Late import of runtime modules (the SimulationApp needs to be created before loading the modules)
import omni.replicator.core as rep
import omni.usd
import carb

# Custom util functions for the example
from isaacsim.core.utils import prims
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.stage import get_current_stage, open_stage
from isaacsim.core.utils.bounds import compute_combined_aabb, compute_obb, create_bbox_cache, get_obb_corners
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.semantics import remove_all_semantics
from pxr import Gf, UsdGeom

import my_utils

# ============================= 초기 세팅 ===================================

# Get server path
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not get nucleus server path, closing application..")
    simulation_app.close()

# Open the given environment in a new stage
print(f"[scene_based_sdg] Loading Stage {config['env_url']}")
if not open_stage(assets_root_path + config["env_url"]):
    carb.log_error(f"Could not open stage{config['env_url']}, closing application..")
    simulation_app.close()

# Disable capture on play (data generation will be triggered manually)
rep.orchestrator.set_capture_on_play(False)

# Clear any previous semantic data in the loaded stage
if config["clear_previous_semantics"]:
    stage = get_current_stage()
    my_utils.remove_previous_semantics(stage)



# =========================== object Prim 스폰 ===============================

pallet_prim = prims.create_prim(
    prim_path="/World/Pallet",
    position=(random.uniform(-20, -2), random.uniform(-1, 3), 0),
    orientation=euler_angles_to_quat([0, 0, random.uniform(0, math.pi)]),
    usd_path=assets_root_path + config["pallet"]["url"],
    semantic_label=config["pallet"]["class"])

table_tf = omni.usd.get_world_transform_matrix(pallet_prim)
table_loc_gf = table_tf.ExtractTranslation()
table_quat_gf = table_tf.ExtractRotationQuat()


# ========================= 카메라 Prim 스폰 =============================

cam_1 = rep.create.camera(name="Cam1")
cam_2 = rep.create.camera(name="Cam2")
cam_3 = rep.create.camera(name="Cam3")
cam_4 = rep.create.camera(name="Cam4")


# ========================= Randomization graphs 등록 ===========================

# Object Randomization (Pose)
my_utils.register_scatter_objs(pallet_prim, assets_root_path, config, "cube_1")
# Light Parameter
my_utils.register_lights_placement(pallet_prim)


# ==================== 데이터셋 생성을 위한 Render Product 생성 ===================

# Create render products for the custom cameras and attach them to the writer
resolution = config.get("resolution", (1024, 1024))
rp_1 = rep.create.render_product(cam_1, resolution, name="Cam1")
rp_2 = rep.create.render_product(cam_2, resolution, name="Cam2")
rp_3 = rep.create.render_product(cam_3, resolution, name="Cam3")
rp_4 = rep.create.render_product(cam_4, resolution, name="Cam4")
rps = [rp_1, rp_2, rp_3, rp_4]
for rp in rps:
    rp.hydra_texture.set_updates_enabled(False)


# ================== 데이터셋 파일변환을 위한 Writer Product 생성 ==================

# If output directory is relative, set it relative to the current working directory
if not os.path.isabs(config["writer_config"]["output_dir"]):
    config["writer_config"]["output_dir"] = os.path.join(os.getcwd(), config["writer_config"]["output_dir"])
print(f"[scene_based_sdg] Output directory={config['writer_config']['output_dir']}")

# Make sure the writer type is in the registry
writer_type = config.get("writer", "BasicWriter")
if writer_type not in rep.WriterRegistry.get_writers():
    carb.log_error(f"Writer type {writer_type} not found in the registry, closing application..")
    simulation_app.close()

# Get the writer from the registry and initialize it with the given config parameters
writer = rep.WriterRegistry.get(writer_type)
writer_kwargs = config["writer_config"]
print(f"[scene_based_sdg] Initializing {writer_type} with: {writer_kwargs}")
writer.initialize(**writer_kwargs)

# Attach writer to the render products
writer.attach(rps)




# ================ Randomizer의 Frame-wise Control Logic 생성 ==================

# Setup the randomizations to be triggered every frame
with rep.trigger.on_frame():
    rep.randomizer.scatter_obj()
    rep.randomizer.randomize_lights()

    rho_min = 1.5
    rho_max = 2.0
    roll = 0
    pitch = 45
    yaw = [0, 90.0, 180.0, 270.0]

    directions = []
    for yaw_deg in yaw:
        yaw_rad = math.radians(yaw_deg)
        pitch_rad = math.radians(pitch)
         
        x = math.sin(pitch_rad) * math.cos(yaw_rad)
        y = math.sin(pitch_rad) * math.sin(yaw_rad)
        z = math.cos(pitch_rad)
        
        directions.append((x, y, z))

    directions_arr = np.array(directions)
    table_loc_arr = np.array([table_loc_gf[0], table_loc_gf[1], table_loc_gf[2]])

    endpoint1 = directions_arr * rho_min + table_loc_arr
    endpoint2 = directions_arr * rho_max + table_loc_arr

    final_min_pos = np.minimum(endpoint1, endpoint2)
    final_max_pos = np.maximum(endpoint1, endpoint2)
    with cam_1:
        rep.modify.pose(
            position=rep.distribution.uniform(final_min_pos[0, :], final_max_pos[0, :]),
            look_at=str(pallet_prim.GetPrimPath()),
        )    

    with cam_2:
        rep.modify.pose(
            position=rep.distribution.uniform(final_min_pos[1, :], final_max_pos[1, :]),
            look_at=str(pallet_prim.GetPrimPath()),
        )    
    
    with cam_3:
        rep.modify.pose(
            position=rep.distribution.uniform(final_min_pos[2, :], final_max_pos[2, :]),
            look_at=str(pallet_prim.GetPrimPath()),
        )    

    with cam_4:
        rep.modify.pose(
            position=rep.distribution.uniform(final_min_pos[3, :], final_max_pos[3, :]),
            look_at=str(pallet_prim.GetPrimPath()),
        ) 


# ========================= Simulation 시작 ==============================

# Increase subframes if materials are not loaded on time, or ghosting artifacts appear on moving objects,
# see: https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/subframes_examples.html
rt_subframes = config.get("rt_subframes", -1)

# Enable the render products for SDG
for rp in rps:
    rp.hydra_texture.set_updates_enabled(True)

# Start the SDG
num_frames = config.get("num_frames", 0)
print(f"[scene_based_sdg] Running SDG for {num_frames} frames")
for i in range(num_frames):
    print(f"[scene_based_sdg] \t Capturing frame {i}")
    # Trigger any on_frame registered randomizers and the writers (delta_time=0.0 to avoid advancing the timeline)
    rep.orchestrator.step(delta_time=0.0, rt_subframes=rt_subframes)
# Wait for the data to be written to disk
rep.orchestrator.wait_until_complete()

# Cleanup writer and render products
writer.detach()
for rp in rps:
    rp.destroy()

# Check if the application should keep running after the data generation (debug purposes)
close_app_after_run = config.get("close_app_after_run", True)
if config["launch_config"]["headless"]:
    if not close_app_after_run:
        print(
            "[scene_based_sdg] 'close_app_after_run' is ignored when running headless. The application will be closed."
        )
elif not close_app_after_run:
    print("[scene_based_sdg] The application will not be closed after the run. Make sure to close it manually.")
    while simulation_app.is_running():
        simulation_app.update()
simulation_app.close()
