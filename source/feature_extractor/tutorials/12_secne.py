
import argparse
import json
import math
import os
import random

import yaml
from isaacsim import SimulationApp


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
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.bounds import compute_combined_aabb, compute_obb, create_bbox_cache, get_obb_corners
from pxr import Gf

import utils


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
    utils.remove_previous_semantics(stage)




# =========================== Prim 스폰 =====================================

# Spawn a new forklift at a random pose
forklift_prim = prims.create_prim(
    prim_path="/World/Forklift",
    position=(random.uniform(-20, -2), random.uniform(-1, 3), 0),
    orientation=euler_angles_to_quat([0, 0, random.uniform(0, math.pi)]),
    usd_path=assets_root_path + config["forklift"]["url"],
    semantic_label=config["forklift"]["class"],
)

# Spawn the pallet in front of the forklift with a random offset on the Y (pallet's forward) axis
forklift_tf = omni.usd.get_world_transform_matrix(forklift_prim)
pallet_offset_tf = Gf.Matrix4d().SetTranslate(Gf.Vec3d(0, random.uniform(-1.2, -1.8), 0))
pallet_pos_gf = (pallet_offset_tf * forklift_tf).ExtractTranslation()
forklift_quat_gf = forklift_tf.ExtractRotationQuat()
forklift_quat_xyzw = (forklift_quat_gf.GetReal(), *forklift_quat_gf.GetImaginary())

pallet_prim = prims.create_prim(
    prim_path="/World/Pallet",
    position=pallet_pos_gf,
    orientation=forklift_quat_xyzw,
    usd_path=assets_root_path + config["pallet"]["url"],
    semantic_label=config["pallet"]["class"],
)

# Spawn a camera in the driver's location looking at the pallet
foklift_pos_gf = forklift_tf.ExtractTranslation()
driver_cam_pos_gf = foklift_pos_gf + Gf.Vec3d(0.0, 0.0, 1.9)

driver_cam = rep.create.camera(
    focus_distance=400.0, focal_length=24.0, clipping_range=(0.1, 10000000.0), name="DriverCam"
)

# Camera looking at the pallet
pallet_cam = rep.create.camera(name="PalletCam")

# Camera looking at the forklift from a top view with large min clipping to see the scene through the ceiling
top_view_cam = rep.create.camera(clipping_range=(6.0, 1000000.0), name="TopCam")




# ========================= Randomization graphs 등록 ===========================

utils.register_scatter_boxes(pallet_prim, assets_root_path, config)
utils.register_cone_placement(forklift_prim, assets_root_path, config)
utils.register_lights_placement(forklift_prim, pallet_prim)




# ==================== 데이터셋 생성을 위한 Render Product 생성 ===================

# Create render products for the custom cameras and attach them to the writer
resolution = config.get("resolution", (512, 512))
forklift_rp = rep.create.render_product(top_view_cam, resolution, name="TopView")
driver_rp = rep.create.render_product(driver_cam, resolution, name="DriverView")
pallet_rp = rep.create.render_product(pallet_cam, resolution, name="PalletView")
# Disable the render products until SDG to improve perf by avoiding unnecessary rendering
rps = [forklift_rp, driver_rp, pallet_rp]
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
    rep.randomizer.scatter_boxes()
    rep.randomizer.randomize_lights()

    # Randomize the camera position in the given area above the pallet and look at the pallet prim
    pallet_cam_min = (pallet_pos_gf[0] - 2, pallet_pos_gf[1] - 2, 2)
    pallet_cam_max = (pallet_pos_gf[0] + 2, pallet_pos_gf[1] + 2, 4)
    with pallet_cam:
        rep.modify.pose(
            position=rep.distribution.uniform(pallet_cam_min, pallet_cam_max),
            look_at=str(pallet_prim.GetPrimPath()),
        )

    # Randomize the camera position in the given height above the forklift driver's seat and look at the pallet prim
    driver_cam_min = (driver_cam_pos_gf[0], driver_cam_pos_gf[1], driver_cam_pos_gf[2] - 0.25)
    driver_cam_max = (driver_cam_pos_gf[0], driver_cam_pos_gf[1], driver_cam_pos_gf[2] + 0.25)
    with driver_cam:
        rep.modify.pose(
            position=rep.distribution.uniform(driver_cam_min, driver_cam_max),
            look_at=str(pallet_prim.GetPrimPath()),
        )

# Setup the randomizations to be triggered at every nth frame (interval)
with rep.trigger.on_frame(interval=4):
    top_view_cam_min = (foklift_pos_gf[0], foklift_pos_gf[1], 9)
    top_view_cam_max = (foklift_pos_gf[0], foklift_pos_gf[1], 11)
    with top_view_cam:
        rep.modify.pose(
            position=rep.distribution.uniform(top_view_cam_min, top_view_cam_max),
            rotation=rep.distribution.uniform((0, -90, -30), (0, -90, 30)),
        )

# Setup the randomizations to be manually triggered at specific times
with rep.trigger.on_custom_event("randomize_cones"):
    rep.randomizer.place_cones()



# ========================= Simulation 시작 ==============================

# Run a simulation by dropping randomly placed boxes on a pallet next to the forklift
utils.simulate_falling_objects(forklift_prim, assets_root_path, config)

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
    # Trigger the custom event to randomize the cones at specific frames
    if i % 2 == 0:
        rep.utils.send_og_event(event_name="randomize_cones")
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