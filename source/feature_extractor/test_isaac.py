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

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on adding sensors on a robot.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
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
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, quat_from_angle_axis, quat_mul

FRONT_ROT = (0.61237, 0.35355, 0.35355, 0.61237)
LEFT_ROT = (0.81013, 0.53848, -0.13613, -0.18761)
RIGHT_ROT = (0.18761, 0.13613, -0.53848, -0.81013)

# Experimental known (Configuration -> Simulation Transformation)
QUAT_CON_TO_SIM = (0.5, 0.5, -0.5, -0.5)
QUAT_SIM_TO_CON = (0.5, -0.5, 0.5, 0.5)

FRONT_ROT_CON = tuple(quat_mul(torch.tensor(FRONT_ROT), torch.tensor(QUAT_SIM_TO_CON)).numpy())
LEFT_ROT_CON = tuple(quat_mul(torch.tensor(LEFT_ROT), torch.tensor(QUAT_SIM_TO_CON)).numpy())
RIGHT_ROT_CON = tuple(quat_mul(torch.tensor(RIGHT_ROT), torch.tensor(QUAT_SIM_TO_CON)).numpy())

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
                                   scale=(1.0, 1.0, 1.0)),
    )


    # sensor
    front_camera = CameraCfg(
        prim_path=f"/World/envs/env/FrontCam",
        update_period=0.1,
        height=1024,
        width=1024,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            # 데이터 수집 시 파라미터와 동일하게 설정
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.954999923706055,
            vertical_aperture=15.290800094604492,  # 이 값을 명시적으로 추가합니다.
            clipping_range=(1.0, 1000000.0)      # Near/Far 값을 정확히 맞춰줍니다.
        ),
        offset=CameraCfg.OffsetCfg(pos=(1.6, 0.0, 0.7), rot=FRONT_ROT_CON, convention="world"),
    )

    # sensor
    left_behind_camera = CameraCfg(
        prim_path="/World/envs/env/Leftcam",
        update_period=0.1,
        height=1024,
        width=1024,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            # 데이터 수집 시 파라미터와 동일하게 설정
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.954999923706055,
            vertical_aperture=15.290800094604492,  # 이 값을 명시적으로 추가합니다.
            clipping_range=(1.0, 1000000.0)      # Near/Far 값을 정확히 맞춰줍니다.
        ),
        offset=CameraCfg.OffsetCfg(pos=(-0.13, -1.2, 0.5), rot=LEFT_ROT_CON, convention="world"),
    )

    # sensor
    right_behind_camera = CameraCfg(
        prim_path="/World/envs/env/Rightcam",
        update_period=0.1,
        height=1024,
        width=1024,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            # 데이터 수집 시 파라미터와 동일하게 설정
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.954999923706055,
            vertical_aperture=15.290800094604492,  # 이 값을 명시적으로 추가합니다.
            clipping_range=(1.0, 1000000.0)      # Near/Far 값을 정확히 맞춰줍니다.
        ),
        offset=CameraCfg.OffsetCfg(pos=(-0.13, 1.2, 0.5), rot=RIGHT_ROT_CON, convention="world"),
    )



def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the simulator."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Simulate physics
    while simulation_app.is_running():
        # Reset
        if count % 500 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            # root_state = scene["robot"].data.default_root_state.clone()
            # root_state[:, :3] += scene.env_origins
            # scene["robot"].write_root_pose_to_sim(root_state[:, :7])
            # scene["robot"].write_root_velocity_to_sim(root_state[:, 7:])
            # # set joint positions with some noise
            # joint_pos, joint_vel = (
            #     scene["robot"].data.default_joint_pos.clone(),
            #     scene["robot"].data.default_joint_vel.clone(),
            # )
            # joint_pos += torch.rand_like(joint_pos) * 0.1
            # scene["robot"].write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting robot state...")
        # Apply default actions to the robot
        # -- generate actions/commands
        # targets = scene["robot"].data.default_joint_pos
        # # -- apply action to the robot
        # scene["robot"].set_joint_position_target(targets)
        # # -- write data to sim
        # scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        scene.update(sim_dt)

        # print information from the sensors
        # print("-------------------------------")
        # print(scene["camera"])
        # print("Received shape of rgb   image: ", scene["camera"].data.output["rgb"].shape)
        # print("-------------------------------")


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