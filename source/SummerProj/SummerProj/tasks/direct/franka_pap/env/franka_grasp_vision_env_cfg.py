# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from ..base.franka_base_vision_cfg import FrankaVisionBaseCfg


@configclass
class FrankaGraspVisionEnvCfg(FrankaVisionBaseCfg):
    # camera
    # num_feature = 7 # (x,y,z, gx, gy, gz, label) -> object and background
    num_feature = 6 # (x,y,z, gx, gy, gz) -> only object
    num_obj_points = 200
    num_bg_points = 1000

    
    # env
    episode_length_s = 10.0
    decimation = 10
    action_space = 21
    observation_space = 35
    state_space = 0


    high_level_observation_space = [num_obj_points, num_feature]
    high_level_action_space = {
        "where": num_obj_points,
        "how": 3
    }
    high_level_state_space = 0
    high_level_goal_space = {
        "obj_state" : [num_obj_points, 3],
        "tcp_state" : 3 
    }

    low_level_observation_space = observation_space
    low_level_action_space = action_space
    low_level_state_space = state_space



    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    
    # goal marker
    goal_pos_marker_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="Visuals/goal_marker",
        markers={
        "frame": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.1, 0.1, 0.1),
            )
        }
    )
    
    # reward hyperparameter
    alpha, beta = 3.0, 4.0
    w_pos = 50.0
    w_rot = 25.0
    w_penalty = 0.01
    w_success = 10.0