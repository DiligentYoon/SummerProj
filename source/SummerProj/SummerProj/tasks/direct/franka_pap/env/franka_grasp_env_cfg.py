# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import  RigidObjectCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim import SimulationCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from ..base.franka_base_env_cfg import FrankaBaseEnvCfg


@configclass
class FrankaGraspEnvCfg(FrankaBaseEnvCfg):
    # env
    episode_length_s = 10.0
    decimation = 5
    action_space = 22
    observation_space = 40
    state_space = 0

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

    # object
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.67, 0.0, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(1.2, 1.2, 1.2),
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

    # object marker
    object_pos_marker_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="Visuals/object_marker",
        markers={
        "frame": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.1, 0.1, 0.1),
            )
        }
    )
    

    # reward hyperparameter
    alpha, beta = 3.0, 3.0
    w_pos = 15.0
    w_rot = 10.0
    w_penalty = 0.001
    w_contact = 0.01
    w_loc_retract = 60.0
    w_rot_retract = 10.0
    # w_grasp = 3.0
    w_grasp = 1.5
    w_reach = 1.0
    # w_success = 100.0
    w_success = 300.0