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
    episode_length_s = 15.0
    decimation = 5
    action_space = 22
    observation_space = 52
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
    
    # marker
    lift_pos_marker_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="Visuals/goal_marker",
        markers={
        "frame": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.1, 0.1, 0.1),
            )
        }
    )

    place_pos_marker_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="Visuals/object_marker",
        markers={
        "frame": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.1, 0.1, 0.1),
            )
        }
    )

    # logging
    logging_interval = 100
    
    # threshold
    loc_th = 5e-2
    rot_th = 1e-1

    place_loc_th = 5e-2
    place_rot_th = 1e-1

    retract_loc_th = 5e-2
    retract_rot_th = 1e-1


    # reward hyperparameter
    alpha, beta = 3.0, 3.0
    alpha_lift, beta_lift = 1.5, 1.5
    alpha_place, beta_place = 0.5, 0.5
    alpha_retract, beta_retract = 0.25, 0.25

    w_pos = 20.0
    w_rot = 10.0

    # w_loc_lift = 45.0
    # w_rot_lift = 10.0
    w_loc_lift = 20.0
    w_rot_lift = 10.0

    # w_loc_place = 50.0
    # w_rot_place = 25.0
    w_loc_place = 20.0
    w_rot_place = 10.0

    w_loc_retract = 20.0
    w_rot_retract = 10.0

    # w_grasp = 1.5
    # w_lift = 4.5
    w_grasp = 6.0
    w_lift = 6.0
    w_place = 6.0
    w_success = 2000.0

    wx = 4.0
    wy = 4.0
    wz = 1.0

    w_penalty = 0.05
    w_gripper = 1.0
    # w_ps = 2.0
    w_ps = 1.0

    # curriculum learning
    place_loc_th_max = 5e-2
    place_loc_th_min = 1e-2
    decay_ratio = 2.0