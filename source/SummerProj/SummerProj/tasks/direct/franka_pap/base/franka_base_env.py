# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import torch
from abc import abstractmethod

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, AssetBase
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.controllers.joint_impedance import JointImpedanceController
from isaaclab.controllers import DifferentialIKController
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_from_angle_axis, quat_mul, sample_uniform

from .franka_base_env_cfg import FrankaBaseEnvCfg

class FrankaBaseEnv(DirectRLEnv):
    cfg: FrankaBaseEnvCfg

    def __init__(self, cfg: FrankaBaseEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Total env ids
        self.total_env_ids = torch.arange(self.num_envs, device=self.device)

        # Joint & Link Index
        self.joint_idx = self._robot.find_joints("panda_joint.*")[0]
        self.hand_link_idx = self._robot.find_bodies("panda_hand")[0][0]
        self.left_finger_link_idx = self._robot.find_bodies("panda_leftfinger")[0][0]
        self.left_finger_joint_idx = self._robot.find_joints("panda_finger_joint1")[0][0]
        self.right_finger_link_idx = self._robot.find_bodies("panda_rightfinger")[0][0]
        self.right_finger_joint_idx = self._robot.find_joints("panda_finger_joint2")[0][0]
        self.finger_open_joint_pos =  0.04 * torch.ones((self.num_envs, 2), device=self.device)
        self.finger_close_joint_pos = torch.zeros(1, device=self.device)

        # Physics Limits
        self.num_active_joints = len(self.joint_idx)
        self.robot_dof_res_lower_limits = torch.tensor(-self.cfg.joint_res_clipping, device=self.device)
        self.robot_dof_res_upper_limits = torch.tensor(self.cfg.joint_res_clipping, device=self.device)
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_dof_stiffness_lower_limits = torch.tensor(self.cfg.imp_controller.stiffness_limits[0], device=self.device)
        self.robot_dof_stiffness_upper_limits = torch.tensor(self.cfg.imp_controller.stiffness_limits[1], device=self.device)
        self.robot_dof_damping_lower_limits = torch.tensor(self.cfg.imp_controller.damping_ratio_limits[0], device=self.device)
        self.robot_dof_damping_upper_limits = torch.tensor(self.cfg.imp_controller.damping_ratio_limits[1], device=self.device)

        # Default Object and Robot Pose
        self.robot_joint_pos = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.robot_joint_vel = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.object_pos_w = torch.zeros((self.num_envs, 7), device=self.device)
        self.object_pos_b = torch.zeros((self.num_envs, 7), device=self.device)
        self.object_linvel = torch.zeros((self.num_envs, 3), device=self.device)
        self.object_angvel = torch.zeros((self.num_envs, 3), device=self.device)

        # Default TCP Offset
        self.tcp_offset = torch.tensor([0.0, 0.0, 0.045], device=self.device).repeat([self.scene.num_envs, 1])
        self.tcp_offset_hand = torch.tensor([0.0, 0.0, 0.107, 1.0, 0.0, 0.0, 0.0], device=self.device).repeat([self.scene.num_envs, 1])
        
        # Joint Impedance Controller for Torque Control
        self.imp_controller = JointImpedanceController(cfg=self.cfg.imp_controller,
                                                       num_robots=self.num_envs,
                                                       dof_pos_limits=self._robot.data.soft_joint_pos_limits[:, 0:self.num_active_joints, :],
                                                       device=self.device)
        
        # IK Controller for Position Control
        self.ik_controller = DifferentialIKController(cfg=self.cfg.ik_controller,
                                                      num_envs=self.num_envs,
                                                      device=self.device)

        # TCP Marker
        self.tcp_marker = VisualizationMarkers(self.cfg.tcp_cfg)

        # unit tensors
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        spawn_ground_plane(prim_path=self.cfg.plane.prim_path, cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))
        spawn = self.cfg.table.spawn
        spawn.func(self.cfg.table.prim_path, spawn, translation=(0.0, 0.0, 0.0), orientation=(1.0, 0.0, 0.0, 0.0))
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # def step(self, action: torch.Tensor) -> VecEnvStepReturn:
    #     # From Superclass (DirectRLEnv)
    #     action.to(self.device)

    #     # add action noies
    #     if self.cfg.action_noise_model:
    #         action = self._action_noise_model.apply(action)
        
    #     # process actions
    #     self._pre_physics_step(action)

    #     # check if we need to do rendering within the physics loop
    #     # note: checked here once to avoid multiple checks within the loop
    #     is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

    #     # perform physics stepping
    #     for _ in range(self.cfg.decimation):
    #         pass

    
    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        # Initialize robot joint state with pose randomization
        pos_noise = sample_uniform(
            -0.125, 0.125,
            (len(env_ids), self.num_active_joints),
            self.device,)
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_pos[:, :self.num_active_joints] += pos_noise
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)

        # Publish to simulator
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


    # ====================== Abstract Functions ================================
    @abstractmethod
    def _pre_physics_step(self, actions):
        raise NotImplementedError(f"Please implement the '_pre_physics_step' method for {self.__class__.__name__}.")

    @abstractmethod
    def _apply_action(self):
        raise NotImplementedError(f"Please implement the '_apply_action' method for {self.__class__.__name__}.")

    @abstractmethod
    def _get_dones(self):
        raise NotImplementedError(f"Please implement the '_get_done' method for {self.__class__.__name__}.")

    @abstractmethod
    def _get_rewards(self):
        raise NotImplementedError(f"Please implement the '_get_rewards' method for {self.__class__.__name__}.")

    @abstractmethod
    def _get_observations(self):
        raise NotImplementedError(f"Please implement the '_get_observation' method for {self.__class__.__name__}.")