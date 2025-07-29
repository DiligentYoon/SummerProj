# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import torch
from isaaclab.utils.math import quat_error_magnitude, subtract_frame_transforms, \
                                combine_frame_transforms, quat_from_angle_axis, quat_mul, quat_inv, sample_uniform, saturate, \
                                matrix_from_quat, quat_apply
from isaaclab.markers import VisualizationMarkers

from ..base.franka_base_vision_env import FrankaVisionBaseEnv
from .franka_grasp_vision_env_cfg import FrankaGraspVisionEnvCfg

class FrankaGraspVisionEnv(FrankaVisionBaseEnv):
    """Franka Pap Approach Environment for the Franka Emika Panda robot."""
    cfg: FrankaGraspVisionEnvCfg
    def __init__(self, cfg: FrankaGraspVisionEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Controller Commands & Scene Entity
        self._robot_entity = self.cfg.robot_entity
        self._robot_entity.resolve(self.scene)
        self.processed_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.imp_commands = torch.zeros((self.num_envs, self.imp_controller.num_actions), device=self.device)

        # Goal pose
        self.goal_pos_w = torch.zeros((self.num_envs, 7), device=self.device)
        self.goal_pos_b = torch.zeros((self.num_envs, 7), device=self.device)

        # Robot and Object Grasp Poses
        self.robot_grasp_pos_w = torch.zeros((self.num_envs, 7), device=self.device)
        self.robot_grasp_pos_b = torch.zeros((self.num_envs, 7), device=self.device)

        # Object Move Checker & Success Checker
        self.prev_loc_error = torch.zeros(self.num_envs, device=self.device)
        self.prev_rot_error = torch.zeros(self.num_envs, device=self.device)
        self.loc_error = torch.zeros(self.num_envs, device=self.device)
        self.rot_error = torch.zeros(self.num_envs, device=self.device)
        self.is_grasp = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # High Level Info
        self.extras["high_level_obs"] = torch.zeros((self.num_envs, 
                                                     self.cfg.high_level_observation_space[0], 
                                                     self.cfg.high_level_observation_space[1]), device=self.device)
        self.extras["high_level_reward"] = torch.zeros((self.num_envs, 1), device=self.device)

        # Goal point & Via point marker
        self.target_marker = VisualizationMarkers(self.cfg.goal_pos_marker_cfg)


    # ================= Impedance Control Gain =================
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        actions.shape  =  (N, 21)
        0:7      →  Delta Joint angle for Impedance Control (rad)
        7:14     →  Joint-stiffness for Impedance Control   (N·m/rad)
        14:21    →  Damping-ratio for Impedance Control     (-)
        """
        self.actions = actions.clone()
        # ── 슬라이스 & 즉시 in-place clip ──────────────────────────
        self.processed_actions[:, :7] = torch.clamp(self.actions[:, :7] * self.cfg.joint_res_clipping,
                                                    self.robot_dof_res_lower_limits,
                                                    self.robot_dof_res_upper_limits)
        self.processed_actions[:, 7:14] = torch.clamp(self.actions[:, 7:14] * self.cfg.stiffness_scale,
                                                      self.robot_dof_stiffness_lower_limits,
                                                      self.robot_dof_stiffness_upper_limits)
        self.processed_actions[:, 14:] = torch.clamp(self.actions[:, 14:] * self.cfg.damping_scale,
                                                     self.robot_dof_damping_lower_limits,
                                                     self.robot_dof_damping_upper_limits) 
        
        # ===== Impedance Controller Parameter 세팅 =====
        self.imp_commands[:, :self.num_active_joints] = self.processed_actions[:, :7] + self._robot.data.joint_pos[:, :7]
        self.imp_commands[:,   self.num_active_joints : 2*self.num_active_joints] = self.processed_actions[:, 7:14]
        self.imp_commands[:, 2*self.num_active_joints : 3*self.num_active_joints] = self.processed_actions[:, 14:]
        self.imp_controller.set_command(self.imp_commands)
        

    def _apply_action(self) -> None:
        """
            최종 커맨드 [N x 21] 생성 후 Controller API 호출.
        """
        # ========= Data 세팅 ==========
        robot_joint_pos = self._robot.data.joint_pos[:, :self.num_active_joints]
        robot_joint_vel = self._robot.data.joint_vel[:, :self.num_active_joints]

        gen_mass = self._robot.root_physx_view.get_generalized_mass_matrices()[:, :self.num_active_joints, :self.num_active_joints]
        gen_grav = self._robot.root_physx_view.get_gravity_compensation_forces()[:, :self.num_active_joints]
    
        # ======== Joint Impedance Regulator ========
        des_torque = self.imp_controller.compute(dof_pos=robot_joint_pos,
                                                 dof_vel=robot_joint_vel,
                                                 mass_matrix=gen_mass,
                                                 gravity=gen_grav)
        
        # ===== Target Torque 버퍼에 저장 =====
        self._robot.set_joint_effort_target(des_torque, joint_ids=self.joint_idx)

        # ===== Gripper는 곧바로 Joint Position 버퍼에 저장 =====
        self._robot.set_joint_position_target(self.finger_open_joint_pos, 
                                               joint_ids=[self.left_finger_joint_idx, self.right_finger_joint_idx])
        
        
    def _get_dones(self):
        self._compute_intermediate_values()
        # self.is_reach = torch.logical_and(self.loc_error < 1e-2, self.rot_error < 1e-1)
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        terminated = truncated
        return terminated, truncated
        
    def _get_rewards(self):

        reward = -1 * torch.ones((self.num_envs, 1), device=self.device)

        return reward
    
    def _get_observations(self):
        # Object 및 Robot의 상태를 포함한 Observation vector 생성
        # joint_pos_scaled = (
        #     2.0
        #     * (self.robot_joint_pos - self.robot_dof_lower_limits)
        #     / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
        #     - 1.0
        # )

        # object_loc_tcp, object_rot_tcp = subtract_frame_transforms(
        # self.robot_grasp_pos_w[:, :3], self.robot_grasp_pos_w[:, 3:7], self.goal_pos_w[:, :3], self.goal_pos_w[:, 3:7])
        # goal_pos_tcp = torch.cat([object_loc_tcp, object_rot_tcp], dim=1)

        # obs = torch.cat(
        #     (   
        #         # robot joint pose (7)
        #         joint_pos_scaled[:, 0:self.num_active_joints],
        #         # robot joint velocity (7)
        #         self.robot_joint_vel[:, 0:self.num_active_joints],
        #         # TCP 6D pose w.r.t Root frame (7)
        #         self.robot_grasp_pos_b,
        #         # object position w.r.t Root frame (7)
        #         self.goal_pos_b,
        #         # object position w.r.t TCP frame (7)
        #         goal_pos_tcp
        #     ), dim=1
        # )
        obs = torch.zeros((self.num_envs, self.cfg.observation_space), dtype=torch.float32, device=self.device)

        return {"policy": obs}

    
    def _reset_idx(self, env_ids: torch.Tensor):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        # ============ Robot State & Scene 리셋 ===============
        super()._reset_idx(env_ids)
        self._compute_intermediate_values(env_ids)

    
    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES


        # ========= TCP 업데이트 ===========
        root_pos_w = self._robot.data.root_state_w[env_ids, :7]
        hand_pos_w = self._robot.data.body_state_w[env_ids, self.hand_link_idx, :7]
        # data for joint
        self.robot_joint_pos[env_ids] = self._robot.data.joint_pos[env_ids]
        self.robot_joint_vel[env_ids] = self._robot.data.joint_vel[env_ids]
        # data for TCP (world & root Frame)
        self.robot_grasp_pos_b[env_ids] = calculate_robot_tcp(hand_pos_w, root_pos_w, self.tcp_offset_hand[env_ids])
        self.robot_grasp_pos_w[env_ids, 3:7] = quat_mul(root_pos_w[:, 3:7], self.robot_grasp_pos_b[env_ids, 3:7])
        self.robot_grasp_pos_w[env_ids, :3] = root_pos_w[:, :3] + quat_apply(root_pos_w[:, 3:7], self.robot_grasp_pos_b[env_ids, :3])

        # ========= Object 업데이트 ==========
        self.goal_pos_w[env_ids] = self._object.data.root_state_w[env_ids, :7]
        self.goal_pos_b[env_ids] = torch.cat(subtract_frame_transforms(
            root_pos_w[:, :3], root_pos_w[:, 3:7], self.goal_pos_w[env_ids, :3], self.goal_pos_w[env_ids, 3:7]
        ), dim=1)
        
        # ========= Position Error 업데이트 =========
        # Location
        self.prev_loc_error[env_ids] = self.loc_error[env_ids]
        self.loc_error[env_ids] = torch.norm(
            self.robot_grasp_pos_b[env_ids, :3] - self.goal_pos_b[env_ids, :3], dim=1)
        # Rotation
        self.prev_rot_error[env_ids] = self.rot_error[env_ids]
        self.rot_error[env_ids] = quat_error_magnitude(self.robot_grasp_pos_b[env_ids, 3:7], self.goal_pos_b[env_ids, 3:7])
        
        # ======== Vision Data 업데이트 =========
        all_clouds = []
        all_labels = []
        all_normals = []
        total_clouds = None
        total_labels = None
        total_normals = None
        for i, cam in enumerate(self.cam_list):
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
        

        # ======== Visualization ==========
        # self.tcp_marker.visualize(self.robot_grasp_pos_w[:, :3], self.robot_grasp_pos_w[:, 3:7])
        # self.target_marker.visualize(self.goal_pos_w[:, :3], self.goal_pos_w[:, 3:7])
    

    def sample_discrete_uniform(self,
                                low:  torch.Tensor | float,
                                high: torch.Tensor | float, 
                                delta: torch.Tensor | float, 
                                size):
        if isinstance(size, int):
            size = (size,)
        # return tensor
        n_vals = int(torch.round((high - low) / delta).item()) + 1
        idx = torch.randint(0, n_vals, size, device=self.device, dtype=torch.int64)
        return idx.to(torch.float32) * delta + low
        
    
    def _apply_high_action(self, actions: torch.Tensor):
        action_how = actions["how"] # Motion Parameter (E, k) -> Rotation information
        action_where = actions["where"] # index (E, 1)

        pass



@torch.jit.script
def calculate_robot_tcp(hand_pos_w: torch.Tensor,
                        root_pos_w: torch.Tensor,
                        offset: torch.Tensor | None) -> torch.Tensor:
    
    hand_loc_b, hand_rot_b = subtract_frame_transforms(
        root_pos_w[:, :3], root_pos_w[:, 3:7], hand_pos_w[:, :3], hand_pos_w[:, 3:7])

    if offset is not None:
        tcp_loc_b, tcp_rot_b = combine_frame_transforms(
            hand_loc_b, hand_rot_b, offset[:, :3], offset[:, 3:7])
    else:
        tcp_loc_b = hand_loc_b; tcp_rot_b = hand_rot_b
    

    return torch.cat((tcp_loc_b, tcp_rot_b), dim=1)


# @torch.jit.script
# def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
#     return quat_mul(
#         quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
#     )

# @torch.jit.script
# def compute_target_rot(base_angle: torch.Tensor, delta_angle: torch.Tensor) -> torch.Tensor:
#         """
#             Compute Delta Rotation :
#                 base_angle: (N, 4) quaternion form
#                 target_angle : (N, 3) euler angle form
            
#                 -> we calculate target_rotation by quaternion form in root frame
#         """
#         delta_rot_axis = delta_angle
#         delta_rot_angle = torch.norm(delta_rot_axis, dim=-1)
#         delta_rot_axis_normalized = delta_rot_axis / (delta_rot_angle.unsqueeze(-1) + 1e-6)
#         delta_rot = quat_from_angle_axis(delta_rot_angle, delta_rot_axis_normalized)
#         target_rot_b = quat_mul(delta_rot, base_angle)
#         return target_rot_b