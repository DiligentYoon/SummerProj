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

from isaaclab.sensors.camera.utils import create_pointcloud_from_depth, transform_points
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

        # Robot Grasp and Object Poses
        self.robot_grasp_pos_w = torch.zeros((self.num_envs, 7), device=self.device)
        self.robot_grasp_pos_b = torch.zeros((self.num_envs, 7), device=self.device)
        self.object_pos_w = torch.zeros((self.num_envs, 7), device=self.device)
        self.object_pos_b = torch.zeros((self.num_envs, 7), device=self.device)
        self.object_linvel = torch.zeros((self.num_envs, 3), device=self.device)
        self.object_angvel = torch.zeros((self.num_envs, 3), device=self.device)
        

        # Object Move Checker & Success Checker
        self.prev_loc_error = torch.zeros(self.num_envs, device=self.device)
        self.prev_rot_error = torch.zeros(self.num_envs, device=self.device)
        self.loc_error = torch.zeros(self.num_envs, device=self.device)
        self.rot_error = torch.zeros(self.num_envs, device=self.device)
        self.is_grasp = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Vision Info
        self.sampled_points = torch.zeros((self.num_envs, self.cfg.num_obj_points, 3), device=self.device)
        self.sampled_labels = torch.zeros((self.num_envs, self.cfg.num_obj_points, 1), dtype=torch.long, device=self.device)
        self.goal_flow = torch.zeros((self.num_envs, self.cfg.num_obj_points, 3), device=self.device)

        # High Level Info
        self.extras["high_level_obs"] = torch.zeros((self.num_envs, 
                                                     self.cfg.high_level_observation_space[0], 
                                                     self.cfg.high_level_observation_space[1]), device=self.device)
        self.extras["high_level_reward"] = torch.zeros((self.num_envs, 1), device=self.device)

        # Goal point & Via point marker
        self.target_marker = VisualizationMarkers(self.cfg.goal_pos_marker_cfg)
        self.pcd_marker = VisualizationMarkers(self.cfg.pcd_marker)
        self.goal_pcd_marker = VisualizationMarkers(self.cfg.pcd_marker)


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
        self._compute_intermediate_values(is_reset=False)
        # self.is_reach = torch.logical_and(self.loc_error < 1e-2, self.rot_error < 1e-1)
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        terminated = truncated
        return terminated, truncated
        
    def _get_rewards(self):
        action_norm = torch.norm(self.actions[:, 6:13], dim=1)
        # =========== Approach Reward (1-1): Potential Based Reward Shaping by log scale =============
        gamma = 1.0
        phi_s_prime = -torch.log(self.cfg.alpha * self.loc_error + 1)
        phi_s = -torch.log(self.cfg.alpha * self.prev_loc_error + 1)

        phi_s_prime_rot = -torch.log(self.cfg.alpha * self.rot_error + 1)
        phi_s_rot = -torch.log(self.cfg.alpha * self.prev_rot_error + 1)

        r_pos = gamma*phi_s_prime - phi_s 
        r_rot = gamma*phi_s_prime_rot - phi_s_rot

        # =========== Success Reward : Goal Reach ============
        r_success = self.is_reach.float()
        
        # =========== Summation =============
        reward = self.cfg.w_pos * r_pos + \
                 self.cfg.w_rot * r_rot - \
                 self.cfg.w_penalty * action_norm + \
                 self.cfg.w_success * r_success

        reward = -1 * torch.ones((self.num_envs, 1), device=self.device)

        return reward
    
    def _get_observations(self):
        # Object 및 Robot의 상태를 포함한 Observation vector 생성
        joint_pos_scaled = (
            2.0
            * (self.robot_joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )

        object_loc_tcp, object_rot_tcp = subtract_frame_transforms(
        self.robot_grasp_pos_w[:, :3], self.robot_grasp_pos_w[:, 3:7], self.goal_pos_w[:, :3], self.goal_pos_w[:, 3:7])
        goal_pos_tcp = torch.cat([object_loc_tcp, object_rot_tcp], dim=1)

        obs = torch.cat(
            (   
                # robot joint pose (7)
                joint_pos_scaled[:, 0:self.num_active_joints],
                # robot joint velocity (7)
                self.robot_joint_vel[:, 0:self.num_active_joints],
                # TCP 6D pose w.r.t Root frame (7)
                self.robot_grasp_pos_b,
                # object position w.r.t Root frame (7)
                self.goal_pos_b,
                # object position w.r.t TCP frame (7)
                goal_pos_tcp
            ), dim=1
        )

        self.extras["high_level_obs"] = torch.cat([self.sampled_points,
                                                   self.goal_flow], dim=-1)
        self.extras["high_level_reward"] = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)

        return {"policy": obs}

    
    def _reset_idx(self, env_ids: torch.Tensor):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        # ============ Robot State & Scene 리셋 ===============
        super()._reset_idx(env_ids)
        self._compute_intermediate_values(env_ids, is_reset=True)

    
    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None, is_reset=False):
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
        
        # ========= Position Error 업데이트 =========
        # Location
        self.prev_loc_error[env_ids] = self.loc_error[env_ids]
        self.loc_error[env_ids] = torch.norm(
            self.robot_grasp_pos_b[env_ids, :3] - self.goal_pos_b[env_ids, :3], dim=1)
        # Rotation
        self.prev_rot_error[env_ids] = self.rot_error[env_ids]
        self.rot_error[env_ids] = quat_error_magnitude(self.robot_grasp_pos_b[env_ids, 3:7], self.goal_pos_b[env_ids, 3:7])
        
        # ======== Vision Data 업데이트 =========
        if is_reset:
            # Vision Update를 위한 Single Rendering Step
            self.sim.render()
            all_clouds = []
            all_labels = []
            for i, cam in enumerate(self.cam_list):
                # (E, H, W, 1) -> (E, W, H, 1)
                semantic_labels = cam.data.output["semantic_segmentation"][env_ids].transpose(1, 2)
                # (E, W * H, 1)
                semantic_labels = semantic_labels.reshape(self.num_envs, -1, 1)                                                 
                # Point Cloud : (E, W * H, 3)
                pointcloud = create_pointcloud_from_depth(
                    intrinsic_matrix=self.cam_intrinsic_mat,
                    depth=cam.data.output["distance_to_image_plane"][env_ids],
                    position=cam.data.pos_w[env_ids],
                    orientation=cam.data.quat_w_ros[env_ids],
                    keep_invalid=True,
                    device=self.device,
                )
                all_clouds.append(pointcloud)
                all_labels.append(semantic_labels)
            # Valid Mask : (E, W * H * N_cam, 1)
            all_clouds = torch.cat(all_clouds, dim=1)
            all_labels = torch.cat(all_labels, dim=1)
            if self.object_id is not None:
                _, mapped_labels = extract_valid_mask(all_labels, self.object_id, self.table_id)
            else:
                ValueError("There is no Object Ids. Please check the object spawner")
            # 각 환경에서 얻어지는 Valid Point Cloud Mask가 서로 다르기 때문에 곧바로 브로드캐스팅은 의미가 없음
            # 따라서, 동일한 갯수로 각 환경 별로 uniform 샘플링을 먼저 한 뒤, 합쳐야 한다.
            # (E, W * H * N_cam, 3) & (E, W * H * N_cam, 1) ---> (E, N_obj, 3) & (E, N_obj, 1)
            self.sampled_points[env_ids], self.sampled_labels[env_ids] = uniform_sampling_for_pointnet(all_clouds, 
                                                                                                       mapped_labels,
                                                                                                       num_obj=self.cfg.num_obj_points,
                                                                                                       num_bg=self.cfg.num_bg_points,
                                                                                                       only_obj=True)
            
            # Goal Flow를 위해 Point Clouds의 Goal Position 생성
            # 현재는 위치에 대해서만 수행 -> 수직으로 들어올리는 Grasp Task
            transformed_points = transform_points(self.sampled_points[env_ids], 
                                                  position=(0.0, 0.0, 0.5),
                                                  orientation=(1.0, 0.0, 0.0, 0.0), 
                                                  device=self.device)
            
            self.goal_flow[env_ids] = transformed_points - torch.mean(transformed_points, dim=-1).unsqueeze(-1)
                                   
        # ======== Visualization ==========
        # self.tcp_marker.visualize(self.robot_grasp_pos_w[:, :3], self.robot_grasp_pos_w[:, 3:7])
        self.target_marker.visualize(self.goal_pos_w[:, :3], self.goal_pos_w[:, 3:7])
        self.pcd_marker.visualize(translations=self.sampled_points.reshape(-1, 3))
        # self.goal_pcd_marker.visualize(translations=transformed_points.reshape(-1, 3))
    

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
        
    
    def _apply_high_action(self, actions: torch.Tensor, env_ids: torch.Tensor):
        # 액션 받아오기
        # target point -> 샘플링된 object 포인트 클라우드에 액션 인덱스 매핑
        # motion param -> 회전 정보 매핑
        action_how = torch.rad2deg(torch.pi * actions["how"][env_ids]) # Motion Parameter (E, k) -> Rotation information
        action_where = actions["where"][env_ids] # index (E, 1)

        # Root Pose
        root_pos_w = self._robot.data.root_state_w[env_ids, :7]

        # goal_pos_w : (n, 7)
        target_point_ids = action_where.unsqueeze(-1).expand(-1, -1, 3)
        self.goal_pos_w[env_ids, :3] = torch.gather(self.sampled_points[env_ids], 1, target_point_ids).squeeze(1)
        self.goal_pos_w[env_ids, 3:7] = compute_target_rot(root_pos_w[:, 3:7], action_how, is_world_frame=True)

        # goal_pos_b : (n, 7)
        self.goal_pos_b[env_ids, :3], self.goal_pos_b[env_ids, 3:7] = subtract_frame_transforms(
            root_pos_w[:, :3], root_pos_w[:, 3:7], self.goal_pos_w[env_ids, :3], self.goal_pos_w[env_ids, 3:7]
        )


@torch.jit.script
def extract_valid_mask(label: torch.Tensor, object_id: int, table_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
        유효한 포인트 클라우드 마스크를 추출합니다.
        1. 레이블이 Object여야 합니다.
        2. 모든 병렬 환경에서 일관적으로 검사합니다.

            Inputs:
                label : [E, W * H, 1]
                object_id : Int
                table_id : Int
    """
    # (E, W * H, 1)
    if object_id is not None:
        mapped_labels = (label == object_id).long()
        valid_mask = (label == object_id)
    else:
        mapped_labels = torch.zeros_like(label, dtype=label.dtype, device=label.device)
        valid_mask = None
    
    return valid_mask, mapped_labels


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


def uniform_sampling_for_pointnet(pointcloud: torch.Tensor,
                                  label: torch.Tensor,
                                  num_obj: int = 200, 
                                  num_bg: int = 1000, 
                                  only_obj = False) -> tuple[torch.Tensor, torch.Tensor]:
    """
        Inputs:
            pointcloud : [E, W * H * N_cam, 3]
            label : [E, W * H * N_cam, 1] -> background(=table) : 0 & object : 1
        
        Returns:
            pointcloud : [E, N_k, 3]
            label : [E, N_k, 1]

    """
    num_envs = int(pointcloud.shape[0])
    if only_obj:
        final_pcd = torch.zeros((num_envs, num_obj, 3), dtype=torch.float32, device=pointcloud.device)
        final_lb = torch.zeros((num_envs, num_obj, 1), dtype=torch.long, device=pointcloud.device)
    else:
        final_pcd = torch.zeros((num_envs, num_obj + num_bg, 3), dtype=torch.float32, device=pointcloud.device)
        final_lb = torch.zeros((num_envs, num_obj + num_bg, 1), dtype=torch.long, device=pointcloud.device)        

    sampled_obj_ids = torch.empty((0,), dtype=torch.long, device=pointcloud.device)
    sampled_bg_ids = torch.empty((0,), dtype=torch.long, device=pointcloud.device)
    for i in range(num_envs):
        # (H * W, 3)
        pcd = pointcloud[i]
        # (H * W, 1)
        lb = label[i]
        if pcd.shape[0] == 0:
            continue
        
        # (H * W, )
        obj_ids = torch.where(lb == 1)[0]
        bg_ids = torch.where(lb != 1)[0]

        # --- Object 샘플링 ---
        if len(obj_ids) > 0:
            if len(obj_ids) >= num_obj: # 포인트가 충분하면 비복원 추출
                sampled_obj_ids = obj_ids[torch.randperm(len(obj_ids))[:num_obj]]
            else: # 포인트가 부족하면 복원 추출로 개수를 맞춤
                print(f"Upsampling !")
                indices_to_sample = torch.randint(0, len(obj_ids), (num_obj,), device=pcd.device)
                sampled_obj_ids = obj_ids[indices_to_sample]
        else:
            continue
        
        if only_obj:
            final_pcd[i] = pcd[sampled_obj_ids, ...]
            final_lb[i] = lb[sampled_obj_ids, ...]
        else:
            # --- Background 샘플링 ---
            if len(bg_ids) > 0:
                if len(bg_ids) >= num_bg: # 포인트가 충분하면 비복원 추출
                    sampled_bg_ids = bg_ids[torch.randperm(len(bg_ids))[:num_bg]]
                else: # 포인트가 부족하면 복원 추출
                    print(f"Upsampling !")
                    indices_to_sample = torch.randint(0, len(bg_ids), (num_bg,), device=pointcloud.device)
                    sampled_bg_ids = bg_ids[indices_to_sample]

            # 최종 셔플링 로직은 동일
            cat_sampled_ids = torch.cat((sampled_obj_ids, sampled_bg_ids))
        
            # 합쳐진 인덱스가 없는 경우 처리
            if len(cat_sampled_ids) == 0:
                continue

            sampled_ids = cat_sampled_ids[torch.randperm(len(cat_sampled_ids))]
            
            final_pcd[i] = pcd[sampled_ids, ...]
            final_lb[i] = lb[sampled_ids, ...]

    return final_pcd, final_lb



# @torch.jit.script
# def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
#     return quat_mul(
#         quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
#     )

@torch.jit.script
def compute_target_rot(base_angle: torch.Tensor, delta_angle: torch.Tensor, is_world_frame: bool = False) -> torch.Tensor:
    """
        Compute Delta Rotation :
            base_angle: (N, 4) quaternion form
            delta_angle : (N, 3) axis-angle form (axis * angle)
    """
    # delta_angle(axis-angle)을 delta_rot(quaternion)으로 변환하는 공통 로직
    delta_rot_axis = delta_angle
    delta_rot_angle = torch.norm(delta_rot_axis, dim=-1)
    # 0으로 나누는 것을 방지하기 위한 epsilon 추가
    delta_rot_axis_normalized = delta_rot_axis / (delta_rot_angle.unsqueeze(-1) + 1e-6)
    delta_rot = quat_from_angle_axis(delta_rot_angle, delta_rot_axis_normalized)

    if is_world_frame:
        # World Frame 기준이므로, 변환된 delta_rot 자체가 목표 회전값
        target_rot_b = delta_rot
    else:
        # Base Frame 기준이므로, base_angle에 변환된 delta_rot 곱셈
        target_rot_b = quat_mul(delta_rot, base_angle)
            
    return target_rot_b