# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import collections
import numpy as np
import torch
from isaaclab.utils.math import quat_error_magnitude, subtract_frame_transforms, \
                                combine_frame_transforms, quat_from_angle_axis, quat_mul, quat_inv, sample_uniform, saturate, \
                                matrix_from_quat, quat_apply
from isaaclab.markers import VisualizationMarkers
from isaaclab.assets import RigidObject

from ..base.franka_base_env import FrankaBaseEnv
from .franka_grasp_env_cfg import FrankaGraspEnvCfg

import omni.usd
from isaacsim.core.utils import bounds

class FrankaGraspEnv(FrankaBaseEnv):
    """Franka Pap Approach Environment for the Franka Emika Panda robot."""
    cfg: FrankaGraspEnvCfg
    def __init__(self, cfg: FrankaGraspEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Controller Commands & Scene Entity
        self._robot_entity = self.cfg.robot_entity
        self._robot_entity.resolve(self.scene)
        self.processed_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.imp_commands = torch.zeros((self.num_envs, self.imp_controller.num_actions), device=self.device)

        # Goal Pose
        self.object_target_pos_w = torch.zeros((self.num_envs, 7), device=self.device)
        self.object_target_pos_b = torch.zeros((self.num_envs, 7), device=self.device)

        # Robot and Object Grasp Poses
        self.robot_grasp_pos_w = torch.zeros((self.num_envs, 7), device=self.device)
        self.robot_grasp_pos_b = torch.zeros((self.num_envs, 7), device=self.device)

        # Object Move Checker & Success Checker
        self.prev_retract_error = torch.zeros((self.num_envs, 2), device=self.device)
        self.prev_loc_error = torch.zeros(self.num_envs, device=self.device)
        self.prev_rot_error = torch.zeros(self.num_envs, device=self.device)
        self.retract_error = torch.zeros((self.num_envs, 2), device=self.device)
        self.loc_error = torch.zeros(self.num_envs, device=self.device)
        self.rot_error = torch.zeros(self.num_envs, device=self.device)
        self.is_reach = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_grasp = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Domain Randomization Scale
        self.noise_scale = torch.tensor(
                            [self.cfg.reset_position_noise_x, self.cfg.reset_position_noise_y],
                            device=self.device,)
        

        # metric
        self.extras["log"] = {
            "epi_success_rate": torch.zeros(self.num_envs, device=self.device),
            "grasp_success_rate": torch.zeros(self.num_envs, device=self.device)
        }
        self.success_buffer = collections.deque(maxlen=100) # 최근 100개 에피소드 결과 저장
        self.grasp_buffer = collections.deque(maxlen=100) # 최근 100개 에피소드 결과 저장   
        self.cumulative_episode_success = 0
        self.cumulative_grasp_sucess = 0
        self.log_interval = 10
        self.log_step = 0

        # Goal point & Via point marker
        self.target_marker = VisualizationMarkers(self.cfg.goal_pos_marker_cfg)


    def _setup_scene(self):
        super()._setup_scene()
        self._object = RigidObject(self.cfg.object)
        self.scene.rigid_objects["object"] = self._object

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)

        self.cfg.dome_light.spawn.func(self.cfg.dome_light.prim_path, self.cfg.dome_light.spawn)

        # Object의 Default 상태에서의 BB Spec도 모든 환경에서 동일
        obj_bb_cache = bounds.create_bbox_cache()
        self.obj_prim = omni.usd.get_context().get_stage().GetPrimAtPath(
            f"/World/envs/env_0/Object"
        )
        self.obj_aabb = bounds.compute_aabb(obj_bb_cache, self.obj_prim.GetPrimPath().pathString)
        self.obj_width = torch.full((self.num_envs, 1), self.obj_aabb[5] - self.obj_aabb[2]).to(self.device)


    # ================= Controller Gain =================
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        actions.shape  =  (N, 22)
        0:7      →  Delta Joint angle for Impedance Control (rad)
        7:14     →  Joint-stiffness for Impedance Control   (N·m/rad)
        14:21    →  Damping-ratio for Impedance Control     (-)
        22       →  Gripper Commands                        (-)
        """
        self.actions = actions.clone()
        # ── 슬라이스 & 즉시 in-place clip ──────────────────────────
        self.processed_actions[:, :7] = torch.clamp(self.actions[:, :7] * self.cfg.joint_res_clipping,
                                                    self.robot_dof_res_lower_limits,
                                                    self.robot_dof_res_upper_limits)
        self.processed_actions[:, 7:14] = torch.clamp(self.actions[:, 7:14] * self.cfg.stiffness_scale,
                                                      self.robot_dof_stiffness_lower_limits,
                                                      self.robot_dof_stiffness_upper_limits)
        self.processed_actions[:, 14:21] = torch.clamp(self.actions[:, 14:21] * self.cfg.damping_scale,
                                                       self.robot_dof_damping_lower_limits,
                                                       self.robot_dof_damping_upper_limits)
        
        # self.processed_actions[:, 21] = torch.clamp(self.actions[:, 21] * self.cfg.gripper_scale/2 + self.cfg.gripper_scale/2,
        #                                             0,
        #                                             self.cfg.gripper_scale)
        
        self.processed_actions[:, 21] = torch.where(self.actions[:, 21] > 0, 0.04, 0.0)
        
        # ===== Impedance Controller Parameter 세팅 =====
        self.imp_commands[:, :self.num_active_joints] = self.processed_actions[:, :self.num_active_joints] + \
                                                        self._robot.data.joint_pos[:, :self.num_active_joints]
        self.imp_commands[:,   self.num_active_joints : 2*self.num_active_joints] = self.processed_actions[:, self.num_active_joints : 2*self.num_active_joints]
        self.imp_commands[:, 2*self.num_active_joints : 3*self.num_active_joints] = self.processed_actions[:, 2*self.num_active_joints : 3*self.num_active_joints]
        self.imp_controller.set_command(self.imp_commands)

        # ===== Gripper는 곧바로 Joint Position 버퍼에 저장 =====
        self._robot.set_joint_position_target(self.processed_actions[:, 21].reshape(-1, 1).repeat(1, 2), 
                                               joint_ids=[self.left_finger_joint_idx, self.right_finger_joint_idx])
        
        # act = self.actions[:, 21]
        # print(f"Gripper Commands : {act}")
        

    def _apply_action(self) -> None:
        """
            최종 커맨드 [N x 22] 생성 후 Controller API 호출.
        """
        # ========= Data 세팅 ==========
        robot_root_pos = self._robot.data.root_state_w[:, :7]
        robot_joint_pos = self._robot.data.joint_pos[:, :self.num_active_joints]
        robot_joint_vel = self._robot.data.joint_vel[:, :self.num_active_joints]

        # hand_pos_w = self._robot.data.body_state_w[:, self.hand_link_idx, :7]
        # _, hand_rot_b = subtract_frame_transforms(
        #     robot_root_pos[:, :3], robot_root_pos[:, 3:7], hand_pos_w[:, :3], hand_pos_w[:, 3:7])
        # robot_grasp_pos_b = calculate_robot_tcp(hand_pos_w, robot_root_pos, self.tcp_offset_hand)

        gen_mass = self._robot.root_physx_view.get_generalized_mass_matrices()[:, :self.num_active_joints, :self.num_active_joints]
        gen_grav = self._robot.root_physx_view.get_gravity_compensation_forces()[:, :self.num_active_joints]

        # ========= Inverse Kinematics =========
        # if robot_grasp_pos_b[:, 3:7].norm() != 0:
        #     # World Frame에서 Hand의 Jacobian Matrix 계산
        #     jacobian_w = self._robot.root_physx_view.get_jacobians()[:, self.jacobi_idx, :, :self.num_active_joints]
        #     # Root Frame에서 TCP의 Jacobian Matrix 계산
        #     jacobian_t = self.compute_frame_jacobian(hand_rot_b, jacobian_w)
        #     # Target Joint Angle 계산
        #     joint_pos_des = self.ik_controller.compute(robot_grasp_pos_b[:, :3], 
        #                                                robot_grasp_pos_b[:, 3:7], 
        #                                                jacobian_t, 
        #                                                robot_joint_pos)
        # else:
        #     joint_pos_des = robot_joint_pos.clone()
    
        # ======== Joint Impedance Regulator ========
        des_torque = self.imp_controller.compute(dof_pos=robot_joint_pos,
                                                 dof_vel=robot_joint_vel,
                                                 mass_matrix=gen_mass,
                                                 gravity=gen_grav)
        
        # ===== Target Torque 버퍼에 저장 =====
        self._robot.set_joint_effort_target(des_torque, joint_ids=self.joint_idx)

        
        
    def _get_dones(self):
        self._compute_intermediate_values()
        is_retract = torch.logical_and(self.retract_error[:, 0] < 1e-2, self.retract_error[:, 1] < 1e-2)
        terminated = torch.logical_and(self.is_grasp, is_retract)
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        done = terminated | truncated

        # print(f"Reach : {self.is_reach.float()}")
        # print(f"Distance : {self.loc_error}")
        if torch.any(done):
            done_env_ids = torch.where(done)[0]
            for env_id in done_env_ids:
                self.success_buffer.append(terminated[env_id].float().item())
                self.grasp_buffer.append(self.is_grasp[env_id].float().item())
        
        if len(self.success_buffer) > 0:
            self.extras["log"]["epi_success_rate"] = sum(self.success_buffer) / len(self.success_buffer)

        if len(self.grasp_buffer) > 0:
            self.extras["log"]["grasp_success_rate"] = sum(self.grasp_buffer) / len(self.grasp_buffer)


        return terminated, truncated
        
    def _get_rewards(self):
        # Action Penalty
        # gripper_norm = torch.abs(self.actions[:, 21])
        # action_norm = torch.norm(self.actions[:, 7:14], dim=1)

        # # =========== Approach Reward (1): Potential Based Reward Shaping =============
        # # gamma = 1.0
        # # phi_s_prime = -self.loc_error
        # # phi_s = -self.prev_loc_error

        # # phi_s_prime_rot = -self.rot_error
        # # phi_s_rot = -self.prev_rot_error

        # # r_pos = gamma*phi_s_prime - phi_s 
        # # r_rot = gamma*phi_s_prime_rot - phi_s_rot

        # =========== Approach Reward (1-1): Potential Based Reward Shaping by log scale =============
        gamma = 1.0
        phi_s_prime = -torch.log(self.cfg.alpha * self.loc_error + 1)
        phi_s = -torch.log(self.cfg.alpha * self.prev_loc_error + 1)

        # phi_s_prime_rot = -torch.log(self.cfg.alpha * self.rot_error + 1)
        # phi_s_rot = -torch.log(self.cfg.alpha * self.prev_rot_error + 1)

        r_pos = gamma*phi_s_prime - phi_s 
        # r_rot = gamma*phi_s_prime_rot - phi_s_rot

        # =========== Phase Bonus : Object Grasping ===========
        # 1. Grasp Bonus
        r_grasp = self.is_grasp.float()

        # 2. Retract Error Bonus (단일 거리)
        phi_s_prime_retract_loc = -torch.log(self.cfg.alpha * self.retract_error[:, 0] + 1)
        phi_s_prime_retract_rot = -torch.log(self.cfg.alpha * self.retract_error[:, 1] + 1)
        phi_s_retract_loc = -torch.log(self.cfg.alpha * self.prev_retract_error[:, 0] + 1)
        phi_s_retract_rot = -torch.log(self.cfg.alpha * self.prev_retract_error[:, 1] + 1)

        r_retract_loc = r_grasp * (gamma * phi_s_prime_retract_loc - phi_s_retract_loc)
        r_retract_rot = r_grasp * (gamma * phi_s_prime_retract_rot - phi_s_retract_rot) 

        print(f"retract loc : {r_retract_loc}")
        print(f"retract rot : {r_retract_rot}")
    

        # =========== Contact Penalty =================
        # p_contact = torch.logical_and(~self.is_grasp, torch.norm(self._object.data.root_vel_w, dim=1) > 1e-1)
        
        # =========== Summation =============
        reward = self.cfg.w_pos * r_pos             + \
                 self.cfg.w_grasp * r_grasp         + \
                 self.cfg.w_pos_retract * r_retract_loc + \
                 self.cfg.w_pos_retract * r_retract_rot
                 

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
            self.robot_grasp_pos_w[:, :3], self.robot_grasp_pos_w[:, 3:7], self.object_pos_w[:, :3], self.object_pos_w[:, 3:7])
        object_pos_tcp = torch.cat([object_loc_tcp, object_rot_tcp], dim=1)

        goal_pos_tcp = torch.cat(subtract_frame_transforms(
            self.robot_grasp_pos_w[:, :3], self.robot_grasp_pos_w[:, 3:7], self.object_target_pos_w[:, :3], self.object_target_pos_w[:, 3:7]), dim=1)


        obs = torch.cat(
            (   
                # robot joint pose (9)
                joint_pos_scaled[:, 0:self.num_active_joints+2],
                # robot joint velocity (9)
                self.robot_joint_vel[:, 0:self.num_active_joints+2],
                # TCP 6D pose w.r.t Root frame (7)
                self.robot_grasp_pos_b,
                # object position w.r.t Root frame (7)
                self.object_pos_b,
                # object position w.r.t TCP frame (7)
                object_pos_tcp,
                # object goal position w.r.t Root Frame (7)
                self.object_target_pos_b,
                # object goal position w.r.t TCP frame (7)
                goal_pos_tcp
            ), dim=1
        )

        return {"policy": obs}

    
    def _reset_idx(self, env_ids: torch.Tensor):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        # ============ Robot State & Scene 리셋 ===============
        super()._reset_idx(env_ids)

        # ============ Target Point 리셋 ===============
        # object(=target point) reset : Location
        loc_noise_x = sample_uniform(-0.2, 0.2, (len(env_ids), 1), device=self.device)
        loc_noise_y = sample_uniform(-0.4, 0.4, (len(env_ids), 1), device=self.device)
        loc_noise_z = torch.full((len(env_ids), 1), self.obj_width[0].item()/2, device=self.device)
        loc_noise = torch.cat([loc_noise_x, loc_noise_y, loc_noise_z], dim=-1)
        object_default_state = self._object.data.default_root_state[env_ids]
        object_default_state[:, :3] += loc_noise + self.scene.env_origins[env_ids, :3]

        # object(=target point) reset : Rotation -> Z-axis Randomization
        rot_noise_z = sample_uniform(-1.0, 1.0, (len(env_ids), ), device=self.device)
        object_default_state[:, 3:7] = quat_from_angle_axis(rot_noise_z, self.z_unit_tensor[env_ids])

        # Pose calculation for root frame variables
        object_default_pos_w = object_default_state[:, :7]

        # Setting Final Goal 3D Location
        self.object_target_pos_w[env_ids, :3] = object_default_pos_w[:, :3] + 0.2 * self.z_unit_tensor[env_ids]
        self.object_target_pos_w[env_ids, 3:7] = object_default_pos_w[:, 3:7]

        self.object_target_pos_b[env_ids, :] = torch.cat(subtract_frame_transforms(
            self._robot.data.root_state_w[env_ids, :3], self._robot.data.root_state_w[env_ids, 3:7], 
            self.object_target_pos_w[env_ids, :3], self.object_target_pos_w[env_ids, 3:7]
        ), dim=1)
        
        self._object.write_root_pose_to_sim(object_default_state[:, :7], env_ids)
        self._object.write_root_velocity_to_sim(object_default_state[:, 7:], env_ids)

        # Need to refresh the intermediate values so that _get_observations() can use the latest values
        self._compute_intermediate_values(env_ids)

    
    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None, reset: bool = False):
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
        self.object_pos_w[env_ids] = self._object.data.root_state_w[env_ids, :7]
        self.object_pos_b[env_ids] = torch.cat(subtract_frame_transforms(
            root_pos_w[:, :3], root_pos_w[:, 3:7], self.object_pos_w[env_ids, :3], self.object_pos_w[env_ids, 3:7]
        ), dim=1)
        
        # ========= Position Error 업데이트 =========
        # Location
        self.prev_loc_error[env_ids] = self.loc_error[env_ids]
        self.loc_error[env_ids] = torch.norm(
            self.robot_grasp_pos_b[env_ids, :3] - self.object_pos_b[env_ids, :3], dim=1)
        # Retract
        self.prev_retract_error[env_ids] = self.retract_error[env_ids]
        self.retract_error[env_ids, 0] = torch.norm(self.object_pos_b[env_ids, :3] - \
                                                    self.object_target_pos_b[env_ids, :3], dim=1)
        self.retract_error[env_ids, 1] = quat_error_magnitude(self.robot_grasp_pos_b[env_ids, 3:7], 
                                                              self.object_target_pos_b[env_ids, 3:7])
        # Phase Signal
        self.is_reach[env_ids] = self.loc_error[env_ids] < 5e-2
        self.is_grasp[env_ids] = torch.logical_and(self.is_reach[env_ids], self.object_pos_b[env_ids, 2] > torch.max(torch.tensor(5e-2, device=self.device), self.obj_width[0]/2))
            
        # ======== Visualization ==========
        # self.tcp_marker.visualize(self.robot_grasp_pos_w[:, :3], self.robot_grasp_pos_w[:, 3:7])
        self.target_marker.visualize(self.object_target_pos_w[:, :3], self.object_target_pos_w[:, 3:7])
    

    def compute_frame_jacobian(self, parent_rot_b, jacobian_w: torch.Tensor) -> torch.Tensor:
        """Computes the geometric Jacobian of the target frame in the root frame.

        This function accounts for the target frame offset and applies the necessary transformations to obtain
        the right Jacobian from the parent body Jacobian.
        """
        # ========= 데이터 세팅 =========
        jacobian_b = jacobian_w.clone()
        root_quat = self._robot.data.root_quat_w
        root_rot_matrix = matrix_from_quat(quat_inv(root_quat))

        # ====== Hand Link의 Root Frame에서의 Jacobian 계산 ======
        jacobian_b[:, :3, :] = torch.bmm(root_rot_matrix, jacobian_b[:, :3, :])
        jacobian_b[:, 3:, :] = torch.bmm(root_rot_matrix, jacobian_b[:, 3:, :])

        # ====== TCP의 Offset을 고려한 Frame Jacobian 보정 ======
        # ====== v_b = v_a + w * r_{ba} Kinematics 관계 반영 ======
        offset_b = quat_apply(parent_rot_b, self.tcp_offset_hand[:, :3])
        s_offset = compute_skew_symmetric_matrix(offset_b[:, :3])
        jacobian_b[:, :3, :] += torch.bmm(-s_offset, jacobian_b[:, 3:, :])
        jacobian_b[:, 3:, :] = torch.bmm(matrix_from_quat(self.tcp_offset_hand[:, 3:7]), jacobian_b[:, 3:, :])

        return jacobian_b


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
        

@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )

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

@torch.jit.script
def compute_target_rot(base_angle: torch.Tensor, delta_angle: torch.Tensor) -> torch.Tensor:
        """
            Compute Delta Rotation :
                base_angle: (N, 4) quaternion form
                target_angle : (N, 3) euler angle form
            
                -> we calculate target_rotation by quaternion form in root frame
        """
        delta_rot_axis = delta_angle
        delta_rot_angle = torch.norm(delta_rot_axis, dim=-1)
        delta_rot_axis_normalized = delta_rot_axis / (delta_rot_angle.unsqueeze(-1) + 1e-6)
        delta_rot = quat_from_angle_axis(delta_rot_angle, delta_rot_axis_normalized)
        target_rot_b = quat_mul(delta_rot, base_angle)
        return target_rot_b


@torch.jit.script
def compute_skew_symmetric_matrix(vec: torch.Tensor) -> torch.Tensor:
    """Computes the skew-symmetric matrix of a vector.
        Args:
            vec: The input vector. Shape is (3,) or (N, 3).

        Returns:
            The skew-symmetric matrix. Shape is (1, 3, 3) or (N, 3, 3).

        Raises:
            ValueError: If input tensor is not of shape (..., 3).
    """
    # check input is correct
    if vec.shape[-1] != 3:
        raise ValueError(f"Expected input vector shape mismatch: {vec.shape} != (..., 3).")
    # unsqueeze the last dimension
    if vec.ndim == 1:
        vec = vec.unsqueeze(0)

    S = torch.zeros(vec.shape[0], 3, 3, device=vec.device, dtype=vec.dtype)
    S[:, 0, 1] = -vec[:, 2]
    S[:, 0, 2] =  vec[:, 1]
    S[:, 1, 0] =  vec[:, 2]
    S[:, 1, 2] = -vec[:, 0]
    S[:, 2, 0] = -vec[:, 1]
    S[:, 2, 1] =  vec[:, 0]

    return S