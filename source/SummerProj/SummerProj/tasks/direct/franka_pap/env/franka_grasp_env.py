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
                                matrix_from_quat, quat_apply, quat_from_matrix
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
        self.object_place_pos_w = torch.zeros((self.num_envs, 7), device=self.device)
        self.object_place_pos_b = torch.zeros((self.num_envs, 7), device=self.device)
        self.object_target_pos_w = torch.zeros((self.num_envs, 7), device=self.device)
        self.object_target_pos_b = torch.zeros((self.num_envs, 7), device=self.device)
        self.sub_goal_w = torch.zeros((self.num_envs, 7), device=self.device)
        self.sub_goal_b = torch.zeros((self.num_envs, 7), device=self.device)

        # Robot and Object Grasp Poses
        self.robot_grasp_pos_w = torch.zeros((self.num_envs, 7), device=self.device)
        self.robot_grasp_pos_b = torch.zeros((self.num_envs, 7), device=self.device)

        # Object Move Checker & Success Checker
        self.prev_place_error = torch.zeros((self.num_envs, 2), device=self.device)
        self.prev_retract_error = torch.zeros((self.num_envs, 2), device=self.device)
        self.prev_approach_error = torch.zeros((self.num_envs, 2), device=self.device)

        self.place_error = torch.zeros((self.num_envs, 2), device=self.device)
        self.retract_error = torch.zeros((self.num_envs, 2), device=self.device)
        self.approach_error = torch.zeros((self.num_envs, 2), device=self.device)

        self.weighted_approach_error = torch.zeros((self.num_envs, 2), device=self.device)
        self.weighted_place_error = torch.zeros((self.num_envs, 2), device=self.device)
        self.prev_weighted_approach_error = torch.zeros((self.num_envs, 2), device=self.device)
        self.prev_weighted_place_error = torch.zeros((self.num_envs, 2), device=self.device)

        self.is_reach = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_grasp = torch.zeros_like(self.is_reach, dtype=torch.bool, device=self.device)
        self.is_contact = torch.zeros_like(self.is_reach, dtype=torch.bool, device=self.device)
        self.is_retract = torch.zeros_like(self.is_reach, dtype=torch.bool, device=self.device)
        self.is_in_place = torch.zeros_like(self.is_reach, dtype=torch.bool, device=self.device)
        self.is_success = torch.zeros_like(self.is_reach, dtype=torch.bool, device=self.device)
        
        self.prev_grasp = torch.zeros_like(self.is_reach, dtype=torch.bool, device=self.device)
        self.prev_retract = torch.zeros_like(self.is_reach, dtype=torch.bool, device=self.device)

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
        self.print_count = 0

        # Goal point & Via point marker
        self.retract_marker = VisualizationMarkers(self.cfg.retract_pos_marker_cfg)
        self.place_marker = VisualizationMarkers(self.cfg.place_pos_marker_cfg)


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
        

    def _apply_action(self) -> None:
        """
            최종 커맨드 [N x 22] 생성 후 Controller API 호출.
        """
        # ========= Data 세팅 ==========
        # robot_root_pos = self._robot.data.root_state_w[:, :7]
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

        
        
    def _get_dones(self):
        self._compute_intermediate_values()
        drop = torch.logical_and(self.prev_grasp, ~self.is_grasp) | torch.logical_and(self.object_pos_w[:, 2] < 0, ~self.is_grasp)
        # drop 조건은 retract시에 무효
        in_place = (self.is_retract) & (self.place_error[:, 0] < 5e-2 * 4)

        drop = drop & ~in_place

        terminated = self.is_success | drop | self.is_contact
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        done = terminated | truncated

        if torch.any(done):
            done_ids = torch.where(done)[0]

            self.success_buffer.extend(
                self.is_success.index_select(0, done_ids).float().detach().cpu().tolist()
            )
            self.grasp_buffer.extend(
                self.is_grasp.index_select(0, done_ids).float().detach().cpu().tolist()
            )

        return terminated, truncated
        
    def _get_rewards(self):
        # Action Penalty
        kp_norm = torch.norm(self.actions[:, 7:14], dim=1)

        # =========== Approach Reward : Potential Based Reward Shaping =============
        gamma = 1.0
        phi_s_prime = -torch.log(self.cfg.beta * self.weighted_approach_error[:, 0] + 1)
        phi_s_prime_rot = -torch.log(self.cfg.alpha * self.weighted_approach_error[:, 1] + 1)
        phi_s = -torch.log(self.cfg.beta * self.prev_weighted_approach_error[:, 0] + 1)
        phi_s_rot = -torch.log(self.cfg.alpha * self.prev_weighted_approach_error[:, 1] + 1)

        r_pos = gamma*phi_s_prime - phi_s 
        r_rot = gamma*phi_s_prime_rot - phi_s_rot

        # =========== Phase Reward : Grasping & Retract ===========
        # 1. Grasp Bonus
        r_grasp = self.is_grasp.float()

        # 2. Retract Error Bonus
        phi_s_prime_retract_loc = -torch.log(self.cfg.alpha_retract * self.retract_error[:, 0] + 1)
        phi_s_prime_retract_rot = -torch.log(self.cfg.alpha_retract * self.retract_error[:, 1] + 1)
        phi_s_retract_loc = -torch.log(self.cfg.alpha_retract * self.prev_retract_error[:, 0] + 1)
        phi_s_retract_rot = -torch.log(self.cfg.alpha_retract * self.prev_retract_error[:, 1] + 1)

        r_retract_loc = torch.max(torch.zeros(1, device=self.device), 
                                  (gamma * phi_s_prime_retract_loc - phi_s_retract_loc))
        r_retract_rot = torch.max(torch.zeros(1, device=self.device), 
                                  (gamma * phi_s_prime_retract_rot - phi_s_retract_rot))

        
        # ========== Phase Reward : Place ===========
        # Retract Bonus
        r_retract = self.is_retract.float()

        # Place Error Bonus
        phi_s_prime_place_loc = -torch.log(self.cfg.alpha_place * self.weighted_place_error[:, 0] + 1)
        phi_s_prime_place_rot = -torch.log(self.cfg.alpha_place * self.weighted_place_error[:, 1] + 1)
        phi_s_place_loc = -torch.log(self.cfg.alpha_place * self.prev_weighted_place_error[:, 0] + 1)
        phi_s_place_rot = -torch.log(self.cfg.alpha_place * self.prev_weighted_place_error[:, 1] + 1)

        r_place_loc = torch.max(torch.zeros(1, device=self.device), 
                                (gamma * phi_s_prime_place_loc - phi_s_place_loc))
        r_place_rot = torch.max(torch.zeros(1, device=self.device), 
                                (gamma * phi_s_prime_place_rot - phi_s_place_rot))      

        # Success Bonus
        r_success = self.is_success.float()
    

        # =========== Contact Penalty =================
        # p_contact = torch.logical_and(~self.is_grasp, torch.norm(self._object.data.root_vel_w, dim=1) > 1e-1)
        
        # =========== Summation =============
        reward = torch.where(self.is_retract,
                             self.cfg.w_loc_place * r_place_loc + self.cfg.w_rot_place * r_place_rot,
                             torch.where(self.is_grasp,
                                         self.cfg.w_loc_retract * r_retract_loc + self.cfg.w_rot_retract * r_retract_rot,
                                         self.cfg.w_pos * r_pos + self.cfg.w_rot * r_rot))

        logic_reward = (self.cfg.w_grasp * r_grasp + 
                        self.cfg.w_retract * r_retract + 
                        self.cfg.w_success * r_success -
                        self.cfg.w_penalty * kp_norm -  
                        self.cfg.w_ps)

        reward += logic_reward
                 

        self.extras["log"]["Approach_loc"] = torch.mean(r_pos)
        self.extras["log"]["Approach_rot"] = torch.mean(r_rot)
        self.extras["log"]["Retract_loc"] = torch.mean(r_retract_loc)
        self.extras["log"]["Retract_rot"] = torch.mean(r_retract_rot)
        self.extras["log"]["Place_loc"] = torch.mean(r_place_loc)
        self.extras["log"]["Place_rot"] = torch.mean(r_place_rot)


        return reward
    
    def _get_observations(self):
        # Object 및 Robot의 상태를 포함한 Observation vector 생성
        joint_pos_scaled = (
            2.0
            * (self.robot_joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )

        sub_goal_loc_tcp, sub_goal_rot_tcp = subtract_frame_transforms(
            self.robot_grasp_pos_w[:, :3], self.robot_grasp_pos_w[:, 3:7], self.sub_goal_w[:, :3], self.sub_goal_w[:, 3:7])
        sub_goal_pos_tcp = torch.cat([sub_goal_loc_tcp, sub_goal_rot_tcp], dim=1)

        retract_pos_obj = torch.cat(subtract_frame_transforms(
            self.object_pos_w[:, :3], self.object_pos_w[:, 3:7], self.object_target_pos_w[:, :3], self.object_target_pos_w[:, 3:7]), dim=1)

        retract_pos_tcp = torch.cat(subtract_frame_transforms(
            self.robot_grasp_pos_w[:, :3], self.robot_grasp_pos_w[:, 3:7], self.object_target_pos_w[:, :3], self.object_target_pos_w[:, 3:7]), dim=1)
        
        place_pos_obj = torch.cat(subtract_frame_transforms(
            self.object_pos_w[:, :3], self.object_pos_w[:, 3:7], self.object_place_pos_w[:, :3], self.object_place_pos_w[:, 3:7]), dim=1)
        
        place_pos_tcp = torch.cat(subtract_frame_transforms(
            self.robot_grasp_pos_w[:, :3], self.robot_grasp_pos_w[:, 3:7], self.object_place_pos_w[:, :3], self.object_place_pos_w[:, 3:7]), dim=1)

        current_goal_info_b = torch.where(self.is_retract.unsqueeze(-1),
                                          self.object_place_pos_b,
                                          torch.where(self.is_grasp.unsqueeze(-1),
                                                      self.object_target_pos_b,
                                                      self.sub_goal_b))
        
        current_goal_info_tcp = torch.where(self.is_retract.unsqueeze(-1),
                                            place_pos_tcp,
                                            torch.where(self.is_grasp.unsqueeze(-1),
                                                        retract_pos_tcp,
                                                        sub_goal_pos_tcp))
        
        current_goal_info_obj = torch.where(self.is_retract.unsqueeze(-1),
                                            place_pos_obj,
                                            retract_pos_obj)

        
        if len(self.success_buffer) > 0:
            self.extras["log"]["epi_success_rate"] = torch.tensor(sum(self.success_buffer) / len(self.success_buffer), device=self.device)

        if len(self.grasp_buffer) > 0:
            self.extras["log"]["grasp_success_rate"] = torch.tensor(sum(self.grasp_buffer) / len(self.grasp_buffer), device=self.device)

        obs = torch.cat(
            (   
                # robot joint pose (9)
                joint_pos_scaled[:, 0:self.num_active_joints+2],
                # robot joint velocity (9)
                self.robot_joint_vel[:, 0:self.num_active_joints+2],
                # TCP 6D pose w.r.t Root frame (7)
                self.robot_grasp_pos_b,
                # Goal Info w.r.t body frame (7)
                current_goal_info_b,
                # Goal Info w.r.t TCP frame (7)
                current_goal_info_tcp,
                # Goal Info w.r.t Obj frame (7)
                current_goal_info_obj,
                # Current Phase Info (1)
                self.is_grasp.unsqueeze(-1),
                self.is_retract.unsqueeze(-1)
            ), dim=1
        )

        return {"policy": obs}

    
    def _reset_idx(self, env_ids: torch.Tensor):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        # ============ Robot State & Scene 리셋 ===============
        super()._reset_idx(env_ids)
        object_default_state = self._object.data.default_root_state[env_ids]


        # ============ Object Pose & Sub-goal 리셋 ===============
        # object(=target point) reset : Location
        pick_loc_noise_x = sample_uniform(-0.15, 0.15, (len(env_ids), 1), device=self.device)
        pick_loc_noise_y = sample_uniform(-0.3, 0.3, (len(env_ids), 1), device=self.device)
        pick_loc_noise_z = torch.full((len(env_ids), 1), self.obj_width[0].item()/2, device=self.device)
        pick_loc_noise = torch.cat([pick_loc_noise_x,pick_loc_noise_y, pick_loc_noise_z], dim=-1)

        # Object(=target point) reset : Rotation
        # 1. Alignment with TCP Axis
        tcp_quat = quat_from_matrix(self.tcp_unit_tensor[env_ids])
        # 2. Z-axis Randomization
        rot_noise_z = sample_uniform(-0.8, 0.8, (len(env_ids), ), device=self.device)
        rot_noise = quat_from_angle_axis(rot_noise_z, self.z_unit_tensor[env_ids])
 
        # Apply Randomization
        object_default_state[:, :3] += pick_loc_noise + self.scene.env_origins[env_ids, :3]
        object_default_state[:, 3:7] = quat_mul(tcp_quat, rot_noise)

        # Pose calculation for root frame variables
        object_default_pos_w = object_default_state[:, :7]

        # Setting Sub-goal
        self.sub_goal_w[env_ids] = object_default_pos_w[:, :7]
        self.sub_goal_b[env_ids] = torch.cat(subtract_frame_transforms(
            self._robot.data.root_state_w[env_ids, :3], self._robot.data.root_state_w[env_ids, 3:7],
            object_default_pos_w[:, :3], object_default_pos_w[:, 3:7]
        ), dim=1)

    
        # ================== Retract Point 리셋 =================
        # Setting Final Goal 3D Location
        self.object_target_pos_w[env_ids, :3] = object_default_pos_w[:, :3] + 0.2 * self.z_unit_tensor[env_ids]
        # Setting Final Goal 3D Rotation -> Aligned XYZ axis with TCP
        self.object_target_pos_w[env_ids, 3:7] = object_default_state[:, 3:7]
        self.object_target_pos_b[env_ids, :] = torch.cat(subtract_frame_transforms(
            self._robot.data.root_state_w[env_ids, :3], self._robot.data.root_state_w[env_ids, 3:7], 
            self.object_target_pos_w[env_ids, :3], self.object_target_pos_w[env_ids, 3:7]
        ), dim=1)


        # ================= Place Point 리셋 ==================
        object_default_state_place = self._object.data.default_root_state[env_ids]

        place_loc_noise_x = pick_loc_noise_x
        place_loc_noise_y = -pick_loc_noise_y
        place_loc_noise_z = pick_loc_noise_z
        place_loc_noise = torch.cat([place_loc_noise_x, place_loc_noise_y, place_loc_noise_z], dim=-1)

        tcp_quat = quat_from_matrix(self.tcp_unit_tensor[env_ids])
        rot_noise_z = sample_uniform(-0.8, 0.8, (len(env_ids), ), device=self.device)
        rot_noise = quat_from_angle_axis(rot_noise_z, self.z_unit_tensor[env_ids])

        object_default_state_place[:, :3] += place_loc_noise + self.scene.env_origins[env_ids, :3]
        object_default_state_place[:, 3:7] = quat_mul(tcp_quat, rot_noise)

        self.object_place_pos_w[env_ids] = object_default_state_place[:, :7]
        self.object_place_pos_b[env_ids] = torch.cat(subtract_frame_transforms(
            self._robot.data.root_state_w[env_ids, :3], self._robot.data.root_state_w[env_ids, 3:7],
            self.object_place_pos_w[env_ids, :3], self.object_place_pos_w[env_ids, 3:7]
        ), dim=1)
        
        # ============= State 업데이트 ===============
        self._object.write_root_pose_to_sim(object_default_state[:, :7], env_ids)
        self._object.write_root_velocity_to_sim(object_default_state[:, 7:], env_ids)

        # Need to refresh the intermediate values so that _get_observations() can use the latest values
        self._compute_intermediate_values(env_ids, reset=True)

    
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
        

        # ========= Previous Variables 업데이트 =========
        if reset:
            self.prev_approach_error[env_ids, 0] = torch.norm(self.robot_grasp_pos_b[env_ids, :3] - \
                                                              self.object_pos_b[env_ids, :3], dim=1)

            self.prev_approach_error[env_ids, 1] = quat_error_magnitude(self.robot_grasp_pos_b[env_ids, 3:7],
                                                                self.object_target_pos_b[env_ids, 3:7])
            
            self.prev_retract_error[env_ids, 0] = torch.norm(self.object_pos_b[env_ids, :3] - \
                                                             self.object_target_pos_b[env_ids, :3], dim=1)
            
            self.prev_retract_error[env_ids, 1] = quat_error_magnitude(self.object_pos_b[env_ids, 3:7], 
                                                                       self.object_target_pos_b[env_ids, 3:7])
            
            self.prev_place_error[env_ids, 0] = torch.norm(self.object_pos_b[env_ids, :3] - \
                                                           self.object_place_pos_b[env_ids, :3], dim=1)
            
            self.prev_place_error[env_ids, 1] = quat_error_magnitude(self.object_pos_b[env_ids, 3:7],
                                                                     self.object_place_pos_b[env_ids, 3:7])
            

            loc_error_xyz = torch.abs(self.robot_grasp_pos_b[env_ids, :3] - self.object_pos_b[env_ids, :3])
            self.prev_weighted_approach_error[env_ids, 0] = torch.sqrt(self.cfg.wx * torch.square(loc_error_xyz[:, 0]) +
                                                                       self.cfg.wy * torch.square(loc_error_xyz[:, 1]) + 
                                                                       self.cfg.wz * torch.square(loc_error_xyz[:, 2])).squeeze(-1)

            self.prev_weighted_approach_error[env_ids, 1] = self.prev_approach_error[env_ids, 1]


            place_error_xyz = torch.abs(self.object_pos_b[env_ids, :3] - self.object_place_pos_b[env_ids, :3])
            self.prev_weighted_place_error[env_ids, 0] = torch.sqrt(self.cfg.wx * torch.square(place_error_xyz[:, 0]) +
                                                                    self.cfg.wy * torch.square(place_error_xyz[:, 1]) + 
                                                                    self.cfg.wz * torch.square(place_error_xyz[:, 2])).squeeze(-1)

            self.prev_weighted_place_error[env_ids, 1] = self.prev_place_error[env_ids, 1]


            self.prev_grasp[env_ids] = torch.zeros_like(self.is_reach[env_ids], dtype=torch.bool, device=self.device)
            self.prev_retract[env_ids] = torch.zeros_like(self.is_reach[env_ids], dtype=torch.bool, device=self.device)
        else:
            self.prev_weighted_approach_error[env_ids] = self.weighted_approach_error[env_ids]
            self.prev_weighted_place_error[env_ids] = self.weighted_place_error[env_ids]
            self.prev_approach_error[env_ids] = self.approach_error[env_ids]
            self.prev_retract_error[env_ids] = self.retract_error[env_ids]    
            self.prev_place_error[env_ids] = self.place_error[env_ids]
            self.prev_grasp[env_ids] = self.is_grasp.clone()
            self.prev_retract[env_ids] = self.is_retract.clone()


        # ========= Position Error 업데이트 =========
        # Approach : Location
        loc_error_xyz = torch.abs(self.robot_grasp_pos_b[env_ids, :3] - self.object_pos_b[env_ids, :3])
        self.approach_error[env_ids, 0] = torch.norm(self.robot_grasp_pos_b[env_ids, :3] - self.object_pos_b[env_ids, :3], dim=1)
        # self.approach_error[env_ids, 0] = torch.sqrt(self.cfg.wx * torch.square(loc_error_xyz[:, 0]) +
        #                                              self.cfg.wy * torch.square(loc_error_xyz[:, 1]) + 
        #                                              self.cfg.wz * torch.square(loc_error_xyz[:, 2])).squeeze(-1)  
        # Appraoch : Rotation -> Pre-defined angle (TCP 정렬 각)
        self.approach_error[env_ids, 1] = quat_error_magnitude(self.robot_grasp_pos_b[env_ids, 3:7],
                                                       self.object_target_pos_b[env_ids, 3:7])
        
        self.weighted_approach_error[env_ids, 0] = torch.sqrt(self.cfg.wx * torch.square(loc_error_xyz[:, 0]) +
                                                              self.cfg.wy * torch.square(loc_error_xyz[:, 1]) + 
                                                              self.cfg.wz * torch.square(loc_error_xyz[:, 2])).squeeze(-1)
        
        self.weighted_approach_error[env_ids, 1] = self.approach_error[env_ids, 1]

        
        # Retract : Location
        self.retract_error[env_ids, 0] = torch.norm(self.object_pos_b[env_ids, :3] - \
                                                    self.object_target_pos_b[env_ids, :3], dim=1)
        # Retract : Rotation
        self.retract_error[env_ids, 1] = quat_error_magnitude(self.object_pos_b[env_ids, 3:7], 
                                                              self.object_target_pos_b[env_ids, 3:7])
        

        # Place : Location
        place_error_xyz = torch.abs(self.object_pos_b[env_ids, :3] - self.object_place_pos_b[env_ids, :3])
        self.place_error[env_ids, 0] = torch.norm(self.object_pos_b[env_ids, :3] - self.object_place_pos_b[env_ids, :3], dim=1)
        # self.place_error[env_ids, 0] = torch.sqrt(self.cfg.wx * torch.square(place_error_xyz[:, 0]) +
        #                                           self.cfg.wy * torch.square(place_error_xyz[:, 1]) + 
        #                                           self.cfg.wz * torch.square(place_error_xyz[:, 2])).squeeze(-1)
        # Retract : Rotation
        self.place_error[env_ids, 1] = quat_error_magnitude(self.object_pos_b[env_ids, 3:7],
                                                            self.object_place_pos_b[env_ids, 3:7])

        self.weighted_place_error[env_ids, 0] = torch.sqrt(self.cfg.wx * torch.square(place_error_xyz[:, 0]) +
                                                           self.cfg.wy * torch.square(place_error_xyz[:, 1]) + 
                                                           self.cfg.wz * torch.square(place_error_xyz[:, 2])).squeeze(-1)

        self.weighted_place_error[env_ids, 1] = self.place_error[env_ids, 1]


        # ============ Phase Signal 업데이트 ============
        if self.cfg.w_rot > 0.0:
            self.is_reach[env_ids] = torch.logical_or(torch.logical_and(self.approach_error[env_ids, 0] < 5e-2, self.approach_error[env_ids, 1] < 1e-1),
                                                      self.prev_retract[env_ids])
        else:
            self.is_reach[env_ids] = self.approach_error[env_ids, 0] < 5e-2

        self.is_grasp[env_ids] = torch.logical_and(self.is_reach[env_ids], 
                                                   self.object_pos_b[env_ids, 2] > torch.max(torch.tensor(5e-2, device=self.device), self.obj_width[0]/2))
        
        self.is_retract[env_ids] = torch.logical_or(torch.logical_and(self.is_grasp[env_ids], 
                                                                      torch.logical_and(self.retract_error[env_ids, 0] < 5e-2,
                                                                                        self.retract_error[env_ids, 1] < 1e-1)),
                                                    self.prev_retract[env_ids])

        self.is_success[env_ids] = torch.logical_and(self.is_retract[env_ids], 
                                                     torch.logical_and(self.place_error[env_ids, 0] < 5e-2,
                                                                       self.place_error[env_ids, 1] < 1e-1))
        
        self.is_in_place[env_ids] = (self.is_retract[env_ids]) & (self.place_error[env_ids, 0] < 5e-2 * 4)


        # retract에서 place zone인 경우, grasp 조건 무효
        self.is_grasp[env_ids] = (self.is_grasp[env_ids]) | (self.is_in_place[env_ids])

        
        if not reset:
            self.is_contact[env_ids] = torch.logical_and(torch.logical_and(~self.is_reach[env_ids], 
                                                                           self.approach_error[env_ids, 0] < 5e-2 * 4),
                                                                           torch.norm(self._object.data.root_vel_w[env_ids], dim=1) > 1e-1)
            print(f"\n")
            print(
                    f"Contact / Reach / Grasp / Retract / Success : "
                    f"{self.is_contact.sum().item()} / "
                    f"{self.is_reach.sum().item()} / "
                    f"{self.is_grasp.sum().item()} / "
                    f"{self.is_retract.sum().item()} / "
                    f"{self.is_success.sum().item()}"
                )
        else:
            self.is_contact[env_ids] = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
            
        # ======== Visualization ==========
        self.tcp_marker.visualize(self.object_target_pos_w[:, :3], self.object_target_pos_w[:, 3:7])
        self.retract_marker.visualize(self.object_pos_w[:, :3], self.object_pos_w[:, 3:7])
        self.place_marker.visualize(self.object_place_pos_w[:, :3], self.object_place_pos_w[:, 3:7])
    




    # def compute_frame_jacobian(self, parent_rot_b, jacobian_w: torch.Tensor) -> torch.Tensor:
    #     """Computes the geometric Jacobian of the target frame in the root frame.

    #     This function accounts for the target frame offset and applies the necessary transformations to obtain
    #     the right Jacobian from the parent body Jacobian.
    #     """
    #     # ========= 데이터 세팅 =========
    #     jacobian_b = jacobian_w.clone()
    #     root_quat = self._robot.data.root_quat_w
    #     root_rot_matrix = matrix_from_quat(quat_inv(root_quat))

    #     # ====== Hand Link의 Root Frame에서의 Jacobian 계산 ======
    #     jacobian_b[:, :3, :] = torch.bmm(root_rot_matrix, jacobian_b[:, :3, :])
    #     jacobian_b[:, 3:, :] = torch.bmm(root_rot_matrix, jacobian_b[:, 3:, :])

    #     # ====== TCP의 Offset을 고려한 Frame Jacobian 보정 ======
    #     # ====== v_b = v_a + w * r_{ba} Kinematics 관계 반영 ======
    #     offset_b = quat_apply(parent_rot_b, self.tcp_offset_hand[:, :3])
    #     s_offset = compute_skew_symmetric_matrix(offset_b[:, :3])
    #     jacobian_b[:, :3, :] += torch.bmm(-s_offset, jacobian_b[:, 3:, :])
    #     jacobian_b[:, 3:, :] = torch.bmm(matrix_from_quat(self.tcp_offset_hand[:, 3:7]), jacobian_b[:, 3:, :])

    #     return jacobian_b


    # def sample_discrete_uniform(self,
    #                             low:  torch.Tensor | float,
    #                             high: torch.Tensor | float, 
    #                             delta: torch.Tensor | float, 
    #                             size):
    #     if isinstance(size, int):
    #         size = (size,)
    #     # return tensor
    #     n_vals = int(torch.round((high - low) / delta).item()) + 1
    #     idx = torch.randint(0, n_vals, size, device=self.device, dtype=torch.int64)
    #     return idx.to(torch.float32) * delta + low
        

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


# @torch.jit.script
# def compute_skew_symmetric_matrix(vec: torch.Tensor) -> torch.Tensor:
#     """Computes the skew-symmetric matrix of a vector.
#         Args:
#             vec: The input vector. Shape is (3,) or (N, 3).

#         Returns:
#             The skew-symmetric matrix. Shape is (1, 3, 3) or (N, 3, 3).

#         Raises:
#             ValueError: If input tensor is not of shape (..., 3).
#     """
#     # check input is correct
#     if vec.shape[-1] != 3:
#         raise ValueError(f"Expected input vector shape mismatch: {vec.shape} != (..., 3).")
#     # unsqueeze the last dimension
#     if vec.ndim == 1:
#         vec = vec.unsqueeze(0)

#     S = torch.zeros(vec.shape[0], 3, 3, device=vec.device, dtype=vec.dtype)
#     S[:, 0, 1] = -vec[:, 2]
#     S[:, 0, 2] =  vec[:, 1]
#     S[:, 1, 0] =  vec[:, 2]
#     S[:, 1, 2] = -vec[:, 0]
#     S[:, 2, 0] = -vec[:, 1]
#     S[:, 2, 1] =  vec[:, 0]

#     return S