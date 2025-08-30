# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import collections
import copy
import torch
import math
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
        self.final_goal_pos_w = torch.zeros((self.num_envs, 7), device=self.device)
        self.final_goal_pos_b = torch.zeros((self.num_envs, 7), device=self.device)
        self.object_place_pos_w = torch.zeros((self.num_envs, 7), device=self.device)
        self.object_place_pos_b = torch.zeros((self.num_envs, 7), device=self.device)
        self.object_lift_pos_w = torch.zeros((self.num_envs, 7), device=self.device)
        self.object_lift_pos_b = torch.zeros((self.num_envs, 7), device=self.device)
        self.sub_goal_w = torch.zeros((self.num_envs, 7), device=self.device)
        self.sub_goal_b = torch.zeros((self.num_envs, 7), device=self.device)

        # Robot and Object Grasp Poses
        self.robot_grasp_pos_w = torch.zeros((self.num_envs, 7), device=self.device)
        self.robot_grasp_pos_b = torch.zeros((self.num_envs, 7), device=self.device)

        # Object Move Checker & Success Checker
        self.prev_retract_error = torch.zeros((self.num_envs, 2), device=self.device)
        self.prev_place_error = torch.zeros((self.num_envs, 2), device=self.device)
        self.prev_place_error = torch.zeros((self.num_envs, 2), device=self.device)
        self.prev_lift_error = torch.zeros((self.num_envs, 2), device=self.device)
        self.prev_approach_error = torch.zeros((self.num_envs, 2), device=self.device)

        self.retract_error = torch.zeros((self.num_envs, 2), device=self.device)
        self.place_error = torch.zeros((self.num_envs, 2), device=self.device)
        self.lift_error = torch.zeros((self.num_envs, 2), device=self.device)
        self.approach_error = torch.zeros((self.num_envs, 2), device=self.device)

        self.weighted_approach_error = torch.zeros((self.num_envs, 2), device=self.device)
        self.weighted_place_error = torch.zeros((self.num_envs, 2), device=self.device)
        self.weighted_retract_error = torch.zeros((self.num_envs, 2), device=self.device)
        self.prev_weighted_approach_error = torch.zeros((self.num_envs, 2), device=self.device)
        self.prev_weighted_place_error = torch.zeros((self.num_envs, 2), device=self.device)
        self.prev_weighted_retract_error = torch.zeros((self.num_envs, 2), device=self.device)

        self.is_reach = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_grasp = torch.zeros_like(self.is_reach, dtype=torch.bool, device=self.device)
        self.is_lift = torch.zeros_like(self.is_reach, dtype=torch.bool, device=self.device)
        self.is_in_place = torch.zeros_like(self.is_reach, dtype=torch.bool, device=self.device)
        self.is_place = torch.zeros_like(self.is_reach, dtype=torch.bool, device=self.device)
        self.is_first_place = torch.zeros_like(self.is_reach, dtype=torch.bool, device=self.device)
        self.is_put = torch.zeros_like(self.is_reach, dtype=torch.bool, device=self.device)
        self.is_success = torch.zeros_like(self.is_reach, dtype=torch.bool, device=self.device)
        self.contact = torch.zeros_like(self.is_reach, dtype=torch.bool, device=self.device)

        self.still_lift = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.drop = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.re_pick = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Previous Phase Signal
        self.prev_grasp = torch.zeros_like(self.is_reach, dtype=torch.bool, device=self.device)
        self.prev_lift = torch.zeros_like(self.is_reach, dtype=torch.bool, device=self.device)
        self.prev_place = torch.zeros_like(self.is_reach, dtype=torch.bool, device=self.device)
        self.prev_put = torch.zeros_like(self.is_reach, dtype=torch.bool, device=self.device)
        
        # metric
        self.place_buffer = collections.deque(maxlen=100)
        self.lift_buffer = collections.deque(maxlen=100)
        self.success_buffer = collections.deque(maxlen=100)
        self.grasp_buffer = collections.deque(maxlen=100)
        self.still_lift_buffer = collections.deque(maxlen=100)
        self.logging_count = 0

        # Goal point & Via point marker
        self.lift_marker = VisualizationMarkers(self.cfg.lift_pos_marker_cfg)
        self.place_marker = VisualizationMarkers(self.cfg.place_pos_marker_cfg)

        # Low-Pass Filter Parameter
        self.control_dt = self.physics_dt * self.cfg.decimation
        self.tau = 0.1
        self.omega = 1/self.tau
        self.prev_imp_commands = torch.zeros_like(self.imp_commands, device=self.device)



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
        

        # norm_stiffness_action = (self.actions[:, 7:14] + 1.0) * 0.5
        # # 지수(k)를 적용하여 낮은 값에 대한 정밀도 향상
        # exp_stiffness_action = torch.pow(norm_stiffness_action, self.cfg.k_stiffness)
        # # 최종 stiffness 범위로 매핑
        # stiffness_val = self.cfg.stiffness_range[0] + (self.cfg.stiffness_range[1] - self.cfg.stiffness_range[0]) * exp_stiffness_action
        
        # self.processed_actions[:, 7:14] = torch.clamp(stiffness_val,
        #                                               self.robot_dof_stiffness_lower_limits,
        #                                               self.robot_dof_stiffness_upper_limits)

        # # 2. Damping 처리
        # # action 값을 [0, 1] 범위로 정규화
        # norm_damping_action = (self.actions[:, 14:21] + 1.0) * 0.5
        # # 지수(k)를 적용하여 낮은 값에 대한 정밀도 향상
        # exp_damping_action = torch.pow(norm_damping_action, self.cfg.k_damping)
        # # 최종 damping 범위로 매핑
        # damping_val = self.cfg.damping_range[0] + (self.cfg.damping_range[1] - self.cfg.damping_range[0]) * exp_damping_action

        # self.processed_actions[:, 14:21] = torch.clamp(damping_val,
        #                                                self.robot_dof_damping_lower_limits,
        #                                                self.robot_dof_damping_upper_limits)
        
        self.processed_actions[:, 7:14] = torch.clamp(self.actions[:, 7:14] * self.cfg.stiffness_scale,
                                                self.robot_dof_stiffness_lower_limits,
                                                self.robot_dof_stiffness_upper_limits)
        self.processed_actions[:, 14:21] = torch.clamp(self.actions[:, 14:21] * self.cfg.damping_scale,
                                                       self.robot_dof_damping_lower_limits,
                                                       self.robot_dof_damping_upper_limits)

        self.processed_actions[:, 21] = torch.where(self.actions[:, 21] > 0, 0.04, 0.0)
        
        # ===== Impedance Controller Parameter 세팅 with LPF Smoothing =====
        prev_joint_commands = self.prev_imp_commands[:, :self.num_active_joints]
        cur_joint_commands = self.processed_actions[:, :self.num_active_joints] + self._robot.data.joint_pos[:, :self.num_active_joints]
        self.imp_commands[:, :self.num_active_joints] = (1 / (1+(self.control_dt * self.omega))) * \
                                                        (prev_joint_commands + self.control_dt * self.omega * cur_joint_commands)
        # self.imp_commands[:, :self.num_active_joints] = self.processed_actions[:, :self.num_active_joints] + \
        #                                                 self._robot.data.joint_pos[:, :self.num_active_joints]

        prev_kp_commands = self.prev_imp_commands[:, self.num_active_joints : 2*self.num_active_joints]
        cur_kp_commands = self.processed_actions[:, self.num_active_joints : 2*self.num_active_joints]
        self.imp_commands[:,   self.num_active_joints : 2*self.num_active_joints] = (1 / (1+(self.control_dt * self.omega))) * \
                                                                                    (prev_kp_commands + self.control_dt * self.omega * cur_kp_commands)
        
        prev_zeta_commands = self.prev_imp_commands[:, 2*self.num_active_joints : 3*self.num_active_joints]
        cur_zeta_commands= self.processed_actions[:, 2*self.num_active_joints : 3*self.num_active_joints]
        self.imp_commands[:, 2*self.num_active_joints : 3*self.num_active_joints] = (1 / (1+(self.control_dt * self.omega))) * \
                                                                                    (prev_zeta_commands + self.control_dt * self.omega * cur_zeta_commands)
        # Control Command 세팅
        self.imp_controller.set_command(self.imp_commands)

        # ===== Gripper는 곧바로 Joint Position 버퍼에 저장 =====
        self._robot.set_joint_position_target(self.processed_actions[:, 21].reshape(-1, 1).repeat(1, 2), 
                                               joint_ids=[self.left_finger_joint_idx, self.right_finger_joint_idx])
        

        # ===== LPF 스무딩을 위해 prev값 업데이트 =====
        self.prev_imp_commands = self.imp_commands.clone()

        # print(f"=====================================")
        # print(f"Kp : {self.processed_actions[:, 7:14]}")
        # print(f"Zeta : {self.processed_actions[:, 14:21]}")
        

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

        # terminated = self.is_success | self.contact | self.still_lift | self.drop
        terminated = self.is_success | self.contact | self.drop
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        done = terminated | truncated

        self.logging_process(done)

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

        self.r_pos = gamma*phi_s_prime - phi_s 
        self.r_rot = gamma*phi_s_prime_rot - phi_s_rot

        # =========== Phase Reward : Grasping & Lift ===========
        # 1. Grasp Bonus
        r_grasp = self.is_grasp.float()

        # 2. Lift Error Bonus
        phi_s_prime_lift_loc = -torch.log(self.cfg.alpha_lift * self.lift_error[:, 0] + 1)
        phi_s_prime_lift_rot = -torch.log(self.cfg.alpha_lift * self.lift_error[:, 1] + 1)
        phi_s_lift_loc = -torch.log(self.cfg.alpha_lift * self.prev_lift_error[:, 0] + 1)
        phi_s_lift_rot = -torch.log(self.cfg.alpha_lift * self.prev_lift_error[:, 1] + 1)

        self.r_lift_loc = torch.max(torch.zeros(1, device=self.device), 
                                  (gamma * phi_s_prime_lift_loc - phi_s_lift_loc))
        self.r_lift_rot = torch.max(torch.zeros(1, device=self.device), 
                                  (gamma * phi_s_prime_lift_rot - phi_s_lift_rot))
        
        # self.r_lift_loc = gamma * phi_s_prime_lift_loc - phi_s_lift_loc
        # self.r_lift_rot = gamma * phi_s_prime_lift_rot - phi_s_lift_rot

        
        # ========== Phase Reward : Place ===========
        # Lift Bonus
        # r_lift = (self.is_lift & ~self.prev_lift).float()
        r_lift = self.is_lift.float()

        # Place Error Bonus
        hand_lin_vel = self.robot_hand_lin_vel
        phi_s_prime_place_loc = -torch.log(self.cfg.alpha_place * self.weighted_place_error[:, 0] + 1)
        phi_s_prime_place_rot = -torch.log(self.cfg.alpha_place * self.weighted_place_error[:, 1] + 1)
        phi_s_place_loc = -torch.log(self.cfg.alpha_place * self.prev_weighted_place_error[:, 0] + 1)
        phi_s_place_rot = -torch.log(self.cfg.alpha_place * self.prev_weighted_place_error[:, 1] + 1)

        self.r_place_loc = torch.max(torch.zeros(1, device=self.device), 
                                    (gamma * phi_s_prime_place_loc - phi_s_place_loc))
        self.r_place_rot = torch.max(torch.zeros(1, device=self.device), 
                                (gamma * phi_s_prime_place_rot - phi_s_place_rot))      
        # self.r_place_loc = gamma * phi_s_prime_place_loc - phi_s_place_loc
        # self.r_place_rot = gamma * phi_s_prime_place_rot - phi_s_place_rot   


        # ========= Phase Reward : Put =============
        r_place = self.is_place.float()
        r_gripper = (self.processed_actions[:, 21] > 0)
        r_gripper_state = torch.stack([self.robot_joint_pos[:, self.left_finger_joint_idx], self.robot_joint_pos[:, self.right_finger_joint_idx]], dim=-1).mean(dim=-1)
        self.r_gripper = r_gripper.float()
        self.r_gripper_state = (r_gripper_state > 0.99 * 0.04).float()


        # =========== Phase Reward : Retract ==============
        r_put = self.is_put.float()
        

        # Retract Error Bonus
        phi_s_prime_retract_loc = -torch.log(self.cfg.alpha_retract * self.weighted_retract_error[:, 0] + 1)
        phi_s_prime_retract_rot = -torch.log(self.cfg.alpha_retract * self.weighted_retract_error[:, 1] + 1)
        phi_s_retract_loc = -torch.log(self.cfg.alpha_retract * self.prev_weighted_retract_error[:, 0] + 1)
        phi_s_retract_rot = -torch.log(self.cfg.alpha_retract * self.prev_weighted_retract_error[:, 1] + 1)

        self.r_retract_loc = torch.max(torch.zeros(1, device=self.device),
                               (gamma * phi_s_prime_retract_loc - phi_s_retract_loc))
        self.r_retract_rot = torch.max(torch.zeros(1, device=self.device),
                                    (gamma * phi_s_prime_retract_rot - phi_s_retract_rot))
        # self.r_retract_loc = (gamma * phi_s_prime_retract_loc - phi_s_retract_loc) / (hand_lin_vel*20 + 1)
        # self.r_retract_rot = (gamma * phi_s_prime_retract_rot - phi_s_retract_rot)


        # Success Bonus
        r_success = self.is_success.float()
    
        
        # =========== Summation =============
        # reward = torch.where(self.is_place,
        #                      self.cfg.w_loc_retract * self.r_retract_loc + self.cfg.w_rot_retract * self.r_retract_rot + self.cfg.w_gripper_state * self.r_gripper_state,
        #                      torch.where(self.is_lift,
        #                                  self.cfg.w_loc_place * self.r_place_loc + self.cfg.w_rot_place * self.r_place_rot,
        #                                  torch.where(self.is_grasp,
        #                                              self.cfg.w_loc_lift * self.r_lift_loc + self.cfg.w_rot_lift * self.r_lift_rot,
        #                                              self.cfg.w_pos * self.r_pos + self.cfg.w_rot * self.r_rot)))
        reward = torch.where(self.is_put,
                             self.cfg.w_loc_retract * self.r_retract_loc + self.cfg.w_rot_retract * self.r_retract_rot + self.cfg.w_gripper * self.r_gripper,
                             torch.where(self.is_place,
                                         self.cfg.w_gripper * self.r_gripper - hand_lin_vel,
                                         torch.where(self.is_lift,
                                                     self.cfg.w_loc_place * self.r_place_loc + self.cfg.w_rot_place * self.r_place_rot,
                                                     torch.where(self.is_grasp,
                                                                 self.cfg.w_loc_lift * self.r_lift_loc + self.cfg.w_rot_lift * self.r_lift_rot,
                                                                 self.cfg.w_pos * self.r_pos + self.cfg.w_rot * self.r_rot))))
        # reward = torch.where(self.is_place,
        #                      self.cfg.w_gripper * self.r_gripper - hand_lin_vel,
        #                      torch.where(self.is_lift,
        #                                  self.cfg.w_loc_place * self.r_place_loc + self.cfg.w_rot_place * self.r_place_rot,
        #                                  torch.where(self.is_grasp,
        #                                              self.cfg.w_loc_lift * self.r_lift_loc + self.cfg.w_rot_lift * self.r_lift_rot,
        #                                              self.cfg.w_pos * self.r_pos + self.cfg.w_rot * self.r_rot)))
    
        

        logic_reward = (self.cfg.w_grasp * r_grasp + 
                        self.cfg.w_lift * r_lift + 
                        self.cfg.w_place * r_place +
                        self.cfg.w_place * 0.5 * r_put + 
                        self.cfg.w_success * r_success -
                        self.cfg.w_penalty * kp_norm -  
                        self.cfg.w_ps)
        
        reward += logic_reward

        self.r_total = reward

        # ========== Logging ===========
        self.update_additional_info()


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

        lift_pos_obj = torch.cat(subtract_frame_transforms(
            self.object_pos_w[:, :3], self.object_pos_w[:, 3:7], self.object_lift_pos_w[:, :3], self.object_lift_pos_w[:, 3:7]), dim=1)

        lift_pos_tcp = torch.cat(subtract_frame_transforms(
            self.robot_grasp_pos_w[:, :3], self.robot_grasp_pos_w[:, 3:7], self.object_lift_pos_w[:, :3], self.object_lift_pos_w[:, 3:7]), dim=1)
        
        place_pos_obj = torch.cat(subtract_frame_transforms(
            self.object_pos_w[:, :3], self.object_pos_w[:, 3:7], self.object_place_pos_w[:, :3], self.object_place_pos_w[:, 3:7]), dim=1)
        
        place_pos_tcp = torch.cat(subtract_frame_transforms(
            self.robot_grasp_pos_w[:, :3], self.robot_grasp_pos_w[:, 3:7], self.object_place_pos_w[:, :3], self.object_place_pos_w[:, 3:7]), dim=1)

        final_goal_pos_tcp = torch.cat(subtract_frame_transforms(
            self.robot_grasp_pos_w[:, :3], self.robot_grasp_pos_w[:, 3:7], 
            self.final_goal_pos_w[:, :3], self.final_goal_pos_w[:, 3:7]), dim=1)

        final_goal_pos_obj = torch.cat(subtract_frame_transforms(
            self.object_pos_w[:, :3], self.object_pos_w[:, 3:7],
            self.final_goal_pos_w[:, :3], self.final_goal_pos_w[:, 3:7]), dim=1)


        gripper = torch.where(self.is_place.unsqueeze(-1) | ~self.is_reach.unsqueeze(-1), 1, 0)


        # target_frame_pos_b = torch.where(self.is_place.unsqueeze(-1),
        #                                   self.robot_grasp_pos_b,
        #                                   torch.where(self.is_lift.unsqueeze(-1),
        #                                               self.object_pos_b,
        #                                               torch.where(self.is_grasp.unsqueeze(-1),
        #                                                           self.object_pos_b,
        #                                                           self.robot_grasp_pos_b)))

        target_frame_pos_b = torch.where(self.is_put.unsqueeze(-1),
                                         self.robot_grasp_pos_b,
                                         torch.where(self.is_place.unsqueeze(-1),
                                                     self.object_pos_b,
                                                     torch.where(self.is_lift.unsqueeze(-1),
                                                                 self.object_pos_b,
                                                                 torch.where(self.is_grasp.unsqueeze(-1),
                                                                             self.object_pos_b,
                                                                             self.robot_grasp_pos_b))))

        # target_frame_pos_b = torch.where(self.is_place.unsqueeze(-1),
        #                                  self.object_pos_b,
        #                                  torch.where(self.is_lift.unsqueeze(-1),
        #                                              self.object_pos_b,
        #                                              torch.where(self.is_grasp.unsqueeze(-1),
        #                                                          self.object_pos_b,
        #                                                          self.robot_grasp_pos_b)))
                     

        # current_goal_info_t = torch.where(self.is_place.unsqueeze(-1),
        #                                   final_goal_pos_tcp,
        #                                   torch.where(self.is_lift.unsqueeze(-1),
        #                                               place_pos_obj,
        #                                               torch.where(self.is_grasp.unsqueeze(-1),
        #                                                           lift_pos_obj,
        #                                                           sub_goal_pos_tcp)))

        current_goal_info_t = torch.where(self.is_put.unsqueeze(-1),
                                          final_goal_pos_tcp,
                                          torch.where(self.is_place.unsqueeze(-1),
                                                     place_pos_obj,
                                                     torch.where(self.is_lift.unsqueeze(-1),
                                                                 place_pos_obj,
                                                                 torch.where(self.is_grasp.unsqueeze(-1),
                                                                             lift_pos_obj,
                                                                             sub_goal_pos_tcp))))


        # current_goal_info_t = torch.where(self.is_place.unsqueeze(-1),
        #                                   place_pos_obj,
        #                                   torch.where(self.is_lift.unsqueeze(-1),
        #                                               place_pos_obj,
        #                                               torch.where(self.is_grasp.unsqueeze(-1),
        #                                                           lift_pos_obj,
        #                                                           sub_goal_pos_tcp)))
        
        # current_phase = torch.zeros(self.num_envs, device=self.device)
        # current_phase = torch.where(self.is_grasp, 1.0, current_phase)
        # current_phase = torch.where(self.is_lift,  2.0, current_phase)
        # current_phase = torch.where(self.is_place, 3.0, current_phase)
        # current_phase /= 3.0

        # Appraoch : [0, 0, 0, 0]
        phase_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Grasp : [1, 0, 0, 0]
        phase_indices = torch.where(self.is_reach & ~self.is_grasp, 1, phase_indices)
        # Lift :  [0, 1, 0, 0]
        phase_indices = torch.where(self.is_grasp & ~self.is_lift,  2, phase_indices)
        # Place : [0, 0, 1, 0]
        phase_indices = torch.where(self.is_lift  & ~self.is_place, 3, phase_indices)
        # Open : [0, 0, 0, 1]
        phase_indices = torch.where(self.is_place & ~self.is_put,   4, phase_indices)
        # 
        phase_indices = torch.where(self.is_put,                    5, phase_indices)
        # one-hot encoding
        phase_encoding = torch.nn.functional.one_hot(phase_indices, num_classes=6).float()

        obs = torch.cat(
            (   
                # robot joint pose (9)
                joint_pos_scaled[:, 0:self.num_active_joints+2],
                # robot joint velocity (9)
                self.robot_joint_vel[:, 0:self.num_active_joints+2],
                # TCP 6D pose w.r.t Root frame (7)
                self.robot_grasp_pos_b,
                # Object 6D pose w.r.t Root frame (7)
                self.object_pos_b,
                # Goal Info w.r.t body frame (7)
                target_frame_pos_b,
                # Goal Info w.r.t Target frame (7)
                current_goal_info_t,
                # # Target Gripper State (1)
                # gripper, 
                # Phase Encoding (5)
                phase_encoding
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

    
        # ================== Lift Point 리셋 =================
        # Setting Final Goal 3D Location
        self.object_lift_pos_w[env_ids, :3] = object_default_pos_w[:, :3] + 0.2 * self.z_unit_tensor[env_ids]
        # Setting Final Goal 3D Rotation -> Aligned XYZ axis with TCP
        self.object_lift_pos_w[env_ids, 3:7] = object_default_state[:, 3:7]
        self.object_lift_pos_b[env_ids, :] = torch.cat(subtract_frame_transforms(
            self._robot.data.root_state_w[env_ids, :3], self._robot.data.root_state_w[env_ids, 3:7], 
            self.object_lift_pos_w[env_ids, :3], self.object_lift_pos_w[env_ids, 3:7]
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

        object_default_state_place[:, :3] += place_loc_noise + self.scene.env_origins[env_ids, :3] + 0.0 * self.z_unit_tensor[env_ids]
        object_default_state_place[:, 3:7] = quat_mul(tcp_quat, rot_noise)

        self.object_place_pos_w[env_ids] = object_default_state_place[:, :7]
        self.object_place_pos_b[env_ids] = torch.cat(subtract_frame_transforms(
            self._robot.data.root_state_w[env_ids, :3], self._robot.data.root_state_w[env_ids, 3:7],
            self.object_place_pos_w[env_ids, :3], self.object_place_pos_w[env_ids, 3:7]
        ), dim=1)


        # ================ Retract Point 리셋 ==================
        self.final_goal_pos_w[env_ids, :3] = self.object_place_pos_w[env_ids, :3] + 0.2 * self.z_unit_tensor[env_ids]
        self.final_goal_pos_w[env_ids, 3:7] = self.object_place_pos_w[env_ids, 3:7]
        self.final_goal_pos_b[env_ids] = torch.cat(subtract_frame_transforms(
            self._robot.data.root_state_w[env_ids, :3], self._robot.data.root_state_w[env_ids, 3:7],
            self.final_goal_pos_w[env_ids, :3], self.final_goal_pos_w[env_ids, 3:7]
        ), dim=1)


        # ================ Curriculum =================
        place_success_rate = sum(self.place_buffer) / max(1, len(self.place_buffer))
        # episode_success_rate = sum(self.success_buffer) / max(1, len(self.success_buffer))
        # still_lift_rate = sum(self.still_lift_buffer) / max(1, len(self.still_lift_buffer))
        # self.cfg.place_loc_th = self.cfg.place_loc_th_min + (self.cfg.place_loc_th_max - self.cfg.place_loc_th_min) * math.exp(-self.cfg.decay_ratio * place_success_rate)


        
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
        self.robot_hand_lin_vel[env_ids] = torch.norm(self._robot.data.body_lin_vel_w[env_ids, self.hand_link_idx, :3], dim=-1)
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
                                                                self.object_lift_pos_b[env_ids, 3:7])
            
            self.prev_lift_error[env_ids, 0] = torch.norm(self.object_pos_b[env_ids, :3] - \
                                                             self.object_lift_pos_b[env_ids, :3], dim=1)
            
            self.prev_lift_error[env_ids, 1] = quat_error_magnitude(self.object_pos_b[env_ids, 3:7], 
                                                                       self.object_lift_pos_b[env_ids, 3:7])
            
            self.prev_place_error[env_ids, 0] = torch.norm(self.object_pos_b[env_ids, :3] - \
                                                           self.object_place_pos_b[env_ids, :3], dim=1)
            
            self.prev_place_error[env_ids, 1] = quat_error_magnitude(self.object_pos_b[env_ids, 3:7],
                                                                     self.object_place_pos_b[env_ids, 3:7])

            
            self.prev_retract_error[env_ids, 0] = torch.norm(self.robot_grasp_pos_b[env_ids, :3] - \
                                                             self.final_goal_pos_b[env_ids, :3], dim=1)

            self.prev_retract_error[env_ids, 1] = quat_error_magnitude(self.robot_grasp_pos_b[env_ids, 3:7],
                                                                       self.final_goal_pos_b[env_ids, 3:7])
            

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

            retract_error_xyz = torch.abs(self.robot_grasp_pos_b[env_ids, :3] - self.final_goal_pos_b[env_ids, :3])
            self.prev_weighted_retract_error[env_ids, 0] = torch.sqrt(self.cfg.wx * torch.square(retract_error_xyz[:, 0]) +
                                                                    self.cfg.wy * torch.square(retract_error_xyz[:, 1]) + 
                                                                    self.cfg.wz * torch.square(retract_error_xyz[:, 2])).squeeze(-1)

            self.prev_weighted_retract_error[env_ids, 1] = self.prev_place_error[env_ids, 1]


            self.prev_lift[env_ids] = torch.zeros_like(self.is_reach[env_ids], dtype=torch.bool, device=self.device)
            self.prev_grasp[env_ids] = torch.zeros_like(self.is_reach[env_ids], dtype=torch.bool, device=self.device)
            self.prev_place[env_ids] = torch.zeros_like(self.is_reach[env_ids], dtype=torch.bool, device=self.device)
            self.prev_put[env_ids] = torch.zeros_like(self.is_reach[env_ids], dtype=torch.bool, device=self.device)
            self.prev_imp_commands[env_ids] = torch.zeros_like(self.imp_commands[env_ids], device=self.device)
        else:
            self.prev_weighted_retract_error[env_ids] = self.weighted_retract_error[env_ids]
            self.prev_weighted_approach_error[env_ids] = self.weighted_approach_error[env_ids]
            self.prev_weighted_place_error[env_ids] = self.weighted_place_error[env_ids]
            self.prev_approach_error[env_ids] = self.approach_error[env_ids]
            self.prev_lift_error[env_ids] = self.lift_error[env_ids]    
            self.prev_place_error[env_ids] = self.place_error[env_ids]
            self.prev_retract_error[env_ids] = self.retract_error[env_ids]
            self.prev_lift[env_ids] = self.is_lift[env_ids].clone()
            self.prev_grasp[env_ids] = self.is_grasp[env_ids].clone()
            self.prev_place[env_ids] = self.is_place[env_ids].clone()
            self.prev_put[env_ids] = self.is_put[env_ids].clone()
            


        # ========= Position Error 업데이트 =========
        # Approach : Location
        loc_error_xyz = torch.abs(self.robot_grasp_pos_b[env_ids, :3] - self.object_pos_b[env_ids, :3])
        self.approach_error[env_ids, 0] = torch.norm(self.robot_grasp_pos_b[env_ids, :3] - self.object_pos_b[env_ids, :3], dim=1)
        # self.approach_error[env_ids, 0] = torch.sqrt(self.cfg.wx * torch.square(loc_error_xyz[:, 0]) +
        #                                              self.cfg.wy * torch.square(loc_error_xyz[:, 1]) + 
        #                                              self.cfg.wz * torch.square(loc_error_xyz[:, 2])).squeeze(-1)  
        # Appraoch : Rotation -> Pre-defined angle (TCP 정렬 각)
        self.approach_error[env_ids, 1] = quat_error_magnitude(self.robot_grasp_pos_b[env_ids, 3:7],
                                                       self.object_lift_pos_b[env_ids, 3:7])
        
        self.weighted_approach_error[env_ids, 0] = torch.sqrt(self.cfg.wx * torch.square(loc_error_xyz[:, 0]) +
                                                              self.cfg.wy * torch.square(loc_error_xyz[:, 1]) + 
                                                              self.cfg.wz * torch.square(loc_error_xyz[:, 2])).squeeze(-1)
        
        self.weighted_approach_error[env_ids, 1] = self.approach_error[env_ids, 1]

        
        # Lift : Location
        self.lift_error[env_ids, 0] = torch.norm(self.object_pos_b[env_ids, :3] - \
                                                    self.object_lift_pos_b[env_ids, :3], dim=1)
        # Lift : Rotation
        self.lift_error[env_ids, 1] = quat_error_magnitude(self.object_pos_b[env_ids, 3:7], 
                                                              self.object_lift_pos_b[env_ids, 3:7])
        

        # Place : Location
        place_error_xyz = torch.abs(self.object_pos_b[env_ids, :3] - self.object_place_pos_b[env_ids, :3])
        self.place_error[env_ids, 0] = torch.norm(self.object_pos_b[env_ids, :3] - self.object_place_pos_b[env_ids, :3], dim=1)
        # self.place_error[env_ids, 0] = torch.sqrt(self.cfg.wx * torch.square(place_error_xyz[:, 0]) +
        #                                           self.cfg.wy * torch.square(place_error_xyz[:, 1]) + 
        #                                           self.cfg.wz * torch.square(place_error_xyz[:, 2])).squeeze(-1)
        # Lift : Rotation
        self.place_error[env_ids, 1] = quat_error_magnitude(self.object_pos_b[env_ids, 3:7],
                                                            self.object_place_pos_b[env_ids, 3:7])

        self.weighted_place_error[env_ids, 0] = torch.sqrt(self.cfg.wx * torch.square(place_error_xyz[:, 0]) +
                                                           self.cfg.wy * torch.square(place_error_xyz[:, 1]) + 
                                                           self.cfg.wz * 0.25 * torch.square(place_error_xyz[:, 2])).squeeze(-1)

        self.weighted_place_error[env_ids, 1] = self.place_error[env_ids, 1]


        # Retract : Location
        self.retract_error[env_ids, 0] = torch.norm(self.robot_grasp_pos_b[env_ids, :3] - \
                                                    self.final_goal_pos_b[env_ids, :3], dim=1)

        self.retract_error[env_ids, 1] = quat_error_magnitude(self.robot_grasp_pos_b[env_ids, 3:7],
                                                              self.final_goal_pos_b[env_ids, 3:7])
        

        retract_error_xyz = torch.abs(self.robot_grasp_pos_b[env_ids, :3] - self.final_goal_pos_b[env_ids, :3])
        self.weighted_retract_error[env_ids, 0] = torch.sqrt(0.25 * torch.square(retract_error_xyz[:, 0]) +
                                                             0.25 * torch.square(retract_error_xyz[:, 1]) + 
                                                             4 * torch.square(retract_error_xyz[:, 2])).squeeze(-1)

        self.weighted_retract_error[env_ids, 1] = self.retract_error[env_ids, 1]

        # print(f"retract loc error : {self.retract_error[env_ids, 0]}")


        # ============ Phase Signal 업데이트 ============
        self.update_phase_signal(env_ids, reset)
        self.update_end_condition(env_ids, reset)
            
        # ======== Visualization ==========
        self.tcp_marker.visualize(self.object_lift_pos_w[:, :3], self.object_lift_pos_w[:, 3:7])
        self.lift_marker.visualize(self.final_goal_pos_w[:, :3], self.final_goal_pos_w[:, 3:7])
        self.place_marker.visualize(self.object_place_pos_w[:, :3], self.object_place_pos_w[:, 3:7])



    def update_phase_signal(self, env_ids, reset):
        self.is_reach[env_ids] = torch.logical_or(torch.logical_and(self.approach_error[env_ids, 0] < self.cfg.loc_th, self.approach_error[env_ids, 1] < self.cfg.rot_th),
                                                    self.prev_lift[env_ids])

        self.is_grasp[env_ids] = torch.logical_and(self.is_reach[env_ids], 
                                                   self.object_pos_b[env_ids, 2] > torch.max(torch.tensor(self.cfg.loc_th, device=self.device), self.obj_width[0]/2))
        
        self.is_lift[env_ids] = torch.logical_or(torch.logical_and(self.is_grasp[env_ids], 
                                                                      torch.logical_and(self.lift_error[env_ids, 0] < self.cfg.loc_th,
                                                                                        self.lift_error[env_ids, 1] < self.cfg.rot_th)),
                                                    self.prev_lift[env_ids])

        
        self.is_in_place[env_ids] = (self.is_lift[env_ids]) & \
                                    (self.place_error[env_ids, 0] < self.cfg.place_loc_th * 4) & \
                                    (self.approach_error[env_ids, 0] < self.cfg.loc_th * 2)

        self.is_place[env_ids] = torch.logical_or(torch.logical_and(self.is_lift[env_ids], 
                                                                    torch.logical_and(self.place_error[env_ids, 0] < self.cfg.place_loc_th,
                                                                                      self.place_error[env_ids, 1] < self.cfg.place_rot_th)),
                                                  self.prev_place[env_ids])
        
        self.is_first_place[env_ids] = self.is_place[env_ids] & ~self.prev_place[env_ids]
        
        # place에서 place zone인 경우, grasp 조건 무효
        self.is_grasp[env_ids] = (self.is_grasp[env_ids]) | (self.is_in_place[env_ids]) | (self.is_place[env_ids])

        gripper_state = torch.stack([self.robot_joint_pos[:, self.left_finger_joint_idx], self.robot_joint_pos[:, self.right_finger_joint_idx]], dim=-1).mean(dim=-1)
        # self.is_put[env_ids] = ((self.is_place[env_ids]) & (gripper_state[env_ids] > 0.99 * 0.04)) | self.prev_put[env_ids]
        self.is_put[env_ids] = ((self.is_place[env_ids]) & (gripper_state[env_ids] > 0.99 * 0.04))


    def update_end_condition(self, env_ids, reset):
        if not reset:
            self.contact[env_ids] = torch.logical_and(torch.logical_and(~self.is_reach[env_ids], 
                                                                            self.approach_error[env_ids, 0] < self.cfg.loc_th * 4),
                                                         torch.norm(self._object.data.root_vel_w[env_ids], dim=1) > 1e-1)

            quasi_drop  = (torch.logical_and(self.prev_grasp[env_ids], ~self.is_grasp[env_ids]) | \
                           torch.logical_and(self.object_pos_w[env_ids, 2] < 0, ~self.is_grasp[env_ids]))
            self.drop[env_ids] = quasi_drop & ~self.is_place[env_ids] & ~self.is_in_place[env_ids]

            self.collision[env_ids] = (self.is_lift[env_ids]) & (self.object_pos_b[env_ids, 2] < self.object_place_pos_b[env_ids, 2]) & (~self.is_place[env_ids])

            self.still_lift[env_ids] = (self.is_put[env_ids]) & (self.object_pos_b[env_ids, 2] > self.object_place_pos_b[env_ids, 2] * 2)

            self.is_success[env_ids] = torch.logical_and(self.is_put[env_ids], 
                                                torch.logical_and(self.retract_error[env_ids, 0] < self.cfg.retract_loc_th,
                                                                self.retract_error[env_ids, 1] < self.cfg.retract_rot_th))
            
    

        else:
            self.contact[env_ids] = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
            self.drop[env_ids] = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
            self.collision[env_ids] = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
            self.re_pick[env_ids] = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
            self.still_lift[env_ids] = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
            self.is_success[env_ids] = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
            


    def logging_process(self, done):
        if torch.any(done):
            done_ids = torch.where(done)[0]

            self.success_buffer.extend(
                self.is_success.index_select(0, done_ids).float().detach().cpu().tolist()
            )

            self.place_buffer.extend(
                self.is_place.index_select(0, done_ids).float().detach().cpu().tolist()
            )

            self.lift_buffer.extend(
                self.is_lift.index_select(0, done_ids).float().detach().cpu().tolist()
            )

            self.grasp_buffer.extend(
                self.is_grasp.index_select(0, done_ids).float().detach().cpu().tolist()
            )

            self.still_lift_buffer.extend(
                self.still_lift.index_select(0, done_ids).float().detach().cpu().tolist()
            )

            if self.logging_count % self.cfg.logging_interval == 0:
                print("============================== [INFO] ================================")
                print(
                        f"[Phase INFO] : Reach / Grasp / Lift / Place / Put: "
                        f"{self.is_reach.sum().item()}/"
                        f"{self.is_grasp.sum().item()}/"
                        f"{self.is_lift.sum().item()}/"
                        f"{self.is_place.sum().item()}/"
                        f"{self.is_put.sum().item()}"
                    )
                print(
                        f"[End INFO] : Contact / Drop / Collision / Still Lift / Success : "
                        f"{self.contact.sum().item()}/"
                        f"{self.drop.sum().item()}/"
                        f"{self.collision.sum().item()}/"
                        f"{self.still_lift.sum().item()}/"
                        f"{self.is_success.sum().item()}"
                    )
                
                print(
                        f"[Additional Physics INFO] : MGAS / MHV : "
                        f"{torch.mean(self.r_gripper[self.is_place]).item() if self.is_place.any() else 0 :.3f}/"
                        f"{torch.mean(self.robot_hand_lin_vel[self.is_put]).item() if self.is_put.any() else 0 :.3f}"
                    )         

                self.logging_count = 0
 
            self.logging_count += 1



    def update_additional_info(self, reset: bool = False):

        self.extras["log"] = copy.deepcopy(
            {
                # Episode Info
                "grasp_success_rate": torch.tensor(sum(self.grasp_buffer) / max(1, len(self.grasp_buffer)), device=self.device),
                "lift_success_rate": torch.tensor(sum(self.lift_buffer) / max(1, len(self.lift_buffer)), device=self.device),
                "place_success_rate": torch.tensor(sum(self.place_buffer) / max(1, len(self.place_buffer)), device=self.device),
                "epi_success_rate": torch.tensor(sum(self.success_buffer) / max(1, len(self.success_buffer)), device=self.device),
                "still_lift_rate": torch.tensor(sum(self.still_lift_buffer) / max(1, len(self.still_lift_buffer)), device=self.device),
                "place_threshold": torch.tensor(self.cfg.place_loc_th, device=self.device),
                
                # Rewards Info
                "Approach_loc": torch.mean(self.r_pos),
                "Approach_rot":torch.mean(self.r_rot),
                "Lift_loc": torch.mean(self.r_lift_loc),
                "Lift_rot": torch.mean(self.r_lift_rot),
                "Place_loc": torch.mean(self.r_place_loc),
                "Place_rot": torch.mean(self.r_place_rot),
                "Retract_loc": torch.mean(self.r_retract_loc[self.is_put])  if self.is_put.any() else 0,
                "Retract_rot": torch.mean(self.r_retract_rot[self.is_put])  if self.is_put.any() else 0,
                "Gripper_commands": torch.mean(self.r_gripper[self.is_place]) if self.is_place.any() else 0,
                
            }
        )

        self.extras["robot"] = copy.deepcopy(
            {
                # control Info
                "impedance_desired_joint_pos": self.imp_commands[:, :self.num_active_joints].detach(),
                "impedance_stiffness" : self.imp_commands[:,   self.num_active_joints : 2*self.num_active_joints].detach(),
                "impedance_damping": self.imp_commands[:, 2*self.num_active_joints : 3*self.num_active_joints].detach(),
                # robot info
                "joint_pos": self.robot_joint_pos.detach(),
                "joint_vel" : self.robot_joint_vel.detach(),
                "hand_vel" : self.robot_hand_lin_vel.detach(),
                # reward Info
                "total_reward": torch.mean(self.r_total).unsqueeze(-1).detach()

            }
        )


        if not reset:
            self.extras["probe"] = copy.deepcopy(
                {
                    # phase flags
                    "is_reach":   self.is_reach.detach(),
                    "is_grasp":   self.is_grasp.detach(),
                    "is_lift": self.is_lift.detach(),
                    "is_place": self.is_place.detach(),
                    "is_put": self.is_put.detach(),
                    "is_success": self.is_success.detach(),
                    # errors
                    "approach_loc": self.approach_error[:, 0].detach(),
                    "approach_rot": self.approach_error[:, 1].detach(),
                    "lift_loc":  self.lift_error[:, 0].detach(),
                    "lift_rot":  self.lift_error[:, 1].detach(),
                    "place_loc":    self.place_error[:, 0].detach(),
                    "place_rot":    self.place_error[:, 1].detach(),
                    # weighted versions
                    "w_approach_loc": self.weighted_approach_error[:, 0].detach(),
                    "w_approach_rot": self.weighted_approach_error[:, 1].detach(),
                    "w_place_loc":    self.weighted_place_error[:, 0].detach(),
                    "w_place_rot":    self.weighted_place_error[:, 1].detach(),
                })



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