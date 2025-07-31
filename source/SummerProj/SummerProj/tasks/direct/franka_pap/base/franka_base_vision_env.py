# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from abc import abstractmethod

import omni.usd
from pxr import Sdf
from isaacsim.core.utils import bounds

from isaaclab.sensors import TiledCamera
from isaaclab.assets import RigidObject
from isaaclab.utils.math import sample_uniform, quat_from_angle_axis, matrix_from_quat

from .franka_base_env_diol import FrankaBaseDIOLEnv
from .franka_base_vision_cfg import FrankaVisionBaseCfg

class FrankaVisionBaseEnv(FrankaBaseDIOLEnv):
    cfg: FrankaVisionBaseCfg

    def __init__(self, cfg: FrankaVisionBaseCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.data_type = self.cfg.front_camera.data_types
        self.table_id = None
        self.object_id = None

        # 모든 환경의 table & object semantic label이 동일하다 가정
        semantic_info = self.cam_list[0].data.info["semantic_segmentation"]["idToLabels"]
        for key, value in semantic_info.items():
            if value.get('class') == 'table':
                self.table_id = int(key)
            if value.get('class') == 'object':
                self.object_id = int(key)
            
            if (self.table_id is not None) and (self.object_id is not None):
                break
        
        # Camera Information
        self.cam_intrinsic_mat = self.cam_list[0].data.intrinsic_matrices[0]

        

    def _setup_scene(self):
        super()._setup_scene()
        self._object: RigidObject =  RigidObject(self.cfg.object)
        self.cam_list: list[TiledCamera, TiledCamera, TiledCamera] = [TiledCamera(self.cfg.front_camera), 
                                                                      TiledCamera(self.cfg.left_behind_camera), 
                                                                      TiledCamera(self.cfg.right_behind_camera)]
        self.scene.rigid_objects["object"] = self._object
        self.scene.sensors["front_cam"] = self.cam_list[0]
        self.scene.sensors["left_cam"] = self.cam_list[1]
        self.scene.sensors["right_cam"] = self.cam_list[2]

        self.scene.clone_environments(copy_from_source=False)

        # Spawn Light
        light_cfg = self.cfg.dome_light.spawn
        light_cfg.func(self.cfg.dome_light.prim_path, light_cfg)

        # Table Spec은 모든 환경에서 동일
        table_bb_cache = bounds.create_bbox_cache()
        self.table_prim = omni.usd.get_context().get_stage().GetPrimAtPath(
            f"/World/envs/env_0/Table"
        )
        self.table_aabb = bounds.compute_aabb(bbox_cache=table_bb_cache, 
                                              prim_path=self.table_prim.GetPrimPath().pathString)
        
        # Object의 Default 상태에서의 BB Spec도 모든 환경에서 동일
        obj_bb_cache = bounds.create_bbox_cache()
        self.obj_prim = omni.usd.get_context().get_stage().GetPrimAtPath(
            f"/World/envs/env_0/Object"
        )
        centroid, axes, half_extent = bounds.compute_obb(obj_bb_cache, self.obj_prim.GetPrimPath().pathString)
        self.obj_corners = torch.from_numpy(bounds.get_obb_corners(centroid, axes, half_extent)).to(dtype=torch.float32, device=self.device)
    

    
    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        # 물체의 고유한 Bounding Box 미리 측정
        # Rotation 결정
        # Rotation 정보를 바탕으로 Rotation Matrix 만들고, AABB에 회전 적용
        # 최소 오프셋 결정
        rot_noise = quat_from_angle_axis(torch.pi/2 * torch.ones(len(env_ids), device=self.device), 
                                         self.y_unit_tensor)
        spawn_z_w = []
        for i, env_id in enumerate(env_ids.tolist()):
            rot_mat = matrix_from_quat(rot_noise[i, :])
            rotated_corners = torch.matmul(self.obj_corners, rot_mat.T)

            rotated_bb_delta_z = (torch.max(rotated_corners[:, 2]) - torch.min(rotated_corners[:, 2])).cpu().numpy()
            
            spawn_z_w.append(rotated_bb_delta_z/2 + self.table_aabb[-1])
                
        loc_noise_x = sample_uniform(-0.2, 0.2, (len(env_ids), 1), device=self.device)
        loc_noise_y = sample_uniform(-0.4, 0.4, (len(env_ids), 1), device=self.device)
        loc_noise_z = torch.tensor(spawn_z_w, device=self.device).reshape(-1, 1)
        loc_noise = torch.cat([loc_noise_x, loc_noise_y, loc_noise_z], dim=-1)

        default_obj_state = self._object.data.default_root_state[env_ids, :]
        default_obj_state[:, :3] += self.scene.env_origins[env_ids, :] + loc_noise
        default_obj_state[:, 3:7] = rot_noise

        self._object.write_root_pose_to_sim(default_obj_state[:, :7], env_ids=env_ids)
        self._object.write_root_velocity_to_sim(default_obj_state[:, 7:], env_ids=env_ids)

        

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