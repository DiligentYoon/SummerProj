# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from abc import abstractmethod

from isaaclab.sensors import TiledCamera
from isaaclab.assets import RigidObject
from isaaclab.utils.math import sample_uniform

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
                self.table_id = int(key)
            
            if (self.table_id is not None) and (self.object_id is not None):
                break
        

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

    
    def _reset_idx(self, env_ids: torch.Tensor | None):
        # Reset Robot 
        super()._reset_idx(env_ids)
        # Reset Camera 
        for cam in self.cam_list:
            cam.reset(env_ids)
        # Reset Object : MultipleAsset 영향으로, 물체의 종류도 바뀌는 것을 기대
        loc_noise_x = sample_uniform(-0.2, 0.2, (len(env_ids), 1), device=self.device)
        loc_noise_y = sample_uniform(-0.2, 0.2, (len(env_ids), 1), device=self.device)
        loc_noise_z = sample_uniform(0.0, 0.0, (len(env_ids), 1), device=self.device)
        loc_noise = torch.cat([loc_noise_x, loc_noise_y, loc_noise_z], dim=-1)
        # 난이도 고려, 회전 정보는 일단 그대로
        default_obj_state = self._object.data.default_root_state[env_ids, :]
        default_obj_state[:, :3] += loc_noise + self.scene.env_origins[env_ids, :3]
        # object 상태 업데이트
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