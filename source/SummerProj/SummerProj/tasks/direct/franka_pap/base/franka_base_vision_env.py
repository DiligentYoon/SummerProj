# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from abc import abstractmethod

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import TiledCamera
from isaaclab.assets import RigidObject
from isaaclab.utils.math import sample_uniform

from .franka_base_env import FrankaBaseEnv
from .franka_base_vision_cfg import FrankaVisionBaseCfg

class FrankaVisionBaseEnv(FrankaBaseEnv):
    cfg: FrankaVisionBaseCfg

    def __init__(self, cfg: FrankaVisionBaseCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.data_type = self.cfg.front_camera.data_types
        self.table_id = None
        self.object_id = None
        

    def _setup_scene(self):
        self._object: RigidObject =  RigidObject(self.cfg.object)
        self.cam_list: list[TiledCamera, TiledCamera, TiledCamera] = [TiledCamera(self.cfg.front_camera), 
                                                                      TiledCamera(self.cfg.left_behind_camera), 
                                                                      TiledCamera(self.cfg.right_behind_camera)]
        self.scene.rigid_objects["object"] = self._object
        self.scene.sensors["front_cam"] = self.cam_list[0]
        self.scene.sensors["left_cam"] = self.cam_list[1]
        self.scene.sensors["right_cam"] = self.cam_list[2]

        # 모든 환경의 table & object semantic label이 동일하다 가정
        semantic_info = self.cam_list[0].data.info[0]["semantic_segmentation"]["idToLabels"]
        for key, value in semantic_info.items():
            if value.get('class') == 'table':
                self.table_id = int(key)
            if value.get('class') == 'object':
                self.table_id = int(key)
            
            if self.table_id is not None and self.object_id is not None:
                break
        super()._setup_scene()

    
    def _reset_idx(self, env_ids: torch.Tensor | None):
        # Robot 리셋
        super()._reset_idx(env_ids)
        # Camera 리셋
        for cam in self.cam_list:
            cam.reset(env_ids)
        # Object 리셋
        


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