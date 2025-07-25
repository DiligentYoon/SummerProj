# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import random

import numpy as np
import omni.replicator.core as rep
import omni.usd
from isaacsim.core.api import World
from omni.usd import get_context
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils import prims
from isaacsim.core.utils.bounds import compute_combined_aabb, compute_obb, create_bbox_cache, get_obb_corners
from isaacsim.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles
from isaacsim.core.utils.semantics import remove_all_semantics
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics


# Add colliders to Gprim and Mesh descendants of the root prim
def add_colliders(root_prim, approx_type="convexDecomposition"):
    # Iterate descendant prims (including root) and add colliders to mesh or primitive types
    for desc_prim in Usd.PrimRange(root_prim):
        if desc_prim.IsA(UsdGeom.Mesh) or desc_prim.IsA(UsdGeom.Gprim):
            # Physics
            if not desc_prim.HasAPI(UsdPhysics.CollisionAPI):
                collision_api = UsdPhysics.CollisionAPI.Apply(desc_prim)
            else:
                collision_api = UsdPhysics.CollisionAPI(desc_prim)
            collision_api.CreateCollisionEnabledAttr(True)
        # Add mesh specific collision properties only to mesh types
        if desc_prim.IsA(UsdGeom.Mesh):
            # Add mesh collision properties to the mesh (e.g. collider aproximation type)
            if not desc_prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(desc_prim)
            else:
                mesh_collision_api = UsdPhysics.MeshCollisionAPI(desc_prim)
            mesh_collision_api.CreateApproximationAttr().Set(approx_type)


# Clear any previous semantic data in the stage
def remove_previous_semantics(stage, recursive: bool = False):
    prims = stage.Traverse()
    for prim in prims:
        remove_all_semantics(prim)


# Run a simulation
def simulate_falling_objects(table_prim, assets_root_path, config, max_sim_steps=250, num_boxes=8):
    # Create the isaac sim world to run any physics simulations
    world = World(physics_dt=1.0 / 90.0, stage_units_in_meters=1.0)

    # Set a random relative offset to the pallet using the forklift transform as a base frame
    forklift_tf = omni.usd.get_world_transform_matrix(table_prim)
    pallet_offset_tf = Gf.Matrix4d().SetTranslate(Gf.Vec3d(random.uniform(-1, 1), random.uniform(-1.5, -1.0), 0))
    pallet_pos = (pallet_offset_tf * forklift_tf).ExtractTranslation()

    # Spawn a pallet prim at a random offset from the forklift
    pallet_prim = prims.create_prim(
        prim_path=f"/World/SimulatedPallet",
        position=pallet_pos,
        orientation=euler_angles_to_quat([0, 0, random.uniform(0, math.pi)]),
        usd_path=assets_root_path + config["pallet"]["url"],
        semantic_label=config["pallet"]["class"]
    )

    # Wrap the pallet as simulation ready with a simplified collider
    add_colliders(pallet_prim, approx_type="boundingCube")
    pallet_rigid_prim = SingleRigidPrim(prim_path=str(pallet_prim.GetPrimPath()))
    pallet_rigid_prim.enable_rigid_body_physics()

    # Use the height of the pallet as a spawn base for the boxes
    bb_cache = create_bbox_cache()
    spawn_height = bb_cache.ComputeLocalBound(pallet_prim).GetRange().GetSize()[2] * 1.1

    # Keep track of the last box to stop the simulation early once it stops moving
    last_box = None
    # Spawn boxes falling on the pallet
    for i in range(num_boxes):
        # Spawn the carbox prim by creating a new Xform prim and adding the USD reference to it
        box_prim = prims.create_prim(
            prim_path=f"/World/SimulatedCardbox_{i}",
            position=pallet_pos + Gf.Vec3d(random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2), spawn_height),
            orientation=euler_angles_to_quat([0, 0, random.uniform(0, math.pi)]),
            usd_path=assets_root_path + config["cardbox"]["url"],
            semantic_label=config["cardbox"]["class"],
        )

        # Get the next spawn height for the box
        spawn_height += bb_cache.ComputeLocalBound(box_prim).GetRange().GetSize()[2] * 1.1

        # Wrap the prim as simulation ready with a simplified collider
        add_colliders(box_prim, approx_type="convexDecomposition")
        box_rigid_prim = SingleRigidPrim(prim_path=str(box_prim.GetPrimPath()))
        box_rigid_prim.enable_rigid_body_physics()

        # Cache the rigid prim
        last_box = box_rigid_prim

    # Reset the world to handle the physics of the newly created rigid prims
    world.reset()

    # Simulate the world for the given number of steps or until the highest box stops moving
    for i in range(max_sim_steps):
        world.step(render=False)
        if last_box and np.linalg.norm(last_box.get_linear_velocity()) < 0.001:
            print(f"[scene_based_sdg] Simulation finished at step {i}..")
            break


# Register the boxes and materials randomizer graph
def register_scatter_objs(table_prim, assets_root_path, config, object_class: str):
    """
        박스 세팅 -> 여기서 정의한 투명한 Ground위에 Scattering
    """
    bb_cache = create_bbox_cache()

    # ========== Offset 계산을 위한 임시 object 생성 ============

    temp_prim_path = "/World/obj_for_measurement"
    temp_mug_prim_for_measure = prims.create_prim(
        prim_path=temp_prim_path,
        usd_path=assets_root_path + config[object_class]["url"],
        position=(10000, 10000, 10000)  # 카메라에 보이지 않는 먼 곳에 생성
    )

    obj_bbox = bb_cache.ComputeLocalBound(temp_mug_prim_for_measure)
    obj_range = obj_bbox.GetRange().GetSize()
    obj_height = obj_range[2]
    z_offset = obj_height / 2.0

    prims.delete_prim(temp_prim_path)

    # min_corner = obj_bbox.GetBox().GetMin()
    # max_corner = obj_bbox.GetBox().GetMax()
    # local_corners = [
    #     Gf.Vec3d(min_corner[0], min_corner[1], min_corner[2]),
    #     Gf.Vec3d(max_corner[0], min_corner[1], min_corner[2]),
    #     Gf.Vec3d(min_corner[0], max_corner[1], min_corner[2]),
    #     Gf.Vec3d(max_corner[0], max_corner[1], min_corner[2]),
    #     Gf.Vec3d(min_corner[0], min_corner[1], max_corner[2]),
    #     Gf.Vec3d(max_corner[0], min_corner[1], max_corner[2]),
    #     Gf.Vec3d(min_corner[0], max_corner[1], max_corner[2]),
    #     Gf.Vec3d(max_corner[0], max_corner[1], max_corner[2]),
    # ]
    
    # =============================================================

    # Calculate the bounds of the prim to create a scatter plane of its size
    bbox3d_gf = bb_cache.ComputeLocalBound(table_prim)
    prim_tf_gf = omni.usd.get_world_transform_matrix(table_prim)

    # Calculate the bounds of the prim
    bbox3d_gf.Transform(prim_tf_gf)
    range_size = bbox3d_gf.GetRange().GetSize()

    # Get the quaterion of the prim in xyzw format from usd
    prim_quat_gf = prim_tf_gf.ExtractRotation().GetQuaternion()
    prim_quat_xyzw = (prim_quat_gf.GetReal(), *prim_quat_gf.GetImaginary())

    # Create a plane on the pallet to scatter the boxes on
    plane_scale = (range_size[0] * 0.7, range_size[1] * 0.7, 1)
    plane_pos_gf = prim_tf_gf.ExtractTranslation() + Gf.Vec3d(0, 0, range_size[2])
    plane_rot_euler_deg = quat_to_euler_angles(np.array(prim_quat_xyzw), degrees=True)
    scatter_plane = rep.create.plane(
        scale=plane_scale, position=plane_pos_gf, rotation=plane_rot_euler_deg, visible=False
    )

    def scatter_obj():
        # object의 bbox 기반으로 offset 계산
        count = 1 
        objs = rep.create.from_usd(
            assets_root_path + config[object_class]["url"], 
            semantics=[("class", config[object_class]["class"])], 
            count=count
        )

        with objs:
            # final_positions = []
            # final_orientations = []

            # for i in range(count): 
            #     # a. 랜덤 회전 값(쿼터니언)을 Python으로 직접 생성합니다.
            #     random_euler = np.random.uniform(low=(0,0,0), high=(360,360,360))
            #     random_quat = Gf.Rotation(Gf.Vec3d(1,0,0), random_euler[0]) * \
            #                     Gf.Rotation(Gf.Vec3d(0,1,0), random_euler[1]) * \
            #                     Gf.Rotation(Gf.Vec3d(0,0,1), random_euler[2])

            #     rotation_mat = Gf.Matrix4d().SetRotate(random_quat)
            #     rotated_corners_z = [ rotation_mat.Transform(corner)[2] for corner in local_corners]
            #     z_offset_2 = -min(rotated_corners_z)
            #     final_positions.append((0, 0, z_offset))
            #     quat_xyzw = random_quat.GetQuaternion()
            #     final_orientations.append((quat_xyzw.GetReal(), *quat_xyzw.GetImaginary()))
            rep.randomizer.scatter_2d(scatter_plane, check_for_collisions=True, offset=z_offset)
            rep.modify.pose(rotation=rep.distribution.uniform((0, 0, 0), (360, 360, 360)))
        return objs.node

    rep.randomizer.register(scatter_obj)


def place_and_settle_objects(world, pallet_prim, assets_root_path, config, count: int = 1):
    """
        오브젝트를 팔레트 위 허공에 랜덤하게 배치한 후,
        물리 시뮬레이션을 통해 안정적으로 안착시킵니다.
    """
    # target object 지정
    object_class = config["target_obj"]
    
    # 팔레트에도 물리 속성을 한 번만 부여합니다.
    add_colliders(pallet_prim, approx_type="boundingCube")
    pallet_rigid_prim = SingleRigidPrim(str(pallet_prim.GetPrimPath()))
    pallet_rigid_prim.enable_rigid_body_physics()

    settled_objects = []
    for i in range(count):
        prim_path = f"/World/SettledObject_{i}"

        bb_cache = create_bbox_cache()
        pallet_tf = omni.usd.get_world_transform_matrix(pallet_prim)
        pallet_loc = pallet_tf.ExtractTranslation()
        world_bbox_range = bb_cache.ComputeWorldBound(pallet_prim).GetRange()
        plane_min = world_bbox_range.GetMin()
        plane_max = world_bbox_range.GetMax()
        spawn_base_z = plane_max[2] # 팔레트 상판 높이

        orientation = euler_angles_to_quat(np.random.uniform((0,0,0), (0,0,360)))

        if prims.is_prim_path_valid(prim_path):
            # 오브젝트가 이미 존재하면, 위치와 방향만 재설정합니다.
            rigid_prim = SingleRigidPrim(prim_path)
            rigid_prim.set_world_pose(position=pallet_loc + Gf.Vec3d(random.uniform(plane_min[0], plane_max[0]) * 0.6, 
                                                                     random.uniform(plane_min[1], plane_max[1]) * 0.6, 
                                                                     spawn_base_z + (i + 1) * 0.02),
                                      orientation=orientation)
        else:
            # 오브젝트가 없으면 새로 생성하고 물리 속성을 부여합니다.
            obj_prim = prims.create_prim(
                prim_path=prim_path,
                position=pallet_loc + Gf.Vec3d(random.uniform(plane_min[0], plane_max[0]) * 0.6, 
                                               random.uniform(plane_min[1], plane_max[1]) * 0.6, 
                                               spawn_base_z + (i + 1) * 0.02),
                orientation=orientation,
                usd_path=assets_root_path + config[object_class]["url"],
                semantic_label=config[object_class]["class"]
            )
            add_colliders(obj_prim, approx_type="convexDecomposition")
            rigid_prim = SingleRigidPrim(prim_path)
            rigid_prim.enable_rigid_body_physics()  
                 
        settled_objects.append(rigid_prim)
        print(f"Object Offset : {omni.usd.get_world_transform_matrix(rigid_prim.prim).ExtractTranslation() - pallet_loc}")

    # 5. 물리 시뮬레이션 실행 (안착 단계)
    world.play()
    # 약 1초간 시뮬레이션을 실행하여 오브젝트를 안정화시킵니다.
    for _ in range(60):
        world.step(render=False)
    
    world.pause()
        
    print("물리 시뮬레이션을 통해 오브젝트 안착 완료.")
    print(f"최종 Object 위치 : {omni.usd.get_world_transform_matrix(rigid_prim.prim).ExtractTranslation()}")




# Register the place cones randomizer graph
def register_cone_placement(forklift_prim, assets_root_path, config):
    # Get the bottom corners of the oriented bounding box (OBB) of the forklift
    bb_cache = create_bbox_cache()
    centroid, axes, half_extent = compute_obb(bb_cache, forklift_prim.GetPrimPath())
    larger_xy_extent = (half_extent[0] * 1.3, half_extent[1] * 1.3, half_extent[2])
    obb_corners = get_obb_corners(centroid, axes, larger_xy_extent)
    bottom_corners = [
        obb_corners[0].tolist(),
        obb_corners[2].tolist(),
        obb_corners[4].tolist(),
        obb_corners[6].tolist(),
    ]

    # Orient the cone using the OBB (Oriented Bounding Box)
    obb_quat = Gf.Matrix3d(axes).ExtractRotation().GetQuaternion()
    obb_quat_xyzw = (obb_quat.GetReal(), *obb_quat.GetImaginary())
    obb_euler = quat_to_euler_angles(np.array(obb_quat_xyzw), degrees=True)

    def place_cones():
        cones = rep.create.from_usd(
            assets_root_path + config["cone"]["url"], semantics=[("class", config["cone"]["class"])]
        )
        with cones:
            rep.modify.pose(position=rep.distribution.sequence(bottom_corners), rotation_z=obb_euler[2])
        return cones.node

    rep.randomizer.register(place_cones)


# Register light randomization graph
def register_lights_placement(table_prim):
    bb_cache = create_bbox_cache()
    bbox3d_gf = bb_cache.ComputeLocalBound(table_prim)
    prim_tf_gf = omni.usd.get_world_transform_matrix(table_prim)

    # Calculate the bounds of the prim
    bbox3d_gf.Transform(prim_tf_gf)
    range_size = bbox3d_gf.GetRange().GetSize()
    pos_min = (range_size[0]*0.8, range_size[1]*0.8, 6)
    pos_max = (range_size[1]*1.2, range_size[1]*1.2, 7)

    def randomize_lights():
        lights = rep.create.light(
            light_type="Sphere",
            color=rep.distribution.uniform((0.2, 0.1, 0.1), (0.9, 0.8, 0.8)),
            intensity=rep.distribution.uniform(1000, 3000),
            position=rep.distribution.uniform(pos_min, pos_max),
            scale=rep.distribution.uniform(5, 10),
            count=3,
        )
        return lights.node

    rep.randomizer.register(randomize_lights)


