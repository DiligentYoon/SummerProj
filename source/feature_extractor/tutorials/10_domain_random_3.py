"""
Isaac‑Sim Replicator: 한 물체를 여러 뷰에서 캡처 → 하나의 PLY·NPY 저장
──────────────────────────────────────────────────────────────────────
• 물체·조명은 샘플마다 한 번만 랜덤 배치
• scatter_2d 로 Z‑오프셋을 자동 맞춰 ‘지면에 올려놓음’ ★
• 같은 샘플 안에서 카메라만 움직여 여러 뷰 촬영
"""

from pxr import Gf
import omni.usd
import omni.replicator.core as rep
import numpy as np, open3d as o3d
import os, asyncio, random, math
import omni.physx as physx


# -------------------------------------------------------------- #
# 1. 경로·파라미터
# -------------------------------------------------------------- #
BASE_PATH = os.path.join(os.getcwd(), "Dataset/google_objects_usd")
SAVE_ROOT = os.path.join(os.getcwd(), "Dataset")

ASSET_LIST = {
    "Mug": [
        "025_mug/025_mug.usd",
        "Room_Essentials_Mug_White_Yellow/Room_Essentials_Mug_White_Yellow.usd",
    ],
    "Cube": [
        "010_potted_meat_can/010_potted_meat_can.usd",
        "003_cracker_box/003_cracker_box.usd",
    ],
    "Cylinder": [
        "006_mustard_bottle/006_mustard_bottle.usd",
        "021_bleach_cleanser/021_bleach_cleanser.usd",
    ],
}

SAMPLES_PER_OBJECT = 2
VIEWS_PER_SAMPLE   = 4
RESOLUTION         = (960, 480)

GROUND = "omniverse://localhost/NVIDIA/Assets/Scenes/Templates/Basic/display_riser.usd"
ID_MAP = {"ground": 1, "object": 2}

# -------------------------------------------------------------- #
# 2. 저장 유틸
# -------------------------------------------------------------- #
def save_cloud(points, colors, labels, out_dir, name):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    pc.colors = o3d.utility.Vector3dVector(colors)

    os.makedirs(out_dir, exist_ok=True)
    o3d.io.write_point_cloud(os.path.join(out_dir, f"{name}.ply"), pc)
    np.save(os.path.join(out_dir, f"{name}_labels.npy"), labels)
    print(f"[✓] 저장: {name}")

# -------------------------------------------------------------- #
# 3. 메인 코루틴
# -------------------------------------------------------------- #
async def generate():
    random.seed(42)

    for cat, rel_list in ASSET_LIST.items():
        for rel in rel_list:
            usd_path  = os.path.join(BASE_PATH, rel).replace("\\", "/")
            usd_stem  = os.path.splitext(os.path.basename(rel))[0]

            for s_idx in range(SAMPLES_PER_OBJECT):
                with rep.new_layer(name="Replicator"):

                    # 1. 생성 단계: 프리미티브를 만듭니다.
                    ground = rep.create.from_usd(GROUND, semantics=[("class", "ground")])
                    with ground:
                        rep.modify.pose(rotation=(90, 0, 0), position=(0, 0, 0))
                        rep.physics.collider()

                    obj = rep.create.from_usd(usd_path, semantics=[("class", "object")])
                    with obj:
                        # rep.physics.rigid_body(velocity=rep.distribution.uniform((-0,0,-0),(0,0,0)))
                        rep.physics.collider(approximation_shape="convexHull")
                    
                    light = rep.create.light(light_type="Sphere")
                    
                    cam  = rep.create.camera()
                    rp   = rep.create.render_product(cam, RESOLUTION)
                    anno = rep.annotators.get("pointcloud"); anno.attach(rp)
                    id_anno = rep.annotators.get("semantic_segmentation"); id_anno.attach(rp)

                    # 2. ★★★ Python random으로 모든 랜덤 값 미리 계산 ★★★
                    # 객체 랜덤값
                    obj_pos = (random.uniform(-150, 150), random.uniform(-150, 150), random.uniform(5.0, 10.0))
                    obj_rot = (random.uniform(0, 360), random.uniform(0, 360), random.uniform(0, 360))
                    # 스케일도 너무 커지지 않게 조정
                    obj_scale = (random.uniform(1.0, 10.0),) * 3

                    # 조명 랜덤값
                    light_pos = (random.uniform(-300, 300), random.uniform(-300, 300), random.uniform(400, 800))
                    light_intensity = random.normalvariate(35000, 5000)
                    light_temp = random.normalvariate(6500, 400)

                    # 3. ★★★ 계산된 '고정값'으로 씬을 직접 수정 ★★★
                    with obj:
                        rep.modify.pose(position=obj_pos, rotation=obj_rot, scale=obj_scale)

                    with light:
                        rep.modify.pose(position=light_pos)
                        rep.modify.attribute("intensity", light_intensity)
                        rep.modify.attribute("colorTemperature", light_temp)
                    
                    print(f"Object Position (before): {obj_pos}")

                    # 물리 안정화가 필요하다면 이 시점에 몇 프레임 실행
                    for _ in range(20): await rep.orchestrator.step_async()

                    prim_paths = []
                    if hasattr(obj, "get_output_prims"):
                        print(f"get_output")
                        prim_paths = obj.get_output_prims()          
                    elif hasattr(obj, "get_prims"):              
                        print(f"get")
                        prim_paths = obj.get_prims()
                    else:
                        print(f"get_node")
                        prim_paths = rep.utils.get_node_targets(
                            obj.node, "inputs:prims"
                        )

                    print(f"Prim Paths : {prim_paths}")
                    obj_prim = obj.get_output_prims()["prims"][0]
                    pose    = physx.get_physx_interface().get_rigid_body_pose(obj_prim)
                    print("world pos:", pose.p)
                    # prim_path = obj_prim.GetPath()            
                    # print(f"Object Prim Path: {prim_path}")
                    # stage = omni.usd.get_context().get_stage()
                    # obj_prim = stage.GetPrimAtPath(prim_path)

                    # iface = physx.get_physx_interface()
                    # rb_prim = next(
                    #     p for p in obj_prim.GetDescendants()
                    #     if iface.is_rigid_body(p)               # 물리 API 달린 Prim
                    # )

                    # # 객체의 월드 변환 행렬(World Transform Matrix)을 가져옵니다.
                    # world_transform = omni.usd.get_world_transform_matrix(rb_prim)
                    
                    # # 변환 행렬에서 위치(Translation) 벡터만 추출합니다.
                    # obj_pos_world   = world_transform.ExtractTranslation()
                    
                    # print(f"Object Position (after): {obj_pos_world}")



                    # 4. ★★★ for 루프를 사용한 안정적인 뷰 캡처 ★★★
                    pts, cols, lbls = [], [], []
                    for v_idx in range(VIEWS_PER_SAMPLE):
                        distance = random.uniform(10, 15)
                        # 수평각(theta)과 수직각(phi)을 랜덤하게 선택 (라디안으로 변환)
                        theta = math.radians(random.uniform(0, 360)) # 0~360도
                        phi = math.radians(random.uniform(30, 80))
                        relative_x = distance * math.sin(phi) * math.cos(theta)
                        relative_y = distance * math.sin(phi) * math.sin(theta)
                        relative_z = distance * math.cos(phi)
                        with cam:
                            cam_pos = (
                                obj_pos[0] + relative_x,
                                obj_pos[1] + relative_y,
                                obj_pos[2] + relative_z
                            )
                            rep.modify.pose(position=cam_pos, look_at=obj)
                        
                        await rep.orchestrator.step_async()
                        
                        pc = anno.get_data()
                        if pc["data"].size > 0:
                            pts.append(pc["data"])
                            cols.append(pc["info"]["pointRgb"][:, :3] / 255.0)
                            lbls.append(pc["info"]["pointSemantic"])

                # ---------- 뷰 → 하나로 저장 -------------------------
                print(f"[INFO] Save Point Cloud Data")
                if pts:
                    pts_raw  = np.concatenate(pts,  axis=0)
                    cols_raw = np.concatenate(cols, axis=0)
                    lbls_raw = np.concatenate(lbls, axis=0)
                    
                    # ★★★★★ 최종 필터링 로직 ★★★★★
                    
                    # 1. ID 맵 가져오기 및 변환 (이전과 동일)
                    replicator_map_nested = id_anno.get_data()['info']['idToLabels']
                    label_to_id = {
                        info_dict['class']: int(id_str) 
                        for id_str, info_dict in replicator_map_nested.items()
                    }
                    dynamic_ground_id = label_to_id.get("ground", -1) 
                    dynamic_object_id = label_to_id.get("object", -1)

                    # 2. 원하는 ID로 레이블 재맵핑 (이전과 동일)
                    remapped_labels = np.zeros_like(lbls_raw)
                    if dynamic_ground_id != -1:
                        remapped_labels[lbls_raw == dynamic_ground_id] = ID_MAP["ground"]
                    if dynamic_object_id != -1:
                        remapped_labels[lbls_raw == dynamic_object_id] = ID_MAP["object"]

                    # 3. 유효한 레이블만 필터링 (배경/0 제외)
                    #    remapped_labels 배열에서 0보다 큰 값을 가진 요소들의 인덱스를 찾습니다.
                    valid_indices = np.where(remapped_labels > 0)[0]

                    # 4. 해당 인덱스를 사용해 포인트, 색상, 레이블 모두에서 유효한 데이터만 추출합니다.
                    final_pts = pts_raw[valid_indices]
                    final_cols = cols_raw[valid_indices]
                    final_lbls = remapped_labels[valid_indices]
                    
                    # 5. 최종적으로 필터링된 데이터를 저장합니다.
                    stub = f"{usd_stem}_s{s_idx:03d}"
                    save_cloud(final_pts, final_cols, final_lbls,
                            os.path.join(SAVE_ROOT, cat), stub)
                    
                await asyncio.sleep(0.1)

# -------------------------------------------------------------- #
# 4. 실행
# -------------------------------------------------------------- #
asyncio.ensure_future(generate())
