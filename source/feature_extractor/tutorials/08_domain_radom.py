"""
Isaac‑Sim Replicator: 한 물체를 여러 뷰에서 캡처 → 하나의 PLY·NPY 저장
──────────────────────────────────────────────────────────────────────
• 물체·조명은 샘플마다 한 번만 랜덤 배치
• scatter_2d 로 Z‑오프셋을 자동 맞춰 ‘지면에 올려놓음’ ★
• 같은 샘플 안에서 카메라만 움직여 여러 뷰 촬영
"""

import omni.replicator.core as rep
import numpy as np, open3d as o3d
import os, asyncio, random

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
    random.seed(24)

    for cat, rel_list in ASSET_LIST.items():
        for rel in rel_list:
            usd_path  = os.path.join(BASE_PATH, rel).replace("\\", "/")
            usd_stem  = os.path.splitext(os.path.basename(rel))[0]

            # ---------- 샘플 루프 ----------------------------------------
            for s_idx in range(SAMPLES_PER_OBJECT):

                with rep.new_layer(name="Replicator"):

                    # (1) 바닥
                    ground = rep.create.from_usd(GROUND)
                    with ground:
                        rep.modify.pose(rotation=(90, 0, 0), position=(0, 0, 0))
                        rep.physics.collider()          # ★ static collider
                        rep.modify.semantics([
                            ("class", "ground")])

                    # (2) 물체
                    obj = rep.create.from_usd(usd_path)
                    with obj:
                        rep.physics.collider(approximation_shape="convexDecomposition")
                        rep.modify.semantics([
                            ("class", "object")])
                        rep.modify.pose(
                            position=rep.distribution.uniform((0, 0, 1), (0, 0, 2)),
                            rotation=rep.distribution.uniform((0, 0, 0), (360, 360, 360)),
                            scale=rep.distribution.uniform(5.0, 15.0)
                        )

                    # (3) 조명 – 샘플마다 한 번 배치
                    rep.create.light(
                        light_type="Sphere",
                        intensity   =rep.distribution.normal(3.5e4, 5e3),
                        temperature =rep.distribution.normal(6500 , 400),
                        position    =rep.distribution.uniform((-3, -3, 4),
                                                               ( 3, 3, 8)),
                        scale       =rep.distribution.uniform(40, 100)
                    )

                    # (4) 카메라·어노테이터
                    cam  = rep.create.camera()
                    rp   = rep.create.render_product(cam, RESOLUTION)
                    anno = rep.annotators.get("pointcloud")
                    anno.attach(rp)

                    id_anno = rep.annotators.get("semantic_segmentation")
                    id_anno.attach(rp)

                    # ---------- 뷰 루프 ----------------------------------
                    pts, cols, lbls = [], [], []
                    for v_idx in range(VIEWS_PER_SAMPLE):
                        with cam:
                            rep.modify.pose(
                                position=rep.distribution.uniform((-3, -3, 4),
                                                                   ( 3, 3, 8)),
                                look_at=obj
                            )

                        await rep.orchestrator.step_async()
                        pc = anno.get_data()
                        if pc["data"].size == 0:
                            continue

                        pts .append(pc["data"])
                        cols.append(pc["info"]["pointRgb"][:, :3] / 255.0)
                        lbls.append(pc["info"]["pointSemantic"])

                # ---------- 뷰 → 하나로 저장 -------------------------
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

# -------------------------------------------------------------- #
# 4. 실행
# -------------------------------------------------------------- #
asyncio.ensure_future(generate())
