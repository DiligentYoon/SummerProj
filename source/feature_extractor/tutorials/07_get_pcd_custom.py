import os, asyncio, omni.replicator.core as rep
import numpy as np, open3d as o3d

BASE_PATH = "C:/SummerProj/Dataset/google_objects_usd"

ASSET_LIST = {
    "Mug": [
        "025_mug/025_mug.usd",
        "Room_Essentials_Mug_White_Yellow/Room_Essentials_Mug_White_Yellow.usd",
    ],
    "Cube": [
        "010_potted_meat_can/010_potted_meat_can.usd",
        "003_cracker_box/003_cracker_box.usd",
        "004_sugar_box/004_sugar_box.usd",
        "008_pudding_box/008_pudding_box.usd",
    ],
    "Cylinder": [
        "006_mustard_bottle/006_mustard_bottle.usd",
        "021_bleach_cleanser/021_bleach_cleanser.usd",
        "005_tomato_soup_can/005_tomato_soup_can.usd",
        "Saccharomyces_Boulardii_MOS_Value_Size/Saccharomyces_Boulardii_MOS_Value_Size.usd",
    ],
}

CAM_POS = [           # 카메라 궤적도 재사용
    (10,0,2),(-10,0,2),(0,10,2),(0,-10,2),(0,0,10)
]

async def export_pc(category:str, usd_rel:str):
    """단일 USD 프림을 읽어 point-cloud (.ply) + label (.npy) 저장"""
    # 0) 장면 리셋
    rep.new_layer()                       # 이전 레이어 자동 삭제/초기화
    # 1) 공통 지오메트리(바닥·조명)
    ground = rep.create.plane(scale=100, visible=True)
    with ground:
        rep.physics.collider()
        rep.modify.semantics([("class","ground")])
    rep.create.light(light_type="Sphere", intensity=3e4, scale=80)

    # 2) 자산 불러오기
    usd_path = os.path.join(BASE_PATH, usd_rel).replace("\\","/")
    obj = rep.create.from_usd(usd_path)   # USD 로드
    with obj:
        rep.physics.collider()
        rep.modify.semantics([("class", "object")])
        rep.modify.pose(position=(0,0,0.5))

    # 3) 카메라·렌더러·어노테이터
    cam   = rep.create.camera()
    rp    = rep.create.render_product(cam, (1024,1024))
    anno  = rep.annotators.get("pointcloud")
    anno.attach(rp)

    pts, cols, lbls = [],[],[]
    for pos in CAM_POS:
        with cam: rep.modify.pose(position=pos, look_at=obj)
        await rep.orchestrator.step_async()
        pc = anno.get_data()
        if pc["data"].size == 0: continue
        pts.append(pc["data"])
        cols.append(pc["info"]["pointRgb"][:,:3]/255.0)
        lbls.append(pc["info"]["pointSemantic"])

    if not pts: return
    pts, cols, lbls = map(np.concatenate,(pts,cols,lbls))

    # 4) 저장
    out_dir   = os.path.join(os.getcwd(),"dataset",category)
    os.makedirs(out_dir, exist_ok=True)
    name      = os.path.splitext(os.path.basename(usd_rel))[0]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    o3d.io.write_point_cloud(os.path.join(out_dir,f"{name}.ply"), pcd)
    np.save(os.path.join(out_dir,f"{name}_labels.npy"), lbls)
    print(f"[✓] {category}/{name} complete save")

async def main():
    for cat, usd_list in ASSET_LIST.items():
        for rel in usd_list:
            await export_pc(cat, rel)

asyncio.ensure_future(main())