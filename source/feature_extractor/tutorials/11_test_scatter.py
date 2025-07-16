
config = {
    "launch_config": {
        "renderer": "RayTracedLighting",
        "headless": False,
    },
    "resolution": [512, 512],
    "rt_subframes": 16,
    "num_frames": 20,
    "env_url": "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd",
    "writer": "BasicWriter",
    "writer_config": {
        "output_dir": "_out_scene_based_sdg",
        "rgb": True,
        "bounding_box_2d_tight": True,
        "semantic_segmentation": True,
        "distance_to_image_plane": True,
        "bounding_box_3d": True,
        "occlusion": True,
    },
    "clear_previous_semantics": True,
    "forklift": {
        "url": "/Isaac/Props/Forklift/forklift.usd",
        "class": "Forklift",
    },
    "cone": {
        "url": "/Isaac/Environments/Simple_Warehouse/Props/S_TrafficCone.usd",
        "class": "TrafficCone",
    },
    "pallet": {
        "url": "/Isaac/Environments/Simple_Warehouse/Props/SM_PaletteA_01.usd",
        "class": "Pallet",
    },
    "cardbox": {
        "url": "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_04.usd",
        "class": "Cardbox",
    },
    "close_app_after_run": True,
}




from isaacsim import SimulationApp


simulation_app = SimulationApp(launch_config=config["launch_config"])


from pxr import Gf
import omni.usd
import omni.replicator.core as rep
import numpy as np, open3d as o3d
import os, asyncio, random, math
import omni.physx as physx


# 차근차근 코드 작성해보기

# 목표는 땅에 닿도록 하기

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
VIEWS_PER_SAMPLE   = 3
RESOLUTION         = (1024, 1024)
ID_MAP = {"ground": 1, "object": 2}
GROUND = "omniverse://localhost/NVIDIA/Assets/Scenes/Templates/Basic/display_riser.usd"





def save_cloud(points, colors, labels, out_dir, name):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    pc.colors = o3d.utility.Vector3dVector(colors)

    os.makedirs(out_dir, exist_ok=True)
    o3d.io.write_point_cloud(os.path.join(out_dir, f"{name}.ply"), pc)
    np.save(os.path.join(out_dir, f"{name}_labels.npy"), labels)
    print(f"[✓] 저장: {name}")


async def generate(obj_key, obj_value):
    """
        입력받은 한 물체에 대해서만 데이터 추출을 수행하는 루프
        목적 : Domain Randomization 확인
        고정 요소 : 카메라 위치
        유동 요소 : 물체 및 라이트 위치
    """
    usd_path = os.path.join(BASE_PATH, obj_value.replace("\\", "/"))
    usd_stem  = os.path.splitext(os.path.basename(obj_value))[0]

    # # ======= prim 생성 루프에 대해서 layer 생성 ======
    with rep.new_layer(name="Replicator"):

        # ========== Ground 생성 ============
        ground = rep.create.from_usd(GROUND, semantics=[("class", "ground")])
        with ground:
            rep.modify.pose(rotation=(90, 0, 0), position=(0, 0, 0), scale=(0.05, 0.05, 0.05))
            rep.physics.collider()

        # ============ Object 생성 ===========
        obj = rep.create.from_usd(usd_path, semantics=[("class", "object")])
        with obj:
            rep.modify.semantics([
                            ("class", "object")])
            rep.modify.pose(
                position=rep.distribution.uniform((-150, -150, 10), (150, 150, 30)),
                rotation=rep.distribution.uniform((0, 0, 0), (360, 360, 360)),
                scale=rep.distribution.uniform(5.0, 15.0)
            )
            rep.physics.collider(approximation_shape="convexDecomposition")
        
        obj_prim_paths = obj.get_output_prims() 
        print(f"[INFO] Generate Object")
        print(f"Prim Path : {obj_prim_paths}\n")

        # ========== Light 생성 =============
        light = rep.create.light(light_type="Sphere")
        with light:
            rep.modify.pose(
                position=rep.distribution.uniform((-300, -300, 400), (300, 300, 800)),
            )
            rep.modify.attribute("intensity", rep.distribution.normal(3.5e4, 5e3))
            rep.modify.attribute("colorTemperature", rep.distribution.normal(6500, 400))
        
        light_prim_paths = light.get_output_prims()
        print(f"[INFO] Generate Light")
        print(f"Prim Path : {light_prim_paths}")


        # ========== Annotator 생성 ===========
        cam  = rep.create.camera(position=(0,0,100), look_at=obj)
        rp   = rep.create.render_product(cam, RESOLUTION)
        anno = rep.annotators.get("pointcloud"); anno.attach(rp)
        id_anno = rep.annotators.get("semantic_segmentation"); id_anno.attach(rp)
        print(f"[INFO] Generate Annotator")


        # ========= Randomization 콜백 함수 ===========
        def randomize_obj():
            # 객체 랜덤화
            with obj:
                rep.modify.pose( # 회전 및 크기
                    position=rep.distribution.uniform((-150, -150, 10), (150, 150, 30)),
                    rotation=rep.distribution.uniform((0, 0, 0), (360, 360, 360)),
                    scale=rep.distribution.uniform(5.0, 15.0)
                )

            return obj.node
        
        def randomize_light():
            # 조명 랜덤화
            with light:
                rep.modify.attribute("intensity", rep.distribution.normal(3.5e4, 5e3))
                rep.modify.attribute("colorTemperature", rep.distribution.normal(6500, 400))
                rep.modify.pose(
                position=rep.distribution.uniform((-300, -300, 400), (300, 300, 800)),
                )
            
            return light.node
        
        def randomize_view():
            # 이거는 고정 시키는게 낫지 않나 ?
            with cam:
                rep.modify.pose(
                    position=rep.distribution.uniform((-400, -400, 100), (400, 400, 500)),
                    look_at=obj
                )

        print(f"[INFO] Define Randomize Function")

        # rep.randomizer.register(randomize_obj)
        rep.randomizer.register(randomize_view)
        rep.randomizer.register(randomize_light)
        print(f"[INFO] Register Randomize Function")


        # Randomizer를 세 프레임에 걸쳐서 수행
        print(f"[INFO] Starts to apply randomizer")
        with rep.trigger.on_frame(max_execs=3):
            # with obj:
            #     print(f"[INFO] Apply Object randomization")
            #     rep.randomizer.randomize_obj()
            with light:
                print(f"[INFO] Apply Light randomization")
                rep.randomizer.randomize_light()
            with cam:
                print(f"[INFO] Apply Camera randomization")
                rep.randomizer.randomize_view()
        
        # Randomizer가 변환시키는 세 프레임에 대한 데이터 추출을 위한 step function
        print(f"[INFO] Starts to append point cloud frame : Iter = {3}")
        pts, cols, lbls = [], [], []
        for idx in range(3):
            await rep.orchestrator.step_async()

            pc = anno.get_data()
            if pc["data"].size > 0:
                pts.append(pc["data"])
                cols.append(pc["info"]["pointRgb"][:, :3] / 255.0)
                lbls.append(pc["info"]["pointSemantic"])
            
            print(f"cam Loop : {idx}")

        print(f"[INFO] Starts to save PCD and Semantic info")
        # if pts:
        #     pts_raw  = np.concatenate(pts,  axis=0)
        #     cols_raw = np.concatenate(cols, axis=0)
        #     lbls_raw = np.concatenate(lbls, axis=0)
            
        #     # ★★★★★ 최종 필터링 로직 ★★★★★
        #     # 1. ID 맵 가져오기 및 변환 (이전과 동일)
        #     replicator_map_nested = id_anno.get_data()['info']['idToLabels']
        #     label_to_id = {
        #         info_dict['class']: int(id_str) 
        #         for id_str, info_dict in replicator_map_nested.items()
        #     }
        #     dynamic_ground_id = label_to_id.get("ground", -1) 
        #     dynamic_object_id = label_to_id.get("object", -1)

        #     # 2. 원하는 ID로 레이블 재맵핑 (이전과 동일)
        #     remapped_labels = np.zeros_like(lbls_raw)
        #     if dynamic_ground_id != -1:
        #         remapped_labels[lbls_raw == dynamic_ground_id] = ID_MAP["ground"]
        #     if dynamic_object_id != -1:
        #         remapped_labels[lbls_raw == dynamic_object_id] = ID_MAP["object"]

        #     # 3. 유효한 레이블만 필터링 (배경/0 제외)
        #     #    remapped_labels 배열에서 0보다 큰 값을 가진 요소들의 인덱스를 찾습니다.
        #     valid_indices = np.where(remapped_labels > 0)[0]

        #     # 4. 해당 인덱스를 사용해 포인트, 색상, 레이블 모두에서 유효한 데이터만 추출합니다.
        #     final_pts = pts_raw[valid_indices]
        #     final_cols = cols_raw[valid_indices]
        #     final_lbls = remapped_labels[valid_indices]
            
        #     # 5. 최종적으로 필터링된 데이터를 저장합니다.
        #     stub = f"{usd_stem}_s{s_idx:03d}"
        #     save_cloud(final_pts, final_cols, final_lbls,
        #             os.path.join(SAVE_ROOT, cat_list[obj_idx]), stub)
        print(f"Loop is success ! Application is down...")
        simulation_app.close()

# cls_key : [Mug, Cube, Cylinder]
# cls_val : ["025_mug/025_mug.usd", ...]
cls_idx, obj_idx = 0, 1
cls_key = list(ASSET_LIST)[cls_idx]
cls_val = ASSET_LIST[cls_key][obj_idx]

asyncio.ensure_future(generate(obj_key=cls_key, obj_value=cls_val))