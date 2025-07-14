import os
import asyncio
import omni.replicator.core as rep
import numpy as np
import open3d as o3d


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


async def generate_perfect_pointcloud_dataset():
    """
    하나의 큐브에 대해 여러 각도에서 포인트 클라우드를 생성하고,
    '문제지(.ply)'와 '정답지(.npy)'를 분리하여 저장하는 함수.
    """
    # GROUND = 'omniverse://localhost/NVIDIA/Assets/Scenes/Templates/Basic/display_riser.usd'
    # TABLE = 'omniverse://localhost/NVIDIA/Assets/Scenes/Templates/Basic/display_riser.usd'
    # 1. 씬 구성: 시맨틱 정보가 부여된 큐브와 카메라 생성
    sphere_light = rep.create.light(
        light_type="Sphere",
        temperature=rep.distribution.normal(6500, 500),
        intensity=rep.distribution.normal(35000, 5000),
        position=rep.distribution.uniform((-300, -300, -300), (300, 300, 300)),
        scale=rep.distribution.uniform(50, 100),
        count=2
    )
    ground_table = rep.create.plane(scale=100, visible=True)
    with ground_table:
        # Add physics to the collision floor, the props already have rigid-body physics applied
        rep.physics.collider()
        rep.modify.semantics([('class', 'ground')])
        # 씬의 원점에 배치합니다.
        rep.modify.pose(position=(0, 0, 0))
    
    cube = rep.create.cube()
    with cube:
        rep.physics.collider()
        rep.modify.semantics([("class", "cube")])
        rep.modify.pose(position=(0, 0, 0.5))

    camera = rep.create.camera()
    render_product = rep.create.render_product(camera, (1024, 1024))

    # 2. Annotator 설정: 포인트 클라우드 정보를 추출할 Annotator 부착
    pointcloud_anno = rep.annotators.get("pointcloud")
    pointcloud_anno.attach(render_product)

    # 3. 데이터 수집: 여러 각도에서 큐브 촬영
    # 큐브의 위, 아래, 네 방향 측면을 모두 보기 위한 카메라 위치 정의
    camera_positions = [
        (10, 0, 2), (-10, 0, 2), (0, 10, 2), (0, -10, 2),
        (0, 0, 10)
    ]

    scene_center_target = (0, 0, -100)
    
    # 데이터를 누적할 빈 리스트 초기화
    accumulated_points = []
    accumulated_colors = []
    accumulated_labels = []

    print("여러 각도에서 포인트 클라우드 데이터 수집을 시작합니다...")
    for pos in camera_positions:
        # 카메라 위치 및 시선 설정
        with camera:
            rep.modify.pose(position=pos, look_at=cube)
        
        # 렌더링 및 데이터 추출 실행
        await rep.orchestrator.step_async()
        pc_data = pointcloud_anno.get_data()

        # 데이터가 존재하는 경우에만 리스트에 추가
        if pc_data["data"].size > 0:
            accumulated_points.append(pc_data["data"])
            # RGB 데이터는 마지막 채널(A)을 제외하고 정규화 (0-255 -> 0-1)
            accumulated_colors.append(pc_data["info"]["pointRgb"][:, :3] / 255.0)
            accumulated_labels.append(pc_data["info"]["pointSemantic"])

    print("데이터 수집 완료. 파일을 생성합니다.")

    # 4. 데이터 통합 및 저장
    if not accumulated_points:
        print("수집된 포인트 클라우드 데이터가 없습니다.")
        return

    # 누적된 데이터를 하나의 NumPy 배열로 통합
    final_points = np.concatenate(accumulated_points)
    final_colors = np.concatenate(accumulated_colors)
    final_labels = np.concatenate(accumulated_labels)

    # 저장할 디렉터리 생성
    output_dir = os.path.join(os.getcwd(), "cube_dataset")
    os.makedirs(output_dir, exist_ok=True)

    # 5. '문제지' 저장 (.ply)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(final_points)
    pcd.colors = o3d.utility.Vector3dVector(final_colors)
    ply_filepath = os.path.join(output_dir, "cube.ply")
    o3d.io.write_point_cloud(ply_filepath, pcd)
    print(f"save complete: {ply_filepath}")

    # 6. '정답지' 저장 (.npy)
    labels_filepath = os.path.join(output_dir, "cube_labels.npy")
    np.save(labels_filepath, final_labels)
    print(f"save complete: {labels_filepath}")


# 비동기 함수 실행
asyncio.ensure_future(generate_perfect_pointcloud_dataset())