import open3d as o3d
import numpy as np
import os, time

CUBE_ID   = 2
GROUND_ID = 4

def verify_and_visualize_dataset():
    # ----- 1. 데이터 로드 -----
    dataset_dir     = os.path.join(os.getcwd(), "cube_dataset")
    ply_filepath    = os.path.join(dataset_dir, "cube.ply")
    labels_filepath = os.path.join(dataset_dir, "cube_labels.npy")

    try:
        pcd    = o3d.io.read_point_cloud(ply_filepath)
        labels = np.load(labels_filepath)
    except FileNotFoundError:
        print(f"오류: '{dataset_dir}' 디렉터리에서 데이터 파일을 찾을 수 없습니다.")
        return

    if len(pcd.points) != len(labels):
        print("오류: 포인트 개수와 레이블 개수가 일치하지 않습니다!")
        return

    # ----- 2. 색상 지정 -----
    colors = np.full((len(pcd.points), 3), [0.0, 0.0, 1.0])
    colors[np.where(labels == CUBE_ID)[0]]   = [1, 0, 0]       # 큐브
    colors[np.where(labels == GROUND_ID)[0]] = [0.5, 0.5, 0.5] # 지면
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # ----- 3. 뷰어 준비 -----
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="데이터셋 검증 (800x600)", width=800, height=600)
    vis.add_geometry(pcd)

    # (★) 한 번 렌더러를 업데이트해 줘야 ViewControl이 초기화됩니다
    vis.poll_events()
    vis.update_renderer()

    # ----- 4. 초기 카메라 세팅 -----
    vc      = vis.get_view_control()
    center  = pcd.get_center()                  # 물체 중심
    front   = np.array([0.3, 0.3, 0.3])      # 대각선 위쪽에서 내려다보는 방향
    front   = front / np.linalg.norm(front)     # 단위벡터
    up_vec  = np.array([0.0, 0.0, 1.0])         # 화면 ‘위쪽’을 +z 로
    vc.set_lookat(center.tolist())              # 카메라 시선 고정점
    vc.set_front(front.tolist())                # front 지정
    vc.set_up(up_vec.tolist())                  # up 벡터 지정
    vc.set_zoom(0.3)                            # 확대(0~1, 값이 작을수록 확대)

    # ----- 5. 애니메이션 루프 -----
    angle_step_deg = 3.0
    R_step = pcd.get_rotation_matrix_from_axis_angle([0, 0, np.deg2rad(angle_step_deg)])

    try:
        while True:
            pcd.rotate(R_step, center=center)   # z 축 기준 회전
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.016)                   # ≈60 FPS
    except KeyboardInterrupt:
        pass
    finally:
        vis.destroy_window()

# 실행
verify_and_visualize_dataset()