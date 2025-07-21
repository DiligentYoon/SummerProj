import open3d as o3d
import numpy as np
import imageio.v2 as imageio
import os
import time
import json

# ========== 헬퍼 함수 및 ID 정의 ==========
def rgba_to_int_id_bgra(r, g, b, a):
    """(R, G, B, A) 튜플을 32비트 정수 ID로 변환 (BGRA 순서)"""
    # Red와 Blue의 위치를 바꿔서 비트 연산 수행
    return (a << 24) | (b << 16) | (g << 8) | r

def int_id_to_rgba_bgra(int_id):
    """32비트 정수 ID를 (R, G, B, A) 튜플로 변환 (BGRA 순서)"""
    a = (int_id >> 24) & 0xFF
    b = (int_id >> 16) & 0xFF # 16비트 자리가 Blue
    g = (int_id >> 8) & 0xFF
    r = int_id & 0xFF        # 0비트 자리가 Red
    return (r, g, b, a)


# JSON 파일 기반 ID 정의
CUBE_ID = rgba_to_int_id_bgra(140, 25, 255, 255)
PALLET_ID = rgba_to_int_id_bgra(140, 255, 25, 255)
UNLABELLED_ID = rgba_to_int_id_bgra(0, 0, 0, 255)
BACKGROUND_ID = rgba_to_int_id_bgra(0, 0, 0, 0)

OBJECT_IDS = [CUBE_ID]

COLOR_MAP = {
        CUBE_ID:       [1.0, 0.0, 0.0],  # 큐브   -> 빨간색
        PALLET_ID:     [0.5, 0.5, 0.5],  # 팔레트 -> 회색
        UNLABELLED_ID: [0.0, 0.0, 0.0],  # 라벨없음 -> 녹색 (디버깅용)
    }

N_OBJ = 800
N_BG = 1600

ID_TO_INDEX_MAP = {
    BACKGROUND_ID: 0,
    UNLABELLED_ID: 0, # 라벨 없는 것도 배경(0)으로 취급
    PALLET_ID: 1,
    CUBE_ID: 2,
}

OBJECT_LIST = ["mug_1", "mug_2", "cube_1", "cube_2", "cylinder_1", "cylinder_2"]

PLOT_MODE = False

print("\n")
print("-" * 20)
print(f"Object ID : {CUBE_ID}")
print(f"Pallet ID : {PALLET_ID}")
print(F"Unlabelled ID : {UNLABELLED_ID}")
print("-" * 20)

# ==========================================

# =============== 시각화 함수 ===============
def prepare_and_visualize_scene(base_dir: str, output_dir: str, test_dir: str, frame_id: int, save_gif=True):
    # ---------- 1. 데이터 로드 및 병합 ----------
    cam_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("Cam")]
    cam_dirs.sort()
    if not cam_dirs: return
    all_points, all_labels, all_colors = [], [], []
    frame_str = f"{frame_id:04d}"
    print(f"--- 프레임 {frame_str} 처리 시작 ---")
    for cam_dir in cam_dirs:
        try:
            # ... 파일 로드 ...
            all_points.append(np.load(os.path.join(base_dir, cam_dir, "pointcloud", f"pointcloud_{frame_str}.npy")))
            all_labels.append(np.load(os.path.join(base_dir, cam_dir, "pointcloud", f"pointcloud_semantic_{frame_str}.npy")))
            # all_colors.append(np.load(os.path.join(base_dir, cam_dir, "pointcloud", f"pointcloud_rgb_{frame_str}.npy")))
        except FileNotFoundError:
            continue
    if not all_points:
        print(f"프레임 {frame_str}에 대한 데이터를 찾을 수 없습니다.")
        return
    merged_points = np.vstack(all_points)
    merged_labels = np.concatenate(all_labels)
    # merged_colors = np.vstack(all_colors)

    # ---------- 2. 데이터 전처리 ----------
    object_mask = np.isin(merged_labels, OBJECT_IDS)
    background_mask = np.isin(merged_labels, PALLET_ID)
    object_points, object_labels = merged_points[object_mask], merged_labels[object_mask]
    background_points, background_labels = merged_points[background_mask], merged_labels[background_mask]
    # object_points, object_colors, object_labels = merged_points[object_mask], merged_colors[object_mask], merged_labels[object_mask]
    # background_points, background_colors, background_labels = merged_points[background_mask], merged_colors[background_mask], merged_labels[background_mask]
    
    print(f"분리 결과 - 오브젝트: {len(object_points)}개, 배경: {len(background_points)}개")
    
    if len(object_points) > 0:
        obj_indices = np.random.choice(len(object_points), N_OBJ, replace=len(object_points) < N_OBJ)
        obj_points_ds, obj_labels_ds = object_points[obj_indices], object_labels[obj_indices]
    else:
        obj_points_ds, obj_labels_ds = np.zeros((N_OBJ, 3)), np.zeros(N_OBJ)
    
    if len(background_points) > 0:
        bg_indices = np.random.choice(len(background_points), N_BG, replace=len(background_points) < N_BG)
        bg_points_ds, bg_labels_ds = background_points[bg_indices], background_labels[bg_indices]
    else:
        bg_points_ds, bg_labels_ds = np.zeros((N_BG, 3)), np.zeros(N_BG)

    pos = np.vstack((obj_points_ds, bg_points_ds)).astype(np.float32)
    pos_normalized = pos - np.mean(pos, axis=0)
    y = np.concatenate((obj_labels_ds, bg_labels_ds)).astype(np.int64)
    # 0으로 채워진 새로운 라벨 배열을 생성합니다 (기본값: 배경).
    y_indexed = np.zeros_like(y, dtype=np.int64)

    for id_32bit, index in ID_TO_INDEX_MAP.items():
        y_indexed[y == id_32bit] = index

    seg_mask = np.vstack((np.ones((N_OBJ, 1)), np.zeros((N_BG, 1)))).astype(np.float32)
    x = seg_mask # HACMan++에서는 색상 제외

    # ---------- 3. 데이터 저장 ----------
    if np.random.randn(1) <= 0.8:
        os.makedirs(output_dir, exist_ok=True)
        output_filepath = os.path.join(output_dir, f"sample_{frame_str}.npz")
    else:
        os.makedirs(test_dir, exist_ok=True)
        output_filepath = os.path.join(test_dir, f"sample_{frame_str}.npz")
    np.savez(output_filepath, 
             pos=pos_normalized, 
             x=x, 
             y=y_indexed)
    print(f"[✓] 전처리 완료. 샘플 저장 → {output_filepath}")

    # ========== 4. 시각화 로직 추가 ==========
    if PLOT_MODE:
        print("\n전처리된 데이터 시각화 시작...")
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pos_normalized) # 정규화된 좌표 사용

        # 시맨틱 라벨(y)을 기반으로 색상 지정
        colors = np.ones((len(pos), 3)) * 0.8 # 기본 흰색
        for label_id, color in COLOR_MAP.items():
            indices = np.where(y == label_id)[0]
            if indices.size > 0:
                colors[indices] = color
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # 뷰어 설정 및 애니메이션 (기존 코드 재사용)
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"Preprocessed Data - Frame {frame_str}", width=1280, height=720)
        vis.add_geometry(pcd)
        
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.1, 0.1, 0.1])
        opt.point_size = 3.0 # 포인트를 더 잘 보이게 크기 증가
        
        vis.poll_events(); vis.update_renderer()

        vc = vis.get_view_control()
        center = pcd.get_center()
        vc.set_lookat(center)
        vc.set_front([-0.5, -0.5, 0.5])
        vc.set_up([0, 0, 1])
        vc.set_zoom(0.7)

        frames = []
        gif_filepath = os.path.join(output_dir, f"visualization_{frame_str}.gif")
        try:
            for _ in range(120): # 240도 회전
                pcd.rotate(pcd.get_rotation_matrix_from_axis_angle([0, 0, np.deg2rad(2)]), center=center)
                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()
                if save_gif:
                    buf = vis.capture_screen_float_buffer(do_render=True)
                    frames.append((np.asarray(buf) * 255).astype(np.uint8))
                time.sleep(0.01)
        finally:
            vis.destroy_window()

        if save_gif and frames:
            print("GIF 저장 중...")
            imageio.mimsave(gif_filepath, frames, fps=30)
            print(f"[✓] 시각화 GIF 저장 완료 → {gif_filepath}")



# =========================================


def prepare_scene_for_pointnet(base_dir: str, output_dir: str, frame_id: int):
    """
        한 프레임의 다중 뷰 데이터를 병합하고 전처리하여
        PointNet++ 학습용 샘플 파일(.npz)을 생성합니다.
        
        num_points: 네트워크에 입력하기 전, 씬을 대표하는 포인트 수 (pre-sampling)
    """
    
    # ---------- 1. 데이터 로드 및 병합 (RGB 로딩 추가) ----------
    cam_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("Cam")]
    cam_dirs.sort()
    if not cam_dirs: return

    all_points, all_labels, all_colors = [], [], []

    frame_str = f"{frame_id:04d}"
    print(f"--- 프레임 {frame_str} 처리 시작 ---")

    for cam_dir in cam_dirs:
        cam_path = os.path.join(base_dir, cam_dir)
        try:
            npy_filepath = os.path.join(cam_path, "pointcloud", f"pointcloud_{frame_str}.npy")
            labels_filepath = os.path.join(cam_path, "pointcloud", f"pointcloud_semantic_{frame_str}.npy")
            rgb_filepath = os.path.join(cam_path, "pointcloud", f"pointcloud_rgb_{frame_str}.npy")

            xyz_data = np.load(npy_filepath)
            labels_part = np.load(labels_filepath)
            rgb_data = np.load(rgb_filepath)

            if xyz_data.size > 0:
                all_points.append(xyz_data)
                all_labels.append(labels_part)
                all_colors.append(rgb_data)
        except FileNotFoundError:
            continue
    
    if not all_points:
        print(f"프레임 {frame_str}에 대한 데이터를 찾을 수 없습니다.")
        return

    merged_points = np.vstack(all_points)
    merged_labels = np.concatenate(all_labels)
    merged_colors = np.vstack(all_colors)

    # ---------- 2. 데이터 전처리 (HACMan++ 방식) ----------
    
    # a. 오브젝트 / 배경 분리
    # 배경 ID에 속하지 않는 모든 포인트를 오브젝트로 간주합니다.
    # np.isin을 사용하여 merged_labels의 각 요소가 BACKGROUND_IDS에 포함되는지 확인합니다.
    object_mask = np.isin(merged_labels, OBJECT_IDS)
    background_mask = ~object_mask

    object_points = merged_points[object_mask]
    object_colors = merged_colors[object_mask]
    object_labels = merged_labels[object_mask]

    background_points = merged_points[background_mask]
    background_colors = merged_colors[background_mask]
    background_labels = merged_labels[background_mask]

    print(f"분리 결과 - 오브젝트: {len(object_points)}개, 배경: {len(background_points)}개")

    # 오브젝트 샘플링
    if len(object_points) > 0:
        obj_indices = np.random.choice(len(object_points), N_OBJ, replace=len(object_points) < N_OBJ)
        obj_points_ds = object_points[obj_indices]
        obj_colors_ds = object_colors[obj_indices]
        obj_labels_ds = object_labels[obj_indices]
    else: # 씬에 오브젝트가 없는 경우
        obj_points_ds = np.zeros((N_OBJ, 3))
        obj_colors_ds = np.zeros((N_OBJ, 3))
        obj_labels_ds = np.zeros(N_OBJ)

    # 배경 샘플링
    if len(background_points) > 0:
        bg_indices = np.random.choice(len(background_points), N_BG, replace=len(background_points) < N_BG)
        bg_points_ds = background_points[bg_indices]
        bg_colors_ds = background_colors[bg_indices]
        bg_labels_ds = background_labels[bg_indices]
    else: # 씬에 배경이 없는 경우
        bg_points_ds = np.zeros((N_BG, 3))
        bg_colors_ds = np.zeros((N_BG, 3))
        bg_labels_ds = np.zeros(N_BG)

    # c. 최종 데이터 통합
    #   - 위치(pos): 오브젝트 포인트와 배경 포인트를 합침
    pos = np.vstack((obj_points_ds, bg_points_ds)).astype(np.float32)
    
    #   - 라벨(y): 오브젝트 라벨과 배경 라벨을 합침
    y = np.concatenate((obj_labels_ds, bg_labels_ds)).astype(np.int64)

    # d. 최종 특징(x) 생성 및 정규화
    #   - 색상(RGB)
    colors_combined = np.vstack((obj_colors_ds, bg_colors_ds))
    colors_normalized = (colors_combined / 255.0).astype(np.float32)

    #   - 위치(pos) 정규화
    pos_normalized = pos - np.mean(pos, axis=0)
    
    #   - 최종 특징(x) = 분할 마스크
    x = np.vstack((
        np.ones((N_OBJ, 1)),  # 오브젝트는 1
        np.zeros((N_BG, 1))   # 배경은 0
    )).astype(np.float32)

    # ---------- 3. 데이터 저장 ----------
    os.makedirs(output_dir, exist_ok=True)
    output_filepath = os.path.join(output_dir, f"sample_{frame_str}.npz")
    
    np.savez(
        output_filepath,
        pos=pos_normalized,    # (1400, 3) - 위치
        x=x,                   # (1400, 4) - 특징 (RGB + Mask)
        y=y                    # (1400,)   - 시맨틱 라벨
    )
    print(f"[✓] 전처리 완료. 샘플 저장 → {output_filepath} (Pos: {pos.shape}, X: {x.shape})")


if __name__ == "__main__":
    for target_obj in OBJECT_LIST:
        dataset_root_dir = os.path.join(os.getcwd(), "Dataset", target_obj)
        training_data_dir = os.path.join(os.getcwd(), "Dataset", "TrainingData", target_obj)
        testing_data_dir = os.path.join(os.getcwd(), "Dataset", "TestingData", target_obj)
        
        num_frames_generated = 200
        for i in range(num_frames_generated):
            prepare_and_visualize_scene(
                base_dir=dataset_root_dir,
                output_dir=training_data_dir,
                test_dir=testing_data_dir,
                frame_id=i,
                save_gif=True # 각 프레임에 대한 시각화 GIF 저장
            )