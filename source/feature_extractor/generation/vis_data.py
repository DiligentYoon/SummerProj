import open3d as o3d
import numpy as np
import imageio.v2 as imageio
import os
import time
import json
from PIL import Image


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

COLOR_MAP = {
        CUBE_ID:       [1.0, 0.0, 0.0],  # 큐브   -> 빨간색
        PALLET_ID:     [0.5, 0.5, 0.5],  # 팔레트 -> 회색
        UNLABELLED_ID: [0.0, 0.0, 0.0],  # 라벨없음 -> 녹색 (디버깅용)
    }

print("\n")
print("-" * 20)
print(f"Object ID : {CUBE_ID}")
print(f"Pallet ID : {PALLET_ID}")
print(F"Unlabelled ID : {UNLABELLED_ID}")
print("-" * 20)

# ==========================================



def verify_and_visualize_dataset(save_gif=True, gif_name="merged_dataset.gif"):
    base_dir = os.path.join(os.getcwd(), "Dataset", "cylinder_2")
    
    # ---------- 1. 데이터 로드 및 병합 ----------
    cam_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("Cam")]
    cam_dirs.sort()

    if not cam_dirs:
        print(f"오류: '{base_dir}' 경로에서 'Cam'으로 시작하는 데이터 폴더를 찾을 수 없습니다.")
        return

    all_points = []
    all_labels = []

    print("데이터 로딩 및 병합 시작...")
    total_points = 0
    for cam_dir in cam_dirs:
        cam_path = os.path.join(base_dir, cam_dir)
        try:
            # .npy 파일 경로를 구성합니다.
            npy_filepath = os.path.join(cam_path, "pointcloud", "pointcloud_0000.npy")
            labels_filepath = os.path.join(cam_path, "pointcloud", "pointcloud_semantic_0000.npy")

            # np.load로 데이터를 불러옵니다.
            xyz_data = np.load(npy_filepath)
            labels_part = np.load(labels_filepath)

            if xyz_data.size > 0:
                all_points.append(xyz_data)
                all_labels.append(labels_part)
                print(f"[✓] {cam_dir} 로드 완료 ({len(xyz_data)} points)")
                total_points += len(xyz_data)
            else:
                 print(f"[!] {cam_dir} 에 포인트가 없습니다.")

        except FileNotFoundError:
            print(f"[!] {cam_dir} 에서 필요한 파일을 찾을 수 없습니다. 건너뜁니다.")
            continue
    
    if not all_points:
        print("오류: 로드된 포인트 클라우드가 없습니다.")
        return

    # 불러온 모든 데이터를 하나의 큰 배열로 합칩니다.
    merged_points = np.vstack(all_points)
    merged_labels = np.concatenate(all_labels)

    # 합쳐진 데이터로 최종 포인트 클라우드 객체를 생성합니다.
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged_points)
    
    print("-" * 20)
    print(f"총 {len(merged_points)}개의 포인트로 병합 완료.")
    print("-" * 20)

    # ---------- 2. 색상 지정 (이하 동일) ----------
    colors = np.ones_like(merged_points) * 0.8
    # indices_1 = np.where(merged_labels == UNLABELLED_ID)[0]
    # indices_2 = np.where(~(merged_labels == UNLABELLED_ID))[0]
    find_points = 0
    for label_id, color in COLOR_MAP.items():
        indices = np.where(merged_labels == label_id)[0]
        if indices.size > 0:
            print(f"ID {label_id} 에 해당하는 포인트를 {indices.size}개 찾았습니다.")
            print(f"ID {label_id} =====> RGBA {int_id_to_rgba_bgra(label_id)}")
            colors[indices] = color
            find_points += indices.size
        else:
            print(f"ID {label_id} 에 해당하는 포인트를 찾지 못했습니다.")
            print(f"ID {label_id} =====> RGBA {int_id_to_rgba_bgra(label_id)}")

    print(f"찾은 포인트 개수 및 비율 : {find_points} / {total_points}")
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # ... (이후 뷰어 설정, 애니메이션, GIF 저장 코드는 이전과 동일) ...
    # ---------- 3. 뷰어 설정 ----------
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Merged Dataset Check", width=1280, height=720, visible=True)
    vis.add_geometry(pcd)
    
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.0, 0.0, 0.0])
    opt.point_size = 2.0
    
    vis.poll_events(); vis.update_renderer()

    vc = vis.get_view_control()
    center = pcd.get_center()
    vc.set_lookat(center)
    vc.set_front([-0.5, -0.5, 0.5])
    vc.set_up([0, 0, 1])
    vc.set_zoom(0.2)

    # ---------- 4. 애니메이션 ----------
    R_step = pcd.get_rotation_matrix_from_axis_angle([0, 0, np.deg2rad(3)])

    frames = []
    try:
        for _ in range(180):
            pcd.rotate(R_step, center=center)
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

            if save_gif:
                buf = vis.capture_screen_float_buffer(do_render=True)
                img = (np.asarray(buf) * 255).astype(np.uint8)
                frames.append(img)
            
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        vis.destroy_window()

    # ---------- 5. GIF 저장 ----------
    if save_gif and frames:
        print("GIF 저장 중...")
        imageio.mimsave(gif_name, frames, fps=30)
        print(f"[✓] GIF 저장 완료 → {gif_name}")


# 실행
if __name__ == "__main__":
    verify_and_visualize_dataset()