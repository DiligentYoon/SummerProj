import open3d as o3d
import numpy as np
import imageio.v2 as imageio
import os
import time
import json
from PIL import Image

# ========== 헬퍼 함수 및 ID 정의 ==========
def rgba_to_int_id(r, g, b, a):
    return (a << 24) | (r << 16) | (g << 8) | b

def int_id_to_rgba(int_id):
    """32비트 정수 ID를 (R, G, B, A) 튜플로 변환 (ARGB 순서)"""
    a = (int_id >> 24) & 0xFF
    r = (int_id >> 16) & 0xFF
    g = (int_id >> 8) & 0xFF
    b = int_id & 0xFF
    return (r, g, b, a)

# JSON 파일 기반 ID 정의 (실제 JSON 파일 값에 맞춰야 함)
# JSON 파일 기반 ID 정의
CUBE_ID = rgba_to_int_id(140, 25, 255, 255)
PALLET_ID = rgba_to_int_id(140, 255, 25, 255)
UNLABELLED_ID = rgba_to_int_id(0, 0, 0, 255)
BACKGROUND_ID = rgba_to_int_id(0, 0, 0, 0)

COLOR_MAP = {
        CUBE_ID:       [1.0, 0.0, 0.0],  # 큐브   -> 빨간색
        PALLET_ID:     [0.5, 0.5, 0.5],  # 팔레트 -> 회색
        UNLABELLED_ID: [0.0, 1.0, 0.0],  # 라벨없음 -> 녹색 (디버깅용)
        BACKGROUND_ID: [0.0, 0.0, 0.0],  # 배경
    }

print("\n")
print("-" * 20)
print(f"Object ID : {CUBE_ID}")
print(f"Pallet ID : {PALLET_ID}")
print(F"Unlabelled ID : {UNLABELLED_ID}")
print("-" * 20)

def verify_and_visualize_dataset(save_gif=True, gif_name="final_visualization.gif"):
    base_dir = os.path.join(os.getcwd(), "Dataset")
    cam_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("Cam")]
    cam_dirs.sort()
    
    if not cam_dirs:
        print(f"오류: '{base_dir}' 경로에서 'Cam'으로 시작하는 데이터 폴더를 찾을 수 없습니다.")
        return

    all_points = []
    all_labels = []
    all_colors = []

    print("2D 데이터 기반 포인트 클라우드 재구성 시작...")
    total_points = 0
    for cam_dir in cam_dirs:
        cam_path = os.path.join(base_dir, cam_dir)
        try:
            # 1. 필요한 모든 파일 로드 (이전과 동일)
            params_path = os.path.join(cam_path, "camera_params", "camera_params_0000.json")
            depth_path = os.path.join(cam_path, "distance_to_image_plane", "distance_to_image_plane_0000.npy")
            semantic_path = os.path.join(cam_path, "semantic_segmentation", "semantic_segmentation_0000.png")
            rgb_path = os.path.join(cam_path, "rgb", "rgb_0000.png")
            
            with open(params_path, 'r') as f: params = json.load(f)
            depth_img = np.load(depth_path)
            semantic_img = np.array(Image.open(semantic_path))
            rgb_img = np.array(Image.open(rgb_path))
            
            # 알파 채널 제거
            if rgb_img.shape[2] == 4:
                rgb_img = rgb_img[:, :, :3]

            # 2. 카메라 파라미터 추출 및 계산 (이전과 동일)
            resolution = params["renderProductResolution"]
            width, height = resolution[0], resolution[1]
            focal_length = params["cameraFocalLength"]
            horiz_aperture = params["cameraAperture"][0]
            fx = focal_length * width / horiz_aperture
            fy = fx 
            cx = width / 2
            cy = height / 2

            # ==================== 핵심 해결 코드 ====================
            
            # 4. 뷰 행렬(View Matrix)을 가져옵니다.
            #    Open3D의 create_from_depth_image는 View Matrix(World->View)를 extrinsic으로 기대합니다.
            #    따라서 역행렬을 계산하지 않고 view_transform을 그대로 사용해야 합니다.
            view_transform = np.array(params["cameraViewTransform"]).reshape(4, 4)
            intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
            depth_o3d = o3d.geometry.Image(depth_img) 

            # 5. Unprojection과 월드 변환을 한번에 수행합니다.
            #    extrinsic 파라미터에 원본 뷰 행렬을 직접 전달합니다.
            pcd_world_space = o3d.geometry.PointCloud.create_from_depth_image(
                depth_o3d,
                intrinsics,
                extrinsic=view_transform, # <--- 핵심 수정: 역행렬이 아닌 원본 View Matrix 전달
                project_valid_depth_only=True
            )
            # ======================================================

            # 6. 시맨틱 및 RGB 정보 매핑
            #    Open3D가 생성한 포인트 클라우드와 1:1 매칭을 위해 마스크를 사용합니다.
            valid_points_mask = depth_img > 0
            
            rgb_colors = rgb_img[valid_points_mask]
            semantic_labels = semantic_img[valid_points_mask]
            
            # 생성된 포인트 클라우드의 크기와 색상/라벨 정보의 크기가 다를 수 있으므로,
            # 포인트 클라우드에 맞춰 잘라냅니다.
            num_points = len(pcd_world_space.points)
            pcd_world_space.colors = o3d.utility.Vector3dVector(rgb_colors[:num_points] / 255.0)
            
            semantic_ids = (semantic_labels[:num_points, 3].astype(np.uint32) << 24) | \
                        (semantic_labels[:num_points, 0].astype(np.uint32) << 16) | \
                        (semantic_labels[:num_points, 1].astype(np.uint32) << 8)  | \
                        (semantic_labels[:num_points, 2].astype(np.uint32))
            
            # 병합을 위해 최종 데이터 저장
            all_points.append(np.asarray(pcd_world_space.points))
            all_labels.append(semantic_ids)
            
            print(f"[✓] {cam_dir} 재구성 완료 ({num_points} points)")
            total_points += num_points

        except Exception as e:
            print(f"[!] {cam_dir} 처리 중 오류 발생: {e}")
            continue

        if not all_points:
            print("오류: 재구성된 포인트 클라우드가 없습니다.")
            return
        
    merged_points = np.vstack(all_points)
    merged_labels = np.concatenate(all_labels)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged_points)
    
    # ---------- 색상 지정 (시맨틱 라벨 기준) ----------
    total_find_points = 0
    colors_semantic = np.ones((len(merged_points), 3)) * 0.8
    for label_id, color in COLOR_MAP.items():
        indices = np.where(merged_labels == label_id)[0]
        if indices.size > 0:
            print(f"ID {label_id} 에 해당하는 포인트를 {indices.size}개 찾았습니다.")
            colors_semantic[indices] = color
            total_find_points += indices.size

    print(f"총 찾은 포인트 : {total_find_points}/{total_points}")
    pcd.colors = o3d.utility.Vector3dVector(colors_semantic)


    # ... (뷰어, 애니메이션, GIF 저장 로직은 이전과 동일) ...
    print("시각화 시작...")
    # ---------- 3. 뷰어 설정 ----------
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Merged Dataset Check", width=1280, height=720, visible=True)
    vis.add_geometry(pcd)
    
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.1, 0.1, 0.1])
    opt.point_size = 2.0
    
    vis.poll_events(); vis.update_renderer()

    vc = vis.get_view_control()
    center = pcd.get_center()
    vc.set_lookat(center)
    vc.set_front([-0.5, -0.5, 0.5])
    vc.set_up([0, 0, 1])
    vc.set_zoom(0.3)

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