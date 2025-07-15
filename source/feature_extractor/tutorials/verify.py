import open3d as o3d
import numpy as np
import imageio.v2 as imageio   # ← GIF 저장용
import os, time

CUBE_ID   = 2
GROUND_ID = 1

ASSET_LIST = {
    "Mug": [
        "025_mug",
        "Room_Essentials_Mug_White_Yellow",
    ],
    "Cube": [
        "010_potted_meat_can",
        "003_cracker_box",
    ],
    "Cylinder": [
        "006_mustard_bottle",
        "021_bleach_cleanser",
    ],
}

def verify_and_visualize_dataset(save_gif=True, gif_name="rotation.gif"):
    # ---------- 1. 데이터 로드 ----------
    cls_idx, obj_idx = 0, 1
    cls_key = list(ASSET_LIST)[cls_idx]
    obj_key = ASSET_LIST[cls_key][obj_idx]

    dataset_dir     = os.path.join(os.getcwd(), "Dataset", cls_key)
    ply_filepath    = os.path.join(dataset_dir, f"{obj_key}_s000.ply")
    labels_filepath = os.path.join(dataset_dir, f"{obj_key}_s000_labels.npy")

    pcd    = o3d.io.read_point_cloud(ply_filepath)
    labels = np.load(labels_filepath)

    # ---------- 2. 색상 지정 ----------
    colors = np.zeros((len(pcd.points), 3))
    colors[np.where(labels == CUBE_ID)[0]]   = [1, 0, 0]
    colors[np.where(labels == GROUND_ID)[0]] = [0.5, 0.5, 0.5]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # ---------- 3. 뷰어 ----------
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Dataset Check", width=800, height=600, visible=True)
    vis.add_geometry(pcd)
    vis.poll_events(); vis.update_renderer()

    vc      = vis.get_view_control()
    center  = pcd.get_center()
    vc.set_lookat(center.tolist())
    vc.set_front([0.2, 0.2, 0.2])
    vc.set_up([0, 0, 0.5])
    vc.set_zoom(0.1)

    # ---------- 4. 애니메이션 ----------
    R_step = pcd.get_rotation_matrix_from_axis_angle([0, 0, np.deg2rad(3)])

    frames = []                            # ← GIF용 프레임 저장 리스트
    try:
        for _ in range(120):               # 120 step ≈ 2 s (60 FPS)
            pcd.rotate(R_step, center=center)
            vis.update_geometry(pcd)
            vis.poll_events(); vis.update_renderer()

            if save_gif:
                buf = vis.capture_screen_float_buffer(False)   # (h,w,3) float32
                img = (np.asarray(buf) * 255).astype(np.uint8) # → uint8
                frames.append(img)

            time.sleep(0.016)             # ≈60 FPS
    except KeyboardInterrupt:
        pass
    finally:
        vis.destroy_window()

    # ---------- 5. GIF 저장 ----------
    if save_gif and frames:
        imageio.mimsave(gif_name, frames, fps=30)   # 30 FPS 애니메이션
        print(f"[✓] GIF 저장 완료 → {gif_name}")

# 실행
verify_and_visualize_dataset()
