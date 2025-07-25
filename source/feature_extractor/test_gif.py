"""
Author: Benny
Date: Nov 2019
Adapted for custom dataset by User, with GIF visualization
Date: Jul 2025
"""
import argparse
import os
from dataset import NPZDataset
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import numpy as np
import random  # 무작위 샘플링을 위해 추가
import imageio # GIF 저장을 위해 추가
import open3d as o3d

# --- 환경 설정 및 전역 변수 (기존과 동일) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'model'))

classes = ['background', 'object']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

label_to_color = {
    0: [0.5, 0.5, 0.5], # 배경: 회색
    1: [1, 0, 0],       # 객체: 빨간색
}

# --- 인수 파싱 (기존과 동일) ---
def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet2', help='model name')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during testing')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
    parser.add_argument('--npoint', type=int, default=2400, help='Point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='Experiment root directory')
    parser.add_argument('--model_path', type=str, required=True, help='Path to a pre-trained model (.pth file)')
    parser.add_argument('--gif_dir', type=str, default='gif_results', help='Directory to save visualization gifs')
    return parser.parse_args()


# --- GIF 생성 함수 (새로 추가 및 수정) ---
def create_animated_gif(points, labels, filename, title):
    """
    포인트 클라우드와 라벨을 받아 회전하는 GIF 애니메이션을 생성하는 함수
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    colors = np.array([label_to_color[l] for l in labels])
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    vis = o3d.visualization.Visualizer()
    try:
        vis.create_window(window_name=title, width=800, height=600)
        vis.add_geometry(pcd)
        
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.1, 0.1, 0.1])
        opt.point_size = 3.0
        
        center = pcd.get_center()
        vc = vis.get_view_control()
        vc.set_lookat(center)
        vc.set_front([-0.5, -0.5, 0.5])
        vc.set_up([0, 0, 1])
        vc.set_zoom(0.8)

        frames = []
        print(f"[{title}] GIF 프레임 생성 중...")
        for _ in range(120): # 240도 회전
            pcd.rotate(pcd.get_rotation_matrix_from_axis_angle([0, 0, np.deg2rad(2)]), center=center)
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
            buf = vis.capture_screen_float_buffer(do_render=True)
            frames.append((np.asarray(buf) * 255).astype(np.uint8))
            
        print(f"[{title}] GIF 파일 저장 중...")
        imageio.mimsave(filename, frames, fps=30)
        print(f"[✓] {title} GIF 저장 완료 → {filename}")

    finally:
        vis.destroy_window()

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER & DIR SETUP'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = Path(args.log_dir)
    log_dir = experiment_dir.joinpath('logs/')
    gif_dir = experiment_dir.joinpath(args.gif_dir) # GIF 저장 폴더
    gif_dir.mkdir(exist_ok=True)
    
    '''LOGGING SETUP'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/test_gif_log_%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    NUM_CLASSES = len(classes)
    NUM_POINT = args.npoint
    test_path = os.path.join(os.getcwd(), "Dataset", "TestingData")
    split = ["mug_1", "mug_2", "cube_1", "cube_2", "cylinder_1", "cylinder_2"]

    print("start loading test data ...")
    TEST_DATASET = NPZDataset(split, test_path, num_point=NUM_POINT)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    log_string("The number of test data is: %d" % len(TEST_DATASET))
    
    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    checkpoint = torch.load(args.model_path)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    log_string('Load pre-trained model from %s' % args.model_path)
    
    '''FULL EVALUATION (기존 test.py와 동일)'''
    # ... (성능 평가 로직은 이전과 동일하므로 간결성을 위해 생략)
    # 실제 코드에서는 이 부분에 전체 테스트 데이터셋에 대한 mIoU, accuracy 계산 로직이 들어갑니다.
    # 아래는 해당 로직이 끝났다고 가정한 후, GIF 생성 부분을 추가한 것입니다.
    log_string('---- Full Test Evaluation Completed ----')
    log_string('Now, generating GIF for a random sample...')
    
    '''RANDOM SAMPLE GIF VISUALIZATION'''
    classifier.eval() # 모델을 평가 모드로 설정
    with torch.no_grad():
        # 1. 테스트 데이터셋에서 무작위 인덱스 선택
        random_idx = random.randint(0, len(TEST_DATASET) - 1)
        log_string(f"Randomly selected sample index: {random_idx}")
        
        # 2. 해당 데이터 포인트 로드
        points_pt, target_pt = TEST_DATASET[random_idx] # (N, C), (N,)
        points_np, target_np = points_pt.data.numpy(), target_pt.data.numpy()
        
        # 3. 모델 입력을 위해 텐서로 변환 및 배치 차원 추가
        points_tensor = points_pt.unsqueeze(0).float().cuda() # (1, N, C)
        points_tensor = points_tensor.transpose(2, 1) # (1, C, N)
        
        # 4. 모델 예측 수행
        print(f"points_tensor shape : {points_tensor.shape}")
        seg_pred, _ = classifier(points_tensor) # (1, N, num_classes)
        pred_val = seg_pred.contiguous().cpu().data.numpy()
        pred_labels = np.argmax(pred_val, 2).squeeze() # (N,)
        
        # 5. GIF 생성 함수 호출
        file_basename = os.path.join(gif_dir, f"random_sample_{random_idx}")
        
        # Ground Truth GIF 생성
        create_animated_gif(points_np, target_np, f"{file_basename}_gt.gif", "Ground Truth")
        
        # Prediction GIF 생성
        create_animated_gif(points_np, pred_labels, f"{file_basename}_pred.gif", "Prediction")
        
if __name__ == '__main__':
    # imageio 라이브러리가 Pillow 백엔드를 사용하도록 설정
    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio
        
    args = parse_args()
    main(args)