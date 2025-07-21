"""
Author: Benny
Date: Nov 2019
Adapted for custom dataset by User
Date: Jul 2025
"""
import argparse
import os
from dataset import NPZDataset  # train.py와 동일한 데이터셋 클래스 사용
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import numpy as np
import open3d as o3d # 시각화를 위해 open3d 추가

# 환경 설정 (train.py와 동일)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'model'))

# 클래스 및 라벨 정보 (train.py와 동일)
classes = ['background', 'object']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

# 시각화를 위한 색상 맵 (클래스별로 다른 색상 지정)
# 0:배경(회색), 1:객체(빨간색)
label_to_color = {
    0: [0.5, 0.5, 0.5],
    1: [1, 0, 0],
}


def parse_args():
    '''모델과 데이터 경로를 인수로 받도록 수정'''
    parser = argparse.ArgumentParser('Model')
    # train.py와 공통된 인수
    parser.add_argument('--model', type=str, default='pointnet2', help='model name')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during testing')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
    parser.add_argument('--npoint', type=int, default=2400, help='Point Number')
    
    # test.py에 특화된 인수
    parser.add_argument('--log_dir', type=str, required=True, help='Experiment root directory')
    parser.add_argument('--model_path', type=str, required=True, help='Path to a pre-trained model (.pth file)')
    parser.add_argument('--visual_dir', type=str, default='visual_results', help='Directory to save visualization results')
    
    return parser.parse_args()


def plot_pcd(points, gt_labels, pred_labels, filename):
    """
    포인트 클라우드, 정답, 예측 결과를 Open3D로 시각화하고 저장하는 함수
    """
    # Ground Truth 시각화
    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(points[:, :3])
    gt_colors = np.array([label_to_color[l] for l in gt_labels])
    pcd_gt.colors = o3d.utility.Vector3dVector(gt_colors)
    
    # Prediction 시각화
    pcd_pred = o3d.geometry.PointCloud()
    pcd_pred.points = o3d.utility.Vector3dVector(points[:, :3])
    pred_colors = np.array([label_to_color[l] for l in pred_labels])
    pcd_pred.colors = o3d.utility.Vector3dVector(pred_colors)

    # 이미지 파일로 저장
    vis_gt = o3d.visualization.Visualizer()
    vis_gt.create_window(window_name='Ground Truth', width=800, height=600)
    vis_gt.add_geometry(pcd_gt)
    vis_gt.update_geometry(pcd_gt)
    vis_gt.poll_events()
    vis_gt.update_renderer()
    vis_gt.capture_screen_image(f"{filename}_gt.png")
    vis_gt.destroy_window()

    vis_pred = o3d.visualization.Visualizer()
    vis_pred.create_window(window_name='Prediction', width=800, height=600)
    vis_pred.add_geometry(pcd_pred)
    vis_pred.update_geometry(pcd_pred)
    vis_pred.poll_events()
    vis_pred.update_renderer()
    vis_pred.capture_screen_image(f"{filename}_pred.png")
    vis_pred.destroy_window()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    '''CREATE DIR'''
    experiment_dir = Path(args.log_dir)
    log_dir = experiment_dir.joinpath('logs/')
    
    # 시각화 결과 저장 폴더 생성
    visual_dir = experiment_dir.joinpath(args.visual_dir)
    visual_dir.mkdir(exist_ok=True)
    
    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/test_%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    NUM_CLASSES = len(classes)
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size
    test_path = os.path.join(os.getcwd(), "Dataset", "TestingData")
    split = ["mug_1", "mug_2", "cube_1", "cube_2", "cylinder_1", "cylinder_2"] # train.py와 동일한 데이터 분할 사용

    print("start loading test data ...")
    # 테스트 시에는 데이터 증강(transform)을 적용하지 않음
    TEST_DATASET = NPZDataset(split, test_path, num_point=NUM_POINT)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
                                               pin_memory=True, drop_last=False) # drop_last=False로 모든 데이터 평가

    log_string("The number of test data is: %d" % len(TEST_DATASET))
    
    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    criterion = MODEL.get_loss().cuda() # Loss 계산도 수행
    
    # 저장된 모델 가중치 로드
    checkpoint = torch.load(args.model_path)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    log_string('Load pre-trained model from %s' % args.model_path)
    
    '''EVALUATION'''
    with torch.no_grad():
        num_batches = len(testDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        total_seen_class = [0 for _ in range(NUM_CLASSES)]
        total_correct_class = [0 for _ in range(NUM_CLASSES)]
        total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
        classifier = classifier.eval()

        log_string('---- Test EVALUATION ----')
        for i, (points, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
            points_np = points.data.numpy() # 시각화를 위해 numpy 배열 저장
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1)

            seg_pred, trans_feat = classifier(points)
            pred_val = seg_pred.contiguous().cpu().data.numpy() # (B, N, C)
            seg_pred_for_loss = seg_pred.contiguous().view(-1, NUM_CLASSES)

            batch_label = target.cpu().data.numpy() # (B, N)
            target_for_loss = target.view(-1, 1)[:, 0]
            
            # Loss 계산 (train.py의 평가 부분과 동일)
            # weights는 필요하다면 로드, 여기서는 None으로 가정
            loss = criterion(seg_pred_for_loss, target_for_loss, trans_feat, None)
            loss_sum += loss
            
            pred_val = np.argmax(pred_val, 2) # (B, N)
            correct = np.sum((pred_val == batch_label))
            total_correct += correct
            total_seen += (points.shape[0] * NUM_POINT)

            # 클래스별 성능 지표 계산
            for l in range(NUM_CLASSES):
                total_seen_class[l] += np.sum((batch_label == l))
                total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))

            # 배치 내 각 데이터에 대한 시각화
            for b in range(points.shape[0]):
                pts = points_np[b]
                l_gt = batch_label[b]
                l_pred = pred_val[b]
                file_name = f"batch_{i}_data_{b}"
                plot_pcd(pts, l_gt, l_pred, os.path.join(visual_dir, file_name))
        
        # 전체 성능 지표 계산 및 출력 (train.py의 평가 부분과 동일)
        mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float32) + 1e-6))
        log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
        log_string('eval point accuracy: %f' % (total_correct / float(total_seen)))
        log_string('eval point avg class acc: %f' % (
            np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float32) + 1e-6))))
        log_string('eval point avg class IoU (mIoU): %f' % mIoU)

        iou_per_class_str = '------- IoU per class --------\n'
        for l in range(NUM_CLASSES):
            iou_per_class_str += 'class %s, IoU: %.4f \n' % (
                seg_label_to_cat[l], total_correct_class[l] / float(total_iou_deno_class[l]))
        log_string(iou_per_class_str)

if __name__ == '__main__':
    args = parse_args()
    main(args)