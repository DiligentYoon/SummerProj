import numpy as np, torch, glob, os
from torch.utils.data import Dataset

class NPZDataset(Dataset):
    def __init__(self, split, data_root,
                 num_point=2400, transforms=None):
        """
        Args
        ────
        split       : 'train' or 'test'
        data_root   : data/my_npz_dataset
        num_point   : 학습 코드에서 --npoint 로 지정
        transforms  : callable(points[ :,0:3 ]) → same shape

        저장 데이터 구조 : Dict 타입
            1. pos : (x,y,z) position
            2. x   : semantic masks  (whether object or not)
            3. y   : semantic labels (background=1, object=2)
            
        """
        self.files = []
        super().__init__()
        for sp in split:
            self.files += sorted(glob.glob(
                os.path.join(data_root, sp, "*.npz")))
        assert self.files, f"No .npz found in {data_root}/{split}"
        self.num_point  = num_point
        self.transforms = transforms
        
        print(f"[INFO] Data Loaded. # of files: {len(self.files)}")
    
        # 클래스 가중치 계산용
        # 덜 보이는 object를 더 잘 잡기위해 필요한 가중치
        label_hist = np.zeros(1, dtype=np.int64)
        for f in self.files:
            labels = np.load(f)["y"]-1
            m = labels.max()
            if m >= len(label_hist):
                label_hist = np.pad(label_hist, (0, m-len(label_hist)+1))
            label_hist += np.bincount(labels, minlength=len(label_hist))
        self.labelweights = label_hist / label_hist.sum()

    def __len__(self):  return len(self.files)

    def __getitem__(self, idx):
        data   = np.load(self.files[idx])
        points, feature, labels = data["pos"], data["x"], data["y"] -1  # (2400, C), (2400,)

        # ① 포인트 수 검증
        assert points.shape[0] == self.num_point, \
            f"{self.files[idx]}: 기대 {self.num_point}개, 실제 {points.shape[0]}개"

        # ② (선택) 증강
        if self.transforms:
            points[:, :3] = self.transforms(points[:, :3])

        return torch.hstack((torch.from_numpy(points).float(), torch.from_numpy(feature).float())), \
               torch.from_numpy(labels).long()
    

if __name__ == "__main__":
    dir_path = os.path.join(os.getcwd(), "Dataset", "TrainingData")
    split = ["mug_1", "mug_2", "cube_1", "cube_2", "cylinder_1", "cylinder_2"]

    dataset = NPZDataset(split, dir_path, num_point=2400)

    dataset.__getitem__(1)
