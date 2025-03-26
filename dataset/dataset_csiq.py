import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class CSIQDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.dist_dir = os.path.join(root_dir, "dst_imgs")
        self.src_dir = os.path.join(root_dir, "src_imgs")
        self.transform = transform

        self.distortion_types = ["AWGN", "BLUR", "contrast", "fnoise", "JPEG", "jpeg2000"]
        self.image_list = []

        # 각 왜곡 폴더에서 이미지 경로 수집
        for label, distortion in enumerate(self.distortion_types):
            dist_folder = os.path.join(self.dist_dir, distortion)
            for img_name in os.listdir(dist_folder):
                self.image_list.append({
                    "path": os.path.join(dist_folder, img_name),
                    "label": label
                })

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        item = self.image_list[idx]
        image = Image.open(item["path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, item["label"]
