# dataset_tid2013.py

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class TID2013Dataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

        # ✅ TID2013의 distortion group 정의
        self.distortion_groups = {
            "noise": [1, 2, 3, 4, 5, 6, 7, 19, 20],
            "blur_denoise": [8, 9],
            "compression": [10, 11, 12, 13, 21],
            "color_distortion": [14, 18, 22, 23],
            "brightness_contrast": [16, 17],
            "spatial_artifacts": [15, 24],
        }

    def get_distortion_group(self, distortion_type):
        for group, codes in self.distortion_groups.items():
            if distortion_type in codes:
                return group
        return "unknown"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        dist_img_path = os.path.join(self.img_dir, row["distortedname"])

        # 이미지 로딩
        dist_img = Image.open(dist_img_path).convert("RGB")

        # 변환 적용
        if self.transform:
            dist_img = self.transform(dist_img)

        # 라벨 변환
        distortion_type = int(row["distortion_type"])
        group = self.get_distortion_group(distortion_type)
        label = list(self.distortion_groups.keys()).index(group)

        return dist_img, label

"""
IXX_YY_Z.BMP
XX : 참조 이미지 번호 (1~25번까지 총 25개)
YY : 왜곡 유형 번호 (1~24번)
Z : 왜곡 강도 레벨 (1~5단계)
"""