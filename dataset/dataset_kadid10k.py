# 3/13 & 3/14 수정된거 없음

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class KADID10KDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

        # ✅ 25개 왜곡을 6개 그룹으로 정리
        self.distortion_groups = {
            "blur": ["01", "02"],
            "noise": ["03", "04", "05"],
            "compression": ["06", "07"],
            "color": ["08", "09", "10"],
            "contrast": ["11", "12"],
            "other": ["13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25"],
        }

    def get_distortion_group(self, dist_img):

        distortion_type = dist_img.split("_")[1]  # 예: 'I01_01_01.png' -> '01'
        for group, codes in self.distortion_groups.items():
            if distortion_type in codes:
                return group
        return "unknown"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]
        dist_img_path = os.path.join(self.img_dir, row["dist_img"])

        # ✅ 이미지 로드
        dist_img = Image.open(dist_img_path).convert("RGB")

        # ✅ 변환 적용
        if self.transform:
            dist_img = self.transform(dist_img)

        # ✅ 왜곡 그룹 라벨
        distortion_group = self.get_distortion_group(row["dist_img"])
        label = list(self.distortion_groups.keys()).index(distortion_group)  # 정수 라벨로 변환

        return dist_img, label

# ✅ 데이터셋 테스트 실행
if __name__ == "__main__":
    csv_path = "C:/Users/IIPL02/Desktop/NEW/data/KADID10K/kadid10k.csv"
    img_dir = "C:/Users/IIPL02/Desktop/NEW/data/KADID10K/images"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = KADID10KDataset(csv_path, img_dir, transform=transform)

    # ✅ 샘플 데이터 확인
    dist_img, label = dataset[0]
    print(f"✅ Distorted Image Shape: {dist_img.shape}")
    print(f"✅ Distortion Label: {label}")

# blur, noise, compression, color, contrast, other

""" xx_yy_zz.png:

xx: 원본 이미지 ID
yy: 왜곡 코드 (01~25)
zz: 왜곡 레벨 (1~5) → 이번 프로젝트에서는 고려하지 않음 """