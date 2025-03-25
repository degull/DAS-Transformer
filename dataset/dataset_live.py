import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class LIVEDistortionDataset(Dataset):
    def __init__(self, csv_path, img_root_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.img_root_dir = img_root_dir
        self.transform = transform

        self.distortion_classes = ["jp2k", "jpeg", "wn", "gblur", "fastfading"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        distortion_type = row["distortion_type"]
        distorted_filename = row["distortedname"]

        # ✅ 폴더 경로 포함해서 이미지 경로 구성
        dist_img_path = os.path.join(self.img_root_dir, distortion_type, distorted_filename)

        # ✅ 이미지 로드 및 변환
        image = Image.open(dist_img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # ✅ 클래스 인덱스
        label = self.distortion_classes.index(distortion_type)

        return image, label

# ✅ 테스트 실행
if __name__ == "__main__":
    csv_path = "C:/Users/IIPL02/Desktop/NEW/data/LIVE/live_dmos_full.csv"
    img_root = "C:/Users/IIPL02/Desktop/NEW/data/LIVE"  # images가 아니라 LIVE 최상위 폴더

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = LIVEDistortionDataset(csv_path, img_root, transform)
    sample_img, sample_label = dataset[100]
    print("✅ Image shape:", sample_img.shape)
    print("✅ Label index:", sample_label)
    print("✅ Distortion type:", dataset.distortion_classes[sample_label])
