import os
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class LIVEDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # ✅ 왜곡 유형 폴더 정의
        self.distortion_types = ["jp2k", "jpeg", "wn", "gblur", "fastfading"]
        self.image_list = []
        self.label_list = []

        # ✅ 각 폴더의 이미지 경로 수집 및 라벨 할당
        for idx, distortion in enumerate(self.distortion_types):
            distortion_dir = os.path.join(root_dir, distortion)
            for img_name in os.listdir(distortion_dir):
                if img_name.endswith(('.bmp', '.jpg', '.png')):
                    self.image_list.append(os.path.join(distortion_dir, img_name))
                    self.label_list.append(idx)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        label = self.label_list[idx]

        # ✅ 이미지 로드
        image = Image.open(img_path).convert("RGB")

        # ✅ 변환 적용
        if self.transform:
            image = self.transform(image)

        return image, label

# ✅ 테스트 코드
if __name__ == "__main__":
    root_dir = "C:/Users/IIPL02/Desktop/NEW/data/LIVE"
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = LIVEDataset(root_dir, transform=transform)

    image, label = dataset[10]
    print(f"✅ Distorted Image Shape: {image.shape}")
    print(f"✅ Distortion Label (Index): {label}")
    print(f"✅ Distortion Type: {dataset.distortion_types[label]}")
