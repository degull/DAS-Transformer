import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class KADID10KDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        """
        KADID-10k 데이터셋을 로드하고, 왜곡을 5~6개 그룹으로 정리하는 클래스

        Args:
            csv_path (str): CSV 파일 경로
            img_dir (str): 이미지 폴더 경로
            transform (torchvision.transforms): 이미지 변환
        """
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

        # 25개 왜곡을 6개 그룹으로 정리
        self.distortion_groups = {
            "blur": ["01", "02"],  # Gaussian Blur, Lens Blur
            "noise": ["03", "04", "05"],  # Gaussian Noise, Impulse Noise, Multiplicative Noise
            "compression": ["06", "07"],  # JPEG, JPEG2000
            "color": ["08", "09", "10"],  # Color Quantization, Color Diffusion
            "contrast": ["11", "12"],  # Contrast Change, Brightness Change
            "other": ["13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25"],  # 기타 왜곡
        }

    def get_distortion_group(self, dist_img):
        """
        이미지 파일명에서 왜곡 그룹을 반환
        """
        distortion_type = dist_img.split("_")[1]  # 예: 'I01_01_01.png' -> '01'
        for group, codes in self.distortion_groups.items():
            if distortion_type in codes:
                return group
        return "unknown"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        데이터셋에서 하나의 샘플을 가져오는 메서드
        """
        row = self.data.iloc[idx]
        dist_img_path = os.path.join(self.img_dir, row["dist_img"])
        ref_img_path = os.path.join(self.img_dir, row["ref_img"])

        # 이미지 로드
        dist_img = Image.open(dist_img_path).convert("RGB")
        ref_img = Image.open(ref_img_path).convert("RGB")

        # 변환 적용
        if self.transform:
            dist_img = self.transform(dist_img)
            ref_img = self.transform(ref_img)

        # 왜곡 그룹 라벨
        distortion_group = self.get_distortion_group(row["dist_img"])
        label = list(self.distortion_groups.keys()).index(distortion_group)  # 정수 라벨로 변환

        return dist_img, ref_img, label

# 데이터셋 사용 예시
if __name__ == "__main__":
    csv_path = r"C:\Users\IIPL02\Desktop\NEW\data\KADID10K\kadid10k.csv"
    img_dir = r"C:\Users\IIPL02\Desktop\NEW\data\KADID10K\images"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = KADID10KDataset(csv_path, img_dir, transform=transform)

    # 샘플 데이터 확인
    dist_img, ref_img, label = dataset[0]
    print(f"Distorted Image Shape: {dist_img.shape}")
    print(f"Reference Image Shape: {ref_img.shape}")
    print(f"Distortion Label: {label}")
