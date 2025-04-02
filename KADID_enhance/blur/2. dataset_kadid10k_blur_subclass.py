# blur 이미지들을 subclass 분류용 Dataset으로 만드는 코드
# Blur 클래스(Gaussian, Motion, Lens)에 대한 PyTorch Dataset 클래스 정의
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class KADID10KBlurSubclassDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

        # ✅ Blur subclass distortion 코드 → 라벨 매핑
        self.blur_subclass_map = {
            "01": 0,  # Gaussian
            "02": 1,  # Lens
            "03": 2,  # Motion
        }

        # ✅ Blur subclass에 해당하는 데이터만 필터링
        self.blur_data = self.data[self.data["dist_img"].apply(self.is_blur_image)].reset_index(drop=True)

    def is_blur_image(self, filename):
        """01 ~ 03번 블러만 필터링"""
        distortion_code = filename.split("_")[1]
        return distortion_code in self.blur_subclass_map

    def get_blur_label(self, filename):
        """파일명에서 subclass 라벨 추출"""
        distortion_code = filename.split("_")[1]
        return self.blur_subclass_map[distortion_code]

    def __len__(self):
        return len(self.blur_data)

    def __getitem__(self, idx):
        row = self.blur_data.iloc[idx]
        img_path = os.path.join(self.img_dir, row["dist_img"])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.get_blur_label(row["dist_img"])  # 0~2

        return image, label

if __name__ == "__main__":
    csv_path = "C:/Users/IIPL02/Desktop/NEW/data/KADID10K/kadid10k.csv"
    img_dir = "C:/Users/IIPL02/Desktop/NEW/data/KADID10K/images"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = KADID10KBlurSubclassDataset(csv_path, img_dir, transform=transform)

    print(f"총 Blur 이미지 수: {len(dataset)}")
    for i in range(5):
        img, label = dataset[i]
        print(f"[{i}] 이미지 shape: {img.shape}, 라벨 (0:Gaussian, 1:Lens, 2:Motion): {label}")
