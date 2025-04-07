# KADID-10k에서 Compression (JPEG / JPEG2000) 이미지만 로드하는 데이터셋 정의
# compression_dataloader.py
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms

class CompressionDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

        # ✅ JPEG / JPEG2000 이미지만 필터링
        self.data = self.data[self.data["dist_img"].str.contains("_06_|_07_")]

        print(f"✅ 데이터셋 로드 완료! 총 {len(self.data)}개의 JPEG/JPEG2000 이미지가 포함됨.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row["dist_img"])
        ref_path = os.path.join(self.img_dir, row["ref_img"])  # 원본 이미지

        # 이미지 로드 및 변환
        img = Image.open(img_path).convert("RGB")
        ref = Image.open(ref_path).convert("RGB")

        if self.transform:
            img = self.transform(img)
            ref = self.transform(ref)

        return img, ref  # 압축된 이미지와 원본 이미지 반환

# ✅ 데이터셋 로드 함수
def get_compression_dataloader(csv_path, img_dir, batch_size=16, img_size=256):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    dataset = CompressionDataset(csv_path, img_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# ✅ 데이터셋 로드 테스트 코드
if __name__ == "__main__":
    # 🔹 KADID-10k 경로 설정 (사용자의 데이터 경로에 맞게 수정 필요)
    CSV_PATH = "E:/ARNIQA/ARNIQA/dataset/KADID10K/kadid10k.csv"  # CSV 파일 경로
    IMG_DIR = "E:/ARNIQA/ARNIQA/dataset/KADID10K/images"  # 이미지 폴더 경로

    # 데이터 로더 실행
    dataloader = get_compression_dataloader(CSV_PATH, IMG_DIR, batch_size=4)

    # 🔹 데이터 샘플 확인
    print("✅ 데이터 로드 테스트 시작!")
    for batch_idx, (img, ref) in enumerate(dataloader):
        print(f"🔹 배치 {batch_idx + 1}:")
        print(f" - 압축 이미지 크기: {img.shape}")  # (batch_size, 3, 256, 256)
        print(f" - 원본 이미지 크기: {ref.shape}")  # (batch_size, 3, 256, 256)
        if batch_idx == 2:  # 3개 배치까지만 확인
            break

    print("✅ 데이터 로드 테스트 완료!")
