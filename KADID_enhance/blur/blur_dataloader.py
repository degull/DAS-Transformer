from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms

class BlurDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

        # ✅ Gaussian(01), Lens(02), Motion(03) Blur만 필터링
        self.data = self.data[
            self.data["dist_img"].str.contains("_01_") |
            self.data["dist_img"].str.contains("_02_") |
            self.data["dist_img"].str.contains("_03_")
        ].reset_index(drop=True)

        print(f"✅ Blur Dataset Loaded: 총 {len(self.data)}개 샘플")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        dist_path = os.path.join(self.img_dir, row["dist_img"])
        ref_path = os.path.join(self.img_dir, row["ref_img"])

        # ✅ class_id 부여: 1=Gaussian, 2=Lens, 3=Motion
        if "_01_" in row["dist_img"]:
            class_id = 1
        elif "_02_" in row["dist_img"]:
            class_id = 2
        elif "_03_" in row["dist_img"]:
            class_id = 3
        else:
            class_id = -1  # 예외 처리용

        dist_img = Image.open(dist_path).convert("RGB")
        ref_img = Image.open(ref_path).convert("RGB")

        if self.transform:
            dist_img = self.transform(dist_img)
            ref_img = self.transform(ref_img)

        return dist_img, ref_img, class_id

# ✅ DataLoader 생성 함수
def get_blur_dataloader(csv_path, img_dir, batch_size=16, img_size=256):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # ✅ [-1, 1] 범위
    ])
    dataset = BlurDataset(csv_path, img_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ✅ 테스트 실행 코드
if __name__ == "__main__":
    csv_path = "E:/ARNIQA/ARNIQA/dataset/KADID10K/kadid10k.csv"
    img_dir = "E:/ARNIQA/ARNIQA/dataset/KADID10K/images"

    dataloader = get_blur_dataloader(csv_path, img_dir, batch_size=4)

    print("🔍 Blur DataLoader 테스트 시작!")
    for idx, (dist, ref, cls) in enumerate(dataloader):
        print(f"\n🔹 Batch {idx+1}")
        print(f" - 입력 이미지 크기: {dist.shape}")
        print(f" - 정답 이미지 크기: {ref.shape}")
        print(f" - Blur 종류(class_id): {cls.tolist()}")
        if idx == 2:  # 테스트용 3개 배치만
            break
    print("✅ DataLoader 테스트 완료!")


