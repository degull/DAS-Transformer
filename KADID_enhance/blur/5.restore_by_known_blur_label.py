import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from models.nafnet import NAFNet
from models.restormer import Restormer
import torch.nn.functional as F

# ✅ 경로 설정
csv_path = "C:/Users/IIPL02/Desktop/NEW/data/KADID10K/kadid10k.csv"
img_dir = "C:/Users/IIPL02/Desktop/NEW/data/KADID10K/images"
save_dir = "C:/Users/IIPL02/Desktop/NEW/KADID_enhance/blur/deblurring_results"
os.makedirs(save_dir, exist_ok=True)

# ✅ Blur code → 복원기 매핑
def get_blur_type_code(filename):
    code = filename.split("_")[1]  # 예: "I01_01_01.png" → "01"
    return code

# ✅ 복원기 로드
device = "cuda" if torch.cuda.is_available() else "cpu"

nafnet = NAFNet()
nafnet.load_state_dict(torch.load("C:/Users/IIPL02/Desktop/NEW/KADID_enhance/blur/weights/NAFNet-GoPro-width32.pth", map_location=device))
nafnet = nafnet.to(device).eval()

restormer = Restormer()
restormer.load_state_dict(torch.load("C:/Users/IIPL02/Desktop/NEW/KADID_enhance/blur/weights/restormer_motion_deblurring.pth", map_location=device))
restormer = restormer.to(device).eval()

# ✅ 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ✅ CSV 읽기
df = pd.read_csv(csv_path)

# ✅ 복원 실행
count = 0
for idx, row in df.iterrows():
    filename = row["dist_img"]
    blur_code = get_blur_type_code(filename)

    if blur_code not in ["01", "02", "03"]:
        continue  # blur가 아닌 경우 skip

    image_path = os.path.join(img_dir, filename)
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)  # [1, 3, 224, 224]

    # ✅ 복원기 선택
    if blur_code == "01" or blur_code == "02":
        restored = nafnet(input_tensor)
    elif blur_code == "03":
        restored = restormer(input_tensor)
    else:
        restored = input_tensor  # fallback

    # ✅ 후처리 및 저장
    restored_img = restored.squeeze(0).detach().cpu().clamp(0, 1)
    restored_img = transforms.ToPILImage()(restored_img)
    restored_img.save(os.path.join(save_dir, f"restore_{count:04d}_{blur_code}.png"))

    count += 1
    if count % 20 == 0:
        print(f"✅ 복원 이미지 {count}장 저장 완료")

print("🎉 모든 Blur 이미지 복원 및 저장 완료!")
