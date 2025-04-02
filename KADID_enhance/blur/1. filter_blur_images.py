# KADID-10k에서 blur에 해당하는 이미지(왜곡 코드 01~03)를 필터링
import os
import pandas as pd

# distortion group 정의 (blur 그룹 코드: 01, 02, 03)
distortion_groups = {
    "blur": ["01", "02", "03"],
    "color_distortion": ["04", "05", "06", "07", "08"],
    "compression": ["09", "10"],
    "noise": ["11", "12", "13", "14", "15"],
    "brightness_change": ["16", "17", "18"],
    "spatial_distortion": ["19", "20", "21", "22", "23"],
    "sharpness_contrast": ["24", "25"],
}

# distortion type 코드로 group 이름 리턴하는 함수
def get_distortion_group(dist_img_name):
    distortion_type = dist_img_name.split("_")[1]  # 예: I01_01_01.png → '01'
    for group, codes in distortion_groups.items():
        if distortion_type in codes:
            return group
    return "unknown"

# CSV 경로 및 이미지 폴더 경로
csv_path = "C:/Users/IIPL02/Desktop/NEW/data/KADID10K/kadid10k.csv"
img_dir = "C:/Users/IIPL02/Desktop/NEW/data/KADID10K/images"

# CSV 로드
df = pd.read_csv(csv_path)

# ✅ blur에 해당하는 이미지 필터링
blur_images = []
for idx, row in df.iterrows():
    dist_img_name = row["dist_img"]
    if get_distortion_group(dist_img_name) == "blur":
        full_path = os.path.join(img_dir, dist_img_name)
        blur_images.append(full_path)

# ✅ 결과 출력
print(f"✅ Blur 이미지 개수: {len(blur_images)}")
print("🔍 예시:")
for i in range(min(10, len(blur_images))):
    print(blur_images[i])
