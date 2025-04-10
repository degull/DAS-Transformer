""" import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from ast_compression_model import ASTCompressionRestoration
import numpy as np

# ✅ 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ASTCompressionRestoration().to(device)
model.load_state_dict(torch.load("C:/Users/IIPL02/Desktop/NEW/compression_restoration.pth"))
model.eval()

# ✅ 이미지 로드 및 변환
def load_image(img_path):
    original = Image.open(img_path).convert("RGB")
    original_size = original.size  # (W, H)
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    input_tensor = transform(original).unsqueeze(0).to(device)
    return input_tensor, original, original_size

# ✅ 복원 수행
def restore_image(img_path):
    input_tensor, original_img, original_size = load_image(img_path)

    with torch.no_grad():
        restored = model(input_tensor).squeeze(0).cpu()

    # ✅ 복원된 이미지를 원본 크기로 리사이즈
    restored_img = transforms.ToPILImage()(restored).resize(original_size)

    # ✅ 시각화
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(original_img)
    ax[0].set_title("Compressed Image")
    ax[0].axis('off')

    ax[1].imshow(restored_img)
    ax[1].set_title("Restored Image")
    ax[1].axis('off')

    plt.tight_layout()
    plt.show()

# ✅ 테스트 실행
test_image_path = "C:/Users/IIPL02/Desktop/NEW/data/KADID10K/images/I11_01_03.png"
restore_image(test_image_path) """

# branch 추가
# test_compression.py
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from ast_compression_model import ASTCompressionRestoration
import numpy as np
import os

# ✅ 클래스 ID 추출 함수
def infer_distortion_class(filename):
    """
    파일명에서 distortion class id 추출 (예: I11_09_03.png → class_id = 9)
    """
    try:
        parts = filename.split("_")
        class_str = parts[1]
        return int(class_str)
    except:
        print("⚠️ 파일명에서 class_id 추출 실패, default branch 사용")
        return -1

# ✅ 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ASTCompressionRestoration().to(device)
model.load_state_dict(torch.load("C:/Users/IIPL02/Desktop/NEW/compression_restoration_with_perceptual.pth", map_location=device))
model.eval()

# ✅ 이미지 로드 및 변환
def load_image(img_path):
    original = Image.open(img_path).convert("RGB")
    original_size = original.size  # (W, H)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    input_tensor = transform(original).unsqueeze(0).to(device)
    return input_tensor, original, original_size

# ✅ 복원 수행
def restore_image(img_path):
    input_tensor, original_img, original_size = load_image(img_path)
    class_id = infer_distortion_class(os.path.basename(img_path))
    class_tensor = torch.tensor([class_id], device=device)

    with torch.no_grad():
        restored = model(input_tensor, class_id=class_tensor).squeeze(0).cpu()

    # ✅ 복원된 이미지를 원본 크기로 리사이즈
    restored_img = transforms.ToPILImage()(restored).resize(original_size)

    # ✅ 시각화
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(original_img)
    ax[0].set_title(f"Compressed Image (Class {class_id})")
    ax[0].axis('off')

    ax[1].imshow(restored_img)
    ax[1].set_title("Restored Image")
    ax[1].axis('off')

    plt.tight_layout()
    plt.show()

# ✅ 테스트 실행
test_image_path = "C:/Users/IIPL02/Desktop/NEW/data/KADID10K/images/I43_10_04.png"  # 예: class 9 (JPEG2000)
restore_image(test_image_path)


# PSNR, SSIM, LPIPS 메트릭 계산 추가
# test_compression.py
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from ast_compression_model import ASTCompressionRestoration
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips  # pip install lpips / pip install scikit-image lpips


# ✅ 클래스 ID 추출 함수
def infer_distortion_class(filename):
    try:
        parts = filename.split("_")
        return int(parts[1])
    except:
        print("⚠️ 파일명에서 class_id 추출 실패, default branch 사용")
        return -1

# ✅ 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ASTCompressionRestoration().to(device)
model.load_state_dict(torch.load("C:/Users/IIPL02/Desktop/NEW/compression_restoration_with_perceptual.pth", map_location=device))
model.eval()

# ✅ LPIPS 모델 초기화
lpips_metric = lpips.LPIPS(net='alex').to(device)

# ✅ 이미지 로드 및 변환
def load_image(img_path):
    original = Image.open(img_path).convert("RGB")
    original_size = original.size  # (W, H)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    input_tensor = transform(original).unsqueeze(0).to(device)
    return input_tensor, original, original_size

# ✅ PSNR, SSIM, LPIPS 계산
def calculate_metrics(original_pil, restored_tensor):
    restored_np = restored_tensor.permute(1, 2, 0).numpy()
    original_np = np.array(original_pil.resize((256, 256))) / 255.0

    # Clamp restored_np to [0, 1]
    restored_np = np.clip(restored_np, 0.0, 1.0)

    # PSNR, SSIM
    psnr_val = psnr(original_np, restored_np, data_range=1.0)
    ssim_val = ssim(original_np, restored_np, channel_axis=2, data_range=1.0)

    # LPIPS (requires normalized tensor input)
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    original_t = preprocess(original_pil).unsqueeze(0).to(device)
    restored_t = (restored_tensor.unsqueeze(0).to(device) - 0.5) / 0.5  # Normalize to [-1, 1]
    lpips_val = lpips_metric(original_t, restored_t).item()

    return psnr_val, ssim_val, lpips_val

# ✅ 복원 수행
def restore_image(img_path):
    input_tensor, original_img, original_size = load_image(img_path)
    class_id = infer_distortion_class(os.path.basename(img_path))
    class_tensor = torch.tensor([class_id], device=device)

    with torch.no_grad():
        restored = model(input_tensor, class_id=class_tensor).squeeze(0).cpu()

    # ✅ clamp to prevent black artifacts
    restored = torch.clamp(restored, 0.0, 1.0)

    # ✅ 복원된 이미지를 원본 크기로 리사이즈
    restored_img = transforms.ToPILImage()(restored).resize(original_size)

    # ✅ 메트릭 계산
    psnr_val, ssim_val, lpips_val = calculate_metrics(original_img, restored)

    print(f"\n📊 PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}, LPIPS: {lpips_val:.4f}")

    # ✅ 시각화
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(original_img)
    ax[0].set_title(f"Compressed Image (Class {class_id})")
    ax[0].axis('off')

    ax[1].imshow(restored_img)
    ax[1].set_title("Restored Image")
    ax[1].axis('off')

    plt.tight_layout()
    plt.show()

# ✅ 테스트 실행
test_image_path = "C:/Users/IIPL02/Desktop/NEW/data/KADID10K/images/I43_10_04.png"  # JPEG = class 10
restore_image(test_image_path)
