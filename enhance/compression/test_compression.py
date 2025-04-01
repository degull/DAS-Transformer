import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from ast_compression_model import ASTCompressionRestoration
import numpy as np

# ✅ 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ASTCompressionRestoration().to(device)
model.load_state_dict(torch.load("compression_restoration.pth"))
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
test_image_path = "C:/Users/IIPL02/Desktop/NEW/data/LIVE/jp2k/img13.bmp"
restore_image(test_image_path)
