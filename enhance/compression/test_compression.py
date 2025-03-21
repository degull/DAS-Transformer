# test_compression.py
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from ast_compression_model import ASTCompressionRestoration

# ✅ 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ASTCompressionRestoration().to(device)
model.load_state_dict(torch.load("compression_restoration.pth"))
model.eval()

# ✅ 이미지 로드 및 변환
def load_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    img = Image.open(img_path).convert("RGB")
    return transform(img).unsqueeze(0).to(device)

# ✅ 복원 수행
def restore_image(img_path):
    img = load_image(img_path)
    with torch.no_grad():
        restored = model(img).squeeze(0).permute(1, 2, 0).cpu().numpy()

    # ✅ 시각화
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(Image.open(img_path))
    ax[0].set_title("Compressed Image")
    ax[1].imshow(restored)
    ax[1].set_title("Restored Image")
    plt.show()

# ✅ 테스트 실행
#test_image_path = "C:/Users/IIPL02/Desktop/NEW/data/KADID10K/images/I04_06_01.png"
test_image_path = "C:/Users/IIPL02/Desktop/NEW/data/KADID10K/images/I81_23_04.png"
restore_image(test_image_path)
