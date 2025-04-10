""" import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from collaborate_blur_model import BlurBranchRestorationModel
import torchmetrics
from lpips import LPIPS

# ✅ Normalize 파라미터 (ImageNet 기준)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ✅ 평가용 메트릭 함수 (normalize 유지)
def calculate_metrics(pred, target):
    pred = pred.unsqueeze(0)
    target = target.unsqueeze(0)

    psnr = torchmetrics.functional.peak_signal_noise_ratio(pred, target, data_range=1.0).item()
    ssim = torchmetrics.functional.structural_similarity_index_measure(pred, target, data_range=1.0).item()

    lpips_model = LPIPS(net='alex').to(pred.device)
    lpips_model.eval()
    # ✅ LPIPS는 [-1, 1] 범위를 기대하므로 normalize된 상태 그대로 입력
    lpips_score = lpips_model(pred, target).item()
    return psnr, ssim, lpips_score

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BlurBranchRestorationModel().to(device)
    model.load_state_dict(torch.load("collaborate_blur_model.pth", map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    denormalize = transforms.Normalize(
        mean=[-m/s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
        std=[1/s for s in IMAGENET_STD]
    )

    test_images = [
        ("C:/Users/IIPL02/Desktop/NEW/data/KADID10K/images/I24_01_03.png", 1),  # Gaussian
        ("C:/Users/IIPL02/Desktop/NEW/data/KADID10K/images/I75_02_03.png", 2),  # Lens
        ("C:/Users/IIPL02/Desktop/NEW/data/KADID10K/images/I81_03_04.png", 3),  # Motion
    ]

    for path, cid in test_images:
        img = Image.open(path).convert("RGB")
        ref_path = path.replace(".png", "_ref.png")
        try:
            ref_img = Image.open(ref_path).convert("RGB")
        except:
            ref_img = img

        img_tensor = transform(img).to(device)
        ref_tensor = transform(ref_img).to(device)
        input_tensor = img_tensor.unsqueeze(0)
        ref_tensor = ref_tensor.unsqueeze(0)
        class_id = torch.tensor([cid]).to(device)

        with torch.no_grad():
            restored = model(input_tensor, class_id)
            restored = restored.squeeze(0).cpu()

        # ✅ 메트릭 계산
        psnr, ssim_val, lpips_score = calculate_metrics(restored, ref_tensor.squeeze(0).cpu())

        # ✅ 시각화를 위해 denormalize 및 clamp
        restored_img = denormalize(restored).clamp(0, 1)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(img.resize((256, 256)))
        axes[0].set_title(f"Input Blur (class={cid})")
        axes[0].axis("off")
        axes[1].imshow(transforms.ToPILImage()(restored_img))
        axes[1].set_title(f"Restored\nPSNR={psnr:.2f}, SSIM={ssim_val:.3f}, LPIPS={lpips_score:.3f}")
        axes[1].axis("off")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    test()
 """

# (각 블러마다 encoder + decoder 전부 별도)
# test_blur.py
import torch
import torch.nn as nn
from torchvision import transforms
from blur_dataloader import get_blur_dataloader
from models.collaborate_blur_model import BlurBranchRestorationModel

import lpips
from pytorch_msssim import ssim
from torchvision.utils import save_image
import os
import math

# 평가 지표: PSNR 계산 함수
def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

# 저장 폴더
os.makedirs("results", exist_ok=True)

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BlurBranchRestorationModel().to(device)
    model.load_state_dict(torch.load("first_collaborate_blur_model.pth", map_location=device))
    model.eval()

    lpips_loss = lpips.LPIPS(net='alex').to(device)
    lpips_loss.eval()

    dataloader = get_blur_dataloader(
        csv_path="E:/ARNIQA/ARNIQA/dataset/KADID10K/kadid10k.csv",
        img_dir="E:/ARNIQA/ARNIQA/dataset/KADID10K/images",
        batch_size=1,
        img_size=256
    )

    total_psnr, total_ssim, total_lpips = 0, 0, 0
    with torch.no_grad():
        for idx, (dist_img, ref_img, cls) in enumerate(dataloader):
            dist_img, ref_img, cls = dist_img.to(device), ref_img.to(device), cls.to(device)

            restored = model(dist_img, class_id=cls)

            # 📌 지표 계산
            psnr_val = calculate_psnr(restored, ref_img)
            ssim_val = ssim(restored, ref_img, data_range=1.0, size_average=True).item()
            lpips_val = lpips_loss(restored, ref_img).mean().item()

            total_psnr += psnr_val
            total_ssim += ssim_val
            total_lpips += lpips_val

            # 🔽 샘플 저장 (선택적으로)
            if idx < 5:
                save_image(dist_img, f"results/{idx}_input.png")
                save_image(ref_img, f"results/{idx}_gt.png")
                save_image(restored, f"results/{idx}_restored.png")

    N = len(dataloader)
    print("\n✅ [Test Results]")
    print(f"📈 Avg PSNR:  {total_psnr / N:.4f}")
    print(f"📈 Avg SSIM:  {total_ssim / N:.4f}")
    print(f"📉 Avg LPIPS: {total_lpips / N:.4f}")

if __name__ == "__main__":
    test()
