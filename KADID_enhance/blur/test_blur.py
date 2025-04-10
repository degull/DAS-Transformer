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