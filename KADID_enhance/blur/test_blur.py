import torch
from PIL import Image
from torchvision import transforms
from collaborate_blur_model import BlurBranchRestorationModel

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BlurBranchRestorationModel().to(device)
    model.load_state_dict(torch.load("collaborate_blur_model.pth", map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # ✅ 테스트용 이미지 경로 및 클래스 ID
    test_images = [
        ("E:/ARNIQA/ARNIQA/dataset/KADID10K/images/sample_blur_01.jpg", 1),  # Gaussian
        ("E:/ARNIQA/ARNIQA/dataset/KADID10K/images/sample_blur_02.jpg", 2),  # Lens
        ("E:/ARNIQA/ARNIQA/dataset/KADID10K/images/sample_blur_03.jpg", 3),  # Motion
    ]

    for path, cid in test_images:
        img = Image.open(path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        class_id = torch.tensor([cid]).to(device)
        with torch.no_grad():
            restored = model(img_tensor, class_id)
        save_path = path.replace(".jpg", "_restored.png")
        transforms.ToPILImage()(restored.squeeze(0).cpu().clamp(0, 1)).save(save_path)
        print(f"🔸 복원 결과 저장됨: {save_path}")

if __name__ == "__main__":
    test()
