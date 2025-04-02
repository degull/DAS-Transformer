import os
import torch
import torch.nn as nn
from torchvision import transforms, utils
from PIL import Image
import matplotlib.pyplot as plt
from blur.dataset_kadid10k_blur_subclass import BlurSubclassDataset
from blur.train_blur_subclass_classifier import BlurSubclassClassifier
from models.nafnet import NAFNet  # 너가 사용하는 NAFNet 모델 경로
from models.restormer import Restormer  # 너가 사용하는 Restormer 모델 경로

# ✅ 복원기 로딩
def load_deblurring_models():
    nafnet = NAFNet().eval()
    nafnet.load_state_dict(torch.load("weights/nafnet_gaussian.pth", map_location="cpu"))

    restormer = Restormer().eval()
    restormer.load_state_dict(torch.load("weights/restormer_motion.pth", map_location="cpu"))

    nafnet_lens = NAFNet().eval()
    nafnet_lens.load_state_dict(torch.load("weights/nafnet_lens.pth", map_location="cpu"))

    return nafnet, restormer, nafnet_lens

# ✅ 복원 함수
def restore_with_subclass(x, blur_type, nafnet, restormer, nafnet_lens):
    outputs = []
    for i in range(x.size(0)):
        if blur_type[i] == 0:
            outputs.append(nafnet(x[i].unsqueeze(0)))
        elif blur_type[i] == 1:
            outputs.append(restormer(x[i].unsqueeze(0)))
        elif blur_type[i] == 2:
            outputs.append(nafnet_lens(x[i].unsqueeze(0)))
    return torch.cat(outputs, dim=0)

# ✅ 시각화 함수
def visualize_restore(input_img, restored_img, blur_type, save_dir, idx):
    input_img = input_img * 0.5 + 0.5  # unnormalize
    restored_img = restored_img * 0.5 + 0.5

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(input_img.permute(1, 2, 0).cpu().numpy())
    axs[0].set_title(f"Input (Class {blur_type})")
    axs[1].imshow(restored_img.permute(1, 2, 0).cpu().detach().numpy())
    axs[1].set_title("Restored")
    for ax in axs: ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"restore_{idx}.png"))
    plt.close()

# ✅ 메인 실행
def main():
    # 경로 설정
    save_dir = "results/restore"
    os.makedirs(save_dir, exist_ok=True)

    # 데이터 로드
    dataset = BlurSubclassDataset(
        csv_path="data/KADID10K/kadid10k.csv",
        img_dir="data/KADID10K/images",
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

    # 모델 로드
    classifier = BlurSubclassClassifier(weights_path="blur_subclass_resnet18.pth")
    nafnet, restormer, nafnet_lens = load_deblurring_models()

    # 복원 시작
    for idx, (img, _) in enumerate(loader):
        blur_type = classifier.predict_class(img)
        restored = restore_with_subclass(img, blur_type, nafnet, restormer, nafnet_lens)

        for i in range(img.size(0)):
            visualize_restore(img[i], restored[i], blur_type[i].item(), save_dir, idx * 4 + i)

        if idx >= 5:
            break  # 최대 20개까지만 복원

    print(f"✅ 복원 완료: 결과는 {save_dir} 폴더에 저장됨")

if __name__ == "__main__":
    main()
