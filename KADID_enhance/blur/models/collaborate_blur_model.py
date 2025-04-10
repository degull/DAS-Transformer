# ← 전체 통합 모델 (블러 분기)
import torch
import torch.nn as nn

# ✅ 블러별 개별 모델 import
from .gaussian_unet import GaussianUNet
from .lens_model import LensDeblur
from .motion_diffusion import MotionDeblurDiffusionModel

class BlurBranchRestorationModel(nn.Module):
    def __init__(self):
        super().__init__()
        # ✅ 각 블러 타입에 해당하는 복원 모델 초기화
        self.gaussian_model = GaussianUNet()
        self.lens_model = LensDeblur()
        self.motion_model = MotionDeblurDiffusionModel()

    def forward(self, x, class_id=None):
        outputs = []
        for i in range(x.size(0)):
            cid = class_id[i].item() if class_id is not None else -1
            input_img = x[i].unsqueeze(0)

            if cid == 1:
                print("📌 Gaussian Blur 복원 (High-Frequency Attention)")
                out = self.gaussian_model(input_img)

            elif cid == 2:
                print("📌 Lens Blur 복원 (CVPR2024 위치 기반 모델)")
                out = self.lens_model(input_img)

            elif cid == 3:
                print("📌 Motion Blur 복원 (ID-Blau Diffusion 기반)")
                out = self.motion_model.sample(input_img, device=input_img.device)

            else:
                print("⚠️ Unknown class, 입력 그대로 반환")
                out = input_img

            outputs.append(out)
        return torch.cat(outputs, dim=0)

# ✅ 실행 테스트
if __name__ == "__main__":
    print("🔷 BlurBranchRestorationModel 테스트 중...\n")
    model = BlurBranchRestorationModel().cuda()
    model.eval()

    input_tensor = torch.randn(4, 3, 256, 256).cuda()
    class_ids = torch.tensor([1, 2, 3, 1]).cuda()

    with torch.no_grad():
        output = model(input_tensor, class_ids)
    print("🔸 최종 출력 크기:", output.shape)  # [4, 3, 256, 256]
