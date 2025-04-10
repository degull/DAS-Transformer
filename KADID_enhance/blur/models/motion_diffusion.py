# ID-Blau 기반 DDPM 구조 (Motion 전용)
# motion_diffusion.py
import torch
import torch.nn as nn
import numpy as np
from functools import partial
from .unet_for_diffusion import UNet  # diffusion 전용 UNet

# ✅ Charbonnier Loss 정의
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps))


# ✅ 시간별 계수 추출 함수
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


# ✅ DDPM 모델 정의
class DDPM(nn.Module):
    def __init__(self, model, img_channels, betas, criterion='l1', device='cuda'):
        super().__init__()
        self.model = nn.DataParallel(model).to(device)
        self.img_channels = img_channels
        self.num_timesteps = len(betas)
        self.device = device

        # 손실 함수 설정
        if criterion == 'l1':
            self.criterion = CharbonnierLoss()
        elif criterion == 'l2':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError("criterion must be 'l1' or 'l2'")

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas", to_torch(alphas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("reciprocal_sqrt_alphas", to_torch(np.sqrt(1 / alphas)))
        self.register_buffer("remove_noise_coeff", to_torch(betas / np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("sigma", to_torch(np.sqrt(betas)))

    def perturb_x(self, x, t, noise):
        return (extract(self.sqrt_alphas_cumprod, t, x.shape) * x +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise)

    def compute_loss(self, x, condition, t):
        noise = torch.randn_like(x)
        x_t = self.perturb_x(x, t, noise)
        input = torch.cat([x_t, condition], dim=1)
        pred_noise = self.model(input, t)
        return self.criterion(pred_noise, noise)

    def forward(self, x, condition):
        b = x.size(0)
        t = torch.randint(0, self.num_timesteps, (b,), device=x.device)
        return self.compute_loss(x, condition, t)

    @torch.no_grad()
    def remove_noise(self, x, condition, t):
        input = torch.cat([x, condition], dim=1)
        return (x - extract(self.remove_noise_coeff, t, x.shape) * self.model(input, t)) * extract(self.reciprocal_sqrt_alphas, t, x.shape)

    @torch.no_grad()
    def sample(self, condition, device=None):
        device = device or self.device
        b, c, h, w = condition.shape
        x = torch.randn((b, 3, h, w), device=device)

        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            x = self.remove_noise(x, condition, t_batch)
            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)

        return x.clamp(0.0, 1.0)


# ✅ Motion Blur 전용 복원 클래스
class MotionDeblurDiffusionModel(nn.Module):
    def __init__(self, diffusion_steps=1000, device='cuda'):
        super().__init__()
        betas = np.linspace(1e-4, 0.02, diffusion_steps)
        self.ddpm = DDPM(
            model=UNet(img_channels=6),  # x_t + condition
            img_channels=3,
            betas=betas,
            criterion='l1',
            device=device
        )

    def sample(self, x, device=None):
        device = device or x.device
        return self.ddpm.sample(condition=x, device=device)
