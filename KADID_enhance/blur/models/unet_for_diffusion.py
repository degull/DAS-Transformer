# Motion용 UNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ✅ Sinusoidal Time Embedding
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb  # [B, dim]

# ✅ Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim=None, dropout=0.1):
        super().__init__()
        self.time_dim = time_dim
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )

        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        if time_dim is not None:
            self.time_emb = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_dim, out_channels)
            )
        else:
            self.time_emb = None

    def forward(self, x, t_emb=None):
        h = self.block1(x)
        if self.time_emb and t_emb is not None:
            t = self.time_emb(t_emb).unsqueeze(-1).unsqueeze(-1)
            h = h + t
        h = self.block2(h)
        return h + self.shortcut(x)

# ✅ Downsampling Block
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.res1 = ResidualBlock(in_channels, out_channels, time_dim)
        self.res2 = ResidualBlock(out_channels, out_channels, time_dim)
        self.down = nn.Conv2d(out_channels, out_channels, 4, 2, 1)

    def forward(self, x, t_emb):
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        x_down = self.down(x)
        return x_down, x

# ✅ Upsampling Block
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.res1 = ResidualBlock(in_channels, out_channels, time_dim)
        self.res2 = ResidualBlock(out_channels, out_channels, time_dim)
        self.up = nn.ConvTranspose2d(out_channels, out_channels, 4, 2, 1)

    def forward(self, x, skip, t_emb):
        # ✅ 크기 다르면 보간
        if x.shape[-2:] != skip.shape[-2:]:
            skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)

        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        x = self.up(x)
        return x

# ✅ UNet 구조 (for Diffusion)
class UNet(nn.Module):
    def __init__(self, img_channels=6, base_channels=64, time_dim=256):
        super().__init__()
        self.time_embedding = nn.Sequential(
            TimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        self.init_conv = nn.Conv2d(img_channels, base_channels, 3, padding=1)

        # Encoder
        self.down1 = DownBlock(base_channels, base_channels * 2, time_dim)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, time_dim)

        # Bottleneck
        self.bot1 = ResidualBlock(base_channels * 4, base_channels * 4, time_dim)
        self.bot2 = ResidualBlock(base_channels * 4, base_channels * 4, time_dim)

        # Decoder
        self.up2 = UpBlock(base_channels * 4 + base_channels * 4, base_channels * 2, time_dim)
        self.up1 = UpBlock(base_channels * 2 + base_channels * 2, base_channels, time_dim)

        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, 3, 3, padding=1)
        )

    def forward(self, x, t):
        t_emb = self.time_embedding(t)
        x = self.init_conv(x)

        x1_down, x1 = self.down1(x, t_emb)
        x2_down, x2 = self.down2(x1_down, t_emb)

        x_mid = self.bot1(x2_down, t_emb)
        x_mid = self.bot2(x_mid, t_emb)

        x = self.up2(x_mid, x2, t_emb)
        x = self.up1(x, x1, t_emb)

        return self.out_conv(x)
