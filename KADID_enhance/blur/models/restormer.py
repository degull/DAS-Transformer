import torch
import torch.nn as nn
import torch.nn.functional as F

# Depthwise Convolution
class DepthwiseConv2d(nn.Module):
    def __init__(self, in_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels)

    def forward(self, x):
        return self.depthwise(x)

# MDTA - Multi-Dconv Head Transposed Attention
class MDTA(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.qkv_dw = DepthwiseConv2d(dim * 3)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        x_ln = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        qkv = self.qkv_dw(self.qkv(x_ln))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.reshape(B, self.num_heads, C // self.num_heads, H * W)
        k = k.reshape(B, self.num_heads, C // self.num_heads, H * W)
        v = v.reshape(B, self.num_heads, C // self.num_heads, H * W)
        attn = (q.transpose(-2, -1) @ k) * self.temperature
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
        out = out.reshape(B, C, H, W)
        return self.project_out(out) + x

# GDFN - Gated Depthwise Feed-Forward Network
class GDFN(nn.Module):
    def __init__(self, dim, expansion_factor=2.66):
        super().__init__()
        hidden_features = int(dim * expansion_factor)
        self.norm = nn.LayerNorm(dim)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=False)
        self.dwconv = DepthwiseConv2d(hidden_features * 2)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        x_ln = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.dwconv(self.project_in(x_ln))
        x1, x2 = x.chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        return x + x_ln

# Transformer Block = MDTA + GDFN
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attn = MDTA(dim, num_heads)
        self.ffn = GDFN(dim)

    def forward(self, x):
        x = self.attn(x)
        x = self.ffn(x)
        return x

# Main Restormer
class Restormer(nn.Module):
    def __init__(self, inp_channels=3, out_channels=3, dim=48, num_blocks=[4, 6, 6, 8], heads=[1, 2, 4, 8]):
        super().__init__()
        self.patch_embed = nn.Conv2d(inp_channels, dim, kernel_size=3, padding=1)

        self.encoder1 = nn.Sequential(*[TransformerBlock(dim, heads[0]) for _ in range(num_blocks[0])])
        self.down1 = nn.Conv2d(dim, dim * 2, kernel_size=2, stride=2)

        self.encoder2 = nn.Sequential(*[TransformerBlock(dim * 2, heads[1]) for _ in range(num_blocks[1])])
        self.down2 = nn.Conv2d(dim * 2, dim * 4, kernel_size=2, stride=2)

        self.encoder3 = nn.Sequential(*[TransformerBlock(dim * 4, heads[2]) for _ in range(num_blocks[2])])
        self.down3 = nn.Conv2d(dim * 4, dim * 8, kernel_size=2, stride=2)

        self.latent = nn.Sequential(*[TransformerBlock(dim * 8, heads[3]) for _ in range(num_blocks[3])])

        self.up3 = nn.ConvTranspose2d(dim * 8, dim * 4, kernel_size=2, stride=2)
        self.decoder3 = nn.Sequential(*[TransformerBlock(dim * 4, heads[2]) for _ in range(num_blocks[2])])

        self.up2 = nn.ConvTranspose2d(dim * 4, dim * 2, kernel_size=2, stride=2)
        self.decoder2 = nn.Sequential(*[TransformerBlock(dim * 2, heads[1]) for _ in range(num_blocks[1])])

        self.up1 = nn.ConvTranspose2d(dim * 2, dim, kernel_size=2, stride=2)
        self.decoder1 = nn.Sequential(*[TransformerBlock(dim, heads[0]) for _ in range(num_blocks[0])])

        self.refine = nn.Sequential(*[TransformerBlock(dim, heads[0]) for _ in range(4)])
        self.output = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.patch_embed(x)
        x1 = self.encoder1(x1)
        x2 = self.encoder2(self.down1(x1))
        x3 = self.encoder3(self.down2(x2))
        x4 = self.latent(self.down3(x3))
        x3_up = self.decoder3(self.up3(x4) + x3)
        x2_up = self.decoder2(self.up2(x3_up) + x2)
        x1_up = self.decoder1(self.up1(x2_up) + x1)
        x_refined = self.refine(x1_up)
        out = self.output(x_refined)
        return out + x  # residual connection
