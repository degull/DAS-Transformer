import torch
import torch.nn as nn

class RestormerBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.conv2(x)
        return x + residual

class Restormer(nn.Module):
    def __init__(self, in_channels=3, embed_dim=48, num_blocks=4):
        super().__init__()
        self.embedding = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)
        self.transformer_blocks = nn.Sequential(*[RestormerBlock(embed_dim) for _ in range(num_blocks)])
        self.output_layer = nn.Conv2d(embed_dim, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_blocks(x)
        return self.output_layer(x)
