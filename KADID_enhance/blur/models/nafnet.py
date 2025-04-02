import torch
import torch.nn as nn

class SimpleNAFBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        identity = x
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x + identity

class NAFNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, width=32, num_blocks=4):
        super().__init__()
        self.entry = nn.Conv2d(in_channels, width, 3, padding=1)
        self.encoder = nn.Sequential(*[SimpleNAFBlock(width) for _ in range(num_blocks)])
        self.middle = SimpleNAFBlock(width)
        self.decoder = nn.Sequential(*[SimpleNAFBlock(width) for _ in range(num_blocks)])
        self.exit = nn.Conv2d(width, out_channels, 3, padding=1)

    def forward(self, x):
        x = self.entry(x)
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return self.exit(x)
