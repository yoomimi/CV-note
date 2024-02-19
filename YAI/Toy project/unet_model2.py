import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),  # Batch Normalization
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # 인코딩
        self.enc1 = SimpleConv(1, 32) # 채널 1개
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # MaxPooling 레이어 정의
        self.enc2 = SimpleConv(32, 64)
        self.enc3 = SimpleConv(64, 128)  # 추가 인코딩 층
        # 디코딩
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = SimpleConv(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = SimpleConv(64, 32)
        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # 인코딩
        x1 = self.enc1(x)
        p1 = self.pool(x1)
        x2 = self.enc2(p1)
        p2 = self.pool(x2)
        x3 = self.enc3(p2)
        # 디코딩
        x = self.up1(x3)
        x = torch.cat([x, x2], dim=1)  # skip connection
        x = self.dec1(x)
        x = self.up2(x)
        x = torch.cat([x, x1], dim=1)  # skip connection
        x = self.dec2(x)
        out = self.out_conv(x)
        return out
