import torch
import torch.nn as nn
from typing import Tuple

def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

def up_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
        nn.ReLU(inplace=True)
    )

class UNet(nn.Module):
    """Small U-Net for binary segmentation (1 input channel)."""
    def __init__(self, in_ch: int = 1, base_ch: int = 32):
        super().__init__()
        self.enc1 = conv_block(in_ch, base_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(base_ch, base_ch*2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(base_ch*2, base_ch*4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = conv_block(base_ch*4, base_ch*8)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(base_ch*8, base_ch*16)

        self.up4 = up_block(base_ch*16, base_ch*8)
        self.dec4 = conv_block(base_ch*16, base_ch*8)
        self.up3 = up_block(base_ch*8, base_ch*4)
        self.dec3 = conv_block(base_ch*8, base_ch*4)
        self.up2 = up_block(base_ch*4, base_ch*2)
        self.dec2 = conv_block(base_ch*4, base_ch*2)
        self.up1 = up_block(base_ch*2, base_ch)
        self.dec1 = conv_block(base_ch*2, base_ch)

        self.head = nn.Conv2d(base_ch, 1, kernel_size=1)

        self.gradcam_target = self.enc4[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)
        b = self.bottleneck(p4)
        u4 = self.up4(b)
        d4 = self.dec4(torch.cat([u4, e4], dim=1))
        u3 = self.up3(d4)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))
        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        out = self.head(d1)
        return out
