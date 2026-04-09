import torch
import torch.nn as nn
import torch.nn.functional as F

def _double_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

class _EncoderStage(nn.Module):

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = _double_conv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))

class _DecoderStage(nn.Module):

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = _double_conv(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        if diff_h > 0 or diff_w > 0:
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class UNetPurifier(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.enc0 = _double_conv(3, 64)
        self.enc1 = _EncoderStage(64, 128)
        self.enc2 = _EncoderStage(128, 256)
        self.enc3 = _EncoderStage(256, 512)
        self.bottleneck = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), _double_conv(512, 1024))
        self.dec3 = _DecoderStage(1024, 512)
        self.dec2 = _DecoderStage(512, 256)
        self.dec1 = _DecoderStage(256, 128)
        self.dec0 = _DecoderStage(128, 64)
        self.output_head = nn.Sequential(nn.Conv2d(64, 3, kernel_size=1), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.enc0(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        b = self.bottleneck(s3)
        d3 = self.dec3(b, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        d0 = self.dec0(d1, s0)
        return self.output_head(d0)
