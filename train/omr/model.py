"""Residual U-Net producing per-channel symbol logits at input resolution."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Input height/width must be a multiple of this (4 stride-2 downsamplings).
SIZE_MULTIPLE = 16


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.skip = nn.Conv2d(cin, cout, 1, bias=False) if cin != cout else nn.Identity()

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        return F.relu(y + self.skip(x), inplace=True)


class UNet(nn.Module):
    def __init__(self, out_channels: int, widths=(32, 64, 128, 192, 256)):
        super().__init__()
        self.stem = nn.Sequential(ResBlock(1, widths[0]), ResBlock(widths[0], widths[0]))
        self.down = nn.ModuleList()
        for cin, cout in zip(widths[:-1], widths[1:]):
            self.down.append(nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                ResBlock(cout, cout), ResBlock(cout, cout)))
        # Dilations widen the receptive field to several staff heights, which the
        # network needs to tell e.g. whole from half rests, or count beams.
        self.bottleneck = nn.Sequential(ResBlock(widths[-1], widths[-1], 2), ResBlock(widths[-1], widths[-1], 4))
        self.up = nn.ModuleList()
        for cdeep, cskip in zip(widths[::-1][:-1], widths[::-1][1:]):
            self.up.append(nn.ModuleDict({
                "reduce": nn.Conv2d(cdeep, cskip, 1, bias=False),
                "fuse": nn.Sequential(ResBlock(2 * cskip, cskip), ResBlock(cskip, cskip)),
            }))
        self.head = nn.Conv2d(widths[0], out_channels, 1)

    def forward(self, x):
        x = self.stem(x)
        skips = [x]
        for stage in self.down:
            x = stage(x)
            skips.append(x)
        x = self.bottleneck(skips.pop())
        for stage in self.up:
            skip = skips.pop()
            x = F.interpolate(stage["reduce"](x), scale_factor=2, mode="nearest")
            x = stage["fuse"](torch.cat([x, skip], 1))
        return self.head(x)
