"""
ResNet from Scratch

WHY it exists:
As requested, this is a pedagogical implementation of a Residual Network (ResNet)
from first principles. Understanding this is critical before using advanced models like Swin.

THEORY:
Deep networks suffer from the "vanishing gradient" problem. As errors backpropagate
from the loss function to early layers, they are repeatedly multiplied by small weights,
eventually becoming zero. Early layers stop learning.

ResNets solve this using "Skip Connections" (or shortcuts).
Instead of learning the underlying mapping H(x), we let the network fit a residual
mapping F(x) = H(x) - x. The output becomes F(x) + x.
This allows gradients to flow directly through the skip connections, bypassing the
non-linearities, enabling networks of 100+ layers.

IMPLEMENTATION:
This implements the standard ResNet-50 architecture using Bottleneck blocks.
"""

import torch
import torch.nn as nn
from typing import Type, List, Optional
from ...utils.registry import BACKBONES


class Bottleneck(nn.Module):
    """
    Bottleneck Block.
    WHY bottleneck? A 3x3 conv with 256 channels is computationally expensive.
    Instead, we use a 1x1 conv to compress channels (e.g., to 64), 
    do the 3x3 conv, and then use another 1x1 conv to expand back to 256.
    """
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        
        # 1x1 convolution (Compression)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 3x3 convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 1x1 convolution (Expansion)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # If the input dimensions changed (e.g., due to stride=2 or channel expansion),
        # we must transform the identity shortcut to match the output dimensions
        # before we can add them together.
        if self.downsample is not None:
            identity = self.downsample(x)

        # Skip connection addition
        out += identity
        out = self.relu(out)

        return out


@BACKBONES.register("resnet50_scratch")
class ResNet(nn.Module):
    """
    ResNet-50 Architecture.
    """
    def __init__(self, block: Type[Bottleneck] = Bottleneck, layers: List[int] = [3, 4, 6, 3], in_channels: int = 1):
        super().__init__()
        self.in_channels = 64

        # Initial stem (reduces spatial resolution quickly)
        # 7x7 conv with stride 2, followed by max pool
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet Stages
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # We don't include the final FC layer because this is just a backbone.
        # It should return spatial feature maps.

    def _make_layer(self, block: Type[Bottleneck], out_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        
        # Downsample is needed if stride > 1 or in_channels != out_channels * expansion
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        # First block handles the downsampling
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        
        self.in_channels = out_channels * block.expansion
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Output shape: [B, 2048, H/32, W/32]
        return x
