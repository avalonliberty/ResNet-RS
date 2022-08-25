import torch.nn as nn
import torch


class ConvFixedPadding(nn.Module):
    def __init__(
        self,
        channel_input: int,
        channel_output: int,
        kernel_size: int,
        stride: int,
        bias: bool = False,
    ):
        super(ConvFixedPadding, self).__init__()
        self.conv = nn.Conv2d(
            channel_input,
            channel_output,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class BatchNormReLU(nn.Module):
    def __init__(self, channels: int, relu: bool = True):
        super(BatchNormReLU, self).__init__()
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU() if relu else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)
        output = self.relu(x)

        return output
