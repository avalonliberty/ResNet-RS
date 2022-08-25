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


class StemBlock(nn.Module):
    def __init__(self, channel_intermediate: int):
        super(StemBlock, self).__init__()
        channel_output = channel_intermediate * 2
        self.process = nn.Sequential(
            ConvFixedPadding(3, channel_intermediate, kernel_size=3, stride=2),
            BatchNormReLU(channel_intermediate),
            ConvFixedPadding(
                channel_intermediate, channel_intermediate, kernel_size=3, stride=1
            ),
            BatchNormReLU(channel_intermediate),
            ConvFixedPadding(
                channel_intermediate, channel_output, kernel_size=3, stride=1
            ),
            BatchNormReLU(channel_output),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.process(x)

        return output
