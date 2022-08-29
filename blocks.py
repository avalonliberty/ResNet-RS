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


class DownSample(nn.Module):
    def __init__(self, channel_input: int, channel_output: int, stride: int):
        super(DownSample, self).__init__()
        self.down_sample = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2) if stride == 2 else nn.Identity(),
            ConvFixedPadding(channel_input, channel_output, kernel_size=1, stride=1),
            BatchNormReLU(channel_output, relu=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.down_sample(x)

        return output


class SEBlock(nn.Module):
    """
    SEBlock: a block which allows model to coordinate info in channels

    see https://arxiv.org/abs/1709.01507 for details
    """

    def __init__(self, channel_in: int, channel_out: int, se_ratio: float):
        super(SEBlock, self).__init__()
        channel_reduced = int(channel_out * se_ratio)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Linear(channel_in, channel_reduced),
            nn.Dropout(inplace=True),
            nn.Linear(channel_reduced, channel_out),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4, "x need to be presented as N, C, H, W format"
        batch_size, channel, __, __ = x.shape
        channel_info = self.gap(x).view(batch_size, channel)
        channel_weights = self.se(channel_info).view(batch_size, channel, 1, 1)
        weighted_output = channel_weights * x

        return weighted_output


class BottleNeckBlock(nn.Module):
    expansion = 4

    def __init__(
        self,
        channel_input: str,
        channel_intermediate: int,
        stride: int,
        se_ratio: float = 0.25,
    ):
        super(BottleNeckBlock, self).__init__()
        channel_output = channel_intermediate * self.expansion
        self.down_sample = DownSample(channel_input, channel_output, stride)

        self.conv1 = nn.Sequential(
            ConvFixedPadding(
                channel_input, channel_intermediate, kernel_size=1, stride=1
            ),
            BatchNormReLU(channel_intermediate),
        )
        self.conv2 = nn.Sequential(
            ConvFixedPadding(
                channel_intermediate, channel_intermediate, kernel_size=3, stride=stride
            ),
            BatchNormReLU(channel_intermediate),
        )
        self.conv3 = nn.Sequential(
            ConvFixedPadding(
                channel_intermediate, channel_output, kernel_size=1, stride=1
            ),
            BatchNormReLU(channel_output, relu=False),
        )
        self.se = (
            SEBlock(channel_output, channel_output, se_ratio)
            if se_ratio > 0
            else nn.Identity()
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.down_sample(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.se(x)
        output = self.relu(shortcut + x)

        return output
