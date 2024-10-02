from typing import Tuple, Union

import torch

from torch import nn
from torch.nn import functional as F

# TODO: Make all convolutional layers padding "same" if not stated by the paper.


class SmallBasicBlock(nn.Module):
    """Small Basic Block for LPRNet backbone
    Inspired from Squeeze Fire Blocks and Inception Blocks.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int or tuple): Stride applied to corresponding dimension of height and channel, shaped (H, W).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: Union[int, Tuple[int, int]] = 1,
    ):
        super().__init__()
        out_div4 = out_channels // 4

        if isinstance(stride, int):
            stride_h = stride_w = stride
        else:
            stride_h, stride_w = stride

        self.conv_in = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_div4,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding="same",
        )
        self.batch_norm_in = nn.BatchNorm2d(out_div4)

        self.conv_h = nn.Conv2d(
            in_channels=out_div4,
            out_channels=out_div4,
            kernel_size=(3, 1),
            stride=(1, stride_w),
            padding=(1, 0),
        )
        self.batch_norm_h = nn.BatchNorm2d(out_div4)

        self.conv_w = nn.Conv2d(
            in_channels=out_div4,
            out_channels=out_div4,
            kernel_size=(1, 3),
            stride=(stride_h, 1),
            padding=(0, 1),
        )
        self.batch_norm_w = nn.BatchNorm2d(out_div4)

        self.conv_out = nn.Conv2d(
            in_channels=out_div4,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding="same",
        )
        self.batch_norm_out = nn.BatchNorm2d(out_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): Tensor of shape (N, C, H, W).
        """
        xs = self.conv_in(input)
        xs = self.batch_norm_in(xs)
        xs = F.relu(xs)

        xs = self.conv_h(xs)
        xs = self.batch_norm_h(xs)
        xs = F.relu(xs)

        xs = self.conv_w(xs)
        xs = self.batch_norm_w(xs)
        xs = F.relu(xs)

        xs = self.conv_out(xs)
        xs = self.batch_norm_out(xs)
        xs = F.relu(xs)

        return xs


class LPRNet(nn.Module):
    """LPRNet model as defined in https://arxiv.org/abs/1806.10447, with modifications.

    Args:
        num_classes (int): Number of character classes (including blank). Default: 37
        input_channels (int): Number of input channel / band. 3 for RGB, 1 for Grayscale. Default: 3
        stn (torch.nn.Module): Spatial Transformer Network Layer. Default: None
        dropout_p (float): Dropout ratio of the backbone. Default: 0.5
    """

    def __init__(
        self,
        num_classes: int = 37,
        input_channels: int = 3,
        stn: nn.Module = None,
        dropout_p: float = 0.5,
    ):
        super().__init__()

        assert num_classes > 0, "Number of classes must be greater than 0."
        assert input_channels in [1, 3], "Input channels must be either 1 or 3."

        self.num_classes = num_classes
        self.input_channels = input_channels

        # --- Spatial Transformer Layer ---
        self.stn = stn
        self.using_stn = False

        # --- Backbone ---
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1)),
            SmallBasicBlock(in_channels=64, out_channels=128),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2)),
            SmallBasicBlock(in_channels=64, out_channels=256),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            SmallBasicBlock(in_channels=256, out_channels=256),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 2)),
            nn.Dropout(dropout_p),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 4), stride=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(
                in_channels=256,
                out_channels=self.num_classes,
                kernel_size=(13, 1),
                stride=1,
            ),
            nn.BatchNorm2d(num_features=self.num_classes),
            nn.ReLU(),
        )
        self.container = nn.Conv2d(
            in_channels=448 + self.num_classes,
            out_channels=self.num_classes,
            kernel_size=(1, 1),
            stride=(1, 1),
        )
        # --- Backbone End ---

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): Images of size (N, C, H, W).

        Return
            torch.Tensor of shape (N, C, T) where:
                - N: Number of batches
                - C: Number of classes
                - T: Timesteps
        """

        x = input
        x = self.forward_stn(x)
        keep_features = list()
        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)
            if i in [2, 6, 13, 22]:  # [2, 4, 8, 11, 22]
                keep_features.append(x)

        global_context = list()
        for i, f in enumerate(keep_features):
            if i in [0, 1]:
                f = nn.AvgPool2d(kernel_size=5, stride=5)(f)
            if i in [2]:
                f = nn.AvgPool2d(kernel_size=(4, 10), stride=(4, 2))(f)
            f_pow = torch.pow(f, 2)
            f_mean = torch.mean(f_pow)
            f = torch.div(f, f_mean)
            global_context.append(f)

        x = torch.cat(global_context, 1)
        x = self.container(x)
        logits = torch.mean(x, dim=2)

        return logits

    def forward_stn(self, input: torch.Tensor) -> torch.Tensor:
        if not self.using_stn or self.stn is None:
            return input

        return self.stn(input)

    def use_stn(self, enable: bool = True):
        self.using_stn = enable
