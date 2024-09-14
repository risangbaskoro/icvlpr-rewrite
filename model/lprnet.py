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
        self.conv_1 = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding="same",
        )
        self.batch_norm_1 = nn.BatchNorm2d(num_features=64)
        self.max_pool_1 = nn.MaxPool2d(
            kernel_size=(3, 3),
            stride=(1, 1),
        )
        self.block_1 = SmallBasicBlock(
            in_channels=64,
            out_channels=128,
            stride=(1, 1),
        )
        self.max_pool_2 = nn.MaxPool2d(
            kernel_size=(3, 3),
            stride=(2, 1),
        )
        self.block_2 = SmallBasicBlock(
            in_channels=128,
            out_channels=256,
            stride=(1, 1),
        )
        self.block_3 = SmallBasicBlock(
            in_channels=256,
            out_channels=256,
            stride=(1, 1),
        )
        self.max_pool_3 = nn.MaxPool2d(
            kernel_size=(3, 3),
            stride=(2, 1),
        )
        self.dropout_1 = nn.Dropout2d(p=dropout_p)
        self.conv_2 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=(4, 1),
            stride=(1, 1),
        )  # TODO: check channels in
        self.batch_norm_2 = nn.BatchNorm2d(num_features=256)
        self.dropout_2 = nn.Dropout2d(p=dropout_p)

        self.conv_out = nn.Conv2d(
            in_channels=256,
            out_channels=num_classes,
            kernel_size=(1, 13),
            stride=(1, 1),
            padding="same",
        )
        self.batch_norm_out = nn.BatchNorm2d(num_features=num_classes)
        # --- Backbone End ---

        # Global Context
        self.using_gc = False

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

        xs = input
        xs = self.forward_stn(xs)
        xs = self.conv_1(input)
        xs = self.batch_norm_1(xs)
        xs = F.relu(xs)

        xs = self.max_pool_1(xs)
        xs = self.block_1(xs)
        xs = self.max_pool_2(xs)

        xs = self.block_2(xs)
        xs = self.block_3(xs)
        xs = self.max_pool_3(xs)
        xs = self.dropout_1(xs)

        xs = self.conv_2(xs)
        xs = self.batch_norm_2(xs)
        xs = F.relu(xs)
        xs = self.dropout_2(xs)

        xs = self.forward_gc(xs)

        xs = self.conv_out(xs)
        xs = self.batch_norm_out(xs)
        xs = F.relu(xs)
        xs = xs.squeeze(dim=2)

        return xs

    def forward_stn(self, input: torch.Tensor) -> torch.Tensor:
        if not self.using_stn or self.stn is None:
            return input

        return self.stn(input)

    # TODO: Implement global context
    def forward_gc(self, input: torch.Tensor) -> torch.Tensor:
        if not self.using_gc:
            return input

        raise NotImplementedError(
            "This method is currently not implemented. Please set `use_gc(False)` instead."
        )
        # Adjust layer
        # xs = self.gc_conv(xs)
        # xs = self.gc_batch_norm(xs)
        # xs = F.relu()

    def use_stn(self, enable: bool = True):
        self.using_stn = enable

    def use_gc(self, enable: bool = True):
        self.using_gc = enable


if __name__ == "__main__":
    from stn import LocNet, SpatialTransformerLayer

    loc = LocNet()
    stn = SpatialTransformerLayer(localization=loc, align_corners=True)

    model = LPRNet(37, 3, stn)

    input = torch.randint(0, 255, (32, 3, 24, 94), dtype=torch.float)

    output = model(input)

    print(output.shape)
