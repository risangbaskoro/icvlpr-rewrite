from typing import List, Tuple, Union

import torch

from torch import nn
from torch.nn import functional as F


class LocNet(nn.Module):
    """LocNet architecture for Spatial Transformer Layer"""

    def __init__(self):
        super().__init__()

        # self.avg_pool = nn.AvgPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(24, 94))
        self.conv_l = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=(5, 5), stride=(3, 3)
        )
        self.conv_r = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=(5, 5), stride=(3, 3)
        )

        self.dropout = nn.Dropout2d()

        # FIXME: Fix the in_features. Read below.
        # For in features in fc_1, we can get it from the product of the shape of the concatenation result of the last layer.
        self.fc_1 = nn.Linear(in_features=64 * 7 * 30, out_features=32)
        self.fc_2 = nn.Linear(in_features=32, out_features=6)

        self.fc_2.weight.data.zero_().requires_grad_(False)
        self.fc_2.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)).requires_grad_(False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): Tensor of shape (N, C, H, W), where:
                - N: the number of batch
                - C: channel
                - H: height (pixel) of the image
                - W: width (pixel) of the image
        Return:
            torch.Tensor of affine matrices shape (N, 2, 3).
        """
        x_l = self.avg_pool(input)
        x_l = self.conv_l(x_l)

        x_r = self.conv_r(input)

        xs = torch.cat([x_l, x_r], dim=1)
        xs = self.dropout(xs)

        xs = xs.flatten(start_dim=1)  # Flatten for fully-connected layer
        xs = self.fc_1(xs)
        xs = torch.tanh(xs)  # activation
        xs = self.fc_2(xs)
        xs = torch.tanh(xs)  # activation
        theta = xs.view(-1, 2, 3)  # transform the shape to (N, 2, 3)
        return theta


class SpatialTransformerLayer(nn.Module):
    """Spatial Transformer Layer module

    Args:
        localization (torch.nn.Module): Module to generate localization.
        align_corners (bool):
            Whether to align_corners. See https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
              and https://pytorch.org/docs/stable/generated/torch.nn.functional.affine_grid.html
    """

    def __init__(self, localization: nn.Module, align_corners: bool = False):
        super().__init__()
        self.localization = localization
        self.align_corners = align_corners

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): Tensor of shape (N, C, H, W), where:
                - N: the number of batch
                - C: channel
                - H: height (pixel) of the image
                - W: width (pixel) of the image

        Return:
            torch.Tensor of grid sample.
        """
        theta = self.localization(input)
        grid = F.affine_grid(
            theta=theta, size=input.shape, align_corners=self.align_corners
        )
        return F.grid_sample(input, grid=grid, align_corners=self.align_corners)
