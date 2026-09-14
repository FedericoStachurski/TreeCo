from __future__ import annotations

import torch
import torch.nn as nn


class TreeCoEncoder(nn.Module):
    """
    Small convolutional encoder used to encode individual TreeCo
    image modalities before multimodal fusion.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
        Examples:
            RGB   -> 3
            Depth -> 1
            SAM   -> 1

    hidden_channels : int
        Number of channels after the first convolution.

    out_channels : int
        Number of channels produced by the encoder.

    first_kernel_size : int
        Kernel size of the first convolution.

    Notes
    -----
    Two stride-2 convolutions reduce the spatial resolution by
    approximately a factor of four:

        224 x 224 -> 112 x 112 -> 56 x 56

    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        first_kernel_size: int = 5,
    ):
        super().__init__()

        if in_channels < 1:
            raise ValueError("in_channels must be >= 1")

        if hidden_channels < 1:
            raise ValueError("hidden_channels must be >= 1")

        if out_channels < 1:
            raise ValueError("out_channels must be >= 1")

        first_padding = first_kernel_size // 2

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.encoder = nn.Sequential(

            # ------------------------------------------------
            # First convolution
            # 224 x 224 -> 112 x 112
            # ------------------------------------------------

            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=first_kernel_size,
                stride=2,
                padding=first_padding,
                bias=False,
            ),

            nn.BatchNorm2d(hidden_channels),

            nn.ReLU(inplace=True),


            # ------------------------------------------------
            # Second convolution
            # 112 x 112 -> 56 x 56
            # ------------------------------------------------

            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),
        )


    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if x.ndim != 4:
            raise ValueError(
                "Expected input with shape "
                "[batch, channels, height, width], "
                f"but received {tuple(x.shape)}"
            )

        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Encoder expects {self.in_channels} input channels, "
                f"but received {x.shape[1]}"
            )

        return self.encoder(x)