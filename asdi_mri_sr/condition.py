"""Conditional input modeling for the ASDI MRI super-resolution module.

This file implements the conditional Pixel-Unshuffle operation described in the
manuscript:

    c = PixelUnshuffle(L_IA) in R^{H/2 x W/2 x 4}

For single-image inference from a low-resolution MRI image, L_IA is obtained by
bicubic interpolation to the target super-resolved size. This keeps the public
implementation executable while preserving the conditional modeling interface
reported in the paper.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ASDICondition(nn.Module):
    """Constructs the compact structural condition tensor for ASDI.

    Args:
        upscale_factor: Super-resolution scale. The manuscript uses
            Pixel-Unshuffle with factor 2 for single-channel MRI inputs.
    """

    def __init__(self, upscale_factor: int = 2) -> None:
        super().__init__()
        if upscale_factor != 2:
            raise ValueError("This ASDI reconstruction currently supports scale=2.")
        self.upscale_factor = upscale_factor
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=upscale_factor)

    def forward(self, x_lr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Build L_IA and the condition tensor.

        Args:
            x_lr: Low-resolution MRI tensor in [-1, 1], shape [B, 1, H, W].

        Returns:
            condition: Pixel-unshuffled condition, shape [B, 4, H, W].
            l_ia: Bicubic interpolated image, shape [B, 1, 2H, 2W].
        """
        if x_lr.ndim != 4 or x_lr.shape[1] != 1:
            raise ValueError("x_lr must have shape [B, 1, H, W].")

        l_ia = F.interpolate(
            x_lr,
            scale_factor=self.upscale_factor,
            mode="bicubic",
            align_corners=False,
        )
        l_ia = torch.clamp(l_ia, -1.0, 1.0)
        condition = self.pixel_unshuffle(l_ia)
        return condition, l_ia
