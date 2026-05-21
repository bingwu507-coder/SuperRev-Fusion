"""Lightweight conditional U-Net used for ASDI noise estimation.

The network follows the manuscript interface:

    eps_theta(z_t, t, c) = UNet_theta(z_t, Concat(z_t, c))

The class can load trained weights when available. For public executable
inference without the original weights, the ASDI sampler can instead use the
analytical structural estimator implemented in ddim_sampler.py.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal timestep embedding."""
    half = dim // 2
    device = timesteps.device
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(0, half, device=device).float() / max(half - 1, 1)
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, time_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class ConditionalUNet(nn.Module):
    """Compact U-Net for ASDI conditional noise residual prediction."""

    def __init__(self, latent_channels: int = 4, condition_channels: int = 4, base_channels: int = 64) -> None:
        super().__init__()
        self.time_dim = base_channels * 4
        in_channels = latent_channels + condition_channels

        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
        )

        self.in_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.rb1 = ResidualBlock(base_channels, self.time_dim)
        self.down = nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.rb2 = ResidualBlock(base_channels * 2, self.time_dim)
        self.mid = ResidualBlock(base_channels * 2, self.time_dim)
        self.up = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.rb3 = ResidualBlock(base_channels, self.time_dim)
        self.out = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, latent_channels, kernel_size=3, padding=1),
        )

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t.repeat(z_t.shape[0])
        if t.shape[0] != z_t.shape[0]:
            t = t[:1].repeat(z_t.shape[0])

        t_emb = timestep_embedding(t, self.in_conv.out_channels)
        t_emb = self.time_mlp(t_emb)

        x = torch.cat([z_t, condition], dim=1)
        h1 = self.rb1(self.in_conv(x), t_emb)
        h2 = self.rb2(self.down(h1), t_emb)
        h = self.mid(h2, t_emb)
        h = self.up(h)
        if h.shape[-2:] != h1.shape[-2:]:
            h = F.interpolate(h, size=h1.shape[-2:], mode="bilinear", align_corners=False)
        h = self.rb3(h + h1, t_emb)
        return self.out(h)
