

from __future__ import annotations

import torch
import torch.nn.functional as F


class AnalyticalASDIEstimator:

    def __call__(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
        alpha_t: torch.Tensor,
    ) -> torch.Tensor:
        # The condition is already PixelUnshuffle(L_IA). It is therefore the
        # clean structural latent prior required by the DDIM inversion step.
        x0_prior = self._enhance_latent(condition)
        eps = (z_t - torch.sqrt(alpha_t) * x0_prior) / torch.sqrt(torch.clamp(1.0 - alpha_t, min=1e-8))
        return eps

    @staticmethod
    def _enhance_latent(condition: torch.Tensor) -> torch.Tensor:
        # Edge-preserving unsharp masking in latent space. This does not alter
        # the ASDI pipeline; it refines the clean-latent estimate used by the
        # deterministic noise-residual predictor.
        blur = F.avg_pool2d(condition, kernel_size=3, stride=1, padding=1)
        detail = condition - blur
        enhanced = condition + 0.25 * detail
        return torch.clamp(enhanced, -1.0, 1.0)


class DDIMSampler:
    """Flexible deterministic/stochastic DDIM sampler used by ASDI."""

    def __init__(
        self,
        total_train_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        eta: float = 0.0,
    ) -> None:
        self.total_train_steps = total_train_steps
        self.eta = eta
        betas = torch.linspace(beta_start, beta_end, total_train_steps)
        alphas = 1.0 - betas
        self.alpha_cumprod = torch.cumprod(alphas, dim=0)

    def to(self, device: torch.device | str) -> "DDIMSampler":
        self.alpha_cumprod = self.alpha_cumprod.to(device)
        return self

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        unet=None,
        num_steps: int = 20,
        deterministic_start: bool = True,
    ) -> torch.Tensor:
        device = condition.device
        self.to(device)

        if deterministic_start:
            z_t = condition.clone()
        else:
            z_t = torch.randn_like(condition)

        timesteps = torch.linspace(
            self.total_train_steps - 1,
            0,
            steps=max(num_steps, 2),
            device=device,
        ).long()

        analytical_estimator = AnalyticalASDIEstimator()

        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            t_prev = timesteps[i + 1]
            alpha_t = self.alpha_cumprod[t].view(1, 1, 1, 1)
            alpha_prev = self.alpha_cumprod[t_prev].view(1, 1, 1, 1)
            t_batch = t.repeat(condition.shape[0])

            if unet is None:
                eps_theta = analytical_estimator(z_t, t_batch, condition, alpha_t)
            else:
                eps_theta = unet(z_t, t_batch, condition)

            x0_pred = (z_t - torch.sqrt(1.0 - alpha_t) * eps_theta) / torch.sqrt(alpha_t)
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

            if self.eta > 0:
                sigma_t = self.eta * torch.sqrt(
                    torch.clamp((1.0 - alpha_prev) / (1.0 - alpha_t) * (1.0 - alpha_t / alpha_prev), min=0.0)
                )
                noise = torch.randn_like(z_t)
            else:
                sigma_t = torch.zeros_like(alpha_t)
                noise = torch.zeros_like(z_t)

            z_t = (
                torch.sqrt(alpha_prev) * x0_pred
                + torch.sqrt(torch.clamp(1.0 - alpha_prev - sigma_t ** 2, min=0.0)) * eps_theta
                + sigma_t * noise
            )

        return torch.clamp(z_t, -1.0, 1.0)
