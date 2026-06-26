"""GlioForecastDiffusion — DDPM wrapper for GlioForecast.

Channel split: (conditioning=5, params=3). Loss = DDPM eps-MSE on param
channels + FK reconstruction loss. Conditional sampler clamps the 5
conditioning channels at every reverse step (same pattern as iter-2's
GlioDiffusion image clamp, just with 5 channels instead of 4).
"""
from __future__ import annotations

import math
import torch
from torch import nn


def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(min=1e-5, max=0.999)


def _extract(buf, t, shape):
    out = buf.gather(0, t)
    return out.view(-1, *([1] * (len(shape) - 1)))


class GlioForecastDiffusion(nn.Module):
    def __init__(
        self,
        forecast_model,
        conditioning_channels: int = 5,
        param_channels: int = 3,
        timesteps: int = 1000,
        diff_weight: float = 1.0,
        recon_weight: float = 1.0,
        recon_warmup_iters: int = 0,
    ):
        super().__init__()
        self.forecast_model = forecast_model
        self.conditioning_channels = conditioning_channels
        self.param_channels = param_channels
        self.total_channels = conditioning_channels + param_channels
        self.timesteps = timesteps
        self.diff_weight = diff_weight
        self.recon_weight = recon_weight
        self.recon_warmup_iters = recon_warmup_iters
        self._current_iter = 0

        betas = _cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]], dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt())
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)

    def set_current_iter(self, iter_idx: int) -> None:
        self._current_iter = int(iter_idx)

    def q_sample(self, z0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(z0)
        a = _extract(self.sqrt_alphas_cumprod, t, z0.shape)
        b = _extract(self.sqrt_one_minus_alphas_cumprod, t, z0.shape)
        return a * z0 + b * noise

    def _predict_x0(self, z_t, t, eps_hat):
        a = _extract(self.sqrt_alphas_cumprod, t, z_t.shape)
        b = _extract(self.sqrt_one_minus_alphas_cumprod, t, z_t.shape)
        return (z_t - b * eps_hat) / a

    def forward(self, z0, anatomy, c_obs, s_obs):
        B = z0.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=z0.device)
        noise = torch.randn_like(z0)
        z_t = self.q_sample(z0, t, noise=noise)
        eps_hat = self.forecast_model.forward_denoise(z_t, t)
        assert eps_hat.shape == z0.shape

        # eps-loss on PARAM channels only.
        eps_p = eps_hat[:, self.conditioning_channels:]
        noise_p = noise[:, self.conditioning_channels:]
        L_diff = (eps_p - noise_p).pow(2).mean()

        # FK reconstruction.
        x0_hat = self._predict_x0(z_t, t, eps_hat)
        params_hat = x0_hat[:, self.conditioning_channels:]
        c_decoded = self.forecast_model.decode_fk(params_hat, anatomy, s_obs)
        assert c_decoded.shape == c_obs.shape
        L_recon = (c_decoded - c_obs).pow(2).mean()

        effective_recon = (
            self.recon_weight if self._current_iter >= self.recon_warmup_iters else 0.0
        )
        loss = self.diff_weight * L_diff + effective_recon * L_recon

        self._last_l_diff = float(L_diff.detach().item())
        self._last_l_recon = float(L_recon.detach().item())
        assert torch.isfinite(loss), (
            f"non-finite loss: L_diff={self._last_l_diff}, "
            f"L_recon={self._last_l_recon}"
        )
        return loss

    def _p_sample_ddpm(self, z_t, t, observation):
        eps_hat = self.forecast_model.forward_denoise(z_t, t)
        x0 = self._predict_x0(z_t, t, eps_hat)
        ac_prev = _extract(self.alphas_cumprod_prev, t, z_t.shape)
        ac = _extract(self.alphas_cumprod, t, z_t.shape)
        beta = _extract(self.betas, t, z_t.shape)
        mean = (
            (ac_prev.sqrt() * beta / (1.0 - ac)) * x0
            + ((1.0 - ac_prev) * (1.0 - beta).sqrt() / (1.0 - ac)) * z_t
        )
        var = _extract(self.posterior_variance, t, z_t.shape)
        noise = torch.randn_like(z_t)
        nonzero_mask = (t > 0).float().view(-1, *([1] * (z_t.dim() - 1)))
        z_prev = mean + nonzero_mask * var.sqrt() * noise
        t_prev = (t - 1).clamp(min=0)
        with torch.no_grad():
            cond_clamped = self.q_sample(observation, t_prev)
        z_prev = z_prev.clone()
        z_prev[:, : self.conditioning_channels] = cond_clamped
        return z_prev

    def _p_sample_ddim(self, z_t, t, t_prev, observation):
        eps_hat = self.forecast_model.forward_denoise(z_t, t)
        x0 = self._predict_x0(z_t, t, eps_hat)
        ac_prev = _extract(self.alphas_cumprod, t_prev, z_t.shape)
        z_prev = ac_prev.sqrt() * x0 + (1.0 - ac_prev).sqrt() * eps_hat
        with torch.no_grad():
            cond_clamped = self.q_sample(observation, t_prev)
        z_prev = z_prev.clone()
        z_prev[:, : self.conditioning_channels] = cond_clamped
        return z_prev

    @torch.no_grad()
    def p_sample_loop_conditional(self, observation, num_steps=None, sampler="ddpm"):
        if sampler not in ("ddpm", "ddim"):
            raise ValueError(f"unknown sampler: {sampler!r}")
        if num_steps is None:
            num_steps = self.timesteps if sampler == "ddpm" else 50
        B, _, D, H, W = observation.shape
        device = observation.device
        z = torch.randn(
            B, self.total_channels, D, H, W,
            device=device, dtype=observation.dtype,
        )
        t_init = torch.full((B,), self.timesteps - 1, dtype=torch.long, device=device)
        z[:, : self.conditioning_channels] = self.q_sample(observation, t_init)

        if num_steps >= self.timesteps:
            ts = list(range(self.timesteps))[::-1]
        else:
            ts = torch.linspace(self.timesteps - 1, 0, num_steps).long().tolist()

        for idx, i in enumerate(ts):
            t = torch.full((B,), i, dtype=torch.long, device=device)
            if sampler == "ddpm":
                z = self._p_sample_ddpm(z, t, observation)
            else:
                next_i = ts[idx + 1] if idx + 1 < len(ts) else 0
                t_prev = torch.full((B,), next_i, dtype=torch.long, device=device)
                z = self._p_sample_ddim(z, t, t_prev, observation)

        z = z.clone()
        z[:, : self.conditioning_channels] = observation
        return z
