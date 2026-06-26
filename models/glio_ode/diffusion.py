import math
import torch
from torch import nn


def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Improved DDPM cosine schedule (Nichol & Dhariwal, 2021)."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(min=1e-5, max=0.999)


def _extract(buf: torch.Tensor, t: torch.Tensor, shape) -> torch.Tensor:
    out = buf.gather(0, t)
    return out.view(-1, *([1] * (len(shape) - 1)))


class GlioDiffusion(nn.Module):
    """DDPM wrapper around a GlioODE denoiser.

    Treats (image, mask) as one tensor; loss is ε-MSE on all channels.
    Conditional sampling clamps the first `image_channels` channels to the
    forward-diffused observed image at every reverse step.
    """

    def __init__(
        self,
        denoiser: nn.Module,
        image_channels: int,
        mask_channels: int,
        timesteps: int = 1000,
        image_weight: float = 1.0,
        mask_weight: float = 1.0,
    ):
        super().__init__()
        self.denoiser = denoiser
        self.image_channels = image_channels
        self.mask_channels = mask_channels
        self.total_channels = image_channels + mask_channels
        self.timesteps = timesteps
        self.image_weight = image_weight
        self.mask_weight = mask_weight

        betas = _cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0]), alphas_cumprod[:-1]], dim=0,
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt()
        )
        # Posterior variance for DDPM ancestral sampling.
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)

    def q_sample(
        self,
        z0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(z0)
        a = _extract(self.sqrt_alphas_cumprod, t, z0.shape)
        b = _extract(self.sqrt_one_minus_alphas_cumprod, t, z0.shape)
        return a * z0 + b * noise

    def forward(self, z0: torch.Tensor) -> torch.Tensor:
        B = z0.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=z0.device)
        noise = torch.randn_like(z0)
        z_t = self.q_sample(z0, t, noise=noise)
        eps_hat = self.denoiser.forward_denoise(z_t, t)
        assert eps_hat.shape == z0.shape

        img = (eps_hat[:, : self.image_channels] - noise[:, : self.image_channels]) ** 2
        msk = (eps_hat[:, self.image_channels :] - noise[:, self.image_channels :]) ** 2
        img_mean = img.mean()
        msk_mean = msk.mean()
        loss = self.image_weight * img_mean + self.mask_weight * msk_mean
        assert torch.isfinite(loss), "non-finite diffusion loss"
        # Cache per-group means as Python floats for logging consumers.
        self._last_img_mse = float(img_mean.detach().item())
        self._last_mask_mse = float(msk_mean.detach().item())
        return loss

    def _predict_x0(self, z_t: torch.Tensor, t: torch.Tensor, eps_hat: torch.Tensor) -> torch.Tensor:
        a = _extract(self.sqrt_alphas_cumprod, t, z_t.shape)
        b = _extract(self.sqrt_one_minus_alphas_cumprod, t, z_t.shape)
        return (z_t - b * eps_hat) / a

    def p_sample(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        x_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One DDPM reverse step. If x_cond is given, image channels of the
        output are overwritten by q_sample(x_cond, t-1).
        """
        eps_hat = self.denoiser.forward_denoise(z_t, t)
        x0 = self._predict_x0(z_t, t, eps_hat)

        # Posterior mean.
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

        if x_cond is not None:
            # Clamp image channels with the forward-diffused observed value at t-1.
            t_prev = (t - 1).clamp(min=0)
            with torch.no_grad():
                img_clamped = self.q_sample(x_cond, t_prev)
            z_prev = z_prev.clone()
            z_prev[:, : self.image_channels] = img_clamped
        return z_prev

    def _p_sample_ddim_step(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        x_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One deterministic DDIM reverse step (eta=0).

        Args:
            z_t: current state.
            t: current timestep (B,) long.
            t_prev: next timestep in the schedule (B,) long; may equal t at the boundary.
            x_cond: optional conditional image; if given, image channels are clamped.
        """
        eps_hat = self.denoiser.forward_denoise(z_t, t)
        x0_hat = self._predict_x0(z_t, t, eps_hat)
        ac_prev = _extract(self.alphas_cumprod, t_prev, z_t.shape)
        z_prev = ac_prev.sqrt() * x0_hat + (1.0 - ac_prev).sqrt() * eps_hat

        if x_cond is not None:
            with torch.no_grad():
                img_clamped = self.q_sample(x_cond, t_prev)
            z_prev = z_prev.clone()
            z_prev[:, : self.image_channels] = img_clamped
        return z_prev

    @torch.no_grad()
    def p_sample_loop_conditional(
        self,
        x_cond: torch.Tensor,
        num_steps: int | None = None,
        sampler: str = "ddpm",
    ) -> torch.Tensor:
        """Reverse diffusion conditioned on x_cond via image-channel clamping.

        Args:
            x_cond: (B, C_img, D, H, W) observed image.
            num_steps: optional reduced step count. Defaults: full T for DDPM,
                50 for DDIM.
            sampler: "ddpm" (ancestral) or "ddim" (deterministic).
        Returns:
            z_0 of shape (B, total_channels, D, H, W). Image channels are
            overwritten with x_cond after the final step.
        """
        if sampler not in ("ddpm", "ddim"):
            raise ValueError(f"unknown sampler: {sampler!r}")
        if num_steps is None:
            num_steps = self.timesteps if sampler == "ddpm" else 50

        B, _, D, H, W = x_cond.shape
        device = x_cond.device
        z = torch.randn(
            B, self.total_channels, D, H, W,
            device=device, dtype=x_cond.dtype,
        )
        t_init = torch.full((B,), self.timesteps - 1, dtype=torch.long, device=device)
        z[:, : self.image_channels] = self.q_sample(x_cond, t_init)

        if num_steps >= self.timesteps:
            ts = list(range(self.timesteps))[::-1]
        else:
            ts = torch.linspace(self.timesteps - 1, 0, num_steps).long().tolist()

        for idx, i in enumerate(ts):
            t = torch.full((B,), i, dtype=torch.long, device=device)
            if sampler == "ddpm":
                z = self.p_sample(z, t, x_cond=x_cond)
            else:  # ddim
                next_i = ts[idx + 1] if idx + 1 < len(ts) else 0
                t_prev = torch.full((B,), next_i, dtype=torch.long, device=device)
                z = self._p_sample_ddim_step(z, t, t_prev, x_cond=x_cond)

        # Clean overwrite of image channels — the reverse loop's last step left
        # them at a slightly-noisy q_sample(x_cond, 0); replace with x_cond itself.
        z = z.clone()
        z[:, : self.image_channels] = x_cond
        return z
