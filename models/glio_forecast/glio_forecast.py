"""GlioForecast — Bayesian inverse + forecast model for glioma growth.

Composes an iter-3 GlioODE as the diffusion encoder + a differentiable FK
Neural-ODE decoder. Trained with DDPM eps-loss on parameter channels +
FK reconstruction loss tying decoded c to observed c.
"""
from __future__ import annotations

import torch
from torch import nn

from networks.models.glio_ode.glio_ode import GlioODE
from networks.models.glio_forecast.fk_decoder import FKDecoder


class GlioForecast(nn.Module):
    def __init__(
        self,
        crop_size=(64, 128, 128),
        anatomy_channels: int = 4,
        observed_channels: int = 1,
        param_channels: int = 3,
        patch_size=(4, 4, 4),
        d_model: int = 384,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        ode_t: float = 1.0,
        ode_steps: int = 8,
        ode_method: str = "rk4",
        decoder_channels=(384, 192, 96, 48),
        diffusion_timesteps: int = 1000,
        fk_voxel_size=(1.0, 1.0, 1.0),
        fk_use_residual: bool = True,
        fk_residual_channels: int = 16,
        fk_ode_steps: int = 8,
        fk_seed_max: float = 0.1,
        fk_d_max: float = 1.0,
        fk_rho_max: float = 1.0,
        eval_sampler: str = "ddim",
        eval_num_steps: int = 50,
        eval_overlap: float = 0.5,
        eval_sw_batch_size: int = 1,
        rope_dims=None,
        attn_backend: str = "auto",
    ):
        super().__init__()
        self.anatomy_channels = anatomy_channels
        self.observed_channels = observed_channels
        self.param_channels = param_channels
        self.conditioning_channels = anatomy_channels + observed_channels

        self.encoder = GlioODE(
            crop_size=crop_size,
            in_channels=self.conditioning_channels,
            num_classes=param_channels,
            patch_size=patch_size,
            d_model=d_model,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            ode_t=ode_t,
            ode_steps=ode_steps,
            ode_method=ode_method,
            decoder_channels=decoder_channels,
            diffusion_timesteps=diffusion_timesteps,
            eval_sampler=eval_sampler,
            eval_num_steps=eval_num_steps,
            eval_overlap=eval_overlap,
            eval_sw_batch_size=eval_sw_batch_size,
            rope_dims=rope_dims,
            attn_backend=attn_backend,
        )
        self.fk_decoder = FKDecoder(
            spatial_size=tuple(crop_size),
            voxel_size=tuple(fk_voxel_size),
            use_residual=fk_use_residual,
            residual_channels=fk_residual_channels,
            ode_steps=fk_ode_steps,
            seed_max=fk_seed_max,
        )
        self.fk_d_max = fk_d_max
        self.fk_rho_max = fk_rho_max

    def forward_denoise(self, z_t, t):
        return self.encoder.forward_denoise(z_t, t)

    def attach_diffusion(self, diffusion):
        # Bypass nn.Module submodule registration to break the cycle (mirrors GlioODE.attach_diffusion).
        object.__setattr__(self, "_diffusion", diffusion)

    def decode_fk(self, params, anatomy, s_obs):
        D_raw, rho_raw, seed_map = params.split([1, 1, 1], dim=1)
        # Bounded activations on D and rho so the FK PDE stays physical and stable
        # at random init. Encoder learns to predict logits whose sigmoid yields
        # the target values; data side stores inverse-sigmoid of true D, rho.
        D = self.fk_d_max * torch.sigmoid(D_raw)
        rho = self.fk_rho_max * torch.sigmoid(rho_raw)
        return self.fk_decoder(D, rho, seed_map, anatomy, s_obs)

    @torch.no_grad()
    def forward(self, observation, forecast_horizon=None, num_samples=1):
        assert hasattr(self, "_diffusion"), "call attach_diffusion(diff) first"
        anatomy = observation["anatomy"]
        c_obs = observation["c_obs"]
        if anatomy.dim() == 4:
            anatomy = anatomy.unsqueeze(0)
            c_obs = c_obs.unsqueeze(0)
        cond = torch.cat([anatomy, c_obs], dim=1)
        out = {"D": [], "rho": [], "seed_map": []}
        if forecast_horizon is not None:
            out["c_forecast"] = []
        for _ in range(num_samples):
            z0 = self._diffusion.p_sample_loop_conditional(
                cond,
                num_steps=self.encoder._eval_num_steps,
                sampler=self.encoder._eval_sampler,
            )
            params = z0[:, self.conditioning_channels:]
            D, rho, seed_map = params.split([1, 1, 1], dim=1)
            out["D"].append(D)
            out["rho"].append(rho)
            out["seed_map"].append(seed_map)
            if forecast_horizon is not None:
                B = anatomy.shape[0]
                s_obs = torch.full((B,), float(forecast_horizon), device=anatomy.device)
                c_forecast = self.decode_fk(params, anatomy, s_obs)
                out["c_forecast"].append(c_forecast)
        for k in list(out.keys()):
            out[k] = torch.stack(out[k], dim=0)
        return out
