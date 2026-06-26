import math
import torch
from torch import nn

from networks.models.glio_ode.utilities.timestep_embedding import (
    sinusoidal_timestep_embedding,
)


class PatchEmbed3D(nn.Module):
    """3D patch embedding via strided Conv3d."""

    def __init__(self, in_channels: int, d_model: int, patch_size):
        super().__init__()
        self.patch_size = tuple(patch_size)
        self.proj = nn.Conv3d(
            in_channels, d_model,
            kernel_size=self.patch_size, stride=self.patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        for dim, p in zip((D, H, W), self.patch_size):
            assert dim % p == 0, f"spatial dim {dim} not divisible by patch {p}"
        x = self.proj(x)  # (B, d_model, D/p_d, H/p_h, W/p_w)
        x = x.flatten(2).transpose(1, 2).contiguous()  # (B, N, d_model)
        return x


class TimestepEmbedding(nn.Module):
    """Sinusoidal embedding followed by a 2-layer MLP."""

    def __init__(self, d_model: int, d_t: int):
        super().__init__()
        self.d_model = d_model
        self.d_t = d_t
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_t),
            nn.SiLU(),
            nn.Linear(d_t, d_t),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        sinus = sinusoidal_timestep_embedding(t, self.d_model)
        return self.mlp(sinus)


class AdaLN(nn.Module):
    """AdaLN-Zero with RMSNorm normalization.

    Iter-6 swap: nn.LayerNorm(elementwise_affine=False) -> F.rms_norm.
    Consistent with iter-3's QK Norm choice; the scale and shift learned from
    tau absorb any normalization-shape difference.
    """

    def __init__(self, d_model: int, d_t: int):
        super().__init__()
        self.d_model = d_model
        self.to_scale_shift = nn.Linear(d_t, 2 * d_model)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        scale_shift = self.to_scale_shift(tau)
        scale, shift = scale_shift.chunk(2, dim=-1)
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)
        normed = torch.nn.functional.rms_norm(x, (self.d_model,))
        return (1.0 + scale) * normed + shift


class _MLP(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: float):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.fc1 = nn.Linear(d_model, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class ViTBlock(nn.Module):
    """Pre-norm ViT block with AdaLN-Zero conditioning and 3D-RoPE attention."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_ratio: float,
        d_t: int,
        rope_dims: tuple[int, int, int],
        attn_backend: str = "auto",
    ):
        super().__init__()
        from networks.models.glio_ode.flash_attention import MultiHeadSelfAttention3D
        self.adaln1 = AdaLN(d_model, d_t)
        self.attn = MultiHeadSelfAttention3D(
            d_model=d_model, num_heads=num_heads,
            rope_dims=rope_dims, attn_backend=attn_backend,
        )
        self.adaln2 = AdaLN(d_model, d_t)
        self.mlp = _MLP(d_model, mlp_ratio)

    def forward(self, x, tau, rope_cache):
        h = self.adaln1(x, tau)
        h = self.attn(h, rope_cache)
        x = x + h
        h = self.adaln2(x, tau)
        x = x + self.mlp(h)
        return x
