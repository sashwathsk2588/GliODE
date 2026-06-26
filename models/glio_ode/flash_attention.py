"""3D self-attention with axial RoPE, QK Norm (RMSNorm), and a pluggable backend.

Replaces nn.MultiheadAttention in ViTBlock. The module is bidirectional (no
causal mask), drops the learned positional embedding in favor of RoPE, and
chooses between flash_attn and torch's SDPA at construction time.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


def default_rope_dims(head_dim: int) -> tuple[int, int, int]:
    """Split head_dim into 3 even chunks (rope_d, rope_h, rope_w).

    Strategy:
      1. Anisotropic: depth gets ~head_dim/4 (rounded down to even),
         height and width split the remainder evenly. Used when the
         remainder splits cleanly.
      2. Equal: head_dim/3 if divisible and even.
      3. Brute force: smallest (d, h, w) with d <= h <= w that sum to
         head_dim with all even and each >= 2.

    Raises ValueError if head_dim < 6 or no valid split exists.
    """
    if head_dim < 6:
        raise ValueError(
            f"head_dim={head_dim} too small for 3D RoPE (need >= 6 for 3 even chunks >= 2)"
        )

    # 1. Anisotropic.
    rope_d = (head_dim // 4) & ~1  # round down to even
    if rope_d >= 2:
        remainder = head_dim - rope_d
        if remainder % 2 == 0:
            half = remainder // 2
            if half % 2 == 0 and half >= 2:
                return (rope_d, half, half)

    # 2. Equal.
    if head_dim % 3 == 0:
        third = head_dim // 3
        if third % 2 == 0 and third >= 2:
            return (third, third, third)

    # 3. Brute force.
    for d in range(2, head_dim - 3, 2):
        for h in range(d, head_dim - d - 1, 2):
            w = head_dim - d - h
            if w >= h and w % 2 == 0:
                return (d, h, w)

    raise ValueError(f"cannot split head_dim={head_dim} into 3 even chunks each >= 2")


@dataclass
class Rope3DCache:
    """Precomputed cos/sin tables and per-token coordinates for 3D axial RoPE.

    Tables are half-dim (R_a / 2 entries each): the rotation x_rot = [x1*cos -
    x2*sin, x1*sin + x2*cos] with x = (x1, x2) split-half along the dim.
    """
    cos_d: torch.Tensor   # (D_grid, rope_d // 2)
    sin_d: torch.Tensor
    cos_h: torch.Tensor   # (H_grid, rope_h // 2)
    sin_h: torch.Tensor
    cos_w: torch.Tensor   # (W_grid, rope_w // 2)
    sin_w: torch.Tensor
    coord_d: torch.Tensor # (N,) long; N = D_grid * H_grid * W_grid
    coord_h: torch.Tensor
    coord_w: torch.Tensor

    def to(self, device: torch.device) -> "Rope3DCache":
        return Rope3DCache(
            cos_d=self.cos_d.to(device), sin_d=self.sin_d.to(device),
            cos_h=self.cos_h.to(device), sin_h=self.sin_h.to(device),
            cos_w=self.cos_w.to(device), sin_w=self.sin_w.to(device),
            coord_d=self.coord_d.to(device), coord_h=self.coord_h.to(device),
            coord_w=self.coord_w.to(device),
        )


def _axis_cos_sin(grid_size: int, rope_dim: int, base: float, device) -> tuple[torch.Tensor, torch.Tensor]:
    """For one axis: returns (cos, sin) of shape (grid_size, rope_dim // 2)."""
    if rope_dim % 2 != 0:
        raise ValueError(f"rope_dim={rope_dim} must be even")
    half = rope_dim // 2
    freqs = torch.exp(
        -math.log(base) * torch.arange(0, half, dtype=torch.float32, device=device) / half
    )  # (half,)
    pos = torch.arange(grid_size, dtype=torch.float32, device=device)  # (grid_size,)
    theta = pos[:, None] * freqs[None, :]  # (grid_size, half)
    return torch.cos(theta), torch.sin(theta)


def build_rope_3d_cache(
    patch_grid: tuple[int, int, int],
    rope_dims: tuple[int, int, int],
    base: float = 10000.0,
    device: torch.device = torch.device("cpu"),
) -> Rope3DCache:
    """Build cos/sin tables and per-token (d, h, w) indices for 3D axial RoPE.

    Token order follows PatchEmbed3D's `flatten(2).transpose(1, 2)` which is
    D-major: token n = d * (H*W) + h * W + w.
    """
    D_grid, H_grid, W_grid = patch_grid
    rope_d, rope_h, rope_w = rope_dims
    if any(r % 2 != 0 for r in rope_dims):
        raise ValueError(f"rope_dims={rope_dims} must all be even")

    cos_d, sin_d = _axis_cos_sin(D_grid, rope_d, base, device)
    cos_h, sin_h = _axis_cos_sin(H_grid, rope_h, base, device)
    cos_w, sin_w = _axis_cos_sin(W_grid, rope_w, base, device)

    # Per-token coordinates in D-major order.
    N = D_grid * H_grid * W_grid
    n = torch.arange(N, device=device)
    coord_d = n // (H_grid * W_grid)
    coord_h = (n // W_grid) % H_grid
    coord_w = n % W_grid

    return Rope3DCache(
        cos_d=cos_d, sin_d=sin_d,
        cos_h=cos_h, sin_h=sin_h,
        cos_w=cos_w, sin_w=sin_w,
        coord_d=coord_d.long(),
        coord_h=coord_h.long(),
        coord_w=coord_w.long(),
    )


def _rotate_one_axis(
    x: torch.Tensor,           # (B, N, num_heads, rope_a)
    cos_table: torch.Tensor,   # (G_a, rope_a // 2)
    sin_table: torch.Tensor,   # (G_a, rope_a // 2)
    coord: torch.Tensor,       # (N,) long
) -> torch.Tensor:
    """Apply half-split rotary rotation along one axis."""
    cos = cos_table[coord]  # (N, rope_a // 2)
    sin = sin_table[coord]
    cos = cos[None, :, None, :]  # (1, N, 1, rope_a // 2)
    sin = sin[None, :, None, :]

    x1, x2 = x.chunk(2, dim=-1)  # each (B, N, num_heads, rope_a // 2)
    x_rot_lo = x1 * cos - x2 * sin
    x_rot_hi = x1 * sin + x2 * cos
    return torch.cat([x_rot_lo, x_rot_hi], dim=-1)


def apply_rope_3d(
    q: torch.Tensor,                # (B, N, num_heads, head_dim)
    k: torch.Tensor,
    rope_cache: Rope3DCache,
    rope_dims: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply 3D axial RoPE to q and k.

    Splits the last dim into 3 chunks of sizes rope_dims, rotates each chunk
    by the corresponding axis's cos/sin table indexed by per-token coords,
    concatenates the rotated chunks back.
    """
    rope_d, rope_h, rope_w = rope_dims
    assert q.shape[-1] == rope_d + rope_h + rope_w, (
        f"q head_dim {q.shape[-1]} != sum(rope_dims) {rope_d + rope_h + rope_w}"
    )

    def _rotate(x: torch.Tensor) -> torch.Tensor:
        x_d, x_h, x_w = torch.split(x, [rope_d, rope_h, rope_w], dim=-1)
        x_d = _rotate_one_axis(x_d, rope_cache.cos_d, rope_cache.sin_d, rope_cache.coord_d)
        x_h = _rotate_one_axis(x_h, rope_cache.cos_h, rope_cache.sin_h, rope_cache.coord_h)
        x_w = _rotate_one_axis(x_w, rope_cache.cos_w, rope_cache.sin_w, rope_cache.coord_w)
        return torch.cat([x_d, x_h, x_w], dim=-1)

    return _rotate(q), _rotate(k)


def _resolve_backend(name: str) -> str:
    """Return 'flash' or 'sdpa'.

    'sdpa' always works (CPU + GPU). 'flash' requires the flash_attn package
    AND CUDA availability; raises ImportError otherwise. 'auto' tries flash
    first, falls back to sdpa.
    """
    if name == "sdpa":
        return "sdpa"
    if name == "flash":
        try:
            import flash_attn  # noqa: F401
        except (ImportError, ModuleNotFoundError) as e:
            raise ImportError(
                "attn_backend='flash' requires the flash_attn package; "
                "install it or use 'sdpa'/'auto'"
            ) from e
        return "flash"
    if name == "auto":
        if not torch.cuda.is_available():
            return "sdpa"
        try:
            import flash_attn  # noqa: F401
            return "flash"
        except (ImportError, ModuleNotFoundError):
            return "sdpa"
    raise ValueError(f"unknown attn_backend: {name!r}")


class MultiHeadSelfAttention3D(nn.Module):
    """Bidirectional 3D self-attention with axial RoPE and QK RMS Norm.

    Layout: tokens in (B, N, d_model). Internally projects to (B, N, num_heads,
    head_dim) — FlashAttention's native layout, also accepted by SDPA after a
    transpose. No causal mask, no KV cache (the denoiser sees every token in
    one shot).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope_dims: tuple[int, int, int],
        attn_backend: str = "auto",
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by num_heads {num_heads}")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        if sum(rope_dims) != self.head_dim:
            raise ValueError(
                f"sum(rope_dims)={sum(rope_dims)} != head_dim={self.head_dim}"
            )
        self.rope_dims = tuple(rope_dims)

        self.c_q = nn.Linear(d_model, d_model, bias=False)
        self.c_k = nn.Linear(d_model, d_model, bias=False)
        self.c_v = nn.Linear(d_model, d_model, bias=False)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)

        self._backend = _resolve_backend(attn_backend)

    def forward(self, x: torch.Tensor, rope_cache: Rope3DCache) -> torch.Tensor:
        assert x.dim() == 3, f"expected (B, N, d_model), got shape {tuple(x.shape)}"
        B, N, D = x.shape
        assert D == self.d_model

        q = self.c_q(x).view(B, N, self.num_heads, self.head_dim)
        k = self.c_k(x).view(B, N, self.num_heads, self.head_dim)
        v = self.c_v(x).view(B, N, self.num_heads, self.head_dim)

        q, k = apply_rope_3d(q, k, rope_cache, self.rope_dims)

        # QK Norm via RMSNorm. No learned scale — the natural RMS magnitude
        # interacts cleanly with the SDPA / FlashAttention default 1/sqrt(d) scale.
        q = F.rms_norm(q, (self.head_dim,))
        k = F.rms_norm(k, (self.head_dim,))

        if self._backend == "flash":
            import flash_attn
            y = flash_attn.flash_attn_func(q, k, v, causal=False)
        else:  # sdpa
            # SDPA expects (B, num_heads, N, head_dim).
            q_t = q.transpose(1, 2)
            k_t = k.transpose(1, 2)
            v_t = v.transpose(1, 2)
            y_t = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=False)
            y = y_t.transpose(1, 2)

        y = y.contiguous().view(B, N, D)
        return self.c_proj(y)
