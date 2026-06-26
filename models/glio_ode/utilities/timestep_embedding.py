import math
import torch


def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard DDPM sinusoidal embedding.

    Args:
        t: 1D long/float tensor of shape (B,).
        dim: embedding dimension (must be even).
    Returns:
        (B, dim) float tensor.
    """
    assert dim % 2 == 0, "sinusoidal dim must be even"
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float()[:, None] * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
