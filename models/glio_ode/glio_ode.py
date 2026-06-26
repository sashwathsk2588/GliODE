import torch
from torch import nn

from networks.models.glio_ode.vit_encoder import PatchEmbed3D, TimestepEmbedding
from networks.models.glio_ode.ode_block import ODEViTTrunk
from networks.models.glio_ode.unetr_decoder import UnetrDecoder
from networks.models.glio_ode.flash_attention import (
    Rope3DCache,
    build_rope_3d_cache,
    default_rope_dims,
)


class GlioODE(nn.Module):
    """Joint-diffusion denoiser: ViT trunk integrated by Neural ODE + UNETR decoder.

    Forward signature: forward(z_t, t) -> eps_hat where z_t is the noised
    (image, mask) concatenation and t is the diffusion timestep.
    """

    def __init__(
        self,
        crop_size=(64, 128, 128),
        in_channels: int = 4,        # MRI modalities
        num_classes: int = 4,        # one-hot seg channels
        patch_size=(4, 4, 4),
        d_model: int = 384,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        ode_t: float = 1.0,
        ode_steps: int = 8,
        ode_method: str = "rk4",
        decoder_channels=(384, 192, 96, 48),
        diffusion_timesteps: int = 1000,
        deep_supervision: bool = False,
        eval_sampler: str = "ddim",
        eval_num_steps: int = 50,
        eval_overlap: float = 0.5,
        eval_sw_batch_size: int = 1,
        rope_dims=None,
        attn_backend: str = "auto",
    ):
        super().__init__()
        for i, (dim, p) in enumerate(zip(crop_size, patch_size)):
            assert dim % p == 0, f"crop_size[{i}]={dim} not divisible by patch_size[{i}]={p}"
        assert len(decoder_channels) == 4

        self.crop_size = tuple(crop_size)
        self.patch_size = tuple(patch_size)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.total_channels = in_channels + num_classes
        self.diffusion_timesteps = diffusion_timesteps

        d_t = 4 * d_model
        self.patch_embed = PatchEmbed3D(self.total_channels, d_model, patch_size)
        self.t_embed = TimestepEmbedding(d_model, d_t)

        head_dim = d_model // num_heads
        if rope_dims is None:
            rope_dims = default_rope_dims(head_dim)
        rope_dims = tuple(rope_dims)
        self.rope_dims = rope_dims
        self.attn_backend = attn_backend

        self.trunk = ODEViTTrunk(
            d_model=d_model, num_heads=num_heads, mlp_ratio=mlp_ratio,
            d_t=d_t, rope_dims=rope_dims, attn_backend=attn_backend,
            t_ode=ode_t, n_steps=ode_steps, method=ode_method,
        )
        self.decoder = UnetrDecoder(
            d_model=d_model,
            decoder_channels=decoder_channels,
            out_channels=self.total_channels,
            patch_size=patch_size,
        )

        self.patch_grid = tuple(d // p for d, p in zip(crop_size, patch_size))
        # Build RoPE3D cache once and register as buffers so it moves with .to(device).
        _cache = build_rope_3d_cache(
            patch_grid=self.patch_grid, rope_dims=rope_dims, device=torch.device("cpu"),
        )
        self.register_buffer("rope_cos_d", _cache.cos_d)
        self.register_buffer("rope_sin_d", _cache.sin_d)
        self.register_buffer("rope_cos_h", _cache.cos_h)
        self.register_buffer("rope_sin_h", _cache.sin_h)
        self.register_buffer("rope_cos_w", _cache.cos_w)
        self.register_buffer("rope_sin_w", _cache.sin_w)
        self.register_buffer("rope_coord_d", _cache.coord_d)
        self.register_buffer("rope_coord_h", _cache.coord_h)
        self.register_buffer("rope_coord_w", _cache.coord_w)
        self._eval_sampler = eval_sampler
        self._eval_num_steps = eval_num_steps
        self._eval_overlap = eval_overlap
        self._eval_sw_batch_size = eval_sw_batch_size

    def forward_denoise(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """The DDPM denoiser. Used by GlioDiffusion during training.

        Note: NOT decorated with @torch.no_grad() — gradients must flow.
        """
        assert z_t.dim() == 5, f"expected 5D input, got shape {tuple(z_t.shape)}"
        assert z_t.shape[1] == self.total_channels, (
            f"expected {self.total_channels} channels, got {z_t.shape[1]}"
        )
        tokens = self.patch_embed(z_t)
        tau = self.t_embed(t)
        rope_cache = Rope3DCache(
            cos_d=self.rope_cos_d, sin_d=self.rope_sin_d,
            cos_h=self.rope_cos_h, sin_h=self.rope_sin_h,
            cos_w=self.rope_cos_w, sin_w=self.rope_sin_w,
            coord_d=self.rope_coord_d, coord_h=self.rope_coord_h, coord_w=self.rope_coord_w,
        )
        states = self.trunk(tokens, tau, rope_cache)
        eps_hat = self.decoder(states, patch_grid=self.patch_grid)
        return eps_hat

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Conditional 3D segmentation via reverse diffusion + sliding-window.

        Args:
            x: (B, C_img, D, H, W) or (C_img, D, H, W). May be larger than
               crop_size; sliding-window inference handles it.
        Returns:
            (B, D, H, W) long tensor of class indices.
        """
        from monai.inferers import sliding_window_inference

        assert hasattr(self, "_diffusion"), "call attach_diffusion(diff) first"
        if x.dim() == 4:
            x = x.unsqueeze(0)
        assert x.dim() == 5 and x.shape[1] == self.in_channels, (
            f"expected (B, {self.in_channels}, D, H, W), got {tuple(x.shape)}"
        )

        def _predictor(crop: torch.Tensor) -> torch.Tensor:
            z0 = self._diffusion.p_sample_loop_conditional(
                crop,
                num_steps=self._eval_num_steps,
                sampler=self._eval_sampler,
            )
            return z0[:, self.in_channels :]

        logits = sliding_window_inference(
            inputs=x,
            roi_size=self.crop_size,
            sw_batch_size=self._eval_sw_batch_size,
            predictor=_predictor,
            overlap=self._eval_overlap,
            mode="gaussian",
        )
        return logits.argmax(dim=1)

    def attach_diffusion(self, diffusion) -> None:
        """Late-bind the GlioDiffusion wrapper for inference convenience.

        Stored via ``object.__setattr__`` so the diffusion wrapper is NOT
        registered as a submodule — otherwise the model<->diffusion cycle
        (diffusion holds this model as its denoiser) makes ``.train()`` /
        ``.eval()`` recurse forever.
        """
        object.__setattr__(self, "_diffusion", diffusion)
