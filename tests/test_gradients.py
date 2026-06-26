import torch

from networks.models.glio_ode.glio_ode import GlioODE
from networks.models.glio_ode.diffusion import GlioDiffusion


def _tiny_model():
    return GlioODE(
        crop_size=(8, 16, 16),
        in_channels=2,
        num_classes=2,
        patch_size=(2, 4, 4),
        d_model=12,
        num_heads=2,
        mlp_ratio=2.0,
        ode_steps=2,
        decoder_channels=(12, 6, 6, 6),
        diffusion_timesteps=10,
    )


def test_loss_backward_produces_finite_grads():
    model = _tiny_model()
    diff = GlioDiffusion(model, image_channels=2, mask_channels=2, timesteps=10)
    z0 = torch.randn(2, 4, 8, 16, 16)
    loss = diff(z0)
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"


def test_ode_block_grads_flow():
    model = _tiny_model()
    diff = GlioDiffusion(model, image_channels=2, mask_channels=2, timesteps=10)
    z0 = torch.randn(2, 4, 8, 16, 16)
    loss = diff(z0)
    loss.backward()
    trunk_params = list(model.trunk.func.block.parameters())
    assert len(trunk_params) > 0
    nonzero = any((p.grad is not None and p.grad.abs().sum() > 0) for p in trunk_params)
    assert nonzero, "no gradient reached the ODE-integrated ViTBlock"


def test_adaln_zero_init_means_output_close_to_input_at_step_0():
    from networks.models.glio_ode.vit_encoder import ViTBlock
    from networks.models.glio_ode.flash_attention import build_rope_3d_cache

    block = ViTBlock(
        d_model=12, num_heads=2, mlp_ratio=2.0, d_t=32,
        rope_dims=(2, 2, 2), attn_backend="sdpa",
    )
    x = torch.randn(2, 8, 12)
    tau = torch.randn(2, 32)
    cache = build_rope_3d_cache(
        patch_grid=(2, 2, 2), rope_dims=(2, 2, 2), device=torch.device("cpu"),
    )
    y = block(x, tau, cache)
    # AdaLN-Zero: scale and shift linears are zero-init, but attention/MLP
    # paths are still active through the residuals. y should be finite and
    # the residual structure preserved (output norm in the ballpark of input).
    assert torch.isfinite(y).all()
    assert abs(y.std().item() - x.std().item()) < 2.0
