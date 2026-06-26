import torch

from networks.models.glio_ode.glio_ode import GlioODE


def test_denoiser_is_deterministic_in_eval():
    model = GlioODE(
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
    model.eval()
    z_t = torch.randn(2, 4, 8, 16, 16)
    t = torch.tensor([3, 7])

    out_a = model.forward_denoise(z_t, t)
    out_b = model.forward_denoise(z_t, t)
    assert torch.allclose(out_a, out_b, atol=0, rtol=0), (
        "denoiser is non-deterministic in eval mode — possible state leak in ODEFunc"
    )
