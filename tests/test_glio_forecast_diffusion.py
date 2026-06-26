import pytest
import torch

from networks.models.glio_forecast.glio_forecast import GlioForecast
from networks.models.glio_forecast.diffusion import GlioForecastDiffusion


@pytest.fixture
def tiny():
    m = GlioForecast(
        crop_size=(8, 16, 16),
        anatomy_channels=4, observed_channels=1, param_channels=3,
        patch_size=(2, 4, 4),
        d_model=12, num_heads=2, mlp_ratio=2.0,
        ode_steps=2, decoder_channels=(12, 6, 6, 6),
        diffusion_timesteps=10,
        fk_ode_steps=2,
        eval_sampler="ddim", eval_num_steps=2,
    )
    diff = GlioForecastDiffusion(
        m, conditioning_channels=5, param_channels=3, timesteps=10,
    )
    m.attach_diffusion(diff)
    return m, diff


def test_diffusion_loss_is_finite(tiny):
    _, diff = tiny
    z0 = torch.randn(2, 8, 8, 16, 16)
    anatomy = torch.zeros(2, 4, 8, 16, 16)
    c_obs = torch.zeros(2, 1, 8, 16, 16)
    s_obs = torch.tensor([1.0, 1.0])
    loss = diff(z0, anatomy, c_obs, s_obs)
    assert torch.isfinite(loss)


def test_loss_split_l_diff_and_l_recon_cached(tiny):
    _, diff = tiny
    z0 = torch.randn(2, 8, 8, 16, 16)
    anatomy = torch.zeros(2, 4, 8, 16, 16)
    c_obs = torch.zeros(2, 1, 8, 16, 16)
    s_obs = torch.tensor([1.0, 1.0])
    diff(z0, anatomy, c_obs, s_obs)
    assert hasattr(diff, "_last_l_diff")
    assert hasattr(diff, "_last_l_recon")
    assert diff._last_l_diff > 0.0
    assert isinstance(diff._last_l_recon, float)


def test_recon_warmup_zeros_l_recon(tiny):
    """With warmup_iters=100 and current_iter=10, recon contributes 0."""
    m, _ = tiny
    diff = GlioForecastDiffusion(
        m, conditioning_channels=5, param_channels=3, timesteps=10,
        diff_weight=1.0, recon_weight=10.0, recon_warmup_iters=100,
    )
    m.attach_diffusion(diff)
    diff.set_current_iter(10)
    z0 = torch.randn(2, 8, 8, 16, 16)
    anatomy = torch.zeros(2, 4, 8, 16, 16)
    c_obs = torch.zeros(2, 1, 8, 16, 16)
    s_obs = torch.tensor([1.0, 1.0])
    loss_warmup = float(diff(z0, anatomy, c_obs, s_obs).detach())
    diff.set_current_iter(200)
    loss_full = float(diff(z0, anatomy, c_obs, s_obs).detach())
    assert diff._last_l_recon > 0.0


def test_p_sample_loop_returns_param_shape(tiny):
    m, diff = tiny
    obs = torch.randn(1, 5, 8, 16, 16)
    z0 = diff.p_sample_loop_conditional(obs, num_steps=2, sampler="ddim")
    assert z0.shape == (1, 8, 8, 16, 16)
    assert torch.allclose(z0[:, :5], obs)


def test_p_sample_rejects_unknown_sampler(tiny):
    _, diff = tiny
    obs = torch.randn(1, 5, 8, 16, 16)
    with pytest.raises(ValueError, match="unknown sampler"):
        diff.p_sample_loop_conditional(obs, num_steps=2, sampler="foo")
