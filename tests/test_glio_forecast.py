import pytest
import torch

from networks.models.glio_forecast.glio_forecast import GlioForecast
from networks.models.glio_ode.glio_ode import GlioODE


def _tiny():
    return GlioForecast(
        crop_size=(8, 16, 16),
        anatomy_channels=4, observed_channels=1, param_channels=3,
        patch_size=(2, 4, 4),
        d_model=12, num_heads=2, mlp_ratio=2.0,
        ode_steps=2, decoder_channels=(12, 6, 6, 6),
        diffusion_timesteps=10,
        fk_ode_steps=2,
        eval_sampler="ddim", eval_num_steps=2,
    )


def test_glio_forecast_constructs_with_glio_ode_encoder():
    m = _tiny()
    assert isinstance(m.encoder, GlioODE)
    assert m.encoder.in_channels == 5
    assert m.encoder.num_classes == 3
    assert m.conditioning_channels == 5


def test_glio_forecast_denoise_shape():
    m = _tiny()
    z_t = torch.randn(2, 8, 8, 16, 16)
    t = torch.tensor([0, 5])
    eps_hat = m.forward_denoise(z_t, t)
    assert eps_hat.shape == z_t.shape


def test_glio_forecast_decode_fk_shape():
    m = _tiny()
    params = torch.zeros(2, 3, 8, 16, 16)
    params[:, 0] = 0.1   # D
    params[:, 1] = 0.1   # rho
    params[:, 2] = -2.0  # seed_map logits
    anatomy = torch.zeros(2, 4, 8, 16, 16)
    s_obs = torch.tensor([10.0, 10.0])
    c = m.decode_fk(params, anatomy, s_obs)
    assert c.shape == (2, 1, 8, 16, 16)


def test_glio_forecast_forward_requires_attached_diffusion():
    m = _tiny()
    with pytest.raises(AssertionError, match="attach_diffusion"):
        m({"anatomy": torch.zeros(1, 4, 8, 16, 16),
           "c_obs": torch.zeros(1, 1, 8, 16, 16)})
