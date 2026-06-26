import pytest
import torch

from networks.models.glio_forecast.fk_decoder import FKDecoder


def test_fk_no_tumor_stays_zero():
    decoder = FKDecoder(spatial_size=(8, 16, 16), use_residual=False)
    D = torch.full((1, 1, 8, 16, 16), 0.1)
    rho = torch.full((1, 1, 8, 16, 16), 0.1)
    seed_map = torch.full((1, 1, 8, 16, 16), -50.0)  # sigmoid(-50) ~ 2e-22, truly zero in fp32
    anatomy = torch.zeros(1, 4, 8, 16, 16)
    s_obs = torch.tensor([100.0])
    c_out = decoder(D, rho, seed_map, anatomy, s_obs)
    assert c_out.abs().max() < 1e-4


def test_fk_csf_stays_zero():
    decoder = FKDecoder(spatial_size=(8, 16, 16), use_residual=False)
    D = torch.full((1, 1, 8, 16, 16), 0.1)
    rho = torch.full((1, 1, 8, 16, 16), 0.1)
    seed_map = torch.full((1, 1, 8, 16, 16), 5.0)  # sigmoid(5) ~ 1
    anatomy = torch.zeros(1, 4, 8, 16, 16)
    anatomy[:, 3] = 1.0  # all CSF
    s_obs = torch.tensor([10.0])
    c_out = decoder(D, rho, seed_map, anatomy, s_obs)
    assert c_out.max() < 1e-6


def test_fk_pure_diffusion_conserves_mass():
    decoder = FKDecoder(spatial_size=(16, 16, 16), use_residual=False, ode_steps=4)
    D = torch.full((1, 1, 16, 16, 16), 0.05)
    rho = torch.zeros(1, 1, 16, 16, 16)  # no reaction
    seed_map = torch.full((1, 1, 16, 16, 16), -10.0)
    seed_map[0, 0, 8, 8, 8] = 5.0  # local seed at center
    anatomy = torch.zeros(1, 4, 16, 16, 16)
    s_obs = torch.tensor([5.0])
    c_out = decoder(D, rho, seed_map, anatomy, s_obs)
    csf = (anatomy[:, 3:4] > 0.5).float()
    c_init = torch.sigmoid(seed_map) * decoder.seed_max * (1.0 - csf)
    mass_init = c_init.sum().item()
    mass_final = c_out.sum().item()
    rel_err = abs(mass_final - mass_init) / max(mass_init, 1e-6)
    assert rel_err < 0.10, f"mass changed by {rel_err:.3f} (tol 0.10)"


def test_fk_pure_reaction_caps_at_one():
    decoder = FKDecoder(spatial_size=(8, 16, 16), use_residual=False, ode_steps=20)
    D = torch.zeros(1, 1, 8, 16, 16)  # no diffusion
    rho = torch.full((1, 1, 8, 16, 16), 1.0)
    seed_map = torch.full((1, 1, 8, 16, 16), 5.0)
    anatomy = torch.zeros(1, 4, 8, 16, 16)
    s_obs = torch.tensor([50.0])
    c_out = decoder(D, rho, seed_map, anatomy, s_obs)
    assert c_out.max() <= 1.0 + 1e-3
    assert c_out.max() > 0.9  # asymptote


def test_fk_decoder_gradients_flow():
    decoder = FKDecoder(spatial_size=(8, 16, 16), use_residual=True)
    D = torch.full((1, 1, 8, 16, 16), 0.1, requires_grad=True)
    rho = torch.full((1, 1, 8, 16, 16), 0.1, requires_grad=True)
    seed_map = torch.full((1, 1, 8, 16, 16), -2.0, requires_grad=True)
    anatomy = torch.zeros(1, 4, 8, 16, 16)
    s_obs = torch.tensor([10.0])
    c_out = decoder(D, rho, seed_map, anatomy, s_obs)
    loss = c_out.sum()
    loss.backward()
    assert D.grad is not None and torch.isfinite(D.grad).all()
    assert rho.grad is not None and torch.isfinite(rho.grad).all()
    assert seed_map.grad is not None and torch.isfinite(seed_map.grad).all()
    for n, p in decoder.named_parameters():
        assert p.grad is not None, f"no grad on {n}"
        assert torch.isfinite(p.grad).all()


def test_fk_decoder_residual_zero_init_matches_no_residual():
    """Residual is zero-init, so use_residual=True at init matches use_residual=False."""
    torch.manual_seed(42)
    decoder_off = FKDecoder(spatial_size=(8, 16, 16), use_residual=False, ode_steps=4)
    torch.manual_seed(42)
    decoder_on = FKDecoder(spatial_size=(8, 16, 16), use_residual=True, ode_steps=4)
    D = torch.full((1, 1, 8, 16, 16), 0.1)
    rho = torch.full((1, 1, 8, 16, 16), 0.1)
    seed_map = torch.zeros(1, 1, 8, 16, 16)
    anatomy = torch.zeros(1, 4, 8, 16, 16)
    s_obs = torch.tensor([5.0])
    c_off = decoder_off(D, rho, seed_map, anatomy, s_obs)
    c_on = decoder_on(D, rho, seed_map, anatomy, s_obs)
    assert torch.allclose(c_off, c_on, atol=1e-6)


def test_fk_decoder_shape_and_dtype():
    decoder = FKDecoder(spatial_size=(8, 16, 16), use_residual=False)
    D = torch.full((2, 1, 8, 16, 16), 0.1)
    rho = torch.full((2, 1, 8, 16, 16), 0.1)
    seed_map = torch.zeros(2, 1, 8, 16, 16)
    anatomy = torch.zeros(2, 4, 8, 16, 16)
    s_obs = torch.tensor([10.0, 10.0])
    c_out = decoder(D, rho, seed_map, anatomy, s_obs)
    assert c_out.shape == (2, 1, 8, 16, 16)
    assert c_out.dtype == D.dtype


def test_fk_decoder_rejects_mixed_s_obs():
    decoder = FKDecoder(spatial_size=(8, 16, 16), use_residual=False)
    D = torch.full((2, 1, 8, 16, 16), 0.1)
    rho = torch.full((2, 1, 8, 16, 16), 0.1)
    seed_map = torch.zeros(2, 1, 8, 16, 16)
    anatomy = torch.zeros(2, 4, 8, 16, 16)
    s_obs = torch.tensor([10.0, 20.0])
    with pytest.raises(ValueError, match="per-batch"):
        decoder(D, rho, seed_map, anatomy, s_obs)
