import torch
import pytest

from networks.models.glio_ode.glio_ode import GlioODE
from networks.models.glio_ode.diffusion import GlioDiffusion


@pytest.fixture
def tiny_model():
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


def test_q_sample_zero_noise_matches_signal_scaling(tiny_model):
    """With noise=0 at any t, q_sample(z0) must equal sqrt(alpha_bar_t) * z0.

    The cosine schedule has alpha_bar_0 < 1, so we cannot assume z == z0.
    We check the formula directly instead.
    """
    diff = GlioDiffusion(tiny_model, image_channels=2, mask_channels=2, timesteps=10)
    z0 = torch.randn(2, 4, 8, 16, 16)
    t = torch.tensor([0, 5], dtype=torch.long)
    z = diff.q_sample(z0, t=t, noise=torch.zeros_like(z0))
    scale = diff.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
    assert torch.allclose(z, scale * z0, atol=1e-5)


def test_q_sample_at_tmax_is_near_unit_variance(tiny_model):
    diff = GlioDiffusion(tiny_model, image_channels=2, mask_channels=2, timesteps=10)
    z0 = torch.randn(8, 4, 8, 16, 16)
    t = torch.full((8,), 9, dtype=torch.long)
    z = diff.q_sample(z0, t=t)
    assert abs(z.std().item() - 1.0) < 0.5


def test_training_loss_is_finite(tiny_model):
    diff = GlioDiffusion(tiny_model, image_channels=2, mask_channels=2, timesteps=10)
    z0 = torch.randn(2, 4, 8, 16, 16)
    loss = diff(z0)
    assert torch.isfinite(loss)


def test_conditional_sample_clamps_image_channels(tiny_model, monkeypatch):
    """When x_cond is given, p_sample must overwrite image channels with
    q_sample(x_cond, t-1). We monkeypatch q_sample to return a known stub
    so the assertion is deterministic regardless of RNG state.
    """
    diff = GlioDiffusion(tiny_model, image_channels=2, mask_channels=2, timesteps=10)
    x = torch.randn(1, 2, 8, 16, 16)
    z_T = torch.randn(1, 4, 8, 16, 16)
    t = torch.tensor([5], dtype=torch.long)

    # Sentinel value the stub returns when called on x_cond at t-1.
    expected_img = torch.full_like(x, 0.42)
    original_q_sample = diff.q_sample

    def stub_q_sample(z0, t, noise=None):
        # Only override the clamp call: q_sample(x_cond, t-1) with shape == x's.
        if z0.shape == x.shape and bool((t == 4).all()):
            return expected_img
        return original_q_sample(z0, t, noise=noise)

    monkeypatch.setattr(diff, "q_sample", stub_q_sample)
    z_prev = diff.p_sample(z_T, t, x_cond=x)
    assert torch.allclose(z_prev[:, :2], expected_img)


def test_forward_segment_returns_argmax_mask(tiny_model):
    """Originally tested forward_segment; now tests the new model(x) entry point."""
    # Reconfigure the tiny model for a fast DDIM-2 schedule (test fixture defaults
    # to DDIM-50 which is overkill for a smoke test).
    tiny_model._eval_num_steps = 2
    diff = GlioDiffusion(tiny_model, image_channels=2, mask_channels=2, timesteps=10)
    tiny_model.attach_diffusion(diff)

    x = torch.randn(1, 2, 8, 16, 16)
    mask = tiny_model(x)
    assert mask.shape == (1, 8, 16, 16)
    assert mask.dtype == torch.long
    assert mask.min().item() >= 0 and mask.max().item() < 2


def test_p_sample_loop_conditional_returns_clean_image_at_end(tiny_model):
    """After the full reverse process, image channels of z_0 should equal x_cond exactly."""
    diff = GlioDiffusion(tiny_model, image_channels=2, mask_channels=2, timesteps=10)
    tiny_model.attach_diffusion(diff)
    x = torch.randn(1, 2, 8, 16, 16)
    z0 = diff.p_sample_loop_conditional(x, num_steps=4)
    assert torch.allclose(z0[:, :2], x), "image channels of z_0 must equal x_cond exactly"


def test_p_sample_loop_conditional_visits_final_step():
    """With non-divisible num_steps, the schedule must still include i=0."""
    import torch as _torch
    # We don't need a real model — just patch p_sample to record visited timesteps.
    from networks.models.glio_ode.glio_ode import GlioODE
    from networks.models.glio_ode.diffusion import GlioDiffusion

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
    diff = GlioDiffusion(model, image_channels=2, mask_channels=2, timesteps=1000)
    visited = []
    original_p_sample = diff.p_sample
    def recording_p_sample(z_t, t, x_cond=None):
        visited.append(int(t[0].item()))
        # Return z_t as-is so we don't actually run the model — we only care about scheduling.
        if x_cond is not None:
            return z_t.clone()
        return z_t
    diff.p_sample = recording_p_sample
    x = _torch.randn(1, 2, 8, 16, 16)
    _ = diff.p_sample_loop_conditional(x, num_steps=50)
    assert visited[0] == 999, f"first visited step should be 999, got {visited[0]}"
    assert visited[-1] == 0, f"last visited step should be 0, got {visited[-1]}"
    assert len(visited) == 50


def test_ddim_sampler_returns_clean_image_at_end(tiny_model):
    diff = GlioDiffusion(tiny_model, image_channels=2, mask_channels=2, timesteps=10)
    x = torch.randn(1, 2, 8, 16, 16)
    z0 = diff.p_sample_loop_conditional(x, num_steps=4, sampler="ddim")
    assert z0.shape == (1, 4, 8, 16, 16)
    assert torch.allclose(z0[:, :2], x), "DDIM image channels at z_0 must equal x_cond exactly"


def test_ddim_is_deterministic(tiny_model):
    """DDIM has no random noise term; seeding identically before each call yields
    bit-identical output. (The initial latent z_T is still randomly drawn, so
    seeding makes that draw reproducible too.)
    """
    diff = GlioDiffusion(tiny_model, image_channels=2, mask_channels=2, timesteps=10)
    tiny_model.eval()
    x = torch.randn(1, 2, 8, 16, 16)

    torch.manual_seed(7)
    z0_a = diff.p_sample_loop_conditional(x, num_steps=4, sampler="ddim")
    torch.manual_seed(7)
    z0_b = diff.p_sample_loop_conditional(x, num_steps=4, sampler="ddim")
    assert torch.allclose(z0_a, z0_b, atol=0, rtol=0)


def test_ddim_and_ddpm_produce_different_z0(tiny_model):
    """Sanity check: DDIM and DDPM are different code paths, so different z_0 mask channels."""
    diff = GlioDiffusion(tiny_model, image_channels=2, mask_channels=2, timesteps=10)
    tiny_model.eval()
    x = torch.randn(1, 2, 8, 16, 16)
    z0_ddpm = diff.p_sample_loop_conditional(x, num_steps=4, sampler="ddpm")
    z0_ddim = diff.p_sample_loop_conditional(x, num_steps=4, sampler="ddim")
    # Image channels are clamped to x_cond in both; mask channels should differ.
    assert not torch.allclose(z0_ddpm[:, 2:], z0_ddim[:, 2:])


def test_ddim_rejects_unknown_sampler(tiny_model):
    diff = GlioDiffusion(tiny_model, image_channels=2, mask_channels=2, timesteps=10)
    x = torch.randn(1, 2, 8, 16, 16)
    with pytest.raises(ValueError, match="unknown sampler"):
        diff.p_sample_loop_conditional(x, num_steps=4, sampler="not_a_sampler")
