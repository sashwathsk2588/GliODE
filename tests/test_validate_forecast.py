import math

import pytest
import torch
from torch.utils.data import DataLoader as TorchDataLoader

from networks.models.glio_forecast.diffusion import GlioForecastDiffusion
from networks.models.glio_forecast.glio_forecast import GlioForecast
from networks.training.loop import validate_forecast


def _tiny_attached():
    m = GlioForecast(
        crop_size=(8, 16, 16),
        anatomy_channels=4, observed_channels=1, param_channels=3,
        patch_size=(2, 4, 4),
        d_model=12, num_heads=2, mlp_ratio=2.0,
        ode_steps=2, decoder_channels=(12, 6, 6, 6),
        diffusion_timesteps=10,
        fk_ode_steps=2,
        eval_sampler="ddim", eval_num_steps=2,
        attn_backend="sdpa",
    )
    diff = GlioForecastDiffusion(m, conditioning_channels=5, param_channels=3, timesteps=10)
    m.attach_diffusion(diff)
    return m


class _OneCaseValDataset:
    """Yields one synthetic val sample with the keys validate_forecast expects."""

    def __init__(self, shape=(8, 16, 16)):
        self.shape = shape

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        D, H, W = self.shape
        anatomy = torch.zeros(4, D, H, W)
        anatomy[1, 2:6, 4:12, 4:12] = 1.0  # GM
        anatomy[2, 3:5, 6:10, 6:10] = 1.0  # WM
        c_obs = torch.zeros(1, D, H, W)
        c_future = torch.zeros(1, D, H, W)
        c_future[0, 4, 8, 8] = 1.0          # single bright voxel ground truth
        D_true = torch.full((1, D, H, W), 0.1)
        rho_true = torch.full((1, D, H, W), 0.1)
        brain_mask = ((anatomy[1] + anatomy[2]) > 0.5).float().unsqueeze(0)
        return {
            "anatomy": anatomy,
            "c_obs": c_obs,
            "params": torch.zeros(3, D, H, W),
            "s_obs": torch.tensor(10.0),
            "D_true": D_true,
            "rho_true": rho_true,
            "seed_xyz_true": torch.tensor([4.0, 8.0, 8.0]),
            "c_future": c_future,
            "brain_mask": brain_mask,
            "s_future": torch.tensor(365.0),
        }


def _identity_collator(items):
    out = {}
    for k in items[0]:
        out[k] = torch.stack([it[k] for it in items], dim=0)
    return out


def _one_case_val_loader():
    return TorchDataLoader(
        _OneCaseValDataset(), batch_size=1, collate_fn=_identity_collator,
    )


def test_validate_forecast_returns_metric_dict(monkeypatch):
    model = _tiny_attached()

    def stub_sample(cond, num_steps=None, sampler="ddim"):
        B, _, D, H, W = cond.shape
        z0 = torch.zeros(B, 8, D, H, W)
        z0[:, :5] = cond
        return z0

    monkeypatch.setattr(model._diffusion, "p_sample_loop_conditional", stub_sample)
    loader = _one_case_val_loader()
    metrics = validate_forecast(model, loader, num_samples=1)
    for k in (
        "D_mse", "rho_mse", "seed_xyz_dist", "c_forecast_dice",
        "D_mse_wm", "D_mse_gm", "rho_mse_wm", "rho_mse_gm",
    ):
        assert k in metrics
        assert isinstance(metrics[k], float)


def test_validate_forecast_dice_perfect_when_pred_eq_true(monkeypatch):
    model = _tiny_attached()

    def stub_sample(cond, num_steps=None, sampler="ddim"):
        B, _, D, H, W = cond.shape
        z0 = torch.zeros(B, 8, D, H, W)
        z0[:, :5] = cond
        return z0

    monkeypatch.setattr(model._diffusion, "p_sample_loop_conditional", stub_sample)

    def stub_decode_fk(params, anatomy, s_obs):
        D, H, W = anatomy.shape[-3:]
        c = torch.zeros(anatomy.shape[0], 1, D, H, W)
        c[0, 0, 4, 8, 8] = 1.0
        return c

    monkeypatch.setattr(model, "decode_fk", stub_decode_fk)
    loader = _one_case_val_loader()
    metrics = validate_forecast(model, loader, num_samples=1, dice_threshold=0.1)
    assert abs(metrics["c_forecast_dice"] - 1.0) < 1e-6


def test_validate_forecast_dice_zero_when_pred_empty(monkeypatch):
    model = _tiny_attached()

    def stub_sample(cond, num_steps=None, sampler="ddim"):
        B, _, D, H, W = cond.shape
        z0 = torch.zeros(B, 8, D, H, W)
        z0[:, :5] = cond
        return z0

    monkeypatch.setattr(model._diffusion, "p_sample_loop_conditional", stub_sample)

    def stub_decode_fk(params, anatomy, s_obs):
        D, H, W = anatomy.shape[-3:]
        return torch.zeros(anatomy.shape[0], 1, D, H, W)

    monkeypatch.setattr(model, "decode_fk", stub_decode_fk)
    loader = _one_case_val_loader()
    metrics = validate_forecast(model, loader, num_samples=1, dice_threshold=0.1)
    assert metrics["c_forecast_dice"] == 0.0


def test_validate_forecast_seed_dist_zero_at_truth(monkeypatch):
    model = _tiny_attached()

    def stub_sample(cond, num_steps=None, sampler="ddim"):
        B, _, D, H, W = cond.shape
        z0 = torch.zeros(B, 8, D, H, W)
        z0[:, :5] = cond
        z0[:, 7, 4, 8, 8] = 10.0
        return z0

    monkeypatch.setattr(model._diffusion, "p_sample_loop_conditional", stub_sample)

    def stub_decode_fk(params, anatomy, s_obs):
        D, H, W = anatomy.shape[-3:]
        return torch.zeros(anatomy.shape[0], 1, D, H, W)

    monkeypatch.setattr(model, "decode_fk", stub_decode_fk)
    loader = _one_case_val_loader()
    metrics = validate_forecast(model, loader, num_samples=1)
    assert metrics["seed_xyz_dist"] == 0.0


def test_validate_forecast_d_mse_zero_at_truth(monkeypatch):
    """When predicted D equals true D, D_mse is 0 (within float tolerance)."""
    model = _tiny_attached()

    logit_for_0p1 = math.log(0.1 / 0.9)

    def stub_sample(cond, num_steps=None, sampler="ddim"):
        B, _, D, H, W = cond.shape
        z0 = torch.zeros(B, 8, D, H, W)
        z0[:, :5] = cond
        z0[:, 5] = logit_for_0p1
        z0[:, 6] = 0.0
        z0[:, 7] = -5.0
        return z0

    monkeypatch.setattr(model._diffusion, "p_sample_loop_conditional", stub_sample)

    def stub_decode_fk(params, anatomy, s_obs):
        D, H, W = anatomy.shape[-3:]
        return torch.zeros(anatomy.shape[0], 1, D, H, W)

    monkeypatch.setattr(model, "decode_fk", stub_decode_fk)
    loader = _one_case_val_loader()
    metrics = validate_forecast(model, loader, num_samples=1)
    assert metrics["D_mse"] < 1e-6
