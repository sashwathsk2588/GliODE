from pathlib import Path

import numpy as np
import pytest
import torch

from networks.data.gliomasolver import (
    _write_synthetic_gliomasolver_case,
    build_gliomasolver_datasets,
    build_gliomasolver_dataloaders,
    build_gliomasolver_transforms,
)


def _write_n_cases(root, n, shape=(16, 16, 16)):
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        _write_synthetic_gliomasolver_case(root / f"case_{i:03d}", shape=shape)


def test_gliomasolver_dataset_loads_synthetic_case(tmp_path):
    root = tmp_path / "gs"
    _write_n_cases(root, n=2)
    tf = build_gliomasolver_transforms(crop_size=(16, 16, 16))
    train_ds, val_ds = build_gliomasolver_datasets(str(root), tf, val_fraction=0.5, seed=0)
    assert len(train_ds) == 1
    assert len(val_ds) == 1
    sample = train_ds[0]
    for key in ("t1gd", "gm", "wm", "csf"):
        assert sample[key].shape == (1, 16, 16, 16)


def test_gliomasolver_dataloader_yields_batch_with_keys(tmp_path):
    root = tmp_path / "gs"
    _write_n_cases(root, n=4)
    tf = build_gliomasolver_transforms(crop_size=(16, 16, 16))
    train_ds, val_ds = build_gliomasolver_datasets(str(root), tf, val_fraction=0.5, seed=0)
    train_loader, _ = build_gliomasolver_dataloaders(
        train_ds, val_ds, batch_size=2, num_workers=0,
        s_obs_days_range=(1.0, 2.0),
    )
    batch = next(iter(train_loader))
    for key in ("anatomy", "c_obs", "params", "s_obs"):
        assert key in batch
    assert batch["anatomy"].shape == (2, 4, 16, 16, 16)
    assert batch["c_obs"].shape == (2, 1, 16, 16, 16)
    assert batch["params"].shape == (2, 3, 16, 16, 16)
    assert batch["s_obs"].shape == (2,)


def test_gliomasolver_batch_shares_s_obs(tmp_path):
    root = tmp_path / "gs"
    _write_n_cases(root, n=4)
    tf = build_gliomasolver_transforms(crop_size=(16, 16, 16))
    train_ds, val_ds = build_gliomasolver_datasets(str(root), tf, val_fraction=0.5, seed=0)
    train_loader, _ = build_gliomasolver_dataloaders(
        train_ds, val_ds, batch_size=2, num_workers=0,
        s_obs_days_range=(1.0, 2.0),
    )
    batch = next(iter(train_loader))
    assert batch["s_obs"].unique().numel() == 1


def test_gliomasolver_priors_in_range(tmp_path):
    """Sampled FK params should land inside the configured priors."""
    from networks.data.gliomasolver import _GliomaSolverCollator
    coll = _GliomaSolverCollator(
        priors={
            "d_wm_log10_range": (-2.0, 0.0),
            "d_gm_ratio": 0.1,
            "rho_log10_range": (-2.0, 0.0),
        },
        s_obs_days_range=(30.0, 365.0),
    )
    shape = (8, 8, 8)
    gm = torch.zeros(1, *shape)
    wm = torch.zeros(1, *shape)
    csf = torch.zeros(1, *shape)
    gm[0, 2:6, 2:6, 2:6] = 1.0
    wm[0, 3:5, 3:5, 3:5] = 1.0
    sample = {
        "t1gd": torch.zeros(1, *shape),
        "gm": gm, "wm": wm, "csf": csf,
    }
    batch = coll([sample])
    assert batch["s_obs"][0].item() >= 30.0
    assert batch["s_obs"][0].item() <= 365.0


def test_gliomasolver_params_are_inverse_sigmoid_logits(tmp_path):
    """Params channels must store inverse_sigmoid(true_value / max), not raw values.

    The encoder predicts logits because decode_fk applies sigmoid * max
    activation. Storing raw D would cause a representation mismatch.
    """
    from networks.data.gliomasolver import _GliomaSolverCollator
    coll = _GliomaSolverCollator(
        priors={
            "d_wm_log10_range": (-1.0, -1.0),  # fixed D_wm = 0.1
            "d_gm_ratio": 0.1,
            "rho_log10_range": (-1.0, -1.0),  # fixed rho = 0.1
        },
        s_obs_days_range=(1.0, 1.0),
        fk_d_max=1.0, fk_rho_max=1.0, fk_seed_max=0.1,
    )
    shape = (8, 8, 8)
    gm = torch.zeros(1, *shape)
    wm = torch.zeros(1, *shape)
    csf = torch.zeros(1, *shape)
    wm[0, 3:5, 3:5, 3:5] = 1.0
    sample = {
        "t1gd": torch.zeros(1, *shape),
        "gm": gm, "wm": wm, "csf": csf,
    }
    batch = coll([sample])
    # params[:, 0] = D_logit, params[:, 1] = rho_logit, params[:, 2] = seed_logit.
    # Where WM=1: D_true = 0.1, so D_true / fk_d_max = 0.1.
    # inverse_sigmoid(0.1) = log(0.1 / 0.9) ≈ -2.197.
    import math
    expected = math.log(0.1 / 0.9)
    d_logit_at_wm = batch["params"][0, 0, 4, 4, 4].item()  # WM voxel
    assert abs(d_logit_at_wm - expected) < 0.1, (
        f"expected logit ≈ {expected:.3f}, got {d_logit_at_wm:.3f}"
    )


def test_val_collator_returns_ground_truth_keys(tmp_path):
    """is_val=True collator must add ground-truth fields + c_future for metrics."""
    from networks.data.gliomasolver import _GliomaSolverCollator

    coll = _GliomaSolverCollator(
        priors={
            "d_wm_log10_range": (-2.0, -1.0),
            "d_gm_ratio": 0.1,
            "rho_log10_range": (-2.0, -1.0),
        },
        s_obs_days_range=(1.0, 2.0),
        is_val=True,
        val_forecast_horizon=3.0,
    )
    shape = (8, 8, 8)
    gm = torch.zeros(1, *shape)
    wm = torch.zeros(1, *shape)
    csf = torch.zeros(1, *shape)
    wm[0, 3:5, 3:5, 3:5] = 1.0
    gm[0, 2:6, 2:6, 2:6] = 1.0
    sample = {
        "t1gd": torch.zeros(1, *shape),
        "gm": gm, "wm": wm, "csf": csf,
    }
    batch = coll([sample])
    for key in ("anatomy", "c_obs", "params", "s_obs",
                "D_true", "rho_true", "seed_xyz_true", "c_future",
                "brain_mask", "s_future"):
        assert key in batch, f"missing key {key}"
    assert batch["D_true"].shape == (1, 1, 8, 8, 8)
    assert batch["rho_true"].shape == (1, 1, 8, 8, 8)
    assert batch["seed_xyz_true"].shape == (1, 3)
    assert batch["c_future"].shape == (1, 1, 8, 8, 8)
    assert batch["brain_mask"].shape == (1, 1, 8, 8, 8)
    assert batch["s_future"].shape == (1,)
    assert abs(batch["s_future"][0].item() - 3.0) < 1e-6


def test_val_collator_rejects_zero_horizon():
    from networks.data.gliomasolver import _GliomaSolverCollator
    with pytest.raises(ValueError, match="val_forecast_horizon"):
        _GliomaSolverCollator(is_val=True, val_forecast_horizon=0.0)
