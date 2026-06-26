from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from networks.data.gliomasolver import _write_synthetic_gliomasolver_case
from networks.models.glio_forecast.diffusion import GlioForecastDiffusion
from networks.models.glio_forecast.glio_forecast import GlioForecast
from networks.training.loop import EMA, train_one_iter_forecast


def _tiny_setup():
    model = GlioForecast(
        crop_size=(8, 16, 16),
        anatomy_channels=4, observed_channels=1, param_channels=3,
        patch_size=(2, 4, 4),
        d_model=12, num_heads=2, mlp_ratio=2.0,
        ode_steps=2, decoder_channels=(12, 6, 6, 6),
        diffusion_timesteps=10,
        fk_ode_steps=2,
        eval_sampler="ddim", eval_num_steps=2,
    )
    diff = GlioForecastDiffusion(model, conditioning_channels=5, param_channels=3, timesteps=10)
    model.attach_diffusion(diff)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ema = EMA(model, decay=0.9)
    return model, diff, optim, ema


def test_train_one_iter_forecast_changes_params():
    model, diff, optim, ema = _tiny_setup()
    batch = {
        "anatomy": torch.zeros(2, 4, 8, 16, 16),
        "c_obs": torch.zeros(2, 1, 8, 16, 16),
        "params": torch.zeros(2, 3, 8, 16, 16),
        "s_obs": torch.tensor([1.0, 1.0]),
    }
    pre = {n: p.detach().clone() for n, p in model.named_parameters()}
    metrics = train_one_iter_forecast(
        diff, optim, batch, ema, iter_idx=0, amp_dtype=None, grad_clip=1.0,
    )
    for k in ("loss", "l_diff", "l_recon", "grad_norm"):
        assert k in metrics
        assert metrics[k] == metrics[k]  # not NaN
    changed = 0
    for n, p in model.named_parameters():
        if not torch.equal(pre[n], p.detach()):
            changed += 1
    assert changed > 0


def test_train_forecast_main_runs_two_iters(tmp_path):
    root = tmp_path / "gs"
    root.mkdir()
    for i in range(4):
        _write_synthetic_gliomasolver_case(root / f"case_{i:03d}", shape=(8, 16, 16))

    cfg = OmegaConf.create({
        "model": {
            "_target_": "networks.models.glio_forecast.glio_forecast.GlioForecast",
            "crop_size": [8, 16, 16],
            "anatomy_channels": 4, "observed_channels": 1, "param_channels": 3,
            "patch_size": [2, 4, 4],
            "d_model": 12, "num_heads": 2, "mlp_ratio": 2.0,
            "ode_t": 1.0, "ode_steps": 2, "ode_method": "rk4",
            "decoder_channels": [12, 6, 6, 6],
            "diffusion_timesteps": 10,
            "fk_voxel_size": [1.0, 1.0, 1.0],
            "fk_use_residual": False, "fk_residual_channels": 4, "fk_ode_steps": 2,
            "fk_seed_max": 0.1, "fk_d_max": 1.0, "fk_rho_max": 1.0,
            "eval_sampler": "ddim", "eval_num_steps": 2,
            "eval_overlap": 0.5, "eval_sw_batch_size": 1,
            "attn_backend": "sdpa",
        },
        "data": {
            "root": str(root),
            "val_fraction": 0.5,
            "num_workers": 0,
            "priors": {
                "d_wm_log10_range": [-2.0, -1.0],
                "d_gm_ratio": 0.1,
                "rho_log10_range": [-2.0, -1.0],
            },
            "seed_sigma_voxels": 2.0,
            "voxel_size": [1.0, 1.0, 1.0],
            "s_obs_days_range": [1.0, 2.0],
            "val_forecast_horizon": 1.0,
        },
        "training": {
            "device": "cpu",
            "lr": 1.0e-3, "weight_decay": 1.0e-2,
            "batch_size": 1, "total_iters": 2,
            "ema_decay": 0.9,
            "amp": False, "grad_clip": 1.0,
            "log_every": 1, "val_every": 2,
            "log_dir": str(tmp_path / "runs"),
            "ckpt_dir": str(tmp_path / "ckpt"),
            "resume": None,
            "val_num_samples": 1,
            "val_forecast_horizon": 1.0,
            "val_dice_threshold": 0.1,
        },
        "diffusion": {
            "diff_weight": 1.0,
            "recon_weight": 1.0,
            "recon_warmup_iters": 0,
        },
    })

    import train_forecast
    train_forecast.main.__wrapped__(cfg)
    assert Path(cfg.training.log_dir).exists()
    # With val_every=2 and total_iters=2, validation ran once and last.pt was written.
    assert (Path(cfg.training.ckpt_dir) / "last.pt").exists()
