"""Smoke test for forecast.py inference CLI.

Builds a tiny model, saves a checkpoint, writes a synthetic case, and runs
forecast.main() to assert output NIfTIs are written.
"""
from pathlib import Path

import nibabel as nib
import pytest
import torch
from omegaconf import OmegaConf

from networks.data.gliomasolver import _write_synthetic_gliomasolver_case
from networks.models.glio_forecast.diffusion import GlioForecastDiffusion
from networks.models.glio_forecast.glio_forecast import GlioForecast
from networks.training.loop import EMA, save_checkpoint


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
        attn_backend="sdpa",
    )
    diff = GlioForecastDiffusion(model, conditioning_channels=5, param_channels=3, timesteps=10)
    model.attach_diffusion(diff)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ema = EMA(model, decay=0.9)
    return model, diff, optim, ema


def test_forecast_main_writes_outputs(tmp_path):
    # 1. Build tiny model + save a checkpoint.
    model, diff, optim, ema = _tiny_setup()
    ckpt_path = tmp_path / "test.pt"
    save_checkpoint(
        ckpt_path, model, diff, ema, optim,
        iter_idx=0, best_val_dice=0.0,
    )

    # 2. Write a synthetic case.
    case_dir = tmp_path / "case_000"
    _write_synthetic_gliomasolver_case(case_dir, shape=(8, 16, 16))

    # 3. Run forecast.main() in-process via a config.
    out_dir = tmp_path / "out"
    cfg = OmegaConf.create({
        "ckpt": str(ckpt_path),
        "case": str(case_dir),
        "output_dir": str(out_dir),
        "horizon": 1.0,
        "num_samples": 1,
        "device": "cpu",
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
            "fk_use_residual": True, "fk_residual_channels": 16, "fk_ode_steps": 2,
            "fk_seed_max": 0.1, "fk_d_max": 1.0, "fk_rho_max": 1.0,
            "eval_sampler": "ddim", "eval_num_steps": 2,
            "eval_overlap": 0.5, "eval_sw_batch_size": 1,
            "attn_backend": "sdpa",
        },
    })

    import forecast
    forecast.run(cfg)

    # 4. Assert outputs exist and are valid NIfTI.
    for name in ("c_forecast", "D", "rho", "seed_map"):
        path = out_dir / f"{name}.nii.gz"
        assert path.exists(), f"missing {name}.nii.gz"
        img = nib.load(str(path))
        arr = img.get_fdata()
        assert arr.shape == (8, 16, 16)
