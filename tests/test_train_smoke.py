"""Smoke test for the train.py main loop.

Builds an OmegaConf manually pointing at a synthetic BraTS root, runs 2 iters,
and asserts the loop completes without exception. Does NOT validate (val_every
> total_iters keeps validation off).
"""
from pathlib import Path

import torch
from omegaconf import OmegaConf

from networks.data.brats import _write_synthetic_brats_case


def test_train_main_runs_two_iters(tmp_path, monkeypatch):
    # Build synthetic BraTS root with 4 cases (>= 2 needed after 0.5 val split).
    root = tmp_path / "brats"
    root.mkdir()
    for i in range(4):
        _write_synthetic_brats_case(root / f"case_{i:03d}", shape=(16, 16, 16))

    cfg = OmegaConf.create({
        "model": {
            "_target_": "networks.models.glio_ode.glio_ode.GlioODE",
            "crop_size": [8, 16, 16],
            "in_channels": 4,
            "num_classes": 4,
            "patch_size": [2, 4, 4],
            "d_model": 12,
            "num_heads": 2,
            "mlp_ratio": 2.0,
            "ode_t": 1.0,
            "ode_steps": 2,
            "ode_method": "rk4",
            "decoder_channels": [12, 6, 6, 6],
            "diffusion_timesteps": 10,
            "deep_supervision": False,
            "eval_sampler": "ddim",
            "eval_num_steps": 2,
            "eval_overlap": 0.5,
            "eval_sw_batch_size": 1,
        },
        "data": {
            "root": str(root),
            "val_fraction": 0.5,
            "num_workers": 0,
            "remap_label_4_to_3": True,
        },
        "training": {
            "device": "cpu",
            "lr": 1.0e-3,
            "weight_decay": 1.0e-2,
            "batch_size": 1,
            "total_iters": 2,
            "ema_decay": 0.9,
            "amp": False,
            "grad_clip": 1.0,
            "log_every": 1,
            "val_every": 999,  # disable validation
            "log_dir": str(tmp_path / "runs"),
            "ckpt_dir": str(tmp_path / "ckpt"),
            "resume": None,
        },
        "diffusion": {
            "image_weight": 1.0,
            "mask_weight": 1.0,
        },
    })

    # Import inside the test to avoid hydra import side effects at collection.
    import train
    # Bypass hydra.main: call the inner main with the manually-built cfg.
    train.main.__wrapped__(cfg)  # type: ignore[attr-defined]

    # If we got here, the loop ran end-to-end without exception.
    # Optionally check TensorBoard log files exist.
    assert (Path(cfg.training.log_dir)).exists()
