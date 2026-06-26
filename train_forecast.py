"""GlioForecast training entry point.

Usage:
    python train_forecast.py
    python train_forecast.py training.total_iters=1000 training.batch_size=1
    python train_forecast.py data.root=/path/to/gliomasolver
"""
from __future__ import annotations

import logging
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter

from networks.data.gliomasolver import (
    build_gliomasolver_datasets,
    build_gliomasolver_dataloaders,
    build_gliomasolver_transforms,
)
from networks.models.glio_forecast.diffusion import GlioForecastDiffusion
from networks.training.loop import (
    EMA,
    load_checkpoint,
    save_checkpoint,
    train_one_iter_forecast,
    validate_forecast,
)


log = logging.getLogger(__name__)


def _mid_slice_normalized(volume: torch.Tensor) -> torch.Tensor:
    """Extract a normalized mid-depth slice from (B, C, D, H, W) for TB display.

    Returns (1, H, W) float in [0, 1] suitable for SummaryWriter.add_image.
    """
    d_mid = volume.shape[2] // 2
    sl = volume[0, 0, d_mid].detach().float()
    lo = sl.min()
    hi = sl.max()
    if (hi - lo).item() < 1e-8:
        return torch.zeros_like(sl).unsqueeze(0)
    return ((sl - lo) / (hi - lo + 1e-8)).unsqueeze(0)


def _build(cfg: DictConfig):
    model = instantiate(cfg.model)
    diffusion = GlioForecastDiffusion(
        model,
        conditioning_channels=cfg.model.anatomy_channels + cfg.model.observed_channels,
        param_channels=cfg.model.param_channels,
        timesteps=cfg.model.diffusion_timesteps,
        diff_weight=cfg.diffusion.diff_weight,
        recon_weight=cfg.diffusion.recon_weight,
        recon_warmup_iters=cfg.diffusion.recon_warmup_iters,
    )
    model.attach_diffusion(diffusion)
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    diffusion.to(device)
    ema = EMA(model, decay=cfg.training.ema_decay)
    for n in ema.shadow:
        ema.shadow[n] = ema.shadow[n].to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.total_iters,
    )
    return {
        "model": model, "diffusion": diffusion, "ema": ema,
        "optimizer": optimizer, "scheduler": scheduler, "device": device,
    }


@hydra.main(version_base=None, config_path="conf", config_name="config_forecast")
def main(cfg: DictConfig) -> None:
    bag = _build(cfg)
    model, diffusion, ema = bag["model"], bag["diffusion"], bag["ema"]
    optimizer, scheduler, device = bag["optimizer"], bag["scheduler"], bag["device"]

    tf = build_gliomasolver_transforms(crop_size=tuple(cfg.model.crop_size))
    train_ds, val_ds = build_gliomasolver_datasets(
        root=cfg.data.root, transforms=tf, val_fraction=cfg.data.val_fraction,
    )
    train_loader, val_loader = build_gliomasolver_dataloaders(
        train_ds, val_ds,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        priors=dict(cfg.data.priors),
        seed_sigma_voxels=cfg.data.seed_sigma_voxels,
        fk_seed_max=cfg.model.fk_seed_max,
        fk_d_max=cfg.model.fk_d_max,
        fk_rho_max=cfg.model.fk_rho_max,
        voxel_size=tuple(cfg.data.voxel_size),
        s_obs_days_range=tuple(cfg.data.s_obs_days_range),
        fk_ode_steps=cfg.model.fk_ode_steps,
        val_forecast_horizon=cfg.data.val_forecast_horizon,
    )

    writer = SummaryWriter(cfg.training.log_dir)
    ckpt_dir = Path(cfg.training.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    iter_idx = 0
    best_val = 0.0
    if cfg.training.resume:
        iter_idx, best_val = load_checkpoint(
            Path(cfg.training.resume), model, diffusion, ema, optimizer,
            scheduler=scheduler,
        )
        log.info("resumed from %s at iter %d", cfg.training.resume, iter_idx)

    amp_dtype = torch.bfloat16 if cfg.training.amp else None
    train_iter = iter(train_loader)

    try:
        while iter_idx < cfg.training.total_iters:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            metrics = train_one_iter_forecast(
                diffusion, optimizer, batch, ema, iter_idx=iter_idx,
                amp_dtype=amp_dtype, grad_clip=cfg.training.grad_clip,
            )
            scheduler.step()
            iter_idx += 1
            if iter_idx % cfg.training.log_every == 0:
                for k, v in metrics.items():
                    writer.add_scalar(f"train/{k}", v, iter_idx)
                log.info(
                    "iter %d loss=%.4f l_diff=%.4f l_recon=%.4f grad_norm=%.4f",
                    iter_idx, metrics["loss"], metrics["l_diff"],
                    metrics["l_recon"], metrics["grad_norm"],
                )
            if iter_idx % cfg.training.val_every == 0:
                snapshot = ema.store(model)
                ema.copy_to(model)
                val_metrics = validate_forecast(
                    model, val_loader,
                    num_samples=cfg.training.val_num_samples,
                    forecast_horizon=cfg.training.val_forecast_horizon,
                    dice_threshold=cfg.training.val_dice_threshold,
                )
                # Image summaries: one val case's mid-depth slices (EMA weights still loaded).
                try:
                    val_batch = next(iter(val_loader))
                except StopIteration:
                    val_batch = None
                if val_batch is not None:
                    anatomy_b = val_batch["anatomy"].to(device)
                    c_obs_b = val_batch["c_obs"].to(device)
                    c_future_b = val_batch["c_future"].to(device)
                    D_true_b = val_batch["D_true"].to(device)
                    rho_true_b = val_batch["rho_true"].to(device)
                    cond_b = torch.cat([anatomy_b, c_obs_b], dim=1)
                    with torch.no_grad():
                        z0 = diffusion.p_sample_loop_conditional(
                            cond_b,
                            num_steps=model.encoder._eval_num_steps,
                            sampler=model.encoder._eval_sampler,
                        )
                        params_logits = z0[:, model.conditioning_channels:]
                        D_hat = model.fk_d_max * torch.sigmoid(params_logits[:, 0:1])
                        rho_hat = model.fk_rho_max * torch.sigmoid(params_logits[:, 1:2])
                        s_future_b = torch.full(
                            (1,), float(cfg.training.val_forecast_horizon),
                            device=device,
                        )
                        c_forecast_b = model.decode_fk(params_logits, anatomy_b, s_future_b)
                    writer.add_image("val/D_hat",      _mid_slice_normalized(D_hat),      iter_idx)
                    writer.add_image("val/D_true",    _mid_slice_normalized(D_true_b),    iter_idx)
                    writer.add_image("val/rho_hat",    _mid_slice_normalized(rho_hat),    iter_idx)
                    writer.add_image("val/rho_true",  _mid_slice_normalized(rho_true_b),  iter_idx)
                    writer.add_image("val/c_obs",     _mid_slice_normalized(c_obs_b),     iter_idx)
                    writer.add_image("val/c_forecast", _mid_slice_normalized(c_forecast_b), iter_idx)
                    writer.add_image("val/c_future",  _mid_slice_normalized(c_future_b),  iter_idx)
                ema.copy_from(model, snapshot)
                for k, v in val_metrics.items():
                    writer.add_scalar(f"val/{k}", v, iter_idx)
                log.info(
                    "iter %d val: D_mse=%.4f rho_mse=%.4f seed_dist=%.2f dice@%.2f=%.4f",
                    iter_idx,
                    val_metrics["D_mse"], val_metrics["rho_mse"],
                    val_metrics["seed_xyz_dist"],
                    cfg.training.val_dice_threshold,
                    val_metrics["c_forecast_dice"],
                )
                if val_metrics["c_forecast_dice"] > best_val:
                    best_val = val_metrics["c_forecast_dice"]
                    save_checkpoint(
                        ckpt_dir / "best.pt", model, diffusion, ema,
                        optimizer, iter_idx, best_val, scheduler=scheduler,
                    )
                save_checkpoint(
                    ckpt_dir / "last.pt", model, diffusion, ema,
                    optimizer, iter_idx, best_val, scheduler=scheduler,
                )
    finally:
        writer.close()


if __name__ == "__main__":
    main()
