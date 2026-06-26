"""GlioODE training entry point.

Usage:
    python train.py
    python train.py training.total_iters=1000 training.batch_size=1
    python train.py data.root=/path/to/brats
"""
from __future__ import annotations

import logging
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter

from networks.data.brats import (
    build_brats_datasets,
    build_brats_dataloaders,
    build_brats_transforms,
)
from networks.models.glio_ode.diffusion import GlioDiffusion
from networks.training.loop import (
    EMA,
    load_checkpoint,
    save_checkpoint,
    train_one_iter,
    validate,
)


log = logging.getLogger(__name__)


def _build(cfg: DictConfig):
    """Build model, diffusion, EMA, optimizer, scheduler. Returns a dict bag."""
    model = instantiate(cfg.model)
    diffusion = GlioDiffusion(
        model,
        image_channels=cfg.model.in_channels,
        mask_channels=cfg.model.num_classes,
        timesteps=cfg.model.diffusion_timesteps,
        image_weight=cfg.diffusion.image_weight,
        mask_weight=cfg.diffusion.mask_weight,
    )
    model.attach_diffusion(diffusion)
    device = torch.device(cfg.training.device if torch.cuda.is_available()
                          else "cpu")
    model.to(device)
    ema = EMA(model, decay=cfg.training.ema_decay)
    for n in ema.shadow:
        ema.shadow[n] = ema.shadow[n].to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.total_iters,
    )
    return {
        "model": model,
        "diffusion": diffusion,
        "ema": ema,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "device": device,
    }


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    bag = _build(cfg)
    model, diffusion, ema = bag["model"], bag["diffusion"], bag["ema"]
    optimizer, scheduler, device = bag["optimizer"], bag["scheduler"], bag["device"]

    train_tf, val_tf = build_brats_transforms(
        crop_size=tuple(cfg.model.crop_size),
        num_classes=cfg.model.num_classes,
        remap_label_4_to_3=cfg.data.remap_label_4_to_3,
    )
    train_ds, val_ds = build_brats_datasets(
        root=cfg.data.root,
        train_transforms=train_tf,
        val_transforms=val_tf,
        val_fraction=cfg.data.val_fraction,
    )
    train_loader, val_loader = build_brats_dataloaders(
        train_ds, val_ds,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
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
        log.info("resumed from %s at iter %d (best_val=%.4f)",
                 cfg.training.resume, iter_idx, best_val)

    amp_dtype = torch.bfloat16 if cfg.training.amp else None

    train_iter = iter(train_loader)
    try:
        while iter_idx < cfg.training.total_iters:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            metrics = train_one_iter(
                diffusion, optimizer, batch, ema,
                amp_dtype=amp_dtype, grad_clip=cfg.training.grad_clip,
            )
            scheduler.step()
            iter_idx += 1

            if iter_idx % cfg.training.log_every == 0:
                for k, v in metrics.items():
                    writer.add_scalar(f"train/{k}", v, iter_idx)
                log.info("iter %d loss=%.4f grad_norm=%.4f",
                         iter_idx, metrics["loss"], metrics["grad_norm"])

            if iter_idx % cfg.training.val_every == 0:
                snapshot = ema.store(model)
                ema.copy_to(model)
                val_metrics = validate(model, val_loader, cfg.model.num_classes)
                ema.copy_from(model, snapshot)
                for k, v in val_metrics.items():
                    writer.add_scalar(f"val/{k}", v, iter_idx)
                log.info("iter %d val_dice_mean=%.4f", iter_idx, val_metrics["dice_mean"])
                if val_metrics["dice_mean"] > best_val:
                    best_val = val_metrics["dice_mean"]
                    save_checkpoint(ckpt_dir / "best.pt", model, diffusion, ema,
                                    optimizer, iter_idx, best_val,
                                    scheduler=scheduler)
                save_checkpoint(ckpt_dir / "last.pt", model, diffusion, ema,
                                optimizer, iter_idx, best_val,
                                scheduler=scheduler)
    finally:
        writer.close()


if __name__ == "__main__":
    main()
