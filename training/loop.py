"""GlioODE training-loop helpers: EMA, train_one_iter, validate, checkpointing."""
from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch import nn


class EMA:
    """Exponential-moving-average shadow of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        params = list(model.parameters())
        if not params:
            raise ValueError("EMA requires a model with at least one parameter")
        self.decay = float(decay)
        self.shadow: dict[str, torch.Tensor] = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """Overwrite model params with the EMA shadow."""
        for name, p in model.named_parameters():
            p.copy_(self.shadow[name])

    @torch.no_grad()
    def store(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """Return a snapshot of the current model params (for later copy_from)."""
        return {name: p.detach().clone() for name, p in model.named_parameters()}

    @torch.no_grad()
    def copy_from(self, model: nn.Module, snapshot: dict[str, torch.Tensor]) -> None:
        """Restore model params from a previously-stored snapshot."""
        for name, p in model.named_parameters():
            p.copy_(snapshot[name])


def train_one_iter(
    diffusion,
    optimizer: torch.optim.Optimizer,
    batch: dict,
    ema: EMA | None,
    amp_dtype: torch.dtype | None = torch.bfloat16,
    grad_clip: float = 1.0,
) -> dict[str, float]:
    """One training step.

    Args:
        diffusion: GlioDiffusion instance (treated as nn.Module returning a scalar loss).
            After forward, exposes _last_img_mse / _last_mask_mse for logging.
        optimizer: optimizer to step.
        batch: dict with keys 'image' (B, C_img, D, H, W) and 'label' (B, C_seg, D, H, W).
        ema: optional EMA helper; if given, updated after optimizer.step().
        amp_dtype: dtype for torch.autocast on the forward pass; None disables AMP.
        grad_clip: max grad norm; <=0 disables clipping.
    Returns:
        Metrics dict: loss, img_mse, mask_mse, grad_norm.
    """
    device = next(diffusion.denoiser.parameters()).device
    image = batch["image"].to(device, non_blocking=True)
    label = batch["label"].to(device, non_blocking=True)
    z0 = torch.cat([image, label], dim=1)

    optimizer.zero_grad(set_to_none=True)

    if amp_dtype is not None and device.type == "cuda":
        autocast_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype)
    else:
        autocast_ctx = _NullContext()

    with autocast_ctx:
        loss = diffusion(z0)
    img_mse = getattr(diffusion, "_last_img_mse", 0.0)
    mask_mse = getattr(diffusion, "_last_mask_mse", 0.0)

    loss.backward()

    if grad_clip and grad_clip > 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            diffusion.denoiser.parameters(), max_norm=grad_clip
        ).item()
    else:
        grad_norm = 0.0

    optimizer.step()

    if ema is not None:
        ema.update(diffusion.denoiser)

    return {
        "loss": float(loss.detach().item()),
        "img_mse": float(img_mse),
        "mask_mse": float(mask_mse),
        "grad_norm": float(grad_norm),
    }


class _NullContext:
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc, tb):
        return False


@torch.no_grad()
def validate(
    model,
    val_loader,
    num_classes: int,
) -> dict[str, float]:
    """Sliding-window inference + per-class Dice over the validation loader.

    Returns dict with keys 'dice_class_0', ..., 'dice_class_{num_classes-1}',
    and 'dice_mean'.
    """
    from monai.metrics import DiceMetric
    import torch.nn.functional as F

    metric = DiceMetric(include_background=True, reduction="mean_batch")
    device = next(model.parameters()).device
    model.eval()

    for batch in val_loader:
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        if image.dim() == 4:
            image = image.unsqueeze(0)
            label = label.unsqueeze(0)
        pred_indices = model(image)  # (B, D, H, W) long
        pred_onehot = F.one_hot(pred_indices, num_classes=num_classes)  # (B, D, H, W, C)
        pred_onehot = pred_onehot.permute(0, 4, 1, 2, 3).float()  # (B, C, D, H, W)
        metric(y_pred=pred_onehot, y=label)

    dice_per_class = metric.aggregate()  # (C,) tensor
    metric.reset()

    out: dict[str, float] = {}
    total = 0.0
    for c in range(num_classes):
        v = float(dice_per_class[c].item())
        out[f"dice_class_{c}"] = v
        total += v
    out["dice_mean"] = total / num_classes
    return out


def save_checkpoint(
    path,
    model,
    diffusion,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    iter_idx: int,
    best_val_dice: float,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> None:
    state = {
        "model": model.state_dict(),
        "diffusion_buffers": {
            name: buf.detach().clone()
            for name, buf in diffusion.named_buffers()
        },
        "ema_shadow": {n: t.detach().clone() for n, t in ema.shadow.items()},
        "ema_decay": ema.decay,
        "optimizer": optimizer.state_dict(),
        "iter_idx": int(iter_idx),
        "best_val_dice": float(best_val_dice),
    }
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    torch.save(state, str(path))


def load_checkpoint(
    path,
    model,
    diffusion,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> tuple[int, float]:
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    missing = []
    for key in ("model", "diffusion_buffers", "ema_shadow", "optimizer", "iter_idx", "best_val_dice"):
        if key not in state:
            missing.append(key)
    if missing:
        raise RuntimeError(f"checkpoint at {path} missing keys: {missing}")
    model.load_state_dict(state["model"])
    diffusion_buffers = dict(diffusion.named_buffers())
    for name, buf in state["diffusion_buffers"].items():
        if name in diffusion_buffers:
            diffusion_buffers[name].copy_(buf.to(diffusion_buffers[name].device))
    for n, t in state["ema_shadow"].items():
        if n in ema.shadow:
            ema.shadow[n] = t.clone().to(ema.shadow[n].device)
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    return int(state["iter_idx"]), float(state["best_val_dice"])


def train_one_iter_forecast(
    diffusion,
    optimizer: torch.optim.Optimizer,
    batch: dict,
    ema: EMA | None,
    iter_idx: int,
    amp_dtype: torch.dtype | None = torch.bfloat16,
    grad_clip: float = 1.0,
) -> dict[str, float]:
    """One iter-4 training step (GlioForecast).

    Args:
        diffusion: GlioForecastDiffusion. Exposes set_current_iter, _last_l_diff,
            _last_l_recon.
        optimizer: optimizer to step.
        batch: dict with 'anatomy', 'c_obs', 'params', 's_obs'.
        ema: optional EMA helper; if given, updated after optimizer.step().
        iter_idx: current global iteration (drives recon warmup gate).
        amp_dtype: torch.autocast dtype; None disables AMP.
        grad_clip: max grad norm; <=0 disables.
    Returns:
        Metrics dict: loss, l_diff, l_recon, grad_norm.
    """
    diffusion.set_current_iter(iter_idx)
    device = next(diffusion.forecast_model.encoder.parameters()).device
    anatomy = batch["anatomy"].to(device, non_blocking=True)
    c_obs = batch["c_obs"].to(device, non_blocking=True)
    params = batch["params"].to(device, non_blocking=True)
    s_obs = batch["s_obs"].to(device, non_blocking=True)
    z0 = torch.cat([anatomy, c_obs, params], dim=1)

    optimizer.zero_grad(set_to_none=True)
    if amp_dtype is not None and device.type == "cuda":
        autocast_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype)
    else:
        autocast_ctx = _NullContext()

    with autocast_ctx:
        loss = diffusion(z0, anatomy, c_obs, s_obs)
    l_diff = getattr(diffusion, "_last_l_diff", 0.0)
    l_recon = getattr(diffusion, "_last_l_recon", 0.0)

    loss.backward()
    if grad_clip and grad_clip > 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            diffusion.forecast_model.parameters(), max_norm=grad_clip,
        ).item()
    else:
        grad_norm = 0.0
    optimizer.step()
    if ema is not None:
        ema.update(diffusion.forecast_model)

    return {
        "loss": float(loss.detach().item()),
        "l_diff": float(l_diff),
        "l_recon": float(l_recon),
        "grad_norm": float(grad_norm),
    }


@torch.no_grad()
def validate_forecast(
    model,
    val_loader,
    num_samples: int = 4,
    forecast_horizon: float = 365.0,
    dice_threshold: float = 0.1,
) -> dict[str, float]:
    """Validation loop for GlioForecast.

    For each val case, inverse-sample num_samples parameter realizations,
    average the param LOGITS (not the physical values — bounded activations
    apply inside decode_fk), then run FK forward at forecast_horizon and
    score 4 metrics:
      D_mse: voxelwise MSE between predicted and true D over brain voxels.
      rho_mse: same, for rho.
      seed_xyz_dist: voxel Euclidean distance between argmax of predicted
                     seed_map_logit and seed_xyz_true.
      c_forecast_dice: Dice between (c_forecast > threshold) and
                       (c_future > threshold).

    Returns dict of float means across val cases.
    """
    assert hasattr(model, "_diffusion"), "call attach_diffusion(diff) first"
    model.eval()
    device = next(model.encoder.parameters()).device

    totals = {
        "D_mse": 0.0, "rho_mse": 0.0,
        "D_mse_wm": 0.0, "D_mse_gm": 0.0,
        "rho_mse_wm": 0.0, "rho_mse_gm": 0.0,
        "seed_xyz_dist": 0.0, "c_forecast_dice": 0.0,
    }
    count = 0

    for batch in val_loader:
        anatomy = batch["anatomy"].to(device, non_blocking=True)
        c_obs = batch["c_obs"].to(device, non_blocking=True)
        D_true = batch["D_true"].to(device, non_blocking=True)
        rho_true = batch["rho_true"].to(device, non_blocking=True)
        seed_xyz_true = batch["seed_xyz_true"].to(device, non_blocking=True)
        c_future = batch["c_future"].to(device, non_blocking=True)
        brain_mask = batch["brain_mask"].to(device, non_blocking=True)

        B = anatomy.shape[0]
        cond = torch.cat([anatomy, c_obs], dim=1)  # (B, 5, D, H, W)

        realizations = []
        for _ in range(num_samples):
            z0 = model._diffusion.p_sample_loop_conditional(
                cond,
                num_steps=model.encoder._eval_num_steps,
                sampler=model.encoder._eval_sampler,
            )
            realizations.append(z0[:, model.conditioning_channels:])
        params_logits_mean = torch.stack(realizations, dim=0).mean(dim=0)

        s_future_tensor = torch.full(
            (B,), float(forecast_horizon), device=device,
        )
        c_forecast = model.decode_fk(params_logits_mean, anatomy, s_future_tensor)

        D_logit_mean = params_logits_mean[:, 0:1]
        rho_logit_mean = params_logits_mean[:, 1:2]
        seed_logit_mean = params_logits_mean[:, 2:3]
        D_hat = model.fk_d_max * torch.sigmoid(D_logit_mean)
        rho_hat = model.fk_rho_max * torch.sigmoid(rho_logit_mean)

        for b in range(B):
            bm = brain_mask[b]
            denom = bm.sum().clamp(min=1.0)
            d_mse = ((D_hat[b] - D_true[b]).pow(2) * bm).sum() / denom
            r_mse = ((rho_hat[b] - rho_true[b]).pow(2) * bm).sum() / denom
            # Per-tissue breakdown. anatomy channels: [T1Gd, GM, WM, CSF].
            gm_mask = (anatomy[b, 1:2] > 0.5).float()
            wm_mask = (anatomy[b, 2:3] > 0.5).float()
            gm_denom = gm_mask.sum().clamp(min=1.0)
            wm_denom = wm_mask.sum().clamp(min=1.0)
            d_mse_wm = ((D_hat[b] - D_true[b]).pow(2) * wm_mask).sum() / wm_denom
            d_mse_gm = ((D_hat[b] - D_true[b]).pow(2) * gm_mask).sum() / gm_denom
            r_mse_wm = ((rho_hat[b] - rho_true[b]).pow(2) * wm_mask).sum() / wm_denom
            r_mse_gm = ((rho_hat[b] - rho_true[b]).pow(2) * gm_mask).sum() / gm_denom

            flat_idx = int(seed_logit_mean[b, 0].argmax().item())
            shape = seed_logit_mean[b, 0].shape
            d_idx = flat_idx // (shape[1] * shape[2])
            h_idx = (flat_idx // shape[2]) % shape[1]
            w_idx = flat_idx % shape[2]
            pred_xyz = torch.tensor(
                [d_idx, h_idx, w_idx], dtype=torch.float32, device=device,
            )
            seed_dist = (pred_xyz - seed_xyz_true[b]).pow(2).sum().sqrt()

            pred_mask = (c_forecast[b] > dice_threshold).float()
            true_mask = (c_future[b] > dice_threshold).float()
            inter = (pred_mask * true_mask).sum()
            denom_dice = pred_mask.sum() + true_mask.sum()
            if denom_dice.item() < 1.0:
                dice = 1.0
            else:
                dice = (2.0 * inter / denom_dice).item()

            totals["D_mse"] += float(d_mse.item())
            totals["rho_mse"] += float(r_mse.item())
            totals["D_mse_wm"] += float(d_mse_wm.item())
            totals["D_mse_gm"] += float(d_mse_gm.item())
            totals["rho_mse_wm"] += float(r_mse_wm.item())
            totals["rho_mse_gm"] += float(r_mse_gm.item())
            totals["seed_xyz_dist"] += float(seed_dist.item())
            totals["c_forecast_dice"] += float(dice)
            count += 1

    if count == 0:
        return {k: 0.0 for k in totals}
    return {k: v / count for k, v in totals.items()}
