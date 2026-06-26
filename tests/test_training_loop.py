import torch
from torch import nn

from networks.training.loop import EMA


def test_ema_init_copies_state():
    model = nn.Linear(4, 4)
    ema = EMA(model, decay=0.9)
    for name, p in model.named_parameters():
        assert torch.allclose(ema.shadow[name], p.detach())


def test_ema_update_moves_toward_param():
    model = nn.Linear(4, 4)
    ema = EMA(model, decay=0.9)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)  # bump every param by 1
    ema.update(model)
    # shadow := decay * shadow + (1 - decay) * param
    # delta vs the original shadow (which equaled param - 1.0) should be (1 - decay) * 1.0 = 0.1
    for name, p in model.named_parameters():
        expected = (p.detach() - 1.0) * 0.9 + p.detach() * 0.1
        assert torch.allclose(ema.shadow[name], expected, atol=1e-6)


def test_ema_copy_to_swaps_weights():
    model = nn.Linear(4, 4)
    ema = EMA(model, decay=0.9)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(2.0)
    raw_state = {n: p.detach().clone() for n, p in model.named_parameters()}
    ema.copy_to(model)
    for name, p in model.named_parameters():
        assert torch.allclose(p.detach(), ema.shadow[name])
        assert not torch.allclose(p.detach(), raw_state[name])
    # Now restore
    ema.copy_from = lambda mdl: None  # only testing copy_to here


from networks.models.glio_ode.glio_ode import GlioODE
from networks.models.glio_ode.diffusion import GlioDiffusion
from networks.training.loop import train_one_iter


def _tiny_setup():
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
    diff = GlioDiffusion(model, image_channels=2, mask_channels=2, timesteps=10)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ema = EMA(model, decay=0.9)
    return model, diff, optim, ema


def test_train_one_iter_returns_metrics_dict():
    model, diff, optim, ema = _tiny_setup()
    batch = {
        "image": torch.randn(2, 2, 8, 16, 16),
        "label": torch.randn(2, 2, 8, 16, 16),
    }
    metrics = train_one_iter(diff, optim, batch, ema, amp_dtype=None, grad_clip=1.0)
    for key in ("loss", "img_mse", "mask_mse", "grad_norm"):
        assert key in metrics
        assert isinstance(metrics[key], float)
        assert metrics[key] == metrics[key]  # not NaN
    assert metrics["img_mse"] > 0.0
    assert metrics["mask_mse"] > 0.0


def test_train_one_iter_moves_ema_toward_params():
    model, diff, optim, ema = _tiny_setup()
    batch = {
        "image": torch.randn(2, 2, 8, 16, 16),
        "label": torch.randn(2, 2, 8, 16, 16),
    }
    pre_shadow = {n: t.clone() for n, t in ema.shadow.items()}
    train_one_iter(diff, optim, batch, ema, amp_dtype=None, grad_clip=1.0)
    diffs = [
        not torch.allclose(pre_shadow[n], ema.shadow[n])
        for n in ema.shadow
    ]
    assert any(diffs), "EMA shadow did not advance after train_one_iter"


from torch.utils.data import DataLoader as TorchDataLoader

from networks.training.loop import validate


def test_validate_returns_per_class_dice_dict():
    model, diff, _, _ = _tiny_setup()
    model.attach_diffusion(diff)
    model._eval_num_steps = 2
    model.eval()

    # A tiny one-batch val loader yielding a single dict.
    class _OneShotDataset:
        def __len__(self):
            return 1
        def __getitem__(self, idx):
            return {
                "image": torch.randn(2, 8, 16, 16),
                "label": torch.eye(2)[torch.zeros(8 * 16 * 16, dtype=torch.long)]
                    .view(8, 16, 16, 2).permute(3, 0, 1, 2).float(),
            }

    loader = TorchDataLoader(_OneShotDataset(), batch_size=1)
    metrics = validate(model, loader, num_classes=2)
    assert "dice_mean" in metrics
    for c in range(2):
        assert f"dice_class_{c}" in metrics
    for v in metrics.values():
        assert isinstance(v, float)
        assert 0.0 <= v <= 1.0


def test_checkpoint_round_trip(tmp_path):
    model, diff, optim, ema = _tiny_setup()
    from networks.training.loop import save_checkpoint, load_checkpoint

    ckpt_path = tmp_path / "test.pt"
    save_checkpoint(ckpt_path, model, diff, ema, optim, iter_idx=42, best_val_dice=0.7)
    assert ckpt_path.exists()

    # Build a fresh setup with different initial weights, then load.
    model2, diff2, optim2, ema2 = _tiny_setup()
    iter_idx, best_val = load_checkpoint(ckpt_path, model2, diff2, ema2, optim2)
    assert iter_idx == 42
    assert abs(best_val - 0.7) < 1e-6
    # Loaded model params should equal the saved model params.
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
        assert n1 == n2
        assert torch.allclose(p1.detach(), p2.detach())
    # EMA shadow restored too.
    for n in ema.shadow:
        assert torch.allclose(ema.shadow[n], ema2.shadow[n])


def test_checkpoint_round_trip_scheduler(tmp_path):
    """Scheduler state must round-trip: a resumed scheduler.step() should match
    the saved scheduler's next step LR."""
    model, diff, optim, ema = _tiny_setup()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=100)
    # Advance the scheduler a few steps so it has non-default state.
    for _ in range(7):
        scheduler.step()
    lr_before_save = optim.param_groups[0]["lr"]

    from networks.training.loop import save_checkpoint, load_checkpoint
    ckpt_path = tmp_path / "test.pt"
    save_checkpoint(ckpt_path, model, diff, ema, optim, iter_idx=7,
                    best_val_dice=0.0, scheduler=scheduler)

    # Fresh setup; load the checkpoint.
    model2, diff2, optim2, ema2 = _tiny_setup()
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optim2, T_max=100)
    iter_idx, _ = load_checkpoint(ckpt_path, model2, diff2, ema2, optim2,
                                  scheduler=scheduler2)
    assert iter_idx == 7
    # LR should match after restore.
    assert abs(optim2.param_groups[0]["lr"] - lr_before_save) < 1e-9
