from pathlib import Path
import numpy as np
import nibabel as nib
import torch

from networks.data.brats import (
    build_brats_transforms,
    build_brats_datasets,
    build_brats_dataloaders,
    _write_synthetic_brats_case,
)


def _write_n_synthetic_cases(root: Path, n: int, shape=(32, 32, 32)) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        case_dir = root / f"case_{i:03d}"
        _write_synthetic_brats_case(case_dir, shape=shape)


def test_brats_dataset_loads_synthetic_volume(tmp_path):
    root = tmp_path / "brats"
    _write_n_synthetic_cases(root, n=2, shape=(32, 32, 32))

    train_tf, val_tf = build_brats_transforms(
        crop_size=(16, 16, 16), num_classes=4,
    )
    train_ds, val_ds = build_brats_datasets(
        str(root), train_tf, val_tf, val_fraction=0.5, seed=0,
    )
    assert len(train_ds) == 1
    assert len(val_ds) == 1

    sample = train_ds[0]
    img = sample["image"]
    msk = sample["label"]
    # 4 modalities × (16, 16, 16) after crop.
    assert isinstance(img, torch.Tensor)
    assert img.shape == (4, 16, 16, 16)
    # One-hot mask with 4 classes.
    assert msk.shape == (4, 16, 16, 16)
    # Exactly one channel is 1 per voxel.
    onehot_sum = msk.sum(dim=0)
    assert torch.allclose(onehot_sum, torch.ones_like(onehot_sum))


def test_brats_label_remap_4_to_3(tmp_path):
    """Labels in {0,1,2,4} must become contiguous {0,1,2,3} after the remap."""
    root = tmp_path / "brats"
    _write_n_synthetic_cases(root, n=1, shape=(32, 32, 32))

    _, val_tf = build_brats_transforms(
        crop_size=(16, 16, 16), num_classes=4, remap_label_4_to_3=True,
    )
    _, val_ds = build_brats_datasets(
        str(root), val_tf, val_tf, val_fraction=1.0, seed=0,
    )
    sample = val_ds[0]
    msk = sample["label"]
    # Recover non-one-hot indices via argmax.
    indices = msk.argmax(dim=0)
    assert int(indices.max().item()) <= 3, "labels must be in [0, 3] after remap"


def test_brats_dataloaders_yield_batches(tmp_path):
    root = tmp_path / "brats"
    _write_n_synthetic_cases(root, n=4, shape=(32, 32, 32))

    train_tf, val_tf = build_brats_transforms(
        crop_size=(16, 16, 16), num_classes=4,
    )
    train_ds, val_ds = build_brats_datasets(
        str(root), train_tf, val_tf, val_fraction=0.5, seed=0,
    )
    train_loader, val_loader = build_brats_dataloaders(
        train_ds, val_ds, batch_size=2, num_workers=0,
    )
    batch = next(iter(train_loader))
    assert batch["image"].shape == (2, 4, 16, 16, 16)
    assert batch["label"].shape == (2, 4, 16, 16, 16)
