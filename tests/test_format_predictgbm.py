"""Smoke tests for scripts/format_predictgbm.py.

Writes a fake PredictGBM tree, runs the formatter under three timepoint
policies (baseline / all / exact), and asserts the output matches the
GliomaSolver layout that `_GliomaSolverCollator` expects.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "format_predictgbm.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("format_predictgbm", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec_module so @dataclass at module scope can introspect.
    sys.modules["format_predictgbm"] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_fake_predictgbm(
    root: Path, n_patients: int = 2, n_timepoints: int = 3, shape=(8, 8, 8),
) -> None:
    """Layout: <root>/patient_XXX/tY/{T1c,GM,WM,CSF}.nii.gz."""
    affine = np.eye(4)
    for p in range(n_patients):
        for t in range(n_timepoints):
            tp_dir = root / f"patient_{p:03d}" / f"t{t}"
            tp_dir.mkdir(parents=True, exist_ok=True)
            rng = np.random.default_rng(p * 10 + t)
            nib.save(
                nib.Nifti1Image(rng.standard_normal(shape).astype(np.float32), affine),
                str(tp_dir / "T1c.nii.gz"),
            )
            triplet = rng.random((3, *shape)).astype(np.float32)
            triplet /= triplet.sum(axis=0, keepdims=True)
            nib.save(nib.Nifti1Image(triplet[0], affine), str(tp_dir / "GM.nii.gz"))
            nib.save(nib.Nifti1Image(triplet[1], affine), str(tp_dir / "WM.nii.gz"))
            nib.save(nib.Nifti1Image(triplet[2], affine), str(tp_dir / "CSF.nii.gz"))


def test_format_predictgbm_baseline_one_case_per_patient(tmp_path):
    src = tmp_path / "predictgbm"
    dst = tmp_path / "gliomasolver"
    _write_fake_predictgbm(src, n_patients=2, n_timepoints=3)

    mod = _load_script_module()
    rc = mod.main([
        "--src", str(src),
        "--dst", str(dst),
        "--timepoint", "baseline",
    ])
    assert rc == 0

    case_dirs = sorted(d for d in dst.iterdir() if d.is_dir())
    assert [d.name for d in case_dirs] == ["patient_000", "patient_001"]
    for cd in case_dirs:
        for name in ("t1gd.nii.gz", "gm.nii.gz", "wm.nii.gz", "csf.nii.gz"):
            path = cd / name
            assert path.exists(), f"missing {name} in {cd}"
            arr = nib.load(str(path)).get_fdata()
            assert arr.shape == (8, 8, 8)

    manifest = json.loads((dst / "manifest.json").read_text())
    assert manifest["timepoint_policy"] == "baseline"
    assert len(manifest["cases"]) == 2
    # Each case's source dir should be the t0 (earliest) timepoint.
    for entry in manifest["cases"]:
        assert entry["source_dir"].endswith("/t0")


def test_format_predictgbm_all_emits_one_per_timepoint(tmp_path):
    src = tmp_path / "predictgbm"
    dst = tmp_path / "gliomasolver"
    _write_fake_predictgbm(src, n_patients=2, n_timepoints=3)

    mod = _load_script_module()
    rc = mod.main([
        "--src", str(src),
        "--dst", str(dst),
        "--timepoint", "all",
    ])
    assert rc == 0

    case_dirs = sorted(d for d in dst.iterdir() if d.is_dir())
    assert len(case_dirs) == 6  # 2 patients x 3 timepoints
    # Case names follow {patient}_{timepoint}.
    expected = sorted(
        f"patient_{p:03d}_t{t}" for p in (0, 1) for t in (0, 1, 2)
    )
    assert [d.name for d in case_dirs] == expected


def test_format_predictgbm_exact_timepoint(tmp_path):
    src = tmp_path / "predictgbm"
    dst = tmp_path / "gliomasolver"
    _write_fake_predictgbm(src, n_patients=2, n_timepoints=3)

    mod = _load_script_module()
    rc = mod.main([
        "--src", str(src),
        "--dst", str(dst),
        "--timepoint", "t1",  # middle timepoint
    ])
    assert rc == 0

    case_dirs = sorted(d for d in dst.iterdir() if d.is_dir())
    assert [d.name for d in case_dirs] == ["patient_000", "patient_001"]
    manifest = json.loads((dst / "manifest.json").read_text())
    for entry in manifest["cases"]:
        assert entry["source_dir"].endswith("/t1")


def test_format_predictgbm_dry_run_writes_nothing(tmp_path):
    src = tmp_path / "predictgbm"
    dst = tmp_path / "gliomasolver"
    _write_fake_predictgbm(src, n_patients=1, n_timepoints=1)

    mod = _load_script_module()
    rc = mod.main([
        "--src", str(src),
        "--dst", str(dst),
        "--dry_run",
    ])
    assert rc == 0
    assert not dst.exists() or not any(dst.iterdir())


def test_format_predictgbm_missing_modality_skips_case(tmp_path, caplog):
    src = tmp_path / "predictgbm"
    dst = tmp_path / "gliomasolver"
    _write_fake_predictgbm(src, n_patients=1, n_timepoints=1)
    # Corrupt: remove CSF from the only case.
    (src / "patient_000" / "t0" / "CSF.nii.gz").unlink()

    mod = _load_script_module()
    rc = mod.main([
        "--src", str(src),
        "--dst", str(dst),
    ])
    # No cases produced -> exit code 2.
    assert rc == 2
    assert not dst.exists() or not any(p.is_dir() for p in dst.iterdir())


def test_format_predictgbm_custom_globs(tmp_path):
    """Real PredictGBM files might be named e.g. brats_T1ce_brain.nii.gz —
    confirm we can override the default patterns."""
    src = tmp_path / "predictgbm"
    src.mkdir()
    tp_dir = src / "PG_001" / "preop"
    tp_dir.mkdir(parents=True)
    affine = np.eye(4)
    rng = np.random.default_rng(0)
    nib.save(
        nib.Nifti1Image(rng.standard_normal((8, 8, 8)).astype(np.float32), affine),
        str(tp_dir / "brats_T1ce_brain.nii.gz"),
    )
    nib.save(nib.Nifti1Image(rng.random((8, 8, 8)).astype(np.float32), affine),
             str(tp_dir / "atlas_GM_prob.nii.gz"))
    nib.save(nib.Nifti1Image(rng.random((8, 8, 8)).astype(np.float32), affine),
             str(tp_dir / "atlas_WM_prob.nii.gz"))
    nib.save(nib.Nifti1Image(rng.random((8, 8, 8)).astype(np.float32), affine),
             str(tp_dir / "atlas_CSF_prob.nii.gz"))

    dst = tmp_path / "gliomasolver"
    mod = _load_script_module()
    rc = mod.main([
        "--src", str(src),
        "--dst", str(dst),
        "--timepoint", "preop",
        "--t1gd_glob", "*T1ce*.nii.gz",
        "--gm_glob", "*GM_prob.nii.gz",
        "--wm_glob", "*WM_prob.nii.gz",
        "--csf_glob", "*CSF_prob.nii.gz",
    ])
    assert rc == 0
    case_dir = dst / "PG_001"
    for name in ("t1gd.nii.gz", "gm.nii.gz", "wm.nii.gz", "csf.nii.gz"):
        assert (case_dir / name).exists()
