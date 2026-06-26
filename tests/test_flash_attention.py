import pytest
import torch

from networks.models.glio_ode.flash_attention import default_rope_dims


def test_default_rope_dims_64_head():
    """Production-size head: anisotropic depth-favoring split."""
    assert default_rope_dims(64) == (16, 24, 24)


def test_default_rope_dims_48_head():
    """Alternate head: anisotropic still valid."""
    assert default_rope_dims(48) == (12, 18, 18)


def test_default_rope_dims_6_head_tiny():
    """Tiny test config head: equal split fallback."""
    assert default_rope_dims(6) == (2, 2, 2)


def test_default_rope_dims_72_head_falls_back_to_equal():
    """Anisotropic split has odd h/w; fallback to equal."""
    assert default_rope_dims(72) == (24, 24, 24)


def test_default_rope_dims_8_head_brute_force():
    """No equal split; brute force lands on (2, 2, 4)."""
    assert default_rope_dims(8) == (2, 2, 4)


def test_default_rope_dims_raises_on_too_small():
    """head_dim < 6 cannot host 3 even chunks each >= 2."""
    with pytest.raises(ValueError, match="too small"):
        default_rope_dims(4)


def test_default_rope_dims_raises_on_unsplittable():
    """head_dim = 5 is odd; no valid even-split sum."""
    with pytest.raises(ValueError):
        default_rope_dims(5)


from networks.models.glio_ode.flash_attention import (
    Rope3DCache,
    build_rope_3d_cache,
)


def test_rope3d_cache_shapes():
    cache = build_rope_3d_cache(
        patch_grid=(4, 5, 6), rope_dims=(2, 4, 6), device=torch.device("cpu"),
    )
    # Half-dim cos/sin tables (formulation: x1 = x[..., :d/2], rotated, then concat).
    assert cache.cos_d.shape == (4, 1)
    assert cache.sin_d.shape == (4, 1)
    assert cache.cos_h.shape == (5, 2)
    assert cache.sin_h.shape == (5, 2)
    assert cache.cos_w.shape == (6, 3)
    assert cache.sin_w.shape == (6, 3)
    # Coordinate arrays of shape (N,) = (4*5*6,) = (120,).
    assert cache.coord_d.shape == (120,)
    assert cache.coord_h.shape == (120,)
    assert cache.coord_w.shape == (120,)


def test_rope3d_cache_coords_row_major_d():
    """Token index n -> (d, h, w) using D-major flattening (matches PatchEmbed3D)."""
    cache = build_rope_3d_cache(
        patch_grid=(2, 3, 4), rope_dims=(2, 2, 2), device=torch.device("cpu"),
    )
    # n = d * (H*W) + h * W + w, with H=3, W=4
    # token 0: (0, 0, 0); token 1: (0, 0, 1); token 4: (0, 1, 0); token 12: (1, 0, 0)
    assert int(cache.coord_d[0]) == 0 and int(cache.coord_h[0]) == 0 and int(cache.coord_w[0]) == 0
    assert int(cache.coord_d[1]) == 0 and int(cache.coord_h[1]) == 0 and int(cache.coord_w[1]) == 1
    assert int(cache.coord_d[4]) == 0 and int(cache.coord_h[4]) == 1 and int(cache.coord_w[4]) == 0
    assert int(cache.coord_d[12]) == 1 and int(cache.coord_h[12]) == 0 and int(cache.coord_w[12]) == 0


def test_rope3d_cache_origin_is_identity():
    """At position (0, 0, 0): cos=1, sin=0 everywhere."""
    cache = build_rope_3d_cache(
        patch_grid=(4, 4, 4), rope_dims=(4, 4, 4), device=torch.device("cpu"),
    )
    assert torch.allclose(cache.cos_d[0], torch.ones_like(cache.cos_d[0]))
    assert torch.allclose(cache.sin_d[0], torch.zeros_like(cache.sin_d[0]))
    assert torch.allclose(cache.cos_h[0], torch.ones_like(cache.cos_h[0]))
    assert torch.allclose(cache.sin_h[0], torch.zeros_like(cache.sin_h[0]))
    assert torch.allclose(cache.cos_w[0], torch.ones_like(cache.cos_w[0]))
    assert torch.allclose(cache.sin_w[0], torch.zeros_like(cache.sin_w[0]))


def test_rope3d_cache_rejects_odd_rope_dim():
    with pytest.raises(ValueError, match="even"):
        build_rope_3d_cache(
            patch_grid=(4, 4, 4), rope_dims=(2, 3, 2), device=torch.device("cpu"),
        )


from networks.models.glio_ode.flash_attention import apply_rope_3d


def test_apply_rope_3d_preserves_shape():
    cache = build_rope_3d_cache(
        patch_grid=(2, 3, 4), rope_dims=(2, 2, 2), device=torch.device("cpu"),
    )
    q = torch.randn(2, 24, 4, 6)  # (B, N, num_heads, head_dim)
    k = torch.randn(2, 24, 4, 6)
    q_rot, k_rot = apply_rope_3d(q, k, cache, rope_dims=(2, 2, 2))
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape


def test_apply_rope_3d_identity_at_origin_token():
    """Token 0 sits at (0, 0, 0); rotation should be identity (cos=1, sin=0)."""
    cache = build_rope_3d_cache(
        patch_grid=(4, 4, 4), rope_dims=(2, 2, 2), device=torch.device("cpu"),
    )
    q = torch.randn(1, 64, 2, 6)
    k = torch.randn(1, 64, 2, 6)
    q_rot, k_rot = apply_rope_3d(q, k, cache, rope_dims=(2, 2, 2))
    assert torch.allclose(q_rot[:, 0], q[:, 0], atol=1e-6)
    assert torch.allclose(k_rot[:, 0], k[:, 0], atol=1e-6)


def test_apply_rope_3d_preserves_norm():
    """RoPE is unitary; per-token L2 norm of q must match before/after."""
    cache = build_rope_3d_cache(
        patch_grid=(2, 3, 4), rope_dims=(2, 4, 6), device=torch.device("cpu"),
    )
    q = torch.randn(2, 24, 2, 12)
    k = torch.randn(2, 24, 2, 12)
    q_rot, _ = apply_rope_3d(q, k, cache, rope_dims=(2, 4, 6))
    norms_before = q.pow(2).sum(dim=-1).sqrt()
    norms_after = q_rot.pow(2).sum(dim=-1).sqrt()
    assert torch.allclose(norms_before, norms_after, atol=1e-5)


from networks.models.glio_ode.flash_attention import _resolve_backend


def test_resolve_backend_sdpa_always_works():
    assert _resolve_backend("sdpa") == "sdpa"


def test_resolve_backend_auto_falls_back_to_sdpa_on_cpu():
    """In the test env (CPU, possibly no flash_attn), 'auto' must yield 'sdpa'."""
    backend = _resolve_backend("auto")
    assert backend in ("flash", "sdpa")
    if not torch.cuda.is_available():
        assert backend == "sdpa"


def test_resolve_backend_flash_raises_if_unavailable(monkeypatch):
    """Explicit 'flash' without the package raises ImportError."""
    import sys
    monkeypatch.setitem(sys.modules, "flash_attn", None)
    with pytest.raises(ImportError, match="flash_attn"):
        _resolve_backend("flash")


def test_resolve_backend_rejects_unknown():
    with pytest.raises(ValueError, match="unknown"):
        _resolve_backend("not_a_backend")


from networks.models.glio_ode.flash_attention import MultiHeadSelfAttention3D


def test_mhsa3d_output_shape():
    attn = MultiHeadSelfAttention3D(
        d_model=12, num_heads=2, rope_dims=(2, 2, 2), attn_backend="sdpa",
    )
    cache = build_rope_3d_cache(
        patch_grid=(4, 4, 4), rope_dims=(2, 2, 2), device=torch.device("cpu"),
    )
    x = torch.randn(2, 64, 12)
    y = attn(x, cache)
    assert y.shape == (2, 64, 12)


def test_mhsa3d_sdpa_backend_runs_on_cpu():
    """SDPA path must work on CPU without flash_attn."""
    attn = MultiHeadSelfAttention3D(
        d_model=12, num_heads=2, rope_dims=(2, 2, 2), attn_backend="sdpa",
    )
    cache = build_rope_3d_cache(
        patch_grid=(2, 2, 2), rope_dims=(2, 2, 2), device=torch.device("cpu"),
    )
    x = torch.randn(1, 8, 12)
    y = attn(x, cache)
    assert torch.isfinite(y).all()


def test_mhsa3d_qk_rms_norm():
    """After F.rms_norm, per-token q vectors should have RMS ~ 1."""
    attn = MultiHeadSelfAttention3D(
        d_model=12, num_heads=2, rope_dims=(2, 2, 2), attn_backend="sdpa",
    )
    # Probe internals by running the QK projection and rms_norm manually.
    x = torch.randn(1, 8, 12)
    q_raw = attn.c_q(x).view(1, 8, 2, 6)
    q_normed = torch.nn.functional.rms_norm(q_raw, (6,))
    rms_per_token = q_normed.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms_per_token, torch.ones_like(rms_per_token), atol=1e-5)


def test_mhsa3d_gradients_flow():
    attn = MultiHeadSelfAttention3D(
        d_model=12, num_heads=2, rope_dims=(2, 2, 2), attn_backend="sdpa",
    )
    cache = build_rope_3d_cache(
        patch_grid=(2, 2, 2), rope_dims=(2, 2, 2), device=torch.device("cpu"),
    )
    x = torch.randn(1, 8, 12, requires_grad=True)
    y = attn(x, cache)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for name, p in attn.named_parameters():
        assert p.grad is not None, f"no grad on {name}"
        assert torch.isfinite(p.grad).all()
