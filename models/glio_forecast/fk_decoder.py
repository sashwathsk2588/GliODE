"""Differentiable Fisher-Kolmogorov Neural-ODE decoder.

Integrates dc/ds = nabla.(D(x) nabla c) + rho(x) c (1-c) + f_theta(c, anatomy, s)
from s=0 to s=s_obs. Anisotropic Laplacian via finite differences with
face-centered D averaging. Zero-flux (Neumann) boundary. CSF mask kills RHS
in CSF. Optional learned residual via a small 3D CNN.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchdiffeq import odeint


def _axis_anisotropic_laplacian(c, D, h, axis):
    """Compute nabla.(D nabla c) along one spatial axis using face-centered D
    averaging and replicate (Neumann) padding.

    Args:
        c, D: (B, 1, D, H, W) tensors.
        h: voxel size along this axis.
        axis: 0, 1, or 2 for D, H, W respectively.
    """
    tdim = axis + 2
    # F.pad expects (W_l, W_r, H_l, H_r, D_l, D_r) — last axis first.
    pad_spec = [0, 0, 0, 0, 0, 0]
    reverse_axis = 2 - axis  # 0->2, 1->1, 2->0
    pad_spec[reverse_axis * 2] = 1
    pad_spec[reverse_axis * 2 + 1] = 1
    c_pad = F.pad(c, pad_spec, mode="replicate")
    D_pad = F.pad(D, pad_spec, mode="replicate")
    size = c.shape[tdim]
    c_fwd = c_pad.narrow(tdim, 2, size)
    c_ctr = c_pad.narrow(tdim, 1, size)
    c_bwd = c_pad.narrow(tdim, 0, size)
    D_fwd = D_pad.narrow(tdim, 2, size)
    D_ctr = D_pad.narrow(tdim, 1, size)
    D_bwd = D_pad.narrow(tdim, 0, size)
    D_face_pos = 0.5 * (D_ctr + D_fwd)
    D_face_neg = 0.5 * (D_ctr + D_bwd)
    flux = D_face_pos * (c_fwd - c_ctr) - D_face_neg * (c_ctr - c_bwd)
    return flux / (h * h)


def fk_laplacian(c, D, voxel_size):
    """Sum of axis Laplacians: nabla.(D nabla c) in 3D."""
    h_d, h_h, h_w = voxel_size
    return (
        _axis_anisotropic_laplacian(c, D, h_d, axis=0)
        + _axis_anisotropic_laplacian(c, D, h_h, axis=1)
        + _axis_anisotropic_laplacian(c, D, h_w, axis=2)
    )


class FKResidual(nn.Module):
    """Small 3-layer CNN producing an additive residual to dc/ds."""

    def __init__(self, residual_channels: int = 16):
        super().__init__()
        # Input channels: c (1) + anatomy (4) + s_emb (1) = 6
        self.conv1 = nn.Conv3d(6, residual_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(residual_channels, residual_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(residual_channels, 1, kernel_size=3, padding=1)
        self.act = nn.GELU()
        # Zero-init last conv so initial residual is 0.
        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)

    def forward(self, c, anatomy, s_scalar):
        B = c.shape[0]
        # Broadcast scalar s to (B, 1, D, H, W).
        s_emb = s_scalar.float().reshape(1, 1, 1, 1, 1).expand(B, 1, *c.shape[2:])
        x = torch.cat([c, anatomy, s_emb], dim=1)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return self.conv3(x)


class FKDecoder(nn.Module):
    """Differentiable FK PDE integrator.

    Forward signature: (D, rho, seed_map, anatomy, s_obs) -> c(s_obs).
    Per-batch s_obs only (all elements must be identical).
    """

    def __init__(
        self,
        spatial_size,
        voxel_size=(1.0, 1.0, 1.0),
        use_residual: bool = True,
        residual_channels: int = 16,
        ode_method: str = "rk4",
        ode_steps: int = 8,
        seed_max: float = 0.1,
    ):
        super().__init__()
        for s in spatial_size:
            assert s > 0, f"spatial_size entry must be positive, got {s}"
        for v in voxel_size:
            assert v > 0, f"voxel_size entry must be positive, got {v}"
        assert residual_channels > 0
        assert ode_steps >= 1
        assert seed_max > 0
        self.spatial_size = tuple(spatial_size)
        self.voxel_size = tuple(voxel_size)
        self.use_residual = use_residual
        self.ode_method = ode_method
        self.ode_steps = ode_steps
        self.seed_max = seed_max
        if use_residual:
            self.residual = FKResidual(residual_channels)
        else:
            self.residual = None
        # State cached by forward before each odeint call.
        self._D = None
        self._rho = None
        self._anatomy = None
        self._csf = None

    def _rhs(self, s, c):
        lap = fk_laplacian(c, self._D, self.voxel_size)
        react = self._rho * c * (1.0 - c)
        rhs = lap + react
        if self.residual is not None:
            rhs = rhs + self.residual(c, self._anatomy, s)
        rhs = rhs * (1.0 - self._csf)
        return rhs

    def forward(self, D, rho, seed_map, anatomy, s_obs):
        assert D.shape[1] == 1 and rho.shape[1] == 1 and seed_map.shape[1] == 1
        assert anatomy.shape[1] == 4
        B = D.shape[0]
        assert s_obs.shape == (B,), f"s_obs shape {tuple(s_obs.shape)} != (B={B},)"
        if s_obs.unique().numel() != 1:
            raise ValueError(
                f"FKDecoder requires per-batch s_obs (all identical); "
                f"got distinct values {s_obs.unique().tolist()}"
            )
        s_obs_scalar = float(s_obs[0].item())
        if s_obs_scalar <= 0:
            raise ValueError(f"s_obs must be positive, got {s_obs_scalar}")

        # Build CSF mask from anatomy channel 3 (T1Gd, GM, WM, CSF).
        csf = (anatomy[:, 3:4] > 0.5).float()
        # Initial condition: sigmoid(seed_map) * seed_max, zeroed in CSF.
        c0 = torch.sigmoid(seed_map) * self.seed_max * (1.0 - csf)

        # Cache fp32 versions for the RHS.
        self._D = D.float()
        self._rho = rho.float()
        self._anatomy = anatomy.float()
        self._csf = csf.float()

        s_grid = torch.linspace(0.0, s_obs_scalar, self.ode_steps, device=c0.device)
        if self.ode_method == "rk4":
            options = {"step_size": s_obs_scalar / max(self.ode_steps, 1)}
        else:
            options = None

        with torch.amp.autocast("cuda", enabled=False):
            states = odeint(
                self._rhs, c0.float(), s_grid,
                method=self.ode_method, options=options,
            )
        c_final = states[-1].clamp(0.0, 1.0).to(D.dtype)
        return c_final
