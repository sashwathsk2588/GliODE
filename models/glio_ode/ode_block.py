import torch
from torch import nn
from torchdiffeq import odeint

from networks.models.glio_ode.vit_encoder import ViTBlock


class ODEFunc(nn.Module):
    """Wraps a single ViTBlock as the RHS of an ODE.

    torchdiffeq.odeint passes only (s, x) to the func, so we cache the
    timestep conditioning and the RoPE cache via set_condition before each call.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_ratio: float,
        d_t: int,
        rope_dims,
        attn_backend: str = "auto",
    ):
        super().__init__()
        # +1 dim on tau for the s-embedding the trunk appends.
        self.block = ViTBlock(
            d_model=d_model, num_heads=num_heads, mlp_ratio=mlp_ratio,
            d_t=d_t + 1, rope_dims=rope_dims, attn_backend=attn_backend,
        )
        self._tau = None
        self._rope_cache = None

    def set_condition(self, tau, rope_cache) -> None:
        self._tau = tau
        self._rope_cache = rope_cache

    def forward(self, s, x):
        assert self._tau is not None, "set_condition must be called before integration"
        assert self._rope_cache is not None, "set_condition must include rope_cache"
        B = self._tau.shape[0]
        s_emb = s.float().reshape(1, 1).expand(B, 1)
        tau_s = torch.cat([self._tau, s_emb], dim=-1)
        return self.block(x, tau_s, self._rope_cache)


class ODEViTTrunk(nn.Module):
    """Integrates one ViTBlock over s in [0, t_ode] and returns 4 checkpoints."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_ratio: float,
        d_t: int,
        rope_dims,
        attn_backend: str = "auto",
        t_ode: float = 1.0,
        n_steps: int = 8,
        method: str = "rk4",
    ):
        super().__init__()
        self.func = ODEFunc(
            d_model=d_model, num_heads=num_heads, mlp_ratio=mlp_ratio,
            d_t=d_t, rope_dims=rope_dims, attn_backend=attn_backend,
        )
        self.t_ode = t_ode
        self.n_steps = n_steps
        self.method = method
        s_grid = torch.linspace(0.0, t_ode, steps=4)
        self.register_buffer("s_grid", s_grid)

    def forward(self, x, tau, rope_cache):
        if self.method == "rk4":
            options = {"step_size": self.t_ode / max(self.n_steps, 1)}
        else:
            options = None
        with torch.amp.autocast("cuda", enabled=False):
            x_fp32 = x.float()
            self.func.set_condition(tau.float(), rope_cache)
            states = odeint(
                self.func, x_fp32, self.s_grid,
                method=self.method, options=options,
            )
        return [states[i].to(x.dtype) for i in range(states.shape[0])]
