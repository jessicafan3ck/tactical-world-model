"""
model/learned_action_encoder.py
================================
Two learned action-encoder architectures trained to predict real Δz from
StatsBomb 360-frame transitions.

PerActionAffine
    z' = z + W_a @ z + b_a
    Per-action linear transform. State-dependent (output scales with z).
    Low parameter count, interpretable, trains well on small data.
    L2 loss on Δz.

ConditionedMLP
    Δz ~ N(μ_θ(z, a, ctx), diag(exp(lv_θ(z, a, ctx))))
    Full state + context dependence. Gaussian NLL loss forces the model to
    be honest about actions whose Δz is noisy (HOLD, PRESS, SWITCH_*).
    Uses LayerNorm + GELU + Dropout; regularised for the ~3 k training pairs.

Both expose:
    predict_dz(z, action_idx, ...)  →  Δz mean  (B, Z_DIM)
    apply(z, action_idx, ...)       →  z + Δz   (B, Z_DIM)

The ConditionedMLP additionally exposes:
    predict_dz_with_std(...)        →  (mean, std) for uncertainty display
    nll_loss(...)                   →  scalar training loss
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

Z_DIM     = 256
N_ACTIONS = 11
CTX_DIM   = 3     # [zone/3, phase_open, phase_counter]


# ── Per-action affine ─────────────────────────────────────────────────────────

class PerActionAffine(nn.Module):
    """
    State-dependent per-action linear transform.
    Each action has its own (W_a, b_a); the transform is:
        Δz = W_a @ z + b_a
        z' = z + Δz
    W_a is initialised to zero so the untrained model is the identity.
    """

    def __init__(self, z_dim: int = Z_DIM, n_actions: int = N_ACTIONS):
        super().__init__()
        self.z_dim     = z_dim
        self.n_actions = n_actions
        # W: (n_actions, z_dim, z_dim)  — initialise near zero, not exact zero,
        # to break symmetry during training
        self.W = nn.Parameter(torch.randn(n_actions, z_dim, z_dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(n_actions, z_dim))

    def predict_dz(self, z: torch.Tensor, action_idx: torch.Tensor,
                   ctx: torch.Tensor | None = None) -> torch.Tensor:
        """
        z:          (B, z_dim)
        action_idx: (B,) long
        ctx:        ignored (kept for API compatibility with ConditionedMLP)
        returns:    Δz (B, z_dim)
        """
        W  = self.W[action_idx]                              # (B, z_dim, z_dim)
        b  = self.b[action_idx]                              # (B, z_dim)
        return torch.bmm(W, z.unsqueeze(-1)).squeeze(-1) + b  # (B, z_dim)

    def forward(self, z: torch.Tensor, action_idx: torch.Tensor,
                ctx: torch.Tensor | None = None) -> torch.Tensor:
        return z + self.predict_dz(z, action_idx, ctx)

    def apply(self, z: torch.Tensor, action_idx: torch.Tensor,
              ctx: torch.Tensor | None = None, sample: bool = False) -> torch.Tensor:
        return self.forward(z, action_idx, ctx)

    def l2_loss(self, z: torch.Tensor, action_idx: torch.Tensor,
                ctx: torch.Tensor, dz_target: torch.Tensor) -> torch.Tensor:
        dz_pred = self.predict_dz(z, action_idx, ctx)
        return F.mse_loss(dz_pred, dz_target)


# ── Conditioned MLP with Gaussian NLL ─────────────────────────────────────────

class ConditionedMLP(nn.Module):
    """
    State-and-context-conditioned MLP with Gaussian NLL loss.

    Input:   [z (256) | onehot(action) (11) | ctx (3)]  → 270 dims
    Output:  (Δz_mean, Δz_log_var) both (B, z_dim)

    The log-variance head forces the model to represent uncertainty honestly
    rather than fitting noisy targets with a flat MSE loss.  Actions with
    high irreducible variance (HOLD, PRESS, SWITCH_*) will learn wide
    distributions; deterministic actions (SHOOT, CROSS) will be narrow.
    """

    def __init__(
        self,
        z_dim:     int = Z_DIM,
        n_actions: int = N_ACTIONS,
        ctx_dim:   int = CTX_DIM,
        hidden:    int = 512,
        dropout:   float = 0.3,
    ):
        super().__init__()
        self.z_dim     = z_dim
        self.n_actions = n_actions
        self.ctx_dim   = ctx_dim
        in_dim = z_dim + n_actions + ctx_dim

        def _block(d_in: int, d_out: int) -> list[nn.Module]:
            return [nn.Linear(d_in, d_out), nn.LayerNorm(d_out),
                    nn.GELU(), nn.Dropout(dropout)]

        self.net = nn.Sequential(
            *_block(in_dim, hidden),
            *_block(hidden, hidden),
            nn.Linear(hidden, z_dim * 2),   # mean || log_var
        )

    def _input(self, z: torch.Tensor, action_idx: torch.Tensor,
               ctx: torch.Tensor) -> torch.Tensor:
        a_hot = F.one_hot(action_idx, self.n_actions).float()
        return torch.cat([z, a_hot, ctx], dim=-1)   # (B, in_dim)

    def forward(self, z: torch.Tensor, action_idx: torch.Tensor,
                ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (dz_mean, dz_log_var) both (B, z_dim)."""
        out            = self.net(self._input(z, action_idx, ctx))
        mu, log_var    = out.chunk(2, dim=-1)
        log_var        = log_var.clamp(-6, 4)   # keep variance in [e^-6, e^4]
        return mu, log_var

    def predict_dz(self, z: torch.Tensor, action_idx: torch.Tensor,
                   ctx: torch.Tensor | None = None,
                   sample: bool = False) -> torch.Tensor:
        if ctx is None:
            ctx = torch.zeros(z.shape[0], self.ctx_dim, device=z.device)
        mu, log_var = self(z, action_idx, ctx)
        if sample:
            return mu + (0.5 * log_var).exp() * torch.randn_like(mu)
        return mu

    def predict_dz_with_std(self, z: torch.Tensor, action_idx: torch.Tensor,
                             ctx: torch.Tensor | None = None
                             ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mean, std) for uncertainty-aware simulation."""
        if ctx is None:
            ctx = torch.zeros(z.shape[0], self.ctx_dim, device=z.device)
        mu, log_var = self(z, action_idx, ctx)
        return mu, (0.5 * log_var).exp()

    def apply(self, z: torch.Tensor, action_idx: torch.Tensor,
              ctx: torch.Tensor | None = None, sample: bool = False) -> torch.Tensor:
        dz = self.predict_dz(z, action_idx, ctx, sample=sample)
        return z + dz

    def nll_loss(self, z: torch.Tensor, action_idx: torch.Tensor,
                 ctx: torch.Tensor, dz_target: torch.Tensor) -> torch.Tensor:
        mu, log_var = self(z, action_idx, ctx)
        # Gaussian NLL = 0.5 * (log_var + (target - mean)² / var)
        return 0.5 * (log_var + (dz_target - mu).pow(2) / log_var.exp()).mean()
