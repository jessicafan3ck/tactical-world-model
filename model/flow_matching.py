"""
Conditional Flow Matching — Tactical Freeze Frame Generator
============================================================
Learns to generate realistic 22-player tactical configurations conditioned on:
  - Team fingerprints (z_A, z_B) from the SSE TeamEncoder
  - Game state: score differential, match minute, phase, current zone

Architecture: permutation-equivariant Transformer.
  Given noisy player positions x_t at time t and conditioning signal c,
  predicts the velocity field v_θ(x_t, t, c) for each player.

Training (Conditional Flow Matching):
  Source:  x_0 ~ N(0, σ²I)   shape (B, N_players, 2)
  Target:  x_1 = real freeze frame positions
  Path:    x_t = (1-t)·x_0 + t·x_1       (linear interpolation)
  Target v: u_t = x_1 - x_0
  Loss:    ||v_θ(x_t, t, c) - u_t||²

Inference (Euler ODE integration):
  x_0 ~ N(0, σ²I)
  for i in range(n_steps):
      t = i / n_steps
      x += (1/n_steps) · v_θ(x, t, c)
  return x   ← generated freeze frame positions

References:
  Lipman et al. (2022) "Flow Matching for Generative Modeling"
  Tong et al. (2023)   "Improving and Generalizing Flow Matching"
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Time embedding ─────────────────────────────────────────────────────────────

class SinusoidalTimeEmbedding(nn.Module):
    """Fourier feature embedding for the flow time t ∈ [0, 1]."""

    def __init__(self, d_model: int, max_freq: int = 100):
        super().__init__()
        half = d_model // 2
        freqs = torch.exp(
            -math.log(max_freq) * torch.arange(half) / (half - 1)
        )
        self.register_buffer("freqs", freqs)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t : (B,) → (B, d_model)"""
        t = t.unsqueeze(-1) * self.freqs.unsqueeze(0) * 2 * math.pi
        emb = torch.cat([t.sin(), t.cos()], dim=-1)
        return self.proj(emb)


# ── Conditioning encoder ───────────────────────────────────────────────────────

class ConditionEncoder(nn.Module):
    """
    Encodes the game-state conditioning signal into a fixed-size vector.

    Inputs (concatenated):
      z_A           : (B, fingerprint_dim)  attacking team fingerprint
      z_B           : (B, fingerprint_dim)  defending team fingerprint
      score_diff    : (B, 1)  clipped to [-3, 3] then /3
      minute_norm   : (B, 1)  minute / 90
      phase_onehot  : (B, 4)  [open_play, counter, set_piece, restart]
      zone_onehot   : (B, 4)  [zone 0–3]

    Total input dim = 2·fingerprint_dim + 1 + 1 + 4 + 4
    """

    def __init__(self, fingerprint_dim: int = 256, cond_dim: int = 512):
        super().__init__()
        in_dim = 2 * fingerprint_dim + 1 + 1 + 4 + 4
        self.net = nn.Sequential(
            nn.Linear(in_dim, cond_dim),
            nn.GELU(),
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, cond_dim),
            nn.GELU(),
            nn.LayerNorm(cond_dim),
        )

    def forward(self,
                z_A:          torch.Tensor,
                z_B:          torch.Tensor,
                score_diff:   torch.Tensor,
                minute_norm:  torch.Tensor,
                phase_onehot: torch.Tensor,
                zone_onehot:  torch.Tensor) -> torch.Tensor:
        raw = torch.cat(
            [z_A, z_B, score_diff, minute_norm, phase_onehot, zone_onehot],
            dim=-1
        )
        return self.net(raw)


# ── Velocity field (core model) ────────────────────────────────────────────────

class VelocityField(nn.Module):
    """
    Permutation-equivariant Transformer that predicts the velocity
    dx/dt for each player given noisy positions x_t at time t
    and conditioning signal c.

    Equivariance: permuting the N_players dimension of x_t produces
    the same permuted velocities — correct for an unordered player set.
    """

    def __init__(self,
                 n_players:   int   = 22,
                 player_dim:  int   = 4,    # x, y, is_teammate, is_actor
                 d_model:     int   = 256,
                 nhead:       int   = 8,
                 num_layers:  int   = 4,
                 cond_dim:    int   = 512,
                 dropout:     float = 0.1):
        super().__init__()
        self.n_players = n_players

        # Player token embedding: [x, y, is_teammate, is_actor] → d_model
        self.player_proj = nn.Sequential(
            nn.Linear(player_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # Time embedding
        self.time_emb = SinusoidalTimeEmbedding(d_model)

        # Condition projection (one cross-attention anchor per batch)
        self.cond_proj = nn.Linear(cond_dim, d_model)

        # Self-attention layers (player ↔ player)
        sa_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.self_attn = nn.TransformerEncoder(sa_layer, num_layers=num_layers)

        # Cross-attention layer (player → conditioning anchor)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(d_model)

        # Per-player velocity output (only x, y)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 2),    # (dx/dt, dy/dt)
        )

    def forward(self,
                x_t:   torch.Tensor,
                roles: torch.Tensor,
                t:     torch.Tensor,
                c:     torch.Tensor,
                mask:  torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x_t   : (B, N, 2)  noisy player (x, y) at time t
            roles : (B, N, 2)  [is_teammate, is_actor] — fixed throughout ODE
            t     : (B,)       flow time ∈ [0, 1]
            c     : (B, cond_dim) conditioning signal
            mask  : (B, N) bool — True where padded (absent players)

        Returns:
            v : (B, N, 2) velocity field for each player
        """
        B, N, _ = x_t.shape

        # Concatenate noisy positions with fixed role flags
        player_in = torch.cat([x_t, roles], dim=-1)   # (B, N, 4)
        h = self.player_proj(player_in)                 # (B, N, d_model)

        # Add time embedding (broadcast over players)
        t_emb = self.time_emb(t).unsqueeze(1)           # (B, 1, d_model)
        h = h + t_emb

        # Cross-attention to conditioning anchor
        c_token = self.cond_proj(c).unsqueeze(1)        # (B, 1, d_model)
        h_cross, _ = self.cross_attn(h, c_token, c_token)
        h = self.cross_norm(h + h_cross)

        # Self-attention across all players
        if mask is not None:
            h = self.self_attn(h, src_key_padding_mask=mask)
        else:
            h = self.self_attn(h)

        return self.out_proj(h)   # (B, N, 2)


# ── Full generator ────────────────────────────────────────────────────────────

class TacticalGenerator(nn.Module):
    """
    Complete conditional flow matching generator.
    Combines ConditionEncoder + VelocityField.
    """

    def __init__(self,
                 fingerprint_dim: int   = 256,
                 cond_dim:        int   = 512,
                 d_model:         int   = 256,
                 nhead:           int   = 8,
                 num_layers:      int   = 4,
                 n_players:       int   = 22,
                 sigma:           float = 0.1,
                 dropout:         float = 0.1):
        super().__init__()
        self.sigma     = sigma
        self.n_players = n_players

        self.cond_encoder = ConditionEncoder(
            fingerprint_dim=fingerprint_dim,
            cond_dim=cond_dim,
        )
        self.velocity_field = VelocityField(
            n_players=n_players,
            player_dim=4,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            cond_dim=cond_dim,
            dropout=dropout,
        )

    def encode_condition(self, z_A, z_B, score_diff,
                          minute_norm, phase_onehot, zone_onehot):
        return self.cond_encoder(
            z_A, z_B, score_diff, minute_norm, phase_onehot, zone_onehot
        )

    def flow_matching_loss(self,
                           x_1:   torch.Tensor,
                           roles: torch.Tensor,
                           c:     torch.Tensor,
                           mask:  torch.Tensor | None = None) -> torch.Tensor:
        """
        Compute CFM training loss on a batch of real freeze frames.

        Args:
            x_1   : (B, N, 2) real player (x, y) positions (normalized)
            roles : (B, N, 2) [is_teammate, is_actor]
            c     : (B, cond_dim) conditioning signal
            mask  : (B, N) bool — True where padded

        Returns:
            scalar loss
        """
        B = x_1.shape[0]
        device = x_1.device

        # Sample source noise and flow time
        x_0 = torch.randn_like(x_1) * self.sigma
        t   = torch.rand(B, device=device)

        # Linear interpolation along the flow path
        t_bc = t.view(B, 1, 1)
        x_t  = (1.0 - t_bc) * x_0 + t_bc * x_1

        # Target velocity (constant along linear path)
        u_t = x_1 - x_0

        # Predicted velocity
        v = self.velocity_field(x_t, roles, t, c, mask)

        # MSE only on non-padded players
        if mask is not None:
            valid = ~mask.unsqueeze(-1)                 # (B, N, 1)
            loss  = ((v - u_t) ** 2 * valid).sum() / valid.sum()
        else:
            loss = F.mse_loss(v, u_t)

        return loss

    @torch.no_grad()
    def generate(self,
                 roles:        torch.Tensor,
                 c:            torch.Tensor,
                 mask:         torch.Tensor | None = None,
                 n_steps:      int   = 50,
                 clamp_pitch:  bool  = True) -> torch.Tensor:
        """
        Generate a freeze frame via Euler ODE integration.

        Args:
            roles    : (B, N, 2) [is_teammate, is_actor] — role assignment
            c        : (B, cond_dim) conditioning signal
            mask     : (B, N) bool — True where padded
            n_steps  : Euler integration steps (more = higher quality)
            clamp_pitch : if True, clamp output to [0,1] normalized pitch

        Returns:
            x_1 : (B, N, 2) generated player positions (normalized)
        """
        B, N, _ = roles.shape
        device  = roles.device

        x = torch.randn(B, N, 2, device=device) * self.sigma
        dt = 1.0 / n_steps

        for i in range(n_steps):
            t = torch.full((B,), i / n_steps, device=device)
            v = self.velocity_field(x, roles, t, c, mask)
            x = x + dt * v

        if clamp_pitch:
            x = x.clamp(0.0, 1.0)

        return x


# ── Factory + utilities ────────────────────────────────────────────────────────

def build_generator(fingerprint_dim: int = 256) -> TacticalGenerator:
    return TacticalGenerator(
        fingerprint_dim=fingerprint_dim,
        cond_dim=512,
        d_model=256,
        nhead=8,
        num_layers=4,
        n_players=22,
        sigma=0.1,
        dropout=0.1,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    gen = build_generator()
    print(f"TacticalGenerator parameters: {count_parameters(gen):,}")

    B, N = 4, 22
    x_1   = torch.rand(B, N, 2)
    roles = torch.zeros(B, N, 2)
    roles[:, 0, 1] = 1.0          # player 0 is actor
    roles[:, :11, 0] = 1.0        # players 0-10 are teammates
    mask  = torch.zeros(B, N, dtype=torch.bool)

    z_A = torch.randn(B, 256)
    z_B = torch.randn(B, 256)
    score_diff  = torch.zeros(B, 1)
    minute_norm = torch.full((B, 1), 0.5)
    phase_oh    = torch.zeros(B, 4); phase_oh[:, 0] = 1.0
    zone_oh     = torch.zeros(B, 4); zone_oh[:, 1] = 1.0

    c = gen.encode_condition(z_A, z_B, score_diff, minute_norm, phase_oh, zone_oh)
    loss = gen.flow_matching_loss(x_1, roles, c, mask)
    print(f"Training loss: {loss.item():.4f}")

    gen.eval()
    generated = gen.generate(roles, c, mask, n_steps=20)
    print(f"Generated positions shape: {generated.shape}")   # (4, 22, 2)
    print(f"Position range: [{generated.min():.3f}, {generated.max():.3f}]")
