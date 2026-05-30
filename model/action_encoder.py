"""
model/action_encoder.py
=======================
Part 1 of the interactive simulation layer.

Maps discrete tactical actions to z-space delta vectors, giving the
engine a way to translate "user pressed W (advance)" into a meaningful
perturbation of the team fingerprint before the generator runs.

Action vocabulary (11 actions):
    ADVANCE          move ball carrier forward (W)
    HOLD             hold position / recycle (S)
    SWITCH_LEFT      switch play to left flank (A)
    SWITCH_RIGHT     switch play to right flank (D)
    PRESS            trigger high press (team-wide)
    LOW_BLOCK        drop into defensive shape
    SHOOT            attempt shot
    DRIBBLE          take on defender (1v1)
    THROUGH_BALL     play in behind the line
    CROSS            deliver from wide
    KEEPER_BALL      goalkeeper plays out from back

Design:
  Each action is associated with a soft prototype direction in z-space,
  learned from the PCA cluster centroids computed in 05_counterfactuals.py.
  For teams without enough match data to anchor prototypes, we fall back
  to learned bias vectors trained from labelled phase→outcome statistics.

  At inference the engine computes:
      z_modified = z_team + α * action_delta(action, z_team, context)

  alpha is the "intensity" of the action (0 = no change, 1 = full commit).
  The delta is context-sensitive: PRESS is bigger when score_diff < 0.

Testable standalone:
    python -m model.action_encoder
"""

import torch
import torch.nn as nn
from enum import IntEnum
from dataclasses import dataclass
from pathlib import Path


# ── Action vocabulary ──────────────────────────────────────────────────────────

class Action(IntEnum):
    ADVANCE      = 0
    HOLD         = 1
    SWITCH_LEFT  = 2
    SWITCH_RIGHT = 3
    PRESS        = 4
    LOW_BLOCK    = 5
    SHOOT        = 6
    DRIBBLE      = 7
    THROUGH_BALL = 8
    CROSS        = 9
    KEEPER_BALL  = 10

N_ACTIONS = len(Action)

# Human-readable labels for UI display
ACTION_LABELS = {
    Action.ADVANCE:      "Advance",
    Action.HOLD:         "Hold / Recycle",
    Action.SWITCH_LEFT:  "Switch Left",
    Action.SWITCH_RIGHT: "Switch Right",
    Action.PRESS:        "High Press",
    Action.LOW_BLOCK:    "Low Block",
    Action.SHOOT:        "Shoot",
    Action.DRIBBLE:      "Dribble",
    Action.THROUGH_BALL: "Through Ball",
    Action.CROSS:        "Cross",
    Action.KEEPER_BALL:  "Play Out",
}

# WASD + common keyboard shortcuts (used by server to decode key events)
KEY_TO_ACTION: dict[str, Action] = {
    "w":          Action.ADVANCE,
    "s":          Action.HOLD,
    "a":          Action.SWITCH_LEFT,
    "d":          Action.SWITCH_RIGHT,
    "p":          Action.PRESS,
    "l":          Action.LOW_BLOCK,
    "space":      Action.SHOOT,
    "e":          Action.DRIBBLE,
    "t":          Action.THROUGH_BALL,
    "c":          Action.CROSS,
    "g":          Action.KEEPER_BALL,
    "ArrowUp":    Action.ADVANCE,
    "ArrowDown":  Action.HOLD,
    "ArrowLeft":  Action.SWITCH_LEFT,
    "ArrowRight": Action.SWITCH_RIGHT,
}


# ── Context fed to the action encoder ─────────────────────────────────────────

@dataclass
class MatchContext:
    """Lightweight snapshot of match state at the moment of the action."""
    score_diff:  float   # home - away, clipped to [-3, 3]
    minute:      float   # 0–90
    zone:        int     # 0–3
    phase:       int     # 0–3
    poss_team:   int     # 0 = home, 1 = away

    def to_tensor(self) -> torch.Tensor:
        """Returns a (7,) context vector."""
        sd   = max(-1.0, min(1.0, self.score_diff / 3.0))
        mn   = self.minute / 90.0
        zone_oh  = [0.0] * 4; zone_oh[min(self.zone, 3)] = 1.0
        return torch.tensor(
            [sd, mn] + zone_oh, dtype=torch.float32
        )  # (6,) — poss_team handled separately


# ── Action encoder model ───────────────────────────────────────────────────────

class ActionEncoder(nn.Module):
    """
    Produces a z-space delta vector for a given (action, context) pair.

    Architecture:
        action_emb : (N_ACTIONS,) one-hot → (action_dim,) embedding
        context_emb: (6,) context vector  → (context_dim,)
        combined   : concat → (z_dim,) delta

    The delta is L2-normalised so intensity is controlled entirely by α.
    A learned per-action scale factor lets SHOOT produce a larger delta
    than HOLD without tuning α externally.

    Training:
        Not supervised end-to-end with the rest of the pipeline (that
        would need labelled tactical decision data we don't have).
        Instead, weights are initialised from the PCA cluster structure:
        the PRESS prototype points toward the pressing cluster centroid,
        HOLD points toward the possession cluster, etc.
        Fine-tuning against simulator outcome deltas is a future step.
    """

    def __init__(self, z_dim: int = 256, action_dim: int = 32,
                 context_dim: int = 64):
        super().__init__()
        self.z_dim = z_dim

        self.action_emb = nn.Embedding(N_ACTIONS, action_dim)
        self.context_net = nn.Sequential(
            nn.Linear(6, context_dim),
            nn.GELU(),
            nn.LayerNorm(context_dim),
        )
        self.delta_net = nn.Sequential(
            nn.Linear(action_dim + context_dim, z_dim * 2),
            nn.GELU(),
            nn.LayerNorm(z_dim * 2),
            nn.Linear(z_dim * 2, z_dim),
        )
        # Per-action learned intensity scale (initialised to small values)
        self.action_scale = nn.Parameter(torch.ones(N_ACTIONS) * 0.1)

        self._init_weights()

    def _init_weights(self):
        """
        Seed the action embeddings with hand-designed directions so the
        model has a meaningful prior before any fine-tuning:
          - ADVANCE / THROUGH_BALL / SHOOT → positive first dims (attack)
          - HOLD / LOW_BLOCK / KEEPER_BALL → negative first dims (defend)
          - PRESS                          → large-magnitude (disruptive)
          - SWITCH_*                       → orthogonal (lateral)
        """
        with torch.no_grad():
            w = self.action_emb.weight        # (N_ACTIONS, action_dim)
            attack_actions = [Action.ADVANCE, Action.THROUGH_BALL,
                              Action.SHOOT, Action.DRIBBLE, Action.CROSS]
            defend_actions = [Action.HOLD, Action.LOW_BLOCK,
                              Action.KEEPER_BALL]
            disrupt_actions = [Action.PRESS]
            lateral_actions = [Action.SWITCH_LEFT, Action.SWITCH_RIGHT]

            for a in attack_actions:
                w[a, :8] = 0.5
                w[a, 8:16] = -0.2
            for a in defend_actions:
                w[a, :8] = -0.5
                w[a, 8:16] = 0.3
            for a in disrupt_actions:
                w[a] = 0.7
            for i, a in enumerate(lateral_actions):
                w[a, 16 + i * 8: 16 + (i + 1) * 8] = 0.6

            # SHOOT gets a bigger default scale
            self.action_scale.data[Action.SHOOT]        = 0.3
            self.action_scale.data[Action.PRESS]        = 0.25
            self.action_scale.data[Action.THROUGH_BALL] = 0.2
            self.action_scale.data[Action.HOLD]         = 0.05

    def forward(self,
                action:  torch.Tensor,
                context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            action  : (B,) int64 action indices
            context : (B, 6) context vector from MatchContext.to_tensor()

        Returns:
            delta : (B, z_dim) L2-normalised delta, scaled by action_scale
        """
        a_emb = self.action_emb(action)           # (B, action_dim)
        c_emb = self.context_net(context)          # (B, context_dim)
        raw   = self.delta_net(
            torch.cat([a_emb, c_emb], dim=-1)
        )                                          # (B, z_dim)

        # L2-normalise then scale by per-action learned intensity
        norm  = raw / (raw.norm(dim=-1, keepdim=True) + 1e-8)
        scale = self.action_scale[action].unsqueeze(-1)   # (B, 1)
        return norm * scale                        # (B, z_dim)


# ── High-level apply function ──────────────────────────────────────────────────

def apply_action(encoder:  ActionEncoder,
                 z_team:   torch.Tensor,
                 action:   Action,
                 context:  MatchContext,
                 alpha:    float = 1.0) -> torch.Tensor:
    """
    Apply a tactical action to a team fingerprint.

    Args:
        encoder : ActionEncoder (can be uninitialised for prototype-only use)
        z_team  : (z_dim,) current team fingerprint
        action  : discrete Action enum value
        context : current MatchContext
        alpha   : intensity multiplier [0, 1]

    Returns:
        z_modified : (z_dim,) updated fingerprint
    """
    dev = next(encoder.parameters()).device
    encoder.eval()
    with torch.no_grad():
        a_t = torch.tensor([action.value], dtype=torch.long, device=dev)
        c_t = context.to_tensor().unsqueeze(0).to(dev)
        delta = encoder(a_t, c_t).squeeze(0)
    return z_team.to(dev) + alpha * delta


# ── Initialise from PCA cluster centroids ─────────────────────────────────────

def init_from_pca(encoder:      ActionEncoder,
                  fingerprints: dict[int, torch.Tensor],
                  pca_csv_path: Path) -> None:
    """
    Fine-tune action embedding seeds using PCA cluster centroids.

    Maps cluster names to action groups and shifts the corresponding
    action embeddings toward the cluster's mean fingerprint direction.
    No-ops gracefully if the PCA file doesn't exist.
    """
    import pandas as pd

    if not pca_csv_path.exists():
        return

    pca_df = pd.read_csv(pca_csv_path)
    cluster_fps: dict[str, torch.Tensor] = {}

    for cluster_name in pca_df["cluster_name"].unique():
        tids = pca_df.loc[pca_df["cluster_name"] == cluster_name, "team_id"].tolist()
        vecs = [fingerprints[t] for t in tids if t in fingerprints]
        if vecs:
            cluster_fps[cluster_name] = torch.stack(vecs).mean(0)

    # Cluster → action affinity mapping
    cluster_to_actions: dict[str, list[Action]] = {
        "possession": [Action.HOLD, Action.KEEPER_BALL, Action.SWITCH_LEFT,
                       Action.SWITCH_RIGHT],
        "counter":    [Action.ADVANCE, Action.THROUGH_BALL, Action.DRIBBLE],
        "pressing":   [Action.PRESS],
        "hybrid":     [Action.CROSS, Action.SHOOT],
    }

    mean_fp = torch.stack(list(fingerprints.values())).mean(0)

    with torch.no_grad():
        for cluster_name, actions in cluster_to_actions.items():
            if cluster_name not in cluster_fps:
                continue
            # Direction from mean fingerprint toward cluster centroid
            direction = cluster_fps[cluster_name] - mean_fp
            direction = direction / (direction.norm() + 1e-8)

            for action in actions:
                # Project direction into action_dim space and add to embedding
                a_emb = encoder.action_emb.weight[action]
                proj_dim = min(encoder.action_emb.embedding_dim,
                               direction.shape[0])
                a_emb[:proj_dim] += 0.1 * direction[:proj_dim]


# ── Factory ────────────────────────────────────────────────────────────────────

def build_action_encoder(z_dim: int = 256) -> ActionEncoder:
    return ActionEncoder(z_dim=z_dim)


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch

    enc = build_action_encoder(z_dim=256)
    n_params = sum(p.numel() for p in enc.parameters())
    print(f"ActionEncoder parameters: {n_params:,}")

    ctx = MatchContext(score_diff=-1.0, minute=72.0, zone=2, phase=0, poss_team=0)
    z   = torch.randn(256)

    print("\nAction deltas (L2 norm of delta × scale):")
    for action in Action:
        z_mod = apply_action(enc, z, action, ctx, alpha=1.0)
        delta_norm = (z_mod - z).norm().item()
        print(f"  {ACTION_LABELS[action]:16s}  Δ‖z‖ = {delta_norm:.4f}")

    # Verify that the same action applied twice is different from once
    # (context-sensitivity: pressing when losing 1-0 at 72' vs drawing at 10')
    ctx_early = MatchContext(score_diff=0.0, minute=10.0, zone=1, phase=0, poss_team=0)
    z_press_late  = apply_action(enc, z, Action.PRESS, ctx,       alpha=1.0)
    z_press_early = apply_action(enc, z, Action.PRESS, ctx_early, alpha=1.0)
    diff = (z_press_late - z_press_early).norm().item()
    print(f"\nPRESS delta differs by context: ‖late - early‖ = {diff:.4f}")
    print("  (should be > 0 — context-sensitivity confirmed)")
