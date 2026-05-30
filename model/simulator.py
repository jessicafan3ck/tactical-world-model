"""
model/simulator.py
==================
Reusable SimulatorRNN and helpers shared between training script and
the counterfactual intervention engine.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd


# ── Constants (importable) ─────────────────────────────────────────────────────

FP_DIM      = 256
HIDDEN_DIM  = 512
N_LAYERS    = 2
DROPOUT     = 0.2
SIM_STEPS   = 500

STATE_DIM = (
    1 +   # score_diff (normalised)
    1 +   # minute_norm
    4 +   # zone onehot
    4 +   # phase onehot
    1 +   # poss_team (0/1)
    4     # cumulative: home_shots, away_shots, home_ft_entries, away_ft_entries
)   # = 15


# ── State encoding ──────────────────────────────────────────────────────────────

def build_state(row: dict, home_stats: dict, away_stats: dict) -> np.ndarray:
    """Encodes one possession into a STATE_DIM-dimensional feature vector."""
    score_diff  = np.clip((home_stats["goals"] - away_stats["goals"]) / 3.0, -1, 1)
    minute_norm = float(row.get("minute", 45)) / 90.0

    zone = int(row.get("entry_state", 1))
    zone_oh = np.zeros(4, dtype=np.float32)
    zone_oh[min(zone, 3)] = 1.0

    phase = int(row.get("phase_int", 0))
    phase_oh = np.zeros(4, dtype=np.float32)
    phase_oh[min(phase, 3)] = 1.0

    poss_team = float(row.get("_is_away", 0))

    cum = np.array([
        home_stats["shots"]     / 10.0,
        away_stats["shots"]     / 10.0,
        home_stats["ft_entries"] / 20.0,
        away_stats["ft_entries"] / 20.0,
    ], dtype=np.float32)

    return np.concatenate([
        [score_diff, minute_norm],
        zone_oh, phase_oh,
        [poss_team],
        cum,
    ]).astype(np.float32)


# ── Model ───────────────────────────────────────────────────────────────────────

class ConditionedStateEncoder(nn.Module):
    """Projects team fingerprints into the GRU hidden state initialisation."""

    def __init__(self, fp_dim: int, hidden_dim: int, n_layers: int):
        super().__init__()
        self.n_layers = n_layers
        self.proj = nn.Sequential(
            nn.Linear(2 * fp_dim, hidden_dim * n_layers),
            nn.Tanh(),
        )

    def forward(self, fp_home: torch.Tensor, fp_away: torch.Tensor) -> torch.Tensor:
        B = fp_home.shape[0]
        h = self.proj(torch.cat([fp_home, fp_away], dim=-1))
        return h.view(B, self.n_layers, -1).permute(1, 0, 2).contiguous()


class SimulatorRNN(nn.Module):
    """
    Autoregressive possession simulator (GRU).
    Predicts [p_advance_zone, p_shot, p_goal, p_retain] per possession.
    Team fingerprints initialise the GRU hidden state.
    """

    def __init__(self,
                 state_dim:  int   = STATE_DIM,
                 fp_dim:     int   = FP_DIM,
                 hidden_dim: int   = HIDDEN_DIM,
                 n_layers:   int   = N_LAYERS,
                 dropout:    float = DROPOUT):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.init_encoder = ConditionedStateEncoder(fp_dim, hidden_dim, n_layers)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.out_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 4),
        )

    def forward(self,
                states:  torch.Tensor,
                fp_home: torch.Tensor,
                fp_away: torch.Tensor,
                mask:    torch.Tensor | None = None) -> torch.Tensor:
        h0  = self.init_encoder(fp_home, fp_away)
        x   = self.input_proj(states)
        out, _ = self.gru(x, h0)
        return self.out_head(out)    # (B, T, 4)

    def step(self,
             state:   torch.Tensor,
             fp_home: torch.Tensor,
             fp_away: torch.Tensor,
             h:       torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if h is None:
            h = self.init_encoder(fp_home, fp_away)
        x = self.input_proj(state.unsqueeze(1))
        out, h_new = self.gru(x, h)
        logits = self.out_head(out.squeeze(1))
        return torch.sigmoid(logits), h_new


# ── Inference helper ────────────────────────────────────────────────────────────

@torch.no_grad()
def simulate_match(model:          SimulatorRNN,
                   fp_home:        torch.Tensor,
                   fp_away:        torch.Tensor,
                   device:         torch.device | None = None,
                   n_poss:         int   = SIM_STEPS,
                   seed:           int | None = None,
                   shot_multiplier: float = 2.0,
                   goal_given_shot: float = 0.25) -> pd.DataFrame:
    """
    Simulate a full match via autoregressive rollout.

    Calibration notes:
      shot_multiplier : the model's raw p(shot) is trained on a filtered
        dataset where shots are only 0.4% of possessions. Multiply by ~8
        to reach realistic ~3-4% per possession (~10-15 shots/90 min).
      goal_given_shot : overrides the model's p(goal|shot) because the
        training target _goal_in_poss was always 0. ~0.28 matches real
        conversion rates in women's football.

    Returns DataFrame with columns:
        minute, poss_team, zone, phase, advanced, shot, goal, retain,
        score_home, score_away
    """
    if device is None:
        device = next(model.parameters()).device

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    model.eval()
    fp_home = fp_home.unsqueeze(0).to(device)
    fp_away = fp_away.unsqueeze(0).to(device)

    home_stats = {"goals": 0, "shots": 0, "ft_entries": 0}
    away_stats = {"goals": 0, "shots": 0, "ft_entries": 0}

    h         = None
    rows      = []
    poss_team = 0
    minute    = 0.0
    zone      = 1
    phase     = 3

    for _ in range(n_poss):
        if minute >= 90:
            break

        row_dict = {
            "minute":      minute,
            "entry_state": zone,
            "phase_int":   phase,
            "_is_away":    float(poss_team),
        }
        state = torch.tensor(
            build_state(row_dict, home_stats, away_stats), dtype=torch.float32
        ).unsqueeze(0).to(device)

        probs, h = model.step(state, fp_home, fp_away, h)
        probs = probs.squeeze(0).cpu()

        # Apply calibration: shot prob is dataset-underestimated; goal uses override
        p_shot = float((probs[1] * shot_multiplier).clamp(0, 1))
        adv    = float(torch.bernoulli(probs[0]))
        shot   = float(torch.bernoulli(torch.tensor(p_shot)))
        goal   = float(np.random.random() < goal_given_shot) if shot else 0.0
        retain = float(torch.bernoulli(probs[3]))

        bucket = home_stats if poss_team == 0 else away_stats
        if shot:  bucket["shots"] += 1
        if adv:   bucket["ft_entries"] += 1
        if goal:  bucket["goals"] += 1

        rows.append({
            "minute":    round(minute, 1),
            "poss_team": poss_team,
            "zone":      zone,
            "phase":     phase,
            "advanced":  adv,
            "shot":      shot,
            "goal":      goal,
            "retain":    retain,
            "score_home": home_stats["goals"],
            "score_away": away_stats["goals"],
        })

        # ~15 sec per possession on average (0.25 min), matching real match data
        minute   += np.random.exponential(0.25)
        zone      = min(zone + int(adv), 3)
        phase     = 0
        poss_team = poss_team if retain else 1 - poss_team

        if goal:
            zone   = 1
            phase  = 3
            minute += 1.0

    return pd.DataFrame(rows)


def build_simulator(state_dim: int = STATE_DIM,
                    fp_dim:    int = FP_DIM) -> SimulatorRNN:
    return SimulatorRNN(state_dim=state_dim, fp_dim=fp_dim)
