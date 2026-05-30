"""
04_train_simulator.py
=====================
Autoregressive match simulator — chains possession-level outcomes into
full match trajectories.

Architecture:
    SimulatorRNN: GRU-based model that, at each possession step, takes the
    current match state and predicts the next possession's outcome
    distribution. Conditioned on the same team fingerprints used by the
    generator.

Match state at step k:
    score_diff       : int         home_goals - away_goals
    minute           : float       match minute [0, 90+]
    zone             : int         entry zone of current possession [0–3]
    phase            : int         phase [0–3]
    poss_team        : int         0 = home, 1 = away
    cumulative stats : int×8       [home/away shots, final_third_entries,
                                    total_possessions]

Predictions per possession:
    p_advance_zone   : prob next zone > current zone (intermediate step)
    p_shot           : prob possession ends in shot
    p_goal           : prob shot → goal  (conditional on shot)
    p_retain         : prob same team keeps possession next

Training:
    Supervised on possession_meta.csv sequences per match.
    Possession sequences within a match form one trajectory.
    Loss: weighted BCE on [p_advance_zone, p_shot, p_goal, p_retain]

Outputs:
    model/checkpoints/simulator_best.pt
    data/results/simulator_training.csv
    data/results/sim_match_demo.csv   (one simulated match per team pair)
"""

import sys
import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.sse import TeamEncoder

# ── Config ────────────────────────────────────────────────────────────────────

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "mps"
                           if torch.backends.mps.is_available() else "cpu")
FP_DIM      = 256
HIDDEN_DIM  = 512
N_LAYERS    = 2
DROPOUT     = 0.2
MAX_POSS    = 300        # max possessions per match (pad/truncate)
SIM_STEPS   = 500        # possessions to generate per simulated match

EPOCHS      = 80
BS          = 32         # sequences (matches) per batch
LR          = 5e-4
VAL_FRAC    = 0.15

CKPT_DIR    = Path("model/checkpoints")
RESULTS_DIR = Path("data/results")
META_PATH   = RESULTS_DIR / "possession_meta.csv"
FP_PATH     = CKPT_DIR   / "team_fingerprints.pt"

CKPT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── State encoding ─────────────────────────────────────────────────────────────

STATE_DIM = (
    1 +   # score_diff (normalised)
    1 +   # minute_norm
    4 +   # zone onehot
    4 +   # phase onehot
    1 +   # poss_team (0/1)
    4     # cumulative: home_shots, away_shots, home_ft_entries, away_ft_entries
)   # = 15


def build_state(row: dict, home_stats: dict, away_stats: dict) -> np.ndarray:
    """Encodes one possession into a STATE_DIM-dimensional feature vector."""
    score_diff   = np.clip((home_stats["goals"] - away_stats["goals"]) / 3.0, -1, 1)
    minute_norm  = float(row.get("minute", 45)) / 90.0

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


def build_targets(row: dict) -> np.ndarray:
    """Four binary targets for one possession."""
    s2   = float(row.get("reached_s2",    0) or 0)
    shot = float(row.get("reached_shot",  0) or 0)
    # goal = reached_shot AND outcome: we use a proxy from metadata
    goal = float(row.get("_goal_in_poss", 0) or 0)
    # retain: same team at next step — resolved during sequence build
    retain = float(row.get("_retain", 0) or 0)
    return np.array([s2, shot, goal, retain], dtype=np.float32)


# ── Dataset ────────────────────────────────────────────────────────────────────

class MatchSequenceDataset(Dataset):
    """
    One item = one match's possession sequence.

    Each item:
        states   : (T, STATE_DIM)       per-possession features
        targets  : (T, 4)               binary labels
        fp_home  : (FP_DIM,)            home team fingerprint
        fp_away  : (FP_DIM,)            away team fingerprint
        length   : int                  real sequence length before padding
    """

    def __init__(self, meta: pd.DataFrame, fingerprints: dict,
                 mean_fp: torch.Tensor, max_len: int = MAX_POSS):
        self.max_len      = max_len
        self.fingerprints = fingerprints
        self.mean_fp      = mean_fp.cpu()
        self.sequences    = []

        for match_id, group in meta.groupby("match_id"):
            group = group.sort_values("possession_id").reset_index(drop=True)
            if len(group) < 5:
                continue

            # Infer home/away from first two distinct teams
            teams = group["team_id"].unique()
            if len(teams) < 2:
                continue
            home_id, away_id = int(teams[0]), int(teams[1])

            # Cumulative stats
            home_stats = {"goals": 0, "shots": 0, "ft_entries": 0}
            away_stats = {"goals": 0, "shots": 0, "ft_entries": 0}

            # Annotate retain + goal
            team_ids = group["team_id"].tolist()
            for i in range(len(group) - 1):
                group.at[i, "_retain"] = float(team_ids[i] == team_ids[i + 1])
            group.at[len(group) - 1, "_retain"] = 0.0

            group["_is_away"]     = (group["team_id"] == away_id).astype(float)
            group["_goal_in_poss"] = 0.0   # proxy: no explicit goal flag in meta

            states  = []
            targets = []
            for _, row in group.iterrows():
                row_dict = row.to_dict()
                states.append(build_state(row_dict, home_stats, away_stats))
                targets.append(build_targets(row_dict))

                # Update cumulative stats after this possession
                tid = int(row["team_id"])
                bucket = home_stats if tid == home_id else away_stats
                if row.get("reached_shot", 0):
                    bucket["shots"] += 1
                if row.get("reached_s3", 0):
                    bucket["ft_entries"] += 1

            fp_home = fingerprints.get(home_id, mean_fp).cpu()
            fp_away = fingerprints.get(away_id, mean_fp).cpu()

            self.sequences.append({
                "states":   np.stack(states).astype(np.float32),
                "targets":  np.stack(targets).astype(np.float32),
                "fp_home":  fp_home.numpy(),
                "fp_away":  fp_away.numpy(),
                "length":   len(states),
                "match_id": match_id,
                "home_id":  home_id,
                "away_id":  away_id,
            })

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        seq = self.sequences[idx]
        T   = min(seq["length"], self.max_len)

        states_pad  = np.zeros((self.max_len, STATE_DIM),  dtype=np.float32)
        targets_pad = np.zeros((self.max_len, 4),          dtype=np.float32)
        mask_pad    = np.ones( self.max_len,                dtype=bool)

        states_pad[:T]  = seq["states"][:T]
        targets_pad[:T] = seq["targets"][:T]
        mask_pad[:T]    = False

        return {
            "states":   torch.tensor(states_pad),
            "targets":  torch.tensor(targets_pad),
            "mask":     torch.tensor(mask_pad),
            "fp_home":  torch.tensor(seq["fp_home"]),
            "fp_away":  torch.tensor(seq["fp_away"]),
            "length":   T,
        }


# ── Model ──────────────────────────────────────────────────────────────────────

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
        """Returns (n_layers, B, hidden_dim) GRU init hidden state."""
        B = fp_home.shape[0]
        h = self.proj(torch.cat([fp_home, fp_away], dim=-1))     # (B, H*L)
        return h.view(B, self.n_layers, -1).permute(1, 0, 2).contiguous()


class SimulatorRNN(nn.Module):
    """
    Autoregressive possession simulator.

    At each step receives the encoded match state and predicts:
        [p_advance_zone, p_shot, p_goal, p_retain]
    as independent Bernoulli outputs.

    Team fingerprints initialise the GRU hidden state rather than being
    concatenated at every step — this reduces input dimensionality and
    lets the team identity fade naturally as the match progresses.
    """

    def __init__(self,
                 state_dim:  int = STATE_DIM,
                 fp_dim:     int = FP_DIM,
                 hidden_dim: int = HIDDEN_DIM,
                 n_layers:   int = N_LAYERS,
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
            nn.Linear(hidden_dim // 2, 4),   # 4 binary targets
        )

    def forward(self,
                states:  torch.Tensor,
                fp_home: torch.Tensor,
                fp_away: torch.Tensor,
                mask:    torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            states   : (B, T, STATE_DIM)
            fp_home  : (B, FP_DIM)
            fp_away  : (B, FP_DIM)
            mask     : (B, T) bool, True where padded

        Returns:
            logits : (B, T, 4)  raw logits, apply sigmoid for probabilities
        """
        h0 = self.init_encoder(fp_home, fp_away)    # (n_layers, B, H)
        x  = self.input_proj(states)                 # (B, T, H)
        out, _ = self.gru(x, h0)                     # (B, T, H)
        return self.out_head(out)                     # (B, T, 4)

    def step(self,
             state:   torch.Tensor,
             fp_home: torch.Tensor,
             fp_away: torch.Tensor,
             h:       torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Single autoregressive step for simulation.

        Args:
            state   : (B, STATE_DIM)
            fp_home : (B, FP_DIM)
            fp_away : (B, FP_DIM)
            h       : (n_layers, B, H) or None for first step

        Returns:
            probs : (B, 4)  sigmoid probabilities
            h_new : (n_layers, B, H) updated hidden state
        """
        if h is None:
            h = self.init_encoder(fp_home, fp_away)
        x = self.input_proj(state.unsqueeze(1))          # (B, 1, H)
        out, h_new = self.gru(x, h)                       # (B, 1, H), (L, B, H)
        logits = self.out_head(out.squeeze(1))             # (B, 4)
        return torch.sigmoid(logits), h_new


# ── Loss ───────────────────────────────────────────────────────────────────────

TARGET_WEIGHTS = torch.tensor([1.0, 2.0, 3.0, 1.0])   # upweight shot/goal


def simulator_loss(logits: torch.Tensor,
                   targets: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
    """
    Weighted BCE across 4 targets, ignoring padded steps.

    Args:
        logits  : (B, T, 4)
        targets : (B, T, 4)
        mask    : (B, T) True where padded
    """
    valid = ~mask.unsqueeze(-1)                        # (B, T, 1)
    w = TARGET_WEIGHTS.to(logits.device)               # (4,)
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )                                                  # (B, T, 4)
    loss = (bce * w * valid).sum() / (valid.sum() * 4)
    return loss


# ── Training ───────────────────────────────────────────────────────────────────

def train_simulator(model: SimulatorRNN,
                    train_loader: DataLoader,
                    val_loader:   DataLoader) -> list[dict]:

    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_val = float("inf")
    log = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            states  = batch["states"].to(DEVICE)
            targets = batch["targets"].to(DEVICE)
            mask    = batch["mask"].to(DEVICE)
            fp_home = batch["fp_home"].to(DEVICE)
            fp_away = batch["fp_away"].to(DEVICE)

            logits = model(states, fp_home, fp_away, mask)
            loss   = simulator_loss(logits, targets, mask)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            train_loss += loss.item()

        sched.step()
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds, all_tgts = [[] for _ in range(4)], [[] for _ in range(4)]
        with torch.no_grad():
            for batch in val_loader:
                states  = batch["states"].to(DEVICE)
                targets = batch["targets"].to(DEVICE)
                mask    = batch["mask"].to(DEVICE)
                fp_home = batch["fp_home"].to(DEVICE)
                fp_away = batch["fp_away"].to(DEVICE)

                logits = model(states, fp_home, fp_away, mask)
                val_loss += simulator_loss(logits, targets, mask).item()

                probs = torch.sigmoid(logits)
                valid_mask = ~mask
                for i in range(4):
                    p = probs[valid_mask, i].cpu().numpy()
                    t = targets[valid_mask, i].cpu().numpy()
                    all_preds[i].append(p)
                    all_tgts[i].append(t)

        val_loss /= len(val_loader)

        aucs = []
        tnames = ["advance", "shot", "goal", "retain"]
        for i in range(4):
            p = np.concatenate(all_preds[i])
            t = np.concatenate(all_tgts[i])
            if t.sum() > 0 and t.sum() < len(t):
                aucs.append(roc_auc_score(t, p))
            else:
                aucs.append(float("nan"))

        entry = {
            "epoch":        epoch,
            "train_loss":   train_loss,
            "val_loss":     val_loss,
            **{f"auc_{tnames[i]}": aucs[i] for i in range(4)},
        }
        log.append(entry)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), CKPT_DIR / "simulator_best.pt")

        if epoch % 10 == 0 or epoch == 1:
            auc_str = "  ".join(
                f"{tnames[i]}={aucs[i]:.3f}" for i in range(4) if not math.isnan(aucs[i])
            )
            print(f"  [{epoch:3d}/{EPOCHS}]  train={train_loss:.4f}  "
                  f"val={val_loss:.4f}  {auc_str}")

    return log


# ── Match simulation (inference) ───────────────────────────────────────────────

@torch.no_grad()
def simulate_match(model:   SimulatorRNN,
                   fp_home: torch.Tensor,
                   fp_away: torch.Tensor,
                   n_poss:  int = SIM_STEPS,
                   seed:    int | None = None) -> pd.DataFrame:
    """
    Simulate a full match between two teams.

    Returns a DataFrame with one row per simulated possession:
        minute, poss_team (0=home/1=away), zone, phase,
        advanced_zone, shot, goal, retain,
        score_home, score_away
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    model.eval()
    fp_home = fp_home.unsqueeze(0).to(DEVICE)
    fp_away = fp_away.unsqueeze(0).to(DEVICE)

    home_stats = {"goals": 0, "shots": 0, "ft_entries": 0}
    away_stats = {"goals": 0, "shots": 0, "ft_entries": 0}

    h = None
    rows = []
    poss_team = 0    # 0 = home starts
    minute    = 0.0
    zone      = 1    # typical kick-off zone
    phase     = 3    # restart

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
        ).unsqueeze(0).to(DEVICE)

        probs, h = model.step(state, fp_home, fp_away, h)
        probs = probs.squeeze(0).cpu()

        adv    = float(torch.bernoulli(probs[0]))
        shot   = float(torch.bernoulli(probs[1]))
        goal   = float(torch.bernoulli(probs[2])) if shot else 0.0
        retain = float(torch.bernoulli(probs[3]))

        # Update stats
        bucket = home_stats if poss_team == 0 else away_stats
        if shot:
            bucket["shots"] += 1
        if adv:
            bucket["ft_entries"] += 1
        if goal:
            bucket["goals"] += 1

        rows.append({
            "minute":      round(minute, 1),
            "poss_team":   poss_team,
            "zone":        zone,
            "phase":       phase,
            "advanced":    adv,
            "shot":        shot,
            "goal":        goal,
            "retain":      retain,
            "score_home":  home_stats["goals"],
            "score_away":  away_stats["goals"],
        })

        # Advance match state
        minute    += np.random.exponential(1.5)   # avg ~1.5 min per possession
        zone      = min(zone + int(adv), 3)
        phase     = 0  # all subsequent possessions are open play unless reset
        poss_team = poss_team if retain else 1 - poss_team

        if goal:
            zone = 1
            phase = 3
            minute += 1.0

    return pd.DataFrame(rows)


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_training(log: list[dict]):
    df = pd.DataFrame(log)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(df["epoch"], df["train_loss"], label="train")
    axes[0].plot(df["epoch"], df["val_loss"],   label="val")
    axes[0].set_title("Simulator Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    for col, label in [("auc_advance", "advance"), ("auc_shot", "shot"),
                       ("auc_goal", "goal"), ("auc_retain", "retain")]:
        axes[1].plot(df["epoch"], df[col], label=label)
    axes[1].set_title("Validation AUC per Target")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    # Score distribution for demo (filled after simulation)
    axes[2].set_title("Match Score Distribution (post-sim)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "simulator_training.png", dpi=150)
    plt.close()
    print(f"  Saved training plot: {RESULTS_DIR / 'simulator_training.png'}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Tactical World Model — Match Simulator Training")
    print("=" * 50)

    if not META_PATH.exists():
        print(f"ERROR: {META_PATH} not found — run 02_build_dataset.py first.")
        return
    if not FP_PATH.exists():
        print(f"ERROR: {FP_PATH} not found — run 03_train_generator.py first.")
        return

    print(f"Device: {DEVICE}")

    # Load fingerprints
    fp_data = torch.load(FP_PATH, map_location="cpu")
    fingerprints_raw: dict = fp_data["team_fingerprints"]
    fingerprints: dict[int, torch.Tensor] = {
        k: v.float() for k, v in fingerprints_raw.items()
    }
    mean_fp: torch.Tensor = fp_data.get(
        "mean_fingerprint",
        torch.stack(list(fingerprints.values())).mean(0)
    )
    print(f"Loaded {len(fingerprints)} team fingerprints")

    # Load meta
    meta = pd.read_csv(META_PATH)
    print(f"Loaded possession meta: {len(meta):,} rows, "
          f"{meta['match_id'].nunique()} matches")

    # Split matches into train/val
    match_ids = meta["match_id"].unique()
    rng = np.random.default_rng(42)
    rng.shuffle(match_ids)
    n_val   = max(1, int(len(match_ids) * VAL_FRAC))
    val_ids = set(match_ids[:n_val])

    train_meta = meta[~meta["match_id"].isin(val_ids)]
    val_meta   = meta[ meta["match_id"].isin(val_ids)]
    print(f"Train matches: {train_meta['match_id'].nunique()} | "
          f"Val matches: {val_meta['match_id'].nunique()}")

    train_ds = MatchSequenceDataset(train_meta, fingerprints, mean_fp)
    val_ds   = MatchSequenceDataset(val_meta,   fingerprints, mean_fp)
    print(f"Train sequences: {len(train_ds)} | Val sequences: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True,
                              num_workers=0, pin_memory=DEVICE.type != "cpu")
    val_loader   = DataLoader(val_ds,   batch_size=BS, shuffle=False,
                              num_workers=0, pin_memory=DEVICE.type != "cpu")

    # Build model
    model = SimulatorRNN(
        state_dim=STATE_DIM, fp_dim=FP_DIM,
        hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS, dropout=DROPOUT,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SimulatorRNN parameters: {n_params:,}")

    # Train
    print("\nTraining SimulatorRNN…")
    log = train_simulator(model, train_loader, val_loader)

    # Save training log
    log_df = pd.DataFrame(log)
    log_df.to_csv(RESULTS_DIR / "simulator_training.csv", index=False)
    print(f"Saved training log: {RESULTS_DIR / 'simulator_training.csv'}")
    plot_training(log)

    # Demo simulation: pick two well-represented teams
    model.load_state_dict(
        torch.load(CKPT_DIR / "simulator_best.pt", map_location=DEVICE)
    )

    team_counts = meta.groupby("team_id").size()
    top_teams   = team_counts.nlargest(10).index.tolist()
    if len(top_teams) >= 2:
        tid_a, tid_b = top_teams[0], top_teams[1]
        fp_a = fingerprints.get(tid_a, mean_fp).to(DEVICE)
        fp_b = fingerprints.get(tid_b, mean_fp).to(DEVICE)

        print(f"\nDemo simulation: team {tid_a} (home) vs team {tid_b} (away)")
        sim_df = simulate_match(model, fp_a, fp_b, n_poss=SIM_STEPS, seed=0)
        sim_df.to_csv(RESULTS_DIR / "sim_match_demo.csv", index=False)

        final = sim_df.iloc[-1]
        print(f"  Final score: {int(final['score_home'])}–{int(final['score_away'])}")
        print(f"  Possessions: {len(sim_df)}")
        print(f"  Total shots:  home={int(sim_df['shot'][sim_df['poss_team']==0].sum())}  "
              f"away={int(sim_df['shot'][sim_df['poss_team']==1].sum())}")
        print(f"  Saved: {RESULTS_DIR / 'sim_match_demo.csv'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
