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

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.simulator import (
    SimulatorRNN, build_state, simulate_match,
    STATE_DIM, FP_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT, SIM_STEPS,
)

# ── Config ────────────────────────────────────────────────────────────────────

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "mps"
                           if torch.backends.mps.is_available() else "cpu")
MAX_POSS    = 300        # max possessions per match (pad/truncate)

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


# ── Targets ───────────────────────────────────────────────────────────────────

def build_targets(row: dict) -> np.ndarray:
    s2     = float(row.get("reached_s2",    0) or 0)
    shot   = float(row.get("reached_shot",  0) or 0)
    goal   = float(row.get("_goal_in_poss", 0) or 0)
    retain = float(row.get("_retain",       0) or 0)
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
        self.max_len   = max_len
        self.sequences = []

        for match_id, group in meta.groupby("match_id"):
            group = group.sort_values("possession_id").reset_index(drop=True)
            if len(group) < 5:
                continue

            teams = group["team_id"].unique()
            if len(teams) < 2:
                continue
            home_id, away_id = int(teams[0]), int(teams[1])

            home_stats = {"goals": 0, "shots": 0, "ft_entries": 0}
            away_stats = {"goals": 0, "shots": 0, "ft_entries": 0}

            team_ids = group["team_id"].tolist()
            for i in range(len(group) - 1):
                group.at[i, "_retain"] = float(team_ids[i] == team_ids[i + 1])
            group.at[len(group) - 1, "_retain"] = 0.0

            group["_is_away"]      = (group["team_id"] == away_id).astype(float)
            group["_goal_in_poss"] = 0.0

            states, targets = [], []
            for _, row in group.iterrows():
                row_dict = row.to_dict()
                states.append(build_state(row_dict, home_stats, away_stats))
                targets.append(build_targets(row_dict))

                tid    = int(row["team_id"])
                bucket = home_stats if tid == home_id else away_stats
                if row.get("reached_shot", 0): bucket["shots"] += 1
                if row.get("reached_s3",   0): bucket["ft_entries"] += 1

            fp_home = fingerprints.get(home_id, mean_fp).cpu()
            fp_away = fingerprints.get(away_id, mean_fp).cpu()

            self.sequences.append({
                "states":   np.stack(states).astype(np.float32),
                "targets":  np.stack(targets).astype(np.float32),
                "fp_home":  fp_home.numpy(),
                "fp_away":  fp_away.numpy(),
                "length":   len(states),
                "match_id": match_id,
            })

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        seq = self.sequences[idx]
        T   = min(seq["length"], self.max_len)

        states_pad  = np.zeros((self.max_len, STATE_DIM), dtype=np.float32)
        targets_pad = np.zeros((self.max_len, 4),         dtype=np.float32)
        mask_pad    = np.ones( self.max_len,               dtype=bool)

        states_pad[:T]  = seq["states"][:T]
        targets_pad[:T] = seq["targets"][:T]
        mask_pad[:T]    = False

        return {
            "states":  torch.tensor(states_pad),
            "targets": torch.tensor(targets_pad),
            "mask":    torch.tensor(mask_pad),
            "fp_home": torch.tensor(seq["fp_home"]),
            "fp_away": torch.tensor(seq["fp_away"]),
            "length":  T,
        }


# ── Loss ───────────────────────────────────────────────────────────────────────

TARGET_WEIGHTS = torch.tensor([1.0, 2.0, 3.0, 1.0])   # upweight shot/goal


def simulator_loss(logits: torch.Tensor,
                   targets: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
    valid = ~mask.unsqueeze(-1)
    w     = TARGET_WEIGHTS.to(logits.device)
    bce   = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (bce * w * valid).sum() / (valid.sum() * 4)


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

        model.eval()
        val_loss = 0.0
        all_preds = [[] for _ in range(4)]
        all_tgts  = [[] for _ in range(4)]
        with torch.no_grad():
            for batch in val_loader:
                states  = batch["states"].to(DEVICE)
                targets = batch["targets"].to(DEVICE)
                mask    = batch["mask"].to(DEVICE)
                fp_home = batch["fp_home"].to(DEVICE)
                fp_away = batch["fp_away"].to(DEVICE)

                logits   = model(states, fp_home, fp_away, mask)
                val_loss += simulator_loss(logits, targets, mask).item()

                probs      = torch.sigmoid(logits)
                valid_mask = ~mask
                for i in range(4):
                    all_preds[i].append(probs[valid_mask, i].cpu().numpy())
                    all_tgts[i].append(targets[valid_mask, i].cpu().numpy())

        val_loss /= len(val_loader)

        tnames = ["advance", "shot", "goal", "retain"]
        aucs   = []
        for i in range(4):
            p = np.concatenate(all_preds[i])
            t = np.concatenate(all_tgts[i])
            aucs.append(roc_auc_score(t, p) if 0 < t.sum() < len(t) else float("nan"))

        entry = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                 **{f"auc_{tnames[i]}": aucs[i] for i in range(4)}}
        log.append(entry)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), CKPT_DIR / "simulator_best.pt")

        if epoch % 10 == 0 or epoch == 1:
            auc_str = "  ".join(
                f"{tnames[i]}={aucs[i]:.3f}" for i in range(4)
                if not math.isnan(aucs[i])
            )
            print(f"  [{epoch:3d}/{EPOCHS}]  train={train_loss:.4f}  "
                  f"val={val_loss:.4f}  {auc_str}")

    return log


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_training(log: list[dict]):
    df   = pd.DataFrame(log)
    _, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(df["epoch"], df["train_loss"], label="train")
    axes[0].plot(df["epoch"], df["val_loss"],   label="val")
    axes[0].set_title("Simulator Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    for col, lbl in [("auc_advance", "advance"), ("auc_shot", "shot"),
                     ("auc_goal", "goal"), ("auc_retain", "retain")]:
        axes[1].plot(df["epoch"], df[col], label=lbl)
    axes[1].set_title("Validation AUC per Target")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "simulator_training.png", dpi=150)
    plt.close()
    print(f"  Saved: {RESULTS_DIR / 'simulator_training.png'}")


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

    fp_data      = torch.load(FP_PATH, map_location="cpu")
    fingerprints = {k: v.float() for k, v in fp_data["team_fingerprints"].items()}
    mean_fp: torch.Tensor = fp_data.get(
        "mean_fingerprint",
        torch.stack(list(fingerprints.values())).mean(0)
    )
    print(f"Loaded {len(fingerprints)} team fingerprints")

    meta = pd.read_csv(META_PATH)
    print(f"Loaded possession meta: {len(meta):,} rows, "
          f"{meta['match_id'].nunique()} matches")

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

    model = SimulatorRNN(
        state_dim=STATE_DIM, fp_dim=FP_DIM,
        hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS, dropout=DROPOUT,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SimulatorRNN parameters: {n_params:,}")

    print("\nTraining SimulatorRNN…")
    log = train_simulator(model, train_loader, val_loader)

    pd.DataFrame(log).to_csv(RESULTS_DIR / "simulator_training.csv", index=False)
    print(f"Saved: {RESULTS_DIR / 'simulator_training.csv'}")
    plot_training(log)

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
        sim_df = simulate_match(model, fp_a, fp_b, device=DEVICE,
                                n_poss=SIM_STEPS, seed=0)
        sim_df.to_csv(RESULTS_DIR / "sim_match_demo.csv", index=False)

        final = sim_df.iloc[-1]
        print(f"  Final score: {int(final['score_home'])}–{int(final['score_away'])}")
        print(f"  Possessions: {len(sim_df)}")
        print(f"  Shots — home={int(sim_df.loc[sim_df['poss_team']==0,'shot'].sum())}  "
              f"away={int(sim_df.loc[sim_df['poss_team']==1,'shot'].sum())}")
        print(f"  Saved: {RESULTS_DIR / 'sim_match_demo.csv'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
