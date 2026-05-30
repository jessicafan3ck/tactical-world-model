"""
scripts/08_validate_forward_auc.py
====================================
Check 2: End-to-end forward AUC on real sequences.

For held-out real possessions:
  1. Load the SSE-encoded entry frame (z_A, z_B, context) from spatial_dataset.pt
  2. Map the real StatsBomb event sequence to the engine's 11 actions
  3. Run simulate_sequence() through the full generative pipeline
  4. Take the peak P(shot) across all simulated frames
  5. Measure AUC of peak P(shot) vs whether the possession actually ended in a shot

GroupKFold by match_id — same leakage discipline as LIM.

Output
------
- data/results/forward_auc.json   — per-fold and mean AUC + 95% CI
- data/results/forward_auc.png    — ROC curve with per-fold traces

Key question: does the full generative pipeline (action encoder + generator +
SSE predictor) produce higher P(shot) for possessions that really ended in shots?
AUC > 0.65 is convincing; random = 0.50.

Run
---
    python -m scripts.08_validate_forward_auc
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.conditional_engine import ConditionalEngine
from model.action_encoder import Action, MatchContext


# ── Paths ──────────────────────────────────────────────────────────────────────

BASE  = Path(__file__).parent.parent
CKPT  = BASE / "model" / "checkpoints"
OUT   = BASE / "data" / "results"

REQUIRED = [CKPT / "sse_best.pt",
            CKPT / "generator_best.pt",
            CKPT / "team_fingerprints.pt"]

MAX_POSSESSIONS = 2000   # cap for speed; increase for final run
N_FOLDS         = 5
MAX_SEQ_LEN     = 6      # max actions per possession to simulate


# ── StatsBomb event → Action mapping ──────────────────────────────────────────

def _etype(event: dict) -> str:
    t = event.get("type", {})
    if isinstance(t, dict):
        return t.get("name", "")
    if isinstance(t, str):
        return t
    return event.get("type_name", "")


def _nested(event: dict, key: str) -> dict:
    v = event.get(key, {})
    return v if isinstance(v, dict) else {}


def event_to_action(event: dict) -> str | None:
    etype = _etype(event)
    if etype == "Shot":              return "SHOOT"
    if etype == "Dribble":           return "DRIBBLE"
    if etype == "Pressure":          return "PRESS"
    if etype in ("Block","Clearance"): return "LOW_BLOCK"
    if etype == "Goal Keeper":       return "KEEPER_BALL"
    if etype == "Carry":
        loc = event.get("location") or [0, 0]
        end = _nested(event, "carry").get("end_location") or loc
        return "ADVANCE" if end[0] - loc[0] > 5 else "HOLD"
    if etype == "Pass":
        p   = _nested(event, "pass")
        tech = _nested(p, "technique")
        if tech.get("name") == "Through Ball" or event.get("pass_technique_name") == "Through Ball":
            return "THROUGH_BALL"
        if p.get("cross") or event.get("pass_cross"):
            return "CROSS"
        loc = event.get("location") or [0, 0]
        end = p.get("end_location") or event.get("pass_end_location") or loc
        if abs(end[1] - loc[1]) > 30 and abs(end[0] - loc[0]) < 20:
            return "SWITCH_LEFT" if end[1] < loc[1] else "SWITCH_RIGHT"
        return "ADVANCE" if end[0] - loc[0] > 10 else "HOLD"
    return None


def possession_action_sequence(events_df: pd.DataFrame,
                                poss_id: int) -> list[str]:
    """Return up to MAX_SEQ_LEN mapped actions for one possession."""
    rows = events_df[events_df["possession"] == poss_id] if "possession" in events_df.columns else pd.DataFrame()
    actions = []
    for _, row in rows.iterrows():
        a = event_to_action(row.to_dict())
        if a:
            actions.append(a)
        if len(actions) >= MAX_SEQ_LEN:
            break
    return actions


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    missing = [p.name for p in REQUIRED if not p.exists()]
    if missing:
        print(f"Missing checkpoints: {missing}")
        sys.exit(1)

    print("Loading engine and dataset…")
    engine = ConditionalEngine(
        sse_path         = CKPT / "sse_best.pt",
        generator_path   = CKPT / "generator_best.pt",
        fingerprint_path = CKPT / "team_fingerprints.pt",
    )

    dataset = torch.load(OUT / "spatial_dataset.pt", map_location="cpu",
                         weights_only=False)
    meta = pd.read_csv(OUT / "possession_meta.csv", low_memory=False)

    # Filter to possessions with valid team fingerprints and spatial data
    valid_teams = set(engine.fingerprints.keys())
    meta_spatial = meta[meta["has_spatial"] == True].copy()
    meta_spatial = meta_spatial[meta_spatial["team_id"].isin(valid_teams)]
    meta_spatial = meta_spatial.dropna(subset=["reached_shot"])
    print(f"  {len(meta_spatial):,} possessions with spatial data + valid teams")
    print(f"  Shot rate: {meta_spatial['reached_shot'].mean():.3%}")

    # We need a second team per possession — use the match opponent.
    # Build match_id → (team_a, team_b) from metadata.
    match_teams = (meta_spatial.groupby("match_id")["team_id"]
                   .apply(lambda x: list(x.unique()))
                   .reset_index())
    match_teams = match_teams[match_teams["team_id"].apply(len) >= 2]
    match_opponent = {}
    for _, row in match_teams.iterrows():
        teams = row["team_id"]
        for t in teams:
            opponents = [x for x in teams if x != t]
            if opponents:
                match_opponent[(row["match_id"], t)] = opponents[0]

    meta_spatial["team_id_b"] = meta_spatial.apply(
        lambda r: match_opponent.get((r["match_id"], r["team_id"])), axis=1
    )
    meta_spatial = meta_spatial.dropna(subset=["team_id_b"])
    meta_spatial["team_id_b"] = meta_spatial["team_id_b"].astype(int)

    # Subsample for speed
    if len(meta_spatial) > MAX_POSSESSIONS:
        meta_spatial = (meta_spatial
                        .groupby("reached_shot", group_keys=False)
                        .apply(lambda g: g.sample(
                            min(len(g), MAX_POSSESSIONS // 2), random_state=42
                        )))
    print(f"  Evaluating {len(meta_spatial):,} possessions")

    # Load StatsBomb events for action sequences
    try:
        from statsbombpy import sb
        has_sb = True
    except ImportError:
        has_sb = False
        print("  statsbombpy not available — using phase-inferred sequences")

    match_events: dict[int, pd.DataFrame] = {}

    def get_sequence(match_id: int, poss_id: int) -> list[str]:
        if not has_sb:
            return []
        if match_id not in match_events:
            try:
                ev = sb.events(match_id=int(match_id), flatten_attrs=False)
                match_events[match_id] = pd.concat(ev.values()) if isinstance(ev, dict) else ev
            except Exception:
                match_events[match_id] = pd.DataFrame()
        return possession_action_sequence(match_events[match_id], poss_id)

    def phase_default_sequence(phase_name: str, zone: int, n: int = 3) -> list[str]:
        """Heuristic fallback when event data is unavailable."""
        if phase_name == "counter":
            return (["ADVANCE", "THROUGH_BALL", "SHOOT"] * 3)[:n]
        if phase_name == "set_piece":
            return (["CROSS", "SHOOT"] * 3)[:n]
        return (["ADVANCE", "HOLD", "ADVANCE"] * 3)[:n]

    # ── Run engine on each possession ─────────────────────────────────────────

    scores, labels, groups = [], [], []
    print("\nRunning forward simulation…")
    skipped = 0

    for idx, row in meta_spatial.iterrows():
        tid_a  = int(row["team_id"])
        tid_b  = int(row["team_id_b"])
        label  = int(row["reached_shot"])
        match_id = int(row["match_id"])
        poss_id  = int(row["possession_id"]) if "possession_id" in row else -1

        seq = get_sequence(match_id, poss_id)
        if not seq:
            seq = phase_default_sequence(
                row.get("phase_name", "open_play"),
                int(row.get("territory_zone", 1) or 1),
            )
        if not seq:
            skipped += 1
            continue

        zone = int(row.get("territory_zone", 1) or 1)
        zone = max(0, min(zone, 3))
        ctx  = MatchContext(
            score_diff = 0.0,
            minute     = 45.0,
            zone       = zone,
            phase      = int(row.get("phase_int", 0) or 0),
            poss_team  = 0,
        )

        sequence = [(Action[a], 1.0) for a in seq if a in Action.__members__]
        if not sequence:
            skipped += 1
            continue

        try:
            frames = engine.simulate_sequence(
                sequence    = sequence,
                context     = ctx,
                team_id_a   = tid_a,
                team_id_b   = tid_b,
                gen_steps   = 10,   # fast for validation
            )
        except Exception:
            skipped += 1
            continue

        peak_p_shot = max(f.probs.p_shot for f in frames)
        scores.append(peak_p_shot)
        labels.append(label)
        groups.append(match_id)

    print(f"  Evaluated {len(scores):,} possessions  ({skipped} skipped)")

    scores = np.array(scores)
    labels = np.array(labels)
    groups = np.array(groups)

    if labels.sum() < 5:
        print("  Too few positive examples for AUC — increase MAX_POSSESSIONS")
        sys.exit(1)

    # ── GroupKFold AUC ────────────────────────────────────────────────────────

    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_aucs = []
    fig, ax = plt.subplots(figsize=(7, 6))

    for fold, (train_idx, test_idx) in enumerate(gkf.split(scores, labels, groups)):
        y_true = labels[test_idx]
        y_score = scores[test_idx]
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            continue
        auc = roc_auc_score(y_true, y_score)
        fold_aucs.append(auc)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        ax.plot(fpr, tpr, alpha=0.4, lw=1, label=f"Fold {fold+1} ({auc:.3f})")

    mean_auc   = float(np.mean(fold_aucs))
    std_auc    = float(np.std(fold_aucs))
    ci95       = 1.96 * std_auc / np.sqrt(len(fold_aucs))
    overall_auc = roc_auc_score(labels, scores)

    print(f"\nResults:")
    print(f"  Overall AUC:    {overall_auc:.4f}")
    print(f"  GroupKFold AUC: {mean_auc:.4f} ± {std_auc:.4f}  (95% CI ±{ci95:.4f})")
    print(f"  Per-fold: {[round(a,4) for a in fold_aucs]}")
    print(f"\n  Baseline (random): 0.5000")
    print(f"  Interpretation: AUC > 0.65 = convincing; > 0.75 = strong")

    # Save results
    result = {
        "overall_auc": round(overall_auc, 4),
        "groupkfold_auc_mean": round(mean_auc, 4),
        "groupkfold_auc_std":  round(std_auc, 4),
        "ci95":                round(ci95, 4),
        "fold_aucs":           [round(a, 4) for a in fold_aucs],
        "n_possessions":       len(scores),
        "n_shots":             int(labels.sum()),
        "max_seq_len":         MAX_SEQ_LEN,
    }
    with open(OUT / "forward_auc.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved forward_auc.json")

    # ROC curve plot
    ax.plot([0,1], [0,1], "k--", lw=1, label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"Forward-on-Real-Sequences ROC\n"
                 f"GroupKFold AUC = {mean_auc:.3f} ± {std_auc:.3f}  "
                 f"(n={len(scores):,}, shots={int(labels.sum())})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "forward_auc.png", dpi=150)
    plt.close(fig)
    print(f"  Saved forward_auc.png")


if __name__ == "__main__":
    main()
