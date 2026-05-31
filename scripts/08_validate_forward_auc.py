"""
scripts/08_validate_forward_auc.py
====================================
Check 2 (revised): End-to-end forward AUC on real sequences.

Previous run used `reached_shot` (0.4% positive rate, ~5-6 cases) — too
sparse for a meaningful signal.  This version uses the proper powered targets:

  reached_s2 : did the possession reach zone 2?          (19% base rate)
  reached_s3 : did the possession reach the final third? ( 8.5% base rate)

For each held-out possession:
  1. Map the real StatsBomb event sequence to engine actions.
  2. Run simulate_sequence() with the grounded action encoder.
  3. Track peak P(advance), peak P(final_third), peak P(shot) across frames.
  4. Compute GroupKFold AUC vs each target.

Fallback analysis: sequences that fell back to `phase_default_sequence`
are reported separately so the real-sequence stratum can be isolated.

Output
------
- data/results/forward_auc.json   — AUC for all three targets, strata counts
- data/results/forward_auc.png    — ROC curves (s2, s3 primary)

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
from utils.statsbomb_utils import iter_possessions, loc_to_zone


# ── Paths ──────────────────────────────────────────────────────────────────────

BASE  = Path(__file__).parent.parent
CKPT  = BASE / "model" / "checkpoints"
OUT   = BASE / "data" / "results"

REQUIRED = [CKPT / "sse_best.pt",
            CKPT / "generator_best.pt",
            CKPT / "team_fingerprints.pt"]

MAX_POSSESSIONS = 2000   # per target class (positive + negative balanced)
N_FOLDS         = 5
MAX_SEQ_LEN     = 6
FIXED_HORIZON   = 4      # steps for mean-aggregation modes (holds sequence length constant)


# ── StatsBomb event → Action mapping ──────────────────────────────────────────

def _etype(event: dict) -> str:
    t = event.get("type", {})
    if isinstance(t, dict): return t.get("name", "")
    if isinstance(t, str):  return t
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
        p    = _nested(event, "pass")
        tech = _nested(p, "technique")
        if tech.get("name") == "Through Ball" or event.get("pass_technique_name") == "Through Ball":
            return "THROUGH_BALL"
        if p.get("cross") or event.get("pass_cross"):
            return "CROSS"
        loc = event.get("location") or [0, 0]
        end = p.get("end_location") or event.get("pass_end_location") or loc
        if abs(end[1]-loc[1]) > 30 and abs(end[0]-loc[0]) < 20:
            return "SWITCH_LEFT" if end[1] < loc[1] else "SWITCH_RIGHT"
        return "ADVANCE" if end[0]-loc[0] > 10 else "HOLD"
    return None



def phase_default_sequence(phase_name: str, zone: int, n: int = 3) -> list[str]:
    if phase_name == "counter":
        return (["ADVANCE", "THROUGH_BALL", "SHOOT"] * 3)[:n]
    if phase_name == "set_piece":
        return (["CROSS", "SHOOT"] * 3)[:n]
    return (["ADVANCE", "HOLD", "ADVANCE"] * 3)[:n]


# ── GroupKFold AUC helper ─────────────────────────────────────────────────────

def groupkfold_auc(scores: np.ndarray, labels: np.ndarray,
                   groups: np.ndarray, n_folds: int,
                   ax, title: str) -> dict:
    """Compute GroupKFold AUC and plot ROC traces onto ax."""
    gkf      = GroupKFold(n_splits=n_folds)
    fold_aucs = []
    for fold, (_, test_idx) in enumerate(gkf.split(scores, labels, groups)):
        y_true  = labels[test_idx]
        y_score = scores[test_idx]
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            continue
        auc = roc_auc_score(y_true, y_score)
        fold_aucs.append(auc)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        ax.plot(fpr, tpr, alpha=0.35, lw=1, label=f"Fold {fold+1} ({auc:.3f})")

    if not fold_aucs:
        return {"error": "no valid folds"}
    mean_auc = float(np.mean(fold_aucs))
    std_auc  = float(np.std(fold_aucs))
    ci95     = 1.96 * std_auc / np.sqrt(len(fold_aucs))
    overall  = roc_auc_score(labels, scores)

    ax.plot([0,1],[0,1],"k--",lw=1,label="Random")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"{title}\nGroupKFold AUC = {mean_auc:.3f} ± {std_auc:.3f}  "
                 f"(overall={overall:.3f}, n={len(labels):,})")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    return {
        "overall_auc": round(overall, 4),
        "mean":        round(mean_auc, 4),
        "std":         round(std_auc, 4),
        "ci95":        round(ci95, 4),
        "folds":       [round(a, 4) for a in fold_aucs],
        "n":           int(len(labels)),
        "n_positive":  int(labels.sum()),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    missing = [p.name for p in REQUIRED if not p.exists()]
    if missing:
        print(f"Missing: {missing}"); sys.exit(1)

    print("Loading engine…")
    engine = ConditionalEngine(
        sse_path         = CKPT / "sse_best.pt",
        generator_path   = CKPT / "generator_best.pt",
        fingerprint_path = CKPT / "team_fingerprints.pt",
    )

    meta = pd.read_csv(OUT / "possession_meta.csv", low_memory=False)
    valid_teams   = set(engine.fingerprints.keys())
    meta_spatial  = meta[meta["has_spatial"] == True].copy()
    meta_spatial  = meta_spatial[meta_spatial["team_id"].isin(valid_teams)]
    meta_spatial  = meta_spatial.dropna(subset=["reached_s2", "reached_s3"])
    meta_spatial["reached_shot"] = meta_spatial.get("reached_shot", 0).fillna(0).astype(int)

    print(f"  {len(meta_spatial):,} possessions with spatial data + valid teams")
    for col in ("reached_s2", "reached_s3", "reached_shot"):
        if col in meta_spatial.columns:
            print(f"  {col}: {meta_spatial[col].mean():.3%} positive")

    # Build match → opponent lookup
    match_teams = (meta_spatial.groupby("match_id")["team_id"]
                   .apply(lambda x: list(x.unique())).reset_index())
    match_teams = match_teams[match_teams["team_id"].apply(len) >= 2]
    match_opponent = {}
    for _, row in match_teams.iterrows():
        for t in row["team_id"]:
            opp = [x for x in row["team_id"] if x != t]
            if opp:
                match_opponent[(row["match_id"], t)] = opp[0]
    meta_spatial["team_id_b"] = meta_spatial.apply(
        lambda r: match_opponent.get((r["match_id"], r["team_id"])), axis=1)
    meta_spatial = meta_spatial.dropna(subset=["team_id_b"])
    meta_spatial["team_id_b"] = meta_spatial["team_id_b"].astype(int)

    # Balanced subsample on reached_s2 (primary target)
    if len(meta_spatial) > MAX_POSSESSIONS * 2:
        meta_spatial = (meta_spatial
                        .groupby("reached_s2", group_keys=False)
                        .apply(lambda g: g.sample(
                            min(len(g), MAX_POSSESSIONS), random_state=42)))
    print(f"  Evaluating {len(meta_spatial):,} possessions")

    # Precompute match-scoped possession index.
    # possession_id in possession_meta.csv is a global sequential counter (poss_counter
    # in 02_build_dataset.py).  Within a match, possessions were processed in the order
    # iter_possessions yields them, so rank-within-match = match-scoped index.
    # We rank against the FULL meta (not just spatial rows) because poss_counter increments
    # for all valid possessions regardless of whether they have 360 data.
    print("  Computing match-scoped possession indices…")
    full_meta = pd.read_csv(OUT / "possession_meta.csv",
                            usecols=["possession_id", "match_id"], low_memory=False)
    full_meta["match_poss_idx"] = (
        full_meta.groupby("match_id")["possession_id"]
        .rank(method="first").astype(int) - 1
    )
    poss_idx_map: dict[int, int] = dict(
        zip(full_meta["possession_id"], full_meta["match_poss_idx"])
    )
    del full_meta  # free memory

    RAW_EVENTS = BASE / "data" / "raw" / "statsbomb" / "events"
    N_WINDOW = 8

    _raw_poss_cache: dict[int, list[list[dict]]] = {}

    def _load_match_possessions(match_id: int) -> list[list[dict]]:
        """Run iter_possessions on raw JSON, applying the same filters as 02_build_dataset."""
        path = RAW_EVENTS / f"{match_id}.json"
        if not path.exists():
            return []
        with open(path) as f:
            events = json.load(f)
        result = []
        for poss in iter_possessions(events, match_id, {}):
            pev = poss["events"]
            if len(pev) < 2:
                continue
            # Replicate all-NaN filter from compute_outcomes
            window = pev[:N_WINDOW]
            entry_state = 0
            for ev in window:
                z = loc_to_zone(ev.get("location"), ev.get("type", {}).get("name", ""))
                if z >= 0:
                    entry_state = z
                    break
            shot_in_window = any(ev.get("type", {}).get("name") == "Shot" for ev in window)
            s2_nan = entry_state >= 2
            s3_nan = entry_state >= 3
            shot_nan = shot_in_window
            if s2_nan and s3_nan and shot_nan:
                continue  # all-NaN possession, skipped in build_dataset
            result.append(pev)
        return result

    def get_sequence(match_id: int, global_poss_id: int,
                     phase_name: str, zone: int) -> tuple[list[str], bool]:
        """Returns (action_list, used_real_sequence)."""
        if match_id not in _raw_poss_cache:
            _raw_poss_cache[match_id] = _load_match_possessions(match_id)

        poss_list = _raw_poss_cache[match_id]
        match_idx = poss_idx_map.get(global_poss_id, -1)

        if 0 <= match_idx < len(poss_list):
            actions = []
            for ev in poss_list[match_idx]:
                a = event_to_action(ev)
                if a:
                    actions.append(a)
                if len(actions) >= MAX_SEQ_LEN:
                    break
            if actions:
                return actions, True

        fallback = phase_default_sequence(phase_name, zone)
        return fallback, False

    def run_sequence(seq_actions: list[str], ctx: MatchContext,
                     tid_a: int, tid_b: int) -> list | None:
        """Simulate a sequence; return ActionResult list or None on error."""
        sequence = [(Action[a], 1.0) for a in seq_actions if a in Action.__members__]
        if not sequence:
            return None
        try:
            return engine.simulate_sequence(
                sequence=sequence, context=ctx,
                team_id_a=tid_a, team_id_b=tid_b, gen_steps=10,
            )
        except Exception:
            return None

    def policy_sequence(ctx: MatchContext, tid_a: int, tid_b: int,
                        n_steps: int) -> list[str]:
        """Model's own top-1 action per step from current state (outcome-independent)."""
        actions = []
        cur_ctx = ctx
        z_A = engine.fingerprints.get(tid_a, engine.mean_fp).clone().to(engine.device)
        z_B = engine.fingerprints.get(tid_b, engine.mean_fp).clone().to(engine.device)
        norm_A = z_A.norm().clamp(min=1e-8)
        for _ in range(n_steps):
            suggestions = engine.suggest_action(cur_ctx, tid_a, tid_b)
            top_action  = suggestions[0]["action"] if suggestions else "HOLD"
            actions.append(top_action)
            action_obj = Action[top_action]
            with torch.no_grad():
                z_A = engine._apply_action(z_A, action_obj, cur_ctx, 1.0)
                z_A = z_A * (norm_A / z_A.norm().clamp(min=1e-8))
            new_zone = min(cur_ctx.zone + int(action_obj.value == 0), 3)
            cur_ctx = MatchContext(
                score_diff=cur_ctx.score_diff, minute=min(cur_ctx.minute+0.5, 90.0),
                zone=new_zone, phase=cur_ctx.phase, poss_team=cur_ctx.poss_team,
            )
        return actions

    # ── Simulation loop — four sequence strategies ─────────────────────────────
    # real_peak    : real StatsBomb actions, variable length, peak aggregation   [LEAKY — kept for comparison]
    # real_mean    : real StatsBomb actions, fixed FIXED_HORIZON steps, mean     [removes peak confound]
    # policy_mean  : model's own top-1 per step, fixed horizon, mean             [outcome-independent]
    # fixed_mean   : 4×ADVANCE for every possession, fixed horizon, mean         [null baseline]

    records   = []
    skipped   = 0
    n_real    = 0
    n_fallback = 0
    print(f"\nRunning forward simulation (4 strategies, fixed horizon={FIXED_HORIZON})…")

    for _, row in meta_spatial.iterrows():
        tid_a    = int(row["team_id"])
        tid_b    = int(row["team_id_b"])
        match_id = int(row["match_id"])
        global_poss_id = int(row["possession_id"])
        phase_name     = str(row.get("phase_name", "open_play"))
        zone           = max(0, min(int(row.get("territory_zone", 1) or 1), 3))

        real_seq, used_real = get_sequence(match_id, global_poss_id, phase_name, zone)
        if not real_seq:
            skipped += 1; continue

        ctx = MatchContext(
            score_diff = 0.0, minute = 45.0, zone = zone,
            phase = int(row.get("phase_int", 0) or 0), poss_team = 0,
        )

        # ── real_peak (original Check 2 — leaky) ──────────────────────────────
        frames_real = run_sequence(real_seq, ctx, tid_a, tid_b)
        if frames_real is None:
            skipped += 1; continue

        # ── real_mean (fixed horizon, mean aggregation) ────────────────────────
        # Pad or truncate real sequence to exactly FIXED_HORIZON steps
        real_fixed = (real_seq * FIXED_HORIZON)[:FIXED_HORIZON]
        frames_real_mean = run_sequence(real_fixed, ctx, tid_a, tid_b)

        # ── policy_mean (outcome-independent: model's own top-1 per step) ──────
        pol_seq = policy_sequence(ctx, tid_a, tid_b, FIXED_HORIZON)
        frames_policy = run_sequence(pol_seq, ctx, tid_a, tid_b)

        # ── fixed_mean (null baseline: always ADVANCE×FIXED_HORIZON) ──────────
        frames_fixed = run_sequence(["ADVANCE"] * FIXED_HORIZON, ctx, tid_a, tid_b)

        def _peak(frames, attr):
            return max(getattr(f.probs, attr) for f in frames) if frames else float("nan")
        def _mean(frames, attr):
            vals = [getattr(f.probs, attr) for f in frames] if frames else [float("nan")]
            return float(np.mean(vals))

        records.append({
            # real-sequence peak (kept for reference — LEAKY)
            "real_peak_s2":    _peak(frames_real, "p_advance"),
            "real_peak_s3":    _peak(frames_real, "p_final_third"),
            "real_peak_shot":  _peak(frames_real, "p_shot"),
            # real-sequence mean, fixed horizon
            "real_mean_s2":   _mean(frames_real_mean, "p_advance")   if frames_real_mean else float("nan"),
            "real_mean_s3":   _mean(frames_real_mean, "p_final_third") if frames_real_mean else float("nan"),
            "real_mean_shot": _mean(frames_real_mean, "p_shot")      if frames_real_mean else float("nan"),
            # policy (outcome-independent) mean, fixed horizon
            "pol_mean_s2":    _mean(frames_policy, "p_advance")      if frames_policy else float("nan"),
            "pol_mean_s3":    _mean(frames_policy, "p_final_third")   if frames_policy else float("nan"),
            "pol_mean_shot":  _mean(frames_policy, "p_shot")         if frames_policy else float("nan"),
            # fixed ADVANCE mean (null baseline)
            "fix_mean_s2":    _mean(frames_fixed, "p_advance")       if frames_fixed else float("nan"),
            "fix_mean_s3":    _mean(frames_fixed, "p_final_third")    if frames_fixed else float("nan"),
            "fix_mean_shot":  _mean(frames_fixed, "p_shot")          if frames_fixed else float("nan"),
            # labels
            "reached_s2":   int(row["reached_s2"]),
            "reached_s3":   int(row["reached_s3"]),
            "reached_shot": int(row.get("reached_shot", 0)),
            "match_id":     match_id,
            "used_real":    used_real,
        })
        if used_real: n_real    += 1
        else:         n_fallback += 1

    df = pd.DataFrame(records).dropna(subset=["real_peak_s2","pol_mean_s2","fix_mean_s2"])
    print(f"  Evaluated {len(df):,} possessions  ({skipped} skipped)")
    print(f"  Real sequences: {n_real:,}  |  Phase fallbacks: {n_fallback:,}")

    if len(df) < 20:
        print("Too few results"); sys.exit(1)

    labels_s2   = df["reached_s2"].values
    labels_s3   = df["reached_s3"].values
    labels_shot = df["reached_shot"].values
    groups      = df["match_id"].values

    # ── Leakage diagnostic: four strategies × three targets ───────────────────
    strategies = [
        ("real_peak",  "real_peak_s2",  "real_peak_s3",  "real_peak_shot",
         "Real seq, PEAK agg [LEAKY — contains future actions]"),
        ("real_mean",  "real_mean_s2",  "real_mean_s3",  "real_mean_shot",
         f"Real seq, mean/{FIXED_HORIZON}-step [LEAKY — contains future actions]"),
        ("policy_mean","pol_mean_s2",   "pol_mean_s3",   "pol_mean_shot",
         f"Policy top-1, mean/{FIXED_HORIZON}-step [outcome-independent]"),
        ("fixed_mean", "fix_mean_s2",   "fix_mean_s3",   "fix_mean_shot",
         f"Fixed ADVANCE×{FIXED_HORIZON}, mean [null baseline]"),
    ]

    def quick_gkf(scores, labels, groups, n_folds=5):
        gkf = GroupKFold(n_splits=n_folds)
        aucs = []
        for _, ti in gkf.split(scores, labels, groups):
            yt, ys = labels[ti], scores[ti]
            if yt.sum() > 0 and yt.sum() < len(yt):
                aucs.append(roc_auc_score(yt, ys))
        if not aucs:
            return float("nan"), float("nan")
        return float(np.mean(aucs)), 1.96 * float(np.std(aucs)) / np.sqrt(len(aucs))

    print(f"\n{'Strategy':<46} {'s2 AUC':>8} {'s3 AUC':>8} {'shot AUC':>9}")
    print("─" * 75)
    results = {}
    for strat_key, sc2_col, sc3_col, sshot_col, label in strategies:
        sc2   = df[sc2_col].fillna(0.5).values
        sc3   = df[sc3_col].fillna(0.5).values
        sshot = df[sshot_col].fillna(0.5).values
        a2, ci2   = quick_gkf(sc2,   labels_s2,   groups)
        a3, ci3   = quick_gkf(sc3,   labels_s3,   groups)
        ash, cish = quick_gkf(sshot, labels_shot, groups)
        print(f"  {label[:44]:<44} {a2:>7.4f} {a3:>8.4f} {ash:>9.4f}")
        results[strat_key] = {"s2": round(a2,4), "s3": round(a3,4),
                               "shot": round(ash,4), "n": len(df)}

    print(f"\n  Baseline (random): 0.5000")
    print(f"\n  Leakage diagnosis:")
    rp  = results.get("real_peak",  {})
    pol = results.get("policy_mean",{})
    fix = results.get("fixed_mean", {})
    for t in ("s2","s3","shot"):
        drop = (rp.get(t,0.5) - pol.get(t,0.5))
        print(f"    {t}: real_peak={rp.get(t,'?'):.4f}  policy={pol.get(t,'?'):.4f}  "
              f"fixed={fix.get(t,'?'):.4f}  Δ(real→policy)={drop:+.4f}"
              f"  {'LEAKAGE LIKELY' if drop > 0.08 else 'signal may be genuine' if drop < 0.03 else 'ambiguous'}")

    # GroupKFold ROC plots for real_peak vs policy_mean (the key comparison)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for col_idx, (target, lbl) in enumerate([
        ("reached_s2", "s2"), ("reached_s3", "s3"), ("reached_shot", "shot")
    ]):
        labels_t = df[target].values
        if labels_t.sum() < 5:
            continue
        for row_idx, (strat_key, sc_col, title_pfx, color) in enumerate([
            ("real_peak",   f"real_peak_{lbl}",  "real seq peak",  "steelblue"),
            ("policy_mean", f"pol_mean_{lbl}",   "policy mean",    "tomato"),
        ]):
            ax = axes[row_idx, col_idx]
            scores_t = df[sc_col].fillna(0.5).values
            gkf = GroupKFold(n_splits=N_FOLDS)
            fold_aucs = []
            for fold, (_, ti) in enumerate(gkf.split(scores_t, labels_t, groups)):
                yt, ys = labels_t[ti], scores_t[ti]
                if yt.sum() == 0 or yt.sum() == len(yt):
                    continue
                auc = roc_auc_score(yt, ys)
                fold_aucs.append(auc)
                fpr, tpr, _ = roc_curve(yt, ys)
                ax.plot(fpr, tpr, alpha=0.35, lw=1, color=color,
                        label=f"Fold {fold+1} ({auc:.3f})")
            mean_a = np.mean(fold_aucs) if fold_aucs else float("nan")
            ax.plot([0,1],[0,1],"k--",lw=1)
            ax.set_title(f"{target}\n{title_pfx}  GroupKFold={mean_a:.4f}")
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
            ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

    plt.suptitle("Forward AUC Leakage Diagnostic\n"
                 "Row 1: real sequence+peak [LEAKY]   Row 2: policy top-1+mean [outcome-independent]",
                 fontsize=11)
    plt.tight_layout()

    fig.savefig(OUT / "forward_auc.png", dpi=150)
    plt.close(fig)

    full_result = {
        "encoder": "ConditionedMLP (learned, match-level split)",
        "strategies": results,
        "n_evaluated": len(df),
        "n_real_sequences": n_real,
        "n_fallback_sequences": n_fallback,
        "n_skipped": skipped,
        "fixed_horizon": FIXED_HORIZON,
        "leakage_note": (
            "real_peak and real_mean condition on the possession's actual future actions — "
            "outcome-correlated by construction. policy_mean and fixed_mean are "
            "outcome-independent. Compare real_peak vs policy_mean to diagnose leakage."
        ),
    }
    with open(OUT / "forward_auc.json", "w") as f:
        json.dump(full_result, f, indent=2)
    print(f"\n  Saved forward_auc.json + forward_auc.png")


if __name__ == "__main__":
    main()
