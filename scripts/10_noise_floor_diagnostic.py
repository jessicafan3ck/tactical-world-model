"""
scripts/10_noise_floor_diagnostic.py
======================================
Noise-floor diagnostic for the action encoder.

For each action type, collect real Δz = z_after − z_before from 360 frames,
then measure *how consistent* the direction of Δz is across repeated instances.

Key metric — resultant length R (Von Mises concentration):
    δ_i  = Δz_i / ‖Δz_i‖     (unit-length direction)
    R_a  = ‖ mean_i(δ_i) ‖    (for action a)

R_a ∈ [0, 1].  R_a = 1 → all instances move z in the same direction → a
deterministic point-predictor can saturate Check 1.  R_a ≈ 0 → the action
scatters in many directions → no deterministic encoder can do better than
cosine ≈ R_a on the diagonal, and a probabilistic (Gaussian / MDN) model
is not optional, it is structurally necessary.

State-dependence test:
    Cluster z_before with k-means (k=8).  If within-cluster R > marginal R,
    Δz direction depends on where you start → conditioning on z_before
    (state-dependent transform) recovers meaningful signal.

Magnitude analysis:
    coefficient-of-variation of ‖Δz‖ per action — complements direction.

Output
------
- data/results/noise_floor.json   — per-action R, CV, state-dependence gain
- data/results/noise_floor.png    — visual summary (R bar + scatter)

Run
---
    python -m scripts.10_noise_floor_diagnostic
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.conditional_engine import ConditionalEngine
from model.action_encoder import Action, ACTION_LABELS

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE = Path(__file__).parent.parent
CKPT = BASE / "model" / "checkpoints"
OUT  = BASE / "data" / "results"

REQUIRED = [CKPT / "sse_best.pt",
            CKPT / "generator_best.pt",
            CKPT / "team_fingerprints.pt"]

MAX_PER_ACTION = 300   # ceiling per action — more = tighter estimates
MAX_MATCHES    = 50    # search up to this many matches
N_CLUSTERS     = 8     # k for z_before state clustering


# ── Re-use the data-collection helpers from script 07 ─────────────────────────

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
    if etype == "Shot":                return "SHOOT"
    if etype == "Dribble":            return "DRIBBLE"
    if etype == "Pressure":           return "PRESS"
    if etype in ("Block","Clearance"): return "LOW_BLOCK"
    if etype == "Goal Keeper":        return "KEEPER_BALL"
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
        if abs(end[1] - loc[1]) > 30 and abs(end[0] - loc[0]) < 20:
            return "SWITCH_LEFT" if end[1] < loc[1] else "SWITCH_RIGHT"
        return "ADVANCE" if end[0] - loc[0] > 10 else "HOLD"
    return None


def parse_360_frame(frame_data: list) -> torch.Tensor | None:
    pts = []
    for p in frame_data:
        loc  = p.get("location") or [0, 0]
        mate = 1.0 if p.get("teammate", False) else 0.0
        actor = 1.0 if p.get("actor", False) else 0.0
        pts.append([loc[0] / 120.0, loc[1] / 80.0, mate, actor])
    if len(pts) < 6:
        return None
    t = torch.zeros(23, 4)
    n = min(len(pts), 23)
    t[:n] = torch.tensor(pts[:n], dtype=torch.float32)
    return t


def make_mask(n_visible: int) -> torch.Tensor:
    mask = torch.zeros(23, dtype=torch.bool)
    mask[n_visible:] = True
    return mask


def ctx_from_event(event: dict) -> torch.Tensor:
    loc  = event.get("location") or [60, 40]
    zone = min(int(loc[0] / 30), 3)
    return torch.tensor([[zone / 3.0, 0.0, 0.0]], dtype=torch.float32)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    missing = [p.name for p in REQUIRED if not p.exists()]
    if missing:
        print(f"Missing: {missing}")
        sys.exit(1)

    print("Loading engine…")
    engine = ConditionalEngine(
        sse_path         = CKPT / "sse_best.pt",
        generator_path   = CKPT / "generator_best.pt",
        fingerprint_path = CKPT / "team_fingerprints.pt",
    )

    meta     = pd.read_csv(OUT / "possession_meta.csv", low_memory=False)
    match_ids = meta["match_id"].unique()
    print(f"  {len(match_ids)} matches in metadata")

    try:
        from statsbombpy import sb
    except ImportError:
        print("statsbombpy not installed"); sys.exit(1)

    ACTION_NAMES = [a.name for a in Action]
    # Collect (z_before, z_after, Δz) per action
    z_before_store: dict[str, list[np.ndarray]] = {a: [] for a in ACTION_NAMES}
    dz_store:       dict[str, list[np.ndarray]] = {a: [] for a in ACTION_NAMES}

    print(f"\nCollecting pairs from up to {MAX_MATCHES} matches…")
    matches_done = 0
    for mid in match_ids:
        if matches_done >= MAX_MATCHES:
            break
        if all(len(v) >= MAX_PER_ACTION for v in dz_store.values()):
            break

        try:
            df = sb.events(match_id=int(mid), flatten_attrs=False)
            if isinstance(df, dict):
                df = pd.concat(df.values(), ignore_index=True)
            _f360 = sb.frames(match_id=int(mid))
            frames_360 = {
                eid: grp.to_dict("records")
                for eid, grp in _f360.groupby("id")
            }
        except Exception:
            continue

        matches_done += 1
        df = df.sort_values("index") if "index" in df.columns else df

        for i in range(len(df) - 1):
            row_b = df.iloc[i]
            row_a = df.iloc[i + 1]

            act = event_to_action(row_b.to_dict())
            if act is None or len(dz_store[act]) >= MAX_PER_ACTION:
                continue

            eid_b = row_b.get("id")
            eid_a = row_a.get("id")
            if not eid_b or not eid_a:
                continue

            ff_b = frames_360.get(eid_b)
            ff_a = frames_360.get(eid_a)
            if not ff_b or not ff_a:
                continue

            pos_b = parse_360_frame(ff_b)
            pos_a = parse_360_frame(ff_a)
            if pos_b is None or pos_a is None:
                continue

            msk_b = make_mask(min(len(ff_b), 23))
            msk_a = make_mask(min(len(ff_a), 23))
            ctx_b = ctx_from_event(row_b.to_dict())

            z_b = engine.encode_frame(pos_b.unsqueeze(0), msk_b.unsqueeze(0), ctx_b).squeeze(0)  # (256,)
            z_a = engine.encode_frame(pos_a.unsqueeze(0), msk_a.unsqueeze(0), ctx_b).squeeze(0)

            dz = (z_a - z_b).numpy()
            z_before_store[act].append(z_b.numpy())
            dz_store[act].append(dz)

    n_collected = {a: len(v) for a, v in dz_store.items()}
    print("  Pairs per action:", {a: n for a, n in n_collected.items()})
    total = sum(n_collected.values())
    if total < 11:
        print("Insufficient data — re-run with more matches"); sys.exit(1)

    # ── Per-action noise-floor metrics ────────────────────────────────────────

    print("\nComputing noise-floor metrics…")
    results: dict[str, dict] = {}
    action_labels = [ACTION_LABELS[a] for a in Action]

    for act, label in zip(ACTION_NAMES, action_labels):
        dzs = np.array(dz_store[act])          # (N, 256)
        zbs = np.array(z_before_store[act])    # (N, 256)
        if len(dzs) < 5:
            results[act] = {"label": label, "n": len(dzs), "note": "insufficient data"}
            continue

        # Directional consistency — resultant length R
        norms = np.linalg.norm(dzs, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1e-9, norms)
        directions = dzs / norms             # unit vectors
        mean_dir   = directions.mean(axis=0)
        R          = float(np.linalg.norm(mean_dir))   # resultant length ∈ [0, 1]

        # Mean pairwise cosine (alternative consistency measure)
        # = (R² × N - 1) / (N - 1)  for unit vectors  ← expensive; use R instead

        # Magnitude coefficient of variation
        mag_mean = float(norms.mean())
        mag_std  = float(norms.std())
        mag_cv   = mag_std / max(mag_mean, 1e-9)

        # State-dependence: does within-cluster R > marginal R?
        # Cluster z_before into N_CLUSTERS groups
        state_gain = float("nan")
        if len(zbs) >= N_CLUSTERS * 2:
            km = KMeans(n_clusters=N_CLUSTERS, n_init=5, random_state=42)
            labels_km = km.fit_predict(zbs)
            cluster_Rs = []
            for c in range(N_CLUSTERS):
                mask_c = labels_km == c
                if mask_c.sum() < 3:
                    continue
                dirs_c    = directions[mask_c]
                mean_dir_c = dirs_c.mean(axis=0)
                cluster_Rs.append(np.linalg.norm(mean_dir_c))
            if cluster_Rs:
                state_gain = float(np.mean(cluster_Rs)) - R

        results[act] = {
            "label":       label,
            "n":           len(dzs),
            "R":           round(R, 4),        # resultant length (= achievable cosine ceiling)
            "mag_mean":    round(mag_mean, 4),
            "mag_cv":      round(mag_cv, 4),   # magnitude coefficient of variation
            "state_gain":  round(state_gain, 4) if not np.isnan(state_gain) else None,
        }

    # ── Print summary ─────────────────────────────────────────────────────────

    print(f"\n{'Action':<20} {'N':>5} {'R (ceiling)':>12} {'Mag CV':>8} {'State gain':>12}")
    print("-" * 62)
    for act in ACTION_NAMES:
        r = results[act]
        if "note" in r:
            print(f"  {r['label']:<18} {r['n']:>5}  {'—':>12}")
            continue
        sg = f"{r['state_gain']:+.3f}" if r["state_gain"] is not None else "   n/a"
        print(f"  {r['label']:<18} {r['n']:>5} {r['R']:>12.4f} {r['mag_cv']:>8.3f} {sg:>12}")

    valid_R = [r["R"] for r in results.values() if "R" in r]
    print(f"\n  Mean R across actions: {np.mean(valid_R):.4f}")
    print(f"  Min R: {np.min(valid_R):.4f}   Max R: {np.max(valid_R):.4f}")
    print()
    print("  Interpretation:")
    print("  R > 0.5  → action has strong directional signal; deterministic encoder viable")
    print("  R ≈ 0.3  → partial signal; conditioned MLP likely needed")
    print("  R < 0.2  → noisy; probabilistic encoder (Gaussian / MDN) not optional")
    print()
    state_gains = [r["state_gain"] for r in results.values()
                   if r.get("state_gain") is not None]
    if state_gains:
        mean_sg = np.mean(state_gains)
        print(f"  Mean state-dependence gain: {mean_sg:+.4f}")
        if mean_sg > 0.05:
            print("  → within-cluster R meaningfully higher: state-conditioning is worthwhile")
        else:
            print("  → state-conditioning recovers little: action bias is mostly state-independent")

    # ── Plot ─────────────────────────────────────────────────────────────────

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: R bar chart with state-gain overlay
    ax = axes[0]
    acts_with_data = [a for a in ACTION_NAMES if "R" in results[a]]
    Rs   = [results[a]["R"] for a in acts_with_data]
    labs = [results[a]["label"] for a in acts_with_data]
    sgs  = [results[a]["state_gain"] or 0.0 for a in acts_with_data]

    x = np.arange(len(acts_with_data))
    bars = ax.bar(x, Rs, color="#58a6ff", alpha=0.8, label="Marginal R")
    ax.bar(x, sgs, bottom=Rs, color="#3fb950", alpha=0.7, label="State-dep. gain")
    ax.axhline(0.5, color="orange", lw=1, ls="--", label="R=0.5 (viable floor)")
    ax.axhline(0.2, color="red",    lw=1, ls="--", label="R=0.2 (MDN mandatory)")
    ax.set_xticks(x); ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Resultant length R  (= achievable cosine ceiling)")
    ax.set_title("Action noise floor: directional consistency of real Δz\n"
                 "R close to 1 → low noise; R near 0 → high noise")
    ax.set_ylim(0, max(1.0, max(Rs) + 0.1))
    ax.legend(fontsize=8)

    # Right: scatter — R vs magnitude CV (coloured by state-gain)
    ax2 = axes[1]
    CVs = [results[a]["mag_cv"] for a in acts_with_data]
    sc  = ax2.scatter(Rs, CVs, c=sgs, cmap="RdYlGn", s=80, vmin=-0.1, vmax=0.3, zorder=3)
    for i, lbl in enumerate(labs):
        ax2.annotate(lbl, (Rs[i], CVs[i]), fontsize=7,
                     xytext=(4, 3), textcoords="offset points")
    ax2.axvline(0.5, color="orange", lw=1, ls="--")
    ax2.axvline(0.2, color="red",    lw=1, ls="--")
    ax2.set_xlabel("Resultant length R (direction consistency)")
    ax2.set_ylabel("Magnitude CV  (‖Δz‖ coefficient of variation)")
    ax2.set_title("R vs magnitude variability per action\n"
                  "colour = state-dependence gain (green = worth conditioning)")
    plt.colorbar(sc, ax=ax2, label="State-dep. gain in R")

    plt.suptitle("Action Encoder Noise-Floor Diagnostic\n"
                 "R = achievable cosine ceiling for any deterministic encoder",
                 fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT / "noise_floor.png", dpi=150)
    plt.close(fig)
    print(f"\n  Saved noise_floor.png")

    with open(OUT / "noise_floor.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved noise_floor.json")


if __name__ == "__main__":
    main()
