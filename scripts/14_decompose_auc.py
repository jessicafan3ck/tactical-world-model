"""
scripts/14_decompose_auc.py
============================
Three-way AUC error decomposition.

Isolates where the ~0.05 AUC gap between the direct SSE ceiling and the
full forward pipeline actually lives: generator or action encoder.

Three stages run on the *same* set of possessions as Check 2:

  Stage 1 — SSE on real frames (ceiling)
    Load the real StatsBomb freeze frame for each possession.
    Pass raw positions directly to the SSE → probs.
    AUC = maximum achievable by this encoder on ground truth.

  Stage 2 — SSE on baseline-generated frames (isolates generator error)
    Same z_A fingerprint, same context, NO action transform.
    Generate one frame → SSE → probs.
    AUC drop vs Stage 1 = generator's contribution to error.

  Stage 3 — SSE on action-transformed generated frames (adds encoder error)
    Apply the real first action via the learned encoder to z_A.
    Generate one frame → SSE → probs.
    AUC drop vs Stage 2 = action encoder's contribution to error.

Using the first real action (not the full peak-over-sequence from Check 2)
keeps Stage 2 vs Stage 3 comparable on a single transform step.

Also reports the p_advance score distribution to test for ceiling saturation
(a uniform forward bias only hurts AUC if scores pile up near 1).

Output
------
- data/results/decompose_auc.json   stage AUCs + attribution
- data/results/decompose_auc.png    ROC curves, score histograms

Run
---
    python -m scripts.14_decompose_auc
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

BASE = Path(__file__).parent.parent
CKPT = BASE / "model" / "checkpoints"
OUT  = BASE / "data" / "results"

REQUIRED = [CKPT / "sse_best.pt",
            CKPT / "generator_best.pt",
            CKPT / "team_fingerprints.pt"]

MAX_POSSESSIONS = 2000
N_FOLDS         = 5
MAX_SEQ_LEN     = 6
GEN_STEPS       = 10
N_WINDOW        = 8


# ── StatsBomb helpers ─────────────────────────────────────────────────────────

def _etype(event: dict) -> str:
    t = event.get("type", {})
    if isinstance(t, dict): return t.get("name", "")
    if isinstance(t, str):  return t
    return event.get("type_name", "")


def event_to_action(event: dict) -> str | None:
    e = _etype(event)
    if e == "Shot":              return "SHOOT"
    if e == "Dribble":           return "DRIBBLE"
    if e == "Pressure":          return "PRESS"
    if e in ("Block","Clearance"): return "LOW_BLOCK"
    if e == "Goal Keeper":       return "KEEPER_BALL"
    if e == "Carry":
        loc = event.get("location") or [0, 0]
        end = (event.get("carry") or {}).get("end_location") or loc
        return "ADVANCE" if end[0] - loc[0] > 5 else "HOLD"
    if e == "Pass":
        p    = event.get("pass") or {}
        tech = p.get("technique") or {}
        if tech.get("name") == "Through Ball": return "THROUGH_BALL"
        if p.get("cross"):                     return "CROSS"
        loc = event.get("location") or [0, 0]
        end = p.get("end_location") or loc
        if abs(end[1]-loc[1]) > 30 and abs(end[0]-loc[0]) < 20:
            return "SWITCH_LEFT" if end[1] < loc[1] else "SWITCH_RIGHT"
        return "ADVANCE" if end[0] - loc[0] > 10 else "HOLD"
    return None


RAW_EVENTS = BASE / "data" / "raw" / "statsbomb" / "events"
_poss_cache: dict[int, list[list[dict]]] = {}


def _load_match_possessions(match_id: int) -> list[list[dict]]:
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
        window = pev[:N_WINDOW]
        entry_state = 0
        for ev in window:
            z = loc_to_zone(ev.get("location"), ev.get("type", {}).get("name", ""))
            if z >= 0:
                entry_state = z; break
        shot_in_window = any(_etype(ev) == "Shot" for ev in window)
        if entry_state >= 2 and shot_in_window:
            continue
        result.append(pev)
    return result


def get_first_action(match_id: int, global_poss_id: int,
                     poss_idx_map: dict) -> str | None:
    if match_id not in _poss_cache:
        _poss_cache[match_id] = _load_match_possessions(match_id)
    poss_list = _poss_cache[match_id]
    midx = poss_idx_map.get(global_poss_id, -1)
    if 0 <= midx < len(poss_list):
        for ev in poss_list[midx]:
            a = event_to_action(ev)
            if a:
                return a
    return None


# ── SSE evaluation helper ─────────────────────────────────────────────────────

@torch.no_grad()
def sse_from_positions(engine: ConditionalEngine,
                       pos_full: torch.Tensor,
                       mask: torch.Tensor,
                       ctx: MatchContext):
    """Run SSE directly on provided positions (B=1, N, 4)."""
    ctx_t = torch.tensor(
        [ctx.zone / 3.0, float(ctx.phase == 1), float(ctx.phase >= 2)],
        dtype=torch.float32, device=engine.device,
    ).unsqueeze(0)
    _, logits = engine.sse(
        pos_full.to(engine.device),
        mask.to(engine.device),
        ctx_t,
    )
    probs = torch.sigmoid(logits).squeeze(0).cpu()
    return float(probs[0]), float(probs[1]), float(probs[2])  # s2, s3, shot


# ── GroupKFold AUC ────────────────────────────────────────────────────────────

def gkf_auc(scores, labels, groups, n_folds=5):
    gkf = GroupKFold(n_splits=n_folds)
    aucs = []
    for _, ti in gkf.split(scores, labels, groups):
        yt, ys = labels[ti], scores[ti]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        aucs.append(roc_auc_score(yt, ys))
    return float(np.mean(aucs)) if aucs else float("nan"), float(np.std(aucs)) if aucs else float("nan")


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

    # ── Possession sample (same logic as Check 2) ─────────────────────────────
    meta = pd.read_csv(OUT / "possession_meta.csv", low_memory=False)
    valid_teams  = set(engine.fingerprints.keys())
    meta_spatial = meta[meta["has_spatial"] == True].copy()
    meta_spatial = meta_spatial[meta_spatial["team_id"].isin(valid_teams)]
    meta_spatial = meta_spatial.dropna(subset=["reached_s2", "reached_s3"])
    meta_spatial["reached_shot"] = meta_spatial.get("reached_shot", 0).fillna(0).astype(int)

    # Opponent lookup
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

    if len(meta_spatial) > MAX_POSSESSIONS * 2:
        meta_spatial = (meta_spatial
                        .groupby("reached_s2", group_keys=False)
                        .apply(lambda g: g.sample(
                            min(len(g), MAX_POSSESSIONS), random_state=42)))

    print(f"  {len(meta_spatial):,} possessions selected")

    # ── Possession index maps ─────────────────────────────────────────────────
    print("  Building index maps…")

    # match-scoped possession index (for real sequence lookup)
    full_meta = pd.read_csv(OUT / "possession_meta.csv",
                            usecols=["possession_id", "match_id"], low_memory=False)
    full_meta["match_poss_idx"] = (
        full_meta.groupby("match_id")["possession_id"]
        .rank(method="first").astype(int) - 1
    )
    poss_idx_map = dict(zip(full_meta["possession_id"], full_meta["match_poss_idx"]))
    del full_meta

    # spatial_dataset.pt index (rank among has_spatial==True rows)
    spatial_rows = (meta[meta["has_spatial"] == True]
                    .sort_values("possession_id")
                    .reset_index(drop=True))
    spatial_idx_map = {int(r["possession_id"]): i
                       for i, r in spatial_rows.iterrows()}

    # ── Load spatial dataset ──────────────────────────────────────────────────
    print("  Loading spatial dataset…")
    dataset    = torch.load(OUT / "spatial_dataset.pt", map_location="cpu",
                            weights_only=False)
    all_pos    = dataset["positions"]   # (M, 23, 4)
    all_masks  = dataset["masks"]       # (M, 23)

    # ── Main evaluation loop ──────────────────────────────────────────────────
    records = []
    skipped = 0
    print(f"\nRunning three-stage evaluation on {len(meta_spatial):,} possessions…")

    for _, row in meta_spatial.iterrows():
        tid_a    = int(row["team_id"])
        tid_b    = int(row["team_id_b"])
        match_id = int(row["match_id"])
        gpid     = int(row["possession_id"])
        zone     = max(0, min(int(row.get("territory_zone", 1) or 1), 3))
        ctx      = MatchContext(
            score_diff=0.0, minute=45.0, zone=zone,
            phase=int(row.get("phase_int", 0) or 0), poss_team=0,
        )

        # ── Stage 1: SSE on real frame ────────────────────────────────────────
        sidx = spatial_idx_map.get(gpid, -1)
        if sidx < 0:
            skipped += 1; continue

        real_pos  = all_pos[sidx].unsqueeze(0)     # (1, 23, 4)
        real_mask = all_masks[sidx].unsqueeze(0)   # (1, 23)
        s1_s2, s1_s3, s1_shot = sse_from_positions(engine, real_pos, real_mask, ctx)

        # ── Stage 2: generate from real z (no action transform) ───────────────
        z_A = engine.fingerprints.get(tid_a, engine.mean_fp).to(engine.device)
        z_B = engine.fingerprints.get(tid_b, engine.mean_fp).to(engine.device)
        with torch.no_grad():
            c      = engine._encode_condition(z_A, z_B, ctx)
            gen_xy = engine._debias_positions(
                engine.generator.generate(engine.roles, c, engine.mask, n_steps=GEN_STEPS),
                ctx,
            )
            pos_full = torch.cat([gen_xy, engine.roles], dim=-1)
        s2_s2, s2_s3, s2_shot = sse_from_positions(engine, pos_full, engine.mask, ctx)

        # ── Stage 3: apply first real action, generate ────────────────────────
        first_action = get_first_action(match_id, gpid, poss_idx_map)
        if first_action is None or first_action not in Action.__members__:
            skipped += 1; continue

        action = Action[first_action]
        with torch.no_grad():
            z_A_mod = engine._apply_action(z_A, action, ctx, 1.0)
            c_mod   = engine._encode_condition(z_A_mod, z_B, ctx)
            gen_xy_mod = engine._debias_positions(
                engine.generator.generate(engine.roles, c_mod, engine.mask, n_steps=GEN_STEPS),
                ctx,
            )
            pos_mod = torch.cat([gen_xy_mod, engine.roles], dim=-1)
        s3_s2, s3_s3, s3_shot = sse_from_positions(engine, pos_mod, engine.mask, ctx)

        records.append({
            "s1_s2": s1_s2, "s1_s3": s1_s3, "s1_shot": s1_shot,
            "s2_s2": s2_s2, "s2_s3": s2_s3, "s2_shot": s2_shot,
            "s3_s2": s3_s2, "s3_s3": s3_s3, "s3_shot": s3_shot,
            "reached_s2":   int(row["reached_s2"]),
            "reached_s3":   int(row["reached_s3"]),
            "reached_shot": int(row.get("reached_shot", 0)),
            "match_id":     match_id,
        })

    df = pd.DataFrame(records)
    print(f"  Evaluated {len(df):,}  ({skipped} skipped)")

    if len(df) < 20:
        print("Too few results"); sys.exit(1)

    groups = df["match_id"].values
    targets = [
        ("reached_s2",   "s1_s2",  "s2_s2",  "s3_s2"),
        ("reached_s3",   "s1_s3",  "s2_s3",  "s3_s3"),
        ("reached_shot", "s1_shot","s2_shot", "s3_shot"),
    ]

    print(f"\n{'Target':<14} {'Stage1(ceiling)':>16} {'Stage2(gen)':>12} {'Stage3(enc)':>12}  "
          f"{'Δ(gen)':>8} {'Δ(enc)':>8}")
    print("─" * 78)

    results = {}
    all_stage_data = {}

    for label, c1, c2, c3 in targets:
        y      = df[label].values
        if y.sum() < 5:
            continue
        sc1    = df[c1].values
        sc2    = df[c2].values
        sc3    = df[c3].values

        a1, s1 = gkf_auc(sc1, y, groups, N_FOLDS)
        a2, s2 = gkf_auc(sc2, y, groups, N_FOLDS)
        a3, s3 = gkf_auc(sc3, y, groups, N_FOLDS)
        dgen = a1 - a2
        denc = a2 - a3

        print(f"  {label:<12} {a1:>14.4f}±{s1:.3f}  {a2:>10.4f}±{s2:.3f}  "
              f"{a3:>10.4f}±{s3:.3f}  {dgen:>+8.4f}  {denc:>+8.4f}")

        results[label] = {
            "stage1_ceiling": round(a1, 4), "stage1_std": round(s1, 4),
            "stage2_gen_only": round(a2, 4), "stage2_std": round(s2, 4),
            "stage3_enc_added": round(a3, 4), "stage3_std": round(s3, 4),
            "delta_generator": round(dgen, 4),
            "delta_action_encoder": round(denc, 4),
            "n": int(len(y)), "n_positive": int(y.sum()),
        }
        all_stage_data[label] = (sc1, sc2, sc3, y)

    # ── p_advance saturation check ────────────────────────────────────────────
    pa_scores = df["s3_s2"].values
    print(f"\n  p_advance (stage 3) distribution:")
    print(f"    min={pa_scores.min():.3f}  p25={np.percentile(pa_scores,25):.3f}  "
          f"median={np.median(pa_scores):.3f}  p75={np.percentile(pa_scores,75):.3f}  "
          f"p95={np.percentile(pa_scores,95):.3f}  max={pa_scores.max():.3f}")
    print(f"    frac > 0.90: {(pa_scores > 0.90).mean():.1%}  "
          f"frac > 0.95: {(pa_scores > 0.95).mean():.1%}")
    print(f"    Saturation (>0.95 causing rank loss): "
          f"{'YES — bias likely hurt AUC' if (pa_scores > 0.95).mean() > 0.3 else 'NO — AUC gap is structural, not saturation'}")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    colors = {"Stage 1\n(real frames)": "steelblue",
              "Stage 2\n(gen, no transform)": "darkorange",
              "Stage 3\n(gen + action enc)": "tomato"}

    for col_idx, (label, c1, c2, c3) in enumerate(targets):
        if label not in all_stage_data:
            continue
        sc1, sc2, sc3, y = all_stage_data[label]
        ax_roc  = axes[0, col_idx]
        ax_hist = axes[1, col_idx]

        for sc, name, color in [
            (sc1, "Stage 1\n(real frames)", "steelblue"),
            (sc2, "Stage 2\n(gen, no transform)", "darkorange"),
            (sc3, "Stage 3\n(gen + action enc)", "tomato"),
        ]:
            if y.sum() > 0 and y.sum() < len(y):
                fpr, tpr, _ = roc_curve(y, sc)
                auc_val = roc_auc_score(y, sc)
                ax_roc.plot(fpr, tpr, color=color, lw=1.5,
                            label=f"{name.split(chr(10))[0]} ({auc_val:.3f})")
            ax_hist.hist(sc[y==0], bins=30, alpha=0.4, color=color, density=True, label=f"neg {name.split(chr(10))[0]}")
            ax_hist.hist(sc[y==1], bins=30, alpha=0.6, color=color, density=True, histtype="step", lw=2, label=f"pos {name.split(chr(10))[0]}")

        ax_roc.plot([0,1],[0,1],"k--",lw=1)
        ax_roc.set_title(f"{label} — ROC")
        ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR")
        ax_roc.legend(fontsize=7); ax_roc.grid(True, alpha=0.3)

        ax_hist.set_title(f"{label} — Score Distributions")
        ax_hist.set_xlabel("predicted probability")
        ax_hist.legend(fontsize=6); ax_hist.grid(True, alpha=0.3)

    plt.suptitle("Three-Way AUC Error Decomposition\n"
                 "Stage1=SSE ceiling (real) | Stage2=generator error | Stage3=action encoder error",
                 fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT / "decompose_auc.png", dpi=150)
    plt.close(fig)

    output = {
        "description": "Stage1=SSE on real frames (ceiling); Stage2=SSE on generated frames no transform; Stage3=full pipeline (Check2 single-step)",
        "n_evaluated": len(df),
        "n_skipped": skipped,
        "results": results,
        "saturation_check": {
            "p_advance_p95": round(float(np.percentile(pa_scores, 95)), 4),
            "frac_above_0_95": round(float((pa_scores > 0.95).mean()), 4),
        },
    }
    with open(OUT / "decompose_auc.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Saved decompose_auc.json + decompose_auc.png")
    print(f"\n  Attribution summary:")
    for label, res in results.items():
        total_gap = res["stage1_ceiling"] - res["stage3_enc_added"]
        gen_frac  = res["delta_generator"] / max(total_gap, 1e-6)
        enc_frac  = res["delta_action_encoder"] / max(total_gap, 1e-6)
        print(f"  {label:<14} total_gap={total_gap:+.4f}  "
              f"generator={gen_frac:.0%}  action_enc={enc_frac:.0%}")


if __name__ == "__main__":
    main()
