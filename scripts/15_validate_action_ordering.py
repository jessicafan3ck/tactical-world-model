"""
scripts/15_validate_action_ordering.py
========================================
Check 4: Comparative action effect-ordering test.

Forward-prediction AUC is the wrong bar for a world model. The right bar is
counterfactual: does the simulator rank the *relative* effect of different
actions correctly?

For each (zone, phase) context bucket:
  1. Collect all possessions in that bucket.
  2. For each action type with ≥ MIN_COUNT observations, compute the empirical
     shot-reached rate among possessions where that action was actually taken.
  3. Get the model's predicted p_shot delta for every action from a
     representative state in that bucket (suggest_action with mean fingerprint).
  4. Compare model ranking vs. empirical ranking → Kendall's τ.

If the model orders action effects correctly (sign-agreement > chance, τ > 0),
that is the empirical content of the simulation: it identifies which actions
raise danger more, validated against real possession outcomes.

Honesty caveat printed in output: action choice is not random (teams cross when
crossing looks good), so empirical rates carry selection bias.  Matching on
(zone, phase) controls the observable state but not unobserved confounders.
Sign-agreement is informative, not airtight.

Output
------
  data/results/action_ordering.json   per-bucket rankings + τ + sign-agreement
  data/results/action_ordering.png    visualisation

Run
---
    python -m scripts.15_validate_action_ordering
"""

import json
import sys
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.conditional_engine import ConditionalEngine
from model.action_encoder import MatchContext
from utils.statsbomb_utils import iter_possessions

BASE = Path(__file__).parent.parent
CKPT = BASE / "model" / "checkpoints"
OUT  = BASE / "data" / "results"

REQUIRED = [CKPT / "sse_best.pt",
            CKPT / "generator_best.pt",
            CKPT / "team_fingerprints.pt"]

MIN_COUNT  = 15    # min possessions per action per bucket for reliable rate
N_WINDOW   = 8     # events lookahead for first-action extraction only
N_BOOTSTRAP = 500  # bootstrap samples for τ CI

# Excluded from ordering comparison: SHOOT is definitionally leaky (shooting
# always "reaches a shot"); KEEPER_BALL / LOW_BLOCK are defensive/goalkeeper
# actions that don't belong in the same tactical choice set.
EXCLUDE_ACTIONS = {"SHOOT", "KEEPER_BALL", "LOW_BLOCK"}

VALID_ACTIONS = ["ADVANCE", "HOLD", "THROUGH_BALL", "CROSS", "SHOOT",
                 "DRIBBLE", "PRESS", "SWITCH_LEFT", "SWITCH_RIGHT",
                 "LOW_BLOCK", "KEEPER_BALL"]


# ── StatsBomb helpers (reused from script 14) ─────────────────────────────────

def _etype(event: dict) -> str:
    t = event.get("type", {})
    if isinstance(t, dict): return t.get("name", "")
    if isinstance(t, str):  return t
    return event.get("type_name", "")


def event_to_action(event: dict) -> str | None:
    e = _etype(event)
    if e == "Shot":                return "SHOOT"
    if e == "Dribble":             return "DRIBBLE"
    if e == "Pressure":            return "PRESS"
    if e in ("Block","Clearance"): return "LOW_BLOCK"
    if e == "Goal Keeper":         return "KEEPER_BALL"
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
_poss_cache: dict[int, list[str | None]] = {}   # match_id → [first_action_per_possession]


def _load_match_first_actions(match_id: int) -> list[str | None]:
    """Return first action string per possession for a match."""
    path = RAW_EVENTS / f"{match_id}.json"
    if not path.exists():
        return []
    with open(path) as f:
        events = json.load(f)
    result = []
    for poss in iter_possessions(events, match_id, {}):
        pev = poss["events"]
        first_action = None
        for ev in pev[:N_WINDOW]:
            a = event_to_action(ev)
            if a:
                first_action = a
                break
        result.append(first_action)
    return result


def get_first_action(match_id: int, match_poss_idx: int) -> str | None:
    if match_id not in _poss_cache:
        _poss_cache[match_id] = _load_match_first_actions(match_id)
    plist = _poss_cache[match_id]
    if 0 <= match_poss_idx < len(plist):
        return plist[match_poss_idx]
    return None


# ── Bootstrap Kendall's τ ─────────────────────────────────────────────────────

def kendall_tau_ci(x, y, n_boot=N_BOOTSTRAP, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(x)
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    tau, _ = stats.kendalltau(x, y)
    boot_taus = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if len(set(x[idx])) < 2 or len(set(y[idx])) < 2:
            continue
        t, _ = stats.kendalltau(x[idx], y[idx])
        boot_taus.append(t)
    if not boot_taus:
        return float(tau), float("nan"), float("nan")
    lo, hi = np.percentile(boot_taus, [5, 95])
    return float(tau), float(lo), float(hi)


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

    # ── Load possession metadata ──────────────────────────────────────────────
    meta = pd.read_csv(OUT / "possession_meta.csv", low_memory=False)
    valid_teams = set(engine.fingerprints.keys())

    meta_spatial = meta[meta["has_spatial"] == True].copy()
    meta_spatial = meta_spatial[meta_spatial["team_id"].isin(valid_teams)]
    meta_spatial = meta_spatial.dropna(subset=["reached_s2", "reached_s3"])
    meta_spatial["reached_shot"] = meta_spatial.get("reached_shot", 0).fillna(0).astype(int)

    # Opponent lookup
    match_teams = (meta_spatial.groupby("match_id")["team_id"]
                   .apply(lambda x: list(x.unique())).reset_index())
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

    # Match-scoped possession index
    full_meta = pd.read_csv(OUT / "possession_meta.csv",
                            usecols=["possession_id", "match_id"], low_memory=False)
    full_meta["match_poss_idx"] = (
        full_meta.groupby("match_id")["possession_id"]
        .rank(method="first").astype(int) - 1
    )
    poss_idx_map = dict(zip(full_meta["possession_id"], full_meta["match_poss_idx"]))
    del full_meta

    print(f"  {len(meta_spatial):,} possessions available")

    # ── Collect per-possession (action, outcome, context bucket) ─────────────
    print("\nCollecting action/outcome records…")

    bucket_records: dict[tuple, list[dict]] = defaultdict(list)
    skipped = 0

    for _, row in meta_spatial.iterrows():
        match_id  = int(row["match_id"])
        gpid      = int(row["possession_id"])
        tid_a     = int(row["team_id"])
        tid_b     = int(row["team_id_b"])
        zone      = max(0, min(int(row.get("territory_zone", 1) or 1), 3))
        phase     = int(row.get("phase_int", 0) or 0)

        midx = poss_idx_map.get(gpid, -1)
        if midx < 0:
            skipped += 1; continue

        first_action = get_first_action(match_id, midx)
        if first_action is None or first_action not in VALID_ACTIONS:
            skipped += 1; continue

        bucket = (zone, phase)
        bucket_records[bucket].append({
            "action":        first_action,
            "reached_shot":  int(row["reached_shot"]),
            "reached_s3":    int(row["reached_s3"]),   # ~8% base rate; primary metric
            "tid_a":         tid_a,
            "tid_b":         tid_b,
        })

    total = sum(len(v) for v in bucket_records.values())
    print(f"  {total:,} records collected across {len(bucket_records)} buckets"
          f"  ({skipped} skipped)")

    # ── Per-bucket: empirical rates + model ranking ───────────────────────────
    print("\nComparing model ranking vs. empirical ranking…")
    print(f"  (minimum {MIN_COUNT} obs per action per bucket)\n")

    PHASE_NAMES = {0: "Open play", 1: "Counter", 2: "Set piece", 3: "Restart"}
    ZONE_NAMES  = {0: "Defensive ⅓", 1: "Middle ⅓", 2: "Attacking ⅓", 3: "Box edge"}

    all_results   = []
    global_agree  = []
    global_pairs  = []

    for bucket in sorted(bucket_records.keys()):
        zone, phase = bucket
        records = bucket_records[bucket]
        ctx = MatchContext(score_diff=0.0, minute=45.0,
                           zone=zone, phase=phase, poss_team=0)

        # Empirical outcome rate per action — run on two metrics, use reached_s3
        # as primary (8% base rate) and reached_shot as secondary (0.4% base rate).
        def _empirical_rates(outcome_key: str) -> dict[str, float]:
            by_a: dict[str, list[int]] = defaultdict(list)
            for r in records:
                if r["action"] not in EXCLUDE_ACTIONS:
                    by_a[r["action"]].append(r[outcome_key])
            return {a: float(np.mean(vals))
                    for a, vals in by_a.items()
                    if len(vals) >= MIN_COUNT}

        empirical  = _empirical_rates("reached_s3")     # primary
        empirical2 = _empirical_rates("reached_shot")   # secondary (sparse)

        if len(empirical) < 3:
            continue   # need at least 3 actions to rank meaningfully

        # Model ranking: suggest_action with mean fingerprint
        tid_counts: dict[int, int] = defaultdict(int)
        for r in records:
            tid_counts[r["tid_a"]] += 1
        rep_tid = max(tid_counts, key=tid_counts.__getitem__)
        rep_opp = max({r["tid_b"] for r in records},
                      key=lambda t: sum(1 for r in records if r["tid_b"] == t))

        suggestions = engine.suggest_action(ctx, rep_tid, rep_opp)
        model_delta = {s["action"]: s["p_shot_delta"] for s in suggestions}

        # Keep only actions present in both rankings
        common = sorted(set(empirical) & set(model_delta))
        if len(common) < 3:
            continue

        emp_vals   = np.array([empirical[a]   for a in common])
        model_vals = np.array([model_delta[a] for a in common])

        tau, tau_lo, tau_hi = kendall_tau_ci(emp_vals, model_vals)

        # Secondary metric (reached_shot) — reported but not used for global stats
        common2 = sorted(set(empirical2) & set(model_delta))
        if len(common2) >= 3:
            tau2, _, _ = kendall_tau_ci(
                np.array([empirical2[a] for a in common2]),
                np.array([model_delta[a] for a in common2]))
        else:
            tau2 = float("nan")

        # Sign agreement on all pairs
        pairs_agree, pairs_total = 0, 0
        pair_details = []
        for i in range(len(common)):
            for j in range(i+1, len(common)):
                a1, a2 = common[i], common[j]
                e_diff = empirical[a1] - empirical[a2]
                m_diff = model_delta[a1] - model_delta[a2]
                if abs(e_diff) < 1e-6:
                    continue
                agree = (e_diff > 0) == (m_diff > 0)
                pairs_agree += int(agree)
                pairs_total += 1
                global_agree.append(int(agree))
                global_pairs.append((bucket, a1, a2))
                pair_details.append({
                    "action_a": a1, "action_b": a2,
                    "emp_rate_a": round(float(empirical[a1]), 4),
                    "emp_rate_b": round(float(empirical[a2]), 4),
                    "model_delta_a": round(float(model_delta[a1]), 4),
                    "model_delta_b": round(float(model_delta[a2]), 4),
                    "agree": bool(agree),
                })

        sign_agree = pairs_agree / pairs_total if pairs_total else float("nan")

        bucket_result = {
            "zone": zone, "phase": phase,
            "zone_name": ZONE_NAMES.get(zone, str(zone)),
            "phase_name": PHASE_NAMES.get(phase, str(phase)),
            "n_possessions": len(records),
            "actions_ranked": common,
            "empirical_s3_rates":   {a: round(float(empirical[a]),4) for a in common},
            "empirical_shot_rates": {a: round(float(empirical2[a]),4) for a in common if a in empirical2},
            "model_deltas":         {a: round(float(model_delta[a]),4) for a in common},
            "kendall_tau_s3":   round(tau, 4),
            "kendall_tau_shot": round(tau2, 4) if not (isinstance(tau2, float) and np.isnan(tau2)) else None,
            "tau_ci_90": [round(tau_lo, 4), round(tau_hi, 4)],
            "sign_agreement":  round(sign_agree, 4),
            "n_pairs": pairs_total,
            "pairs": pair_details,
        }
        all_results.append(bucket_result)

        # Console print
        sign_str = f"{sign_agree:.0%}" if not np.isnan(sign_agree) else "—"
        tau_str  = f"{tau:+.3f}" if not np.isnan(tau) else "—"
        tau2_str = f"{tau2:+.3f}" if not np.isnan(tau2) else "—"
        ci_str   = (f"[{tau_lo:+.2f}, {tau_hi:+.2f}]"
                    if not np.isnan(tau_lo) else "")
        print(f"  {ZONE_NAMES[zone]:<16} {PHASE_NAMES.get(phase,'?'):<12}"
              f"  n={len(records):<6}  actions={len(common)}"
              f"  τ(s3)={tau_str} τ(shot)={tau2_str}  {ci_str}  sign={sign_str}  pairs={pairs_total}")

        # Per-action table (sorted by model ranking)
        sorted_common = sorted(common, key=lambda a: model_delta[a], reverse=True)
        emp_rank = {a: i for i, a in enumerate(
            sorted(common, key=lambda a: empirical[a], reverse=True))}
        by_action_all: dict[str, list] = defaultdict(list)
        for r in records:
            if r["action"] in common:
                by_action_all[r["action"]].append(r["reached_s3"])
        for a in sorted_common:
            cnt = len(by_action_all[a])
            print(f"    {a:<22}  emp_s3={empirical[a]:.3f} (n={cnt:<4})"
                  f"  model_Δ={model_delta[a]:+.4f}"
                  f"  emp_rank={emp_rank[a]+1}/{len(common)}")
        print()

    # ── Global summary ────────────────────────────────────────────────────────
    global_sign = np.mean(global_agree) if global_agree else float("nan")
    print("─" * 72)
    print(f"  Global sign-agreement:  {global_sign:.1%}  across {len(global_agree)} action pairs")
    print(f"  Buckets evaluated:      {len(all_results)}")

    if global_sign > 0.65:
        verdict = "PARTIAL PASS — model orders action effects better than chance"
    elif global_sign > 0.5:
        verdict = "WEAK — marginal ordering signal above chance"
    else:
        verdict = "FAIL — model does not reliably order action effects"
    print(f"  Verdict: {verdict}")
    print()
    print("  ⚠  Honesty note: action choice is NOT random. Teams cross when")
    print("     crossing looks good, so empirical rates carry selection bias.")
    print("     Matching on (zone, phase) controls observable state only.")
    print("     Sign-agreement is informative, not a causal claim.")
    print("─" * 72)

    # ── Visualisation ─────────────────────────────────────────────────────────
    if all_results:
        n_plots = len(all_results)
        ncols   = min(3, n_plots)
        nrows   = (n_plots + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(5 * ncols, 4.5 * nrows), squeeze=False)
        axes_flat = [ax for row in axes for ax in row]

        for ax, res in zip(axes_flat, all_results):
            actions = res["actions_ranked"]
            emp_r   = [res["empirical_shot_rates"][a] for a in actions]
            mod_r   = [res["model_deltas"][a] for a in actions]

            x = np.arange(len(actions))
            # Normalise model deltas to [0,1] range for overlay readability
            mod_arr  = np.array(mod_r)
            emp_arr  = np.array(emp_r)
            if mod_arr.max() > mod_arr.min():
                mod_norm = (mod_arr - mod_arr.min()) / (mod_arr.max() - mod_arr.min())
            else:
                mod_norm = np.full_like(mod_arr, 0.5)

            ax.bar(x - 0.18, emp_arr,  width=0.36, label="Empirical shot rate",
                   color="#58a6ff", alpha=0.8)
            ax.bar(x + 0.18, mod_norm, width=0.36, label="Model Δp_shot (norm.)",
                   color="#f85149", alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels([a.replace("_","\n") for a in actions],
                               fontsize=7)
            tau_str = f"{res['kendall_tau_s3']:+.3f}" if not np.isnan(res["kendall_tau_s3"]) else "—"
            ax.set_title(
                f"{res['zone_name']} · {res['phase_name']}\n"
                f"τ={tau_str}  sign={res['sign_agreement']:.0%}  n={res['n_possessions']}",
                fontsize=9)
            ax.legend(fontsize=7)

        for ax in axes_flat[len(all_results):]:
            ax.set_visible(False)

        plt.suptitle(
            f"Action Effect Ordering: Model vs. Empirical\n"
            f"Global sign-agreement = {global_sign:.1%} across {len(global_agree)} pairs",
            fontsize=11)
        plt.tight_layout()
        fig.savefig(OUT / "action_ordering.png", dpi=150)
        plt.close(fig)
        print(f"\n  Saved action_ordering.png")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output = {
        "global_sign_agreement": round(float(global_sign), 4),
        "n_pairs_total":  len(global_agree),
        "n_buckets":      len(all_results),
        "min_count":      MIN_COUNT,
        "verdict":        verdict,
        "caveat": ("Action choice is non-random (selection bias). "
                   "Matching on (zone, phase) controls observable state only. "
                   "Sign-agreement is informative, not a causal claim."),
        "buckets": all_results,
    }
    with open(OUT / "action_ordering.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved action_ordering.json")


if __name__ == "__main__":
    main()
