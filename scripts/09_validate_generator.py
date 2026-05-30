"""
scripts/09_validate_generator.py
==================================
Check 3: Classifier two-sample test for the generator.

Generate synthetic freeze frames conditioned on real (z_A, z_B, context)
from held-out possessions.  Then train a simple logistic regression to
discriminate real from generated frames.  AUC near 0.5 means the generator
produces spatially indistinguishable configurations; AUC near 1.0 reveals
systematic artifacts.

Also compares the two distributions on the LIM probe concepts that have
already been validated as meaningfully representing tactical structure:
    - territory_zone  (forward presence)
    - s_width_asym    (how wide/asymmetric the shape is)
    - s_vert_support  (vertical spacing of teammates)
    - s_pressure      (how compressed the space is)

These were validated in the LIM study, so reusing them here ties the
generator's quality directly to the same ground truth that established
the SSE evaluator was meaningful.

Output
------
- data/results/generator_classifier_auc.json  — AUC, per-concept KS tests
- data/results/generator_feature_dist.png     — distribution comparison plots

Run
---
    python -m scripts.09_validate_generator
"""

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
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.conditional_engine import ConditionalEngine
from model.action_encoder import MatchContext


# ── Paths ──────────────────────────────────────────────────────────────────────

BASE  = Path(__file__).parent.parent
CKPT  = BASE / "model" / "checkpoints"
OUT   = BASE / "data" / "results"

REQUIRED = [CKPT / "sse_best.pt",
            CKPT / "generator_best.pt",
            CKPT / "team_fingerprints.pt"]

N_SAMPLES   = 2000    # real frames to use (plus equal number of generated)
GEN_STEPS   = 30      # ODE steps for generated frames (quality vs speed)


# ── Feature extraction from raw position tensors ──────────────────────────────

def extract_lim_features(positions: torch.Tensor,
                          mask:      torch.Tensor) -> dict[str, float]:
    """
    Re-implement the LIM probe features directly from position tensors so
    we don't need possession_meta here.

    positions : (N, 4)  [x_norm, y_norm, is_teammate, is_actor]
    mask      : (N,)    True where padded
    """
    valid = ~mask                                    # (N,)
    pos   = positions[valid]                         # (K, 4)
    if len(pos) < 3:
        return {}

    x      = pos[:, 0]
    y      = pos[:, 1]
    mates  = pos[pos[:, 2] > 0.5]

    # Territory zone: fraction of teammates in the attacking half (x > 0.5)
    territory = float((mates[:, 0] > 0.5).float().mean()) if len(mates) else 0.5

    # Width asymmetry: std(y) of teammates vs all players
    width_asym = float(mates[:, 1].std()) if len(mates) > 1 else 0.0

    # Vertical support: std(x) of teammates (spread along pitch length)
    vert_support = float(mates[:, 0].std()) if len(mates) > 1 else 0.0

    # Pressure proxy: mean distance from actor to nearest 3 opponents
    actor_mask = pos[:, 3] > 0.5
    opps       = pos[pos[:, 2] < 0.5]
    if actor_mask.any() and len(opps) >= 1:
        actor_pos = pos[actor_mask][0, :2].unsqueeze(0)   # (1, 2)
        opp_pos   = opps[:, :2]                           # (M, 2)
        dists     = torch.cdist(actor_pos, opp_pos)[0]
        pressure  = float(dists.topk(min(3, len(dists)), largest=False).values.mean())
    else:
        pressure = 0.5

    return {
        "territory_zone":  territory,
        "s_width_asym":    width_asym,
        "s_vert_support":  vert_support,
        "s_pressure":      pressure,
    }


def positions_to_feature_vector(positions: torch.Tensor,
                                 mask:      torch.Tensor,
                                 n_cap:     int | None = None) -> np.ndarray:
    """Flat 46-dim feature vector: raw (x,y) of valid players.

    n_cap lets the caller limit to the first n_cap valid players so that
    real and generated vectors have the same density and the classifier
    can only detect spatial differences, not player-count differences.
    """
    valid = positions[~mask].cpu()   # (K, 4)
    xy    = valid[:, :2].numpy()
    k     = min(len(xy), n_cap if n_cap is not None else 23, 23)
    out   = np.zeros(46, dtype=np.float32)
    out[:k*2:2]    = xy[:k, 0]
    out[1:k*2+1:2] = xy[:k, 1]
    return out


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    missing = [p.name for p in REQUIRED if not p.exists()]
    if missing:
        print(f"Missing checkpoints: {missing}")
        sys.exit(1)

    print("Loading engine and dataset…")
    engine  = ConditionalEngine(
        sse_path         = CKPT / "sse_best.pt",
        generator_path   = CKPT / "generator_best.pt",
        fingerprint_path = CKPT / "team_fingerprints.pt",
    )
    dataset = torch.load(OUT / "spatial_dataset.pt", map_location="cpu",
                         weights_only=False)

    positions_all = dataset["positions"]    # (M, 23, 4)
    masks_all     = dataset["masks"]        # (M, 23)
    contexts_all  = dataset["contexts"]     # (M, 3)
    team_ids_all  = dataset["team_ids"]
    match_ids_all = dataset["match_ids"]

    # Only use frames whose team_id has a fingerprint
    valid_teams = set(engine.fingerprints.keys())
    valid_idx   = [i for i, t in enumerate(team_ids_all) if t in valid_teams]
    print(f"  {len(valid_idx):,} frames with valid team fingerprints")

    # Sample held-out frames (last 20% by match to avoid train/test leakage)
    all_matches = sorted(set(match_ids_all))
    n_holdout   = max(1, len(all_matches) // 5)
    holdout_matches = set(all_matches[-n_holdout:])
    held_idx = [i for i in valid_idx if match_ids_all[i] in holdout_matches]
    print(f"  Held-out: {len(held_idx):,} frames from {n_holdout} matches")

    if len(held_idx) < 100:
        print("  Not enough held-out frames — using all valid frames instead")
        held_idx = valid_idx

    rng = np.random.default_rng(42)
    chosen = rng.choice(held_idx, size=min(N_SAMPLES, len(held_idx)), replace=False)

    # ── Generate synthetic frames under the same conditioning ─────────────────

    print(f"\nGenerating {len(chosen)} synthetic frames (gen_steps={GEN_STEPS})…")
    real_feats, gen_feats = [], []
    real_lim,   gen_lim   = [], []

    for idx in chosen:
        tid_a = team_ids_all[idx]
        ctx_3 = contexts_all[idx]           # (3,)  [zone/3, ph1, ph2]

        # Derive opponent: use mean fingerprint if we can't resolve
        z_A = engine.fingerprints[tid_a].to(engine.device)
        z_B = engine.mean_fp.to(engine.device)

        ctx = MatchContext(
            score_diff = 0.0,
            minute     = 45.0,
            zone       = int(round(ctx_3[0].item() * 3)),
            phase      = 1 if ctx_3[1].item() > 0.5 else (2 if ctx_3[2].item() > 0.5 else 0),
            poss_team  = 0,
        )

        c      = engine._encode_condition(z_A, z_B, ctx)
        gen_xy = engine.generator.generate(
            engine.roles, c, engine.mask, n_steps=GEN_STEPS
        )  # (1, N, 2)

        # Build generated positions tensor in the same format as real
        N       = engine.generator.n_players
        gen_pos = torch.cat([gen_xy.squeeze(0), engine.roles.squeeze(0)], dim=-1)  # (N, 4)
        gen_msk = engine.mask.squeeze(0)                                            # (N,)

        real_pos = positions_all[idx]   # (23, 4)
        real_msk = masks_all[idx]       # (23,)

        # Normalise player count: use the real frame's valid player count so
        # the classifier can only detect spatial differences, not count differences
        # (engine always generates n_players=22 with no padding).
        n_valid = int((~real_msk).sum().item())

        # Feature vectors for classifier
        real_feats.append(positions_to_feature_vector(real_pos, real_msk, n_cap=n_valid))
        gen_feats.append(positions_to_feature_vector(gen_pos, gen_msk, n_cap=n_valid))

        # LIM probe features
        rf = extract_lim_features(real_pos, real_msk)
        gf = extract_lim_features(gen_pos, gen_msk)
        if rf and gf:
            real_lim.append(rf)
            gen_lim.append(gf)

    real_feats = np.array(real_feats)
    gen_feats  = np.array(gen_feats)
    X = np.concatenate([real_feats, gen_feats])
    y = np.concatenate([np.ones(len(real_feats)), np.zeros(len(gen_feats))])

    # ── Logistic regression two-sample classifier ─────────────────────────────

    print("Training logistic regression classifier (real vs. generated)…")
    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)

    skf   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs  = []
    for tr, te in skf.split(X_s, y):
        clf = LogisticRegression(max_iter=500, C=0.1, random_state=42)
        clf.fit(X_s[tr], y[tr])
        auc = roc_auc_score(y[te], clf.predict_proba(X_s[te])[:, 1])
        aucs.append(auc)

    clf_auc      = float(np.mean(aucs))
    clf_auc_std  = float(np.std(aucs))
    print(f"\nClassifier AUC: {clf_auc:.4f} ± {clf_auc_std:.4f}")
    print(f"  AUC = 0.50 → indistinguishable from real")
    print(f"  AUC = 1.00 → perfectly distinguishable (systematic artifacts)")

    # ── LIM probe concept comparison (KS tests) ───────────────────────────────

    concept_names = ["territory_zone", "s_width_asym", "s_vert_support", "s_pressure"]
    ks_results    = {}

    print(f"\nLIM probe feature comparison (Kolmogorov–Smirnov test):")
    print(f"  {'Concept':<20} {'Real mean':>10} {'Gen mean':>10} {'KS stat':>9} {'p-value':>9}")
    for c in concept_names:
        r_vals = np.array([d[c] for d in real_lim if c in d])
        g_vals = np.array([d[c] for d in gen_lim  if c in d])
        if len(r_vals) < 2 or len(g_vals) < 2:
            continue
        ks_stat, p_val = stats.ks_2samp(r_vals, g_vals)
        ks_results[c] = {
            "real_mean": float(r_vals.mean()),
            "gen_mean":  float(g_vals.mean()),
            "ks_stat":   round(float(ks_stat), 4),
            "p_value":   round(float(p_val), 4),
        }
        sig = "**" if p_val < 0.01 else ("*" if p_val < 0.05 else "")
        print(f"  {c:<20} {r_vals.mean():>10.3f} {g_vals.mean():>10.3f} "
              f"{ks_stat:>9.4f} {p_val:>9.4f} {sig}")

    print(f"\n  * p<0.05 ** p<0.01 — non-significant means distributions are matched")

    # ── Distribution comparison plot ──────────────────────────────────────────

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    for ax, c in zip(axes, concept_names):
        r_vals = np.array([d[c] for d in real_lim if c in d])
        g_vals = np.array([d[c] for d in gen_lim  if c in d])
        ax.hist(r_vals, bins=30, alpha=0.6, color="#58a6ff", label="Real", density=True)
        ax.hist(g_vals, bins=30, alpha=0.6, color="#f85149", label="Generated", density=True)
        ks_r = ks_results.get(c, {})
        ax.set_title(f"{c}\nKS={ks_r.get('ks_stat','—')}  p={ks_r.get('p_value','—')}")
        ax.legend(fontsize=8)
        ax.set_xlabel("Value"); ax.set_ylabel("Density")

    plt.suptitle(f"Generator Quality: Real vs. Generated Frame Distributions\n"
                 f"Classifier AUC = {clf_auc:.3f} ± {clf_auc_std:.3f} "
                 f"(0.5 = indistinguishable)",
                 fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT / "generator_feature_dist.png", dpi=150)
    plt.close(fig)
    print(f"\n  Saved generator_feature_dist.png")

    # Save JSON
    import json
    result = {
        "classifier_auc_mean": round(clf_auc, 4),
        "classifier_auc_std":  round(clf_auc_std, 4),
        "n_frames":            len(real_feats),
        "gen_steps":           GEN_STEPS,
        "lim_feature_ks":      ks_results,
        "interpretation": {
            "classifier_auc_0.5": "generated frames are spatially indistinguishable from real",
            "classifier_auc_1.0": "generated frames have systematic spatial artifacts",
            "ks_p_gt_0.05":       "generator matches real distribution on this LIM concept",
        },
    }
    with open(OUT / "generator_classifier_auc.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved generator_classifier_auc.json")


if __name__ == "__main__":
    main()
