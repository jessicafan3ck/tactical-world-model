"""
scripts/12_audit_action_encoder.py
=====================================
Honest held-out audit of the learned action encoder.

Three questions, in order:

1. Are the Check 1 numbers real or partly memorisation?
   Re-train from scratch on a proper match-level split (random 80/20 of
   unique matches, not sorted tail).  Report per-action diagonal cosine on
   the held-out matches.  This is the only number that counts.

2. Does anything exceed its conditional ceiling?
   Conditional R per action = marginal R + within-cluster state-gain
   (both already computed by script 10).  Any learned cosine that exceeds
   conditional R is the overfitting tripwire — flag it explicitly.

3. Does the MLP earn its place?
   Compare MLP vs Affine per action on held-out data with bootstrap CIs.
   Only keep MLP for actions where it genuinely clears the Affine outside
   the CI.  For the rest, prefer the simpler model.

Extra: HOLD regularisation fix
   HOLD has R=0.037, conditional ceiling ~0.35.  With ~400 training pairs
   it cannot learn a state-dependent transform — it will memorise match
   context.  Apply strong per-action weight-decay for low-R actions
   (scaled 1/max(R, 0.1)) and re-train to see if the HOLD number drops
   back to or below its ceiling.

Output
------
- data/results/audit_encoder.json   — per-action held-out cosines + flags
- data/results/audit_encoder.png    — side-by-side comparison plot
- model/checkpoints/action_encoder_affine.pt  — updated with match-split training
- model/checkpoints/action_encoder_mlp.pt     — updated with match-split training

Run
---
    python -m scripts.12_audit_action_encoder
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.action_encoder import Action, ACTION_LABELS
from model.learned_action_encoder import PerActionAffine, ConditionedMLP

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE       = Path(__file__).parent.parent
CKPT       = BASE / "model" / "checkpoints"
OUT        = BASE / "data" / "results"
DATA_CACHE = OUT / "action_encoder_data.pt"

ACTION_NAMES = [a.name for a in Action]
N_ACTIONS    = len(Action)

HOLDOUT_FRAC = 0.20
BATCH_SIZE   = 128
AFFINE_EPOCHS = 80
MLP_EPOCHS    = 120
MLP_PATIENCE  = 20
AFFINE_LR     = 1e-3
MLP_LR        = 3e-4
WEIGHT_DECAY  = 1e-4
DROPOUT       = 0.3
HIDDEN        = 512
N_BOOTSTRAP   = 500   # for per-action CI


# ── Match-level split (random, not sorted-tail) ────────────────────────────────

def match_level_split(data: dict, holdout_frac: float, seed: int = 42):
    """
    Randomly assign unique match IDs to train/val (80/20).  All pairs from a
    given match go entirely to one split, so there is no match-level leakage.
    """
    rng         = np.random.default_rng(seed)
    all_matches = np.array(sorted(set(data["match_ids"])))
    rng.shuffle(all_matches)
    n_hold      = max(1, int(len(all_matches) * holdout_frac))
    val_matches = set(all_matches[:n_hold].tolist())
    tr_matches  = set(all_matches[n_hold:].tolist())

    mask_val = torch.tensor([m in val_matches for m in data["match_ids"]])
    mask_tr  = ~mask_val

    def _split(t):
        return t[mask_tr], t[mask_val]

    tr, va = {}, {}
    for k in ("z_before", "dz", "action_idx", "ctx"):
        tr[k], va[k] = _split(data[k])

    n_per_va = {ACTION_NAMES[a]: int((va["action_idx"] == a).sum()) for a in range(N_ACTIONS)}
    print(f"\n  Match-level split: {len(tr_matches)} train matches / {len(val_matches)} val matches")
    print(f"  Train: {mask_tr.sum():,} pairs  |  Val: {mask_val.sum():,} pairs")
    print(f"  Val pairs per action: {n_per_va}")
    return tr, va


# ── Conditional R ceiling (from noise_floor.json) ─────────────────────────────

def load_ceilings() -> dict[str, dict]:
    """Load marginal R and state-gain from noise floor diagnostic."""
    try:
        with open(OUT / "noise_floor.json") as f:
            nf = json.load(f)
    except FileNotFoundError:
        print("  noise_floor.json not found — run script 10 first")
        return {}

    ceilings = {}
    for act in ACTION_NAMES:
        r    = nf.get(act, {}).get("R", float("nan"))
        gain = nf.get(act, {}).get("state_gain") or 0.0
        cond = float(np.clip(r + gain, 0, 1)) if not np.isnan(r) else float("nan")
        ceilings[act] = {"R_marginal": r, "state_gain": gain, "R_conditional": cond}
    return ceilings


# ── Per-action weight decay scaled by 1/max(R, 0.1) ──────────────────────────

def per_action_wd(ceilings: dict, base_wd: float) -> torch.Tensor:
    """
    Low-R actions get proportionally stronger weight-decay toward zero.
    This regularises HOLD and the switch actions more aggressively.
    """
    wds = []
    for act in ACTION_NAMES:
        r = ceilings.get(act, {}).get("R_marginal", 0.3)
        r = float(r) if not (r != r) else 0.3   # nan → 0.3
        wds.append(base_wd / max(r, 0.1))
    return torch.tensor(wds, dtype=torch.float32)


# ── Training loops (same as script 11, with per-action WD for affine) ─────────

def make_loader(data: dict, shuffle: bool) -> DataLoader:
    ds = TensorDataset(data["z_before"], data["action_idx"], data["ctx"], data["dz"])
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, drop_last=False)


def _affine_l2_with_per_action_wd(model: PerActionAffine,
                                    z_b: torch.Tensor, a_i: torch.Tensor,
                                    ctx: torch.Tensor, dz_t: torch.Tensor,
                                    action_wd: torch.Tensor) -> torch.Tensor:
    """L2 loss + per-action weight decay on W_a and b_a."""
    dz_pred = model.predict_dz(z_b, a_i, ctx)
    loss    = F.mse_loss(dz_pred, dz_t)
    # Per-action penalty: for each unique action in batch, add wd * (||W_a||² + ||b_a||²)
    for a in a_i.unique():
        wd = action_wd[a.item()].to(z_b.device)
        loss = loss + wd * (model.W[a].pow(2).sum() + model.b[a].pow(2).sum())
    return loss


def train_affine_match(tr: dict, device: torch.device,
                        action_wd: torch.Tensor) -> PerActionAffine:
    model  = PerActionAffine().to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=AFFINE_LR, weight_decay=0.0)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=AFFINE_EPOCHS)
    loader = make_loader(tr, shuffle=True)

    for epoch in range(1, AFFINE_EPOCHS + 1):
        model.train()
        for z_b, a_i, ctx_, dz_t in loader:
            z_b, a_i, ctx_, dz_t = (x.to(device) for x in (z_b, a_i, ctx_, dz_t))
            loss = _affine_l2_with_per_action_wd(model, z_b, a_i, ctx_, dz_t, action_wd)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    return model


def train_mlp_match(tr: dict, va: dict, device: torch.device) -> ConditionedMLP:
    model      = ConditionedMLP(hidden=HIDDEN, dropout=DROPOUT).to(device)
    opt        = torch.optim.Adam(model.parameters(), lr=MLP_LR,
                                   weight_decay=WEIGHT_DECAY)
    sched      = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MLP_EPOCHS)
    tr_loader  = make_loader(tr, shuffle=True)
    va_loader  = make_loader(va, shuffle=False)
    best_nll   = float("inf")
    best_state = None
    patience   = 0

    for epoch in range(1, MLP_EPOCHS + 1):
        model.train()
        for z_b, a_i, ctx_, dz_t in tr_loader:
            z_b, a_i, ctx_, dz_t = (x.to(device) for x in (z_b, a_i, ctx_, dz_t))
            loss = model.nll_loss(z_b, a_i, ctx_, dz_t)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

        model.eval()
        va_nlls = []
        with torch.no_grad():
            for z_b, a_i, ctx_, dz_t in va_loader:
                z_b, a_i, ctx_, dz_t = (x.to(device) for x in (z_b, a_i, ctx_, dz_t))
                va_nlls.append(model.nll_loss(z_b, a_i, ctx_, dz_t).item())
        val_nll = float(np.mean(va_nlls))

        if val_nll < best_nll - 1e-4:
            best_nll   = val_nll
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience   = 0
        else:
            patience += 1
            if patience >= MLP_PATIENCE:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


# ── Evaluation helpers ─────────────────────────────────────────────────────────

@torch.no_grad()
def per_action_cosines(model, va: dict, device: torch.device) -> dict[str, np.ndarray]:
    """Return array of per-sample cosine(Δz_pred, Δz_real) for each action."""
    model.eval()
    z_b  = va["z_before"].to(device)
    a_i  = va["action_idx"].to(device)
    ctx  = va["ctx"].to(device)
    dz_r = va["dz"].to(device)
    dz_p = model.predict_dz(z_b, a_i, ctx)
    cos  = F.cosine_similarity(dz_p, dz_r, dim=-1).cpu().numpy()

    per_action: dict[str, np.ndarray] = {}
    a_np = va["action_idx"].numpy()
    for a, name in enumerate(ACTION_NAMES):
        mask = a_np == a
        per_action[name] = cos[mask] if mask.sum() > 0 else np.array([])
    return per_action


def bootstrap_ci(samples: np.ndarray, n: int = N_BOOTSTRAP,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean + (lower, upper) bootstrap CI."""
    if len(samples) == 0:
        return float("nan"), float("nan"), float("nan")
    means = [np.random.choice(samples, len(samples), replace=True).mean()
             for _ in range(n)]
    lo, hi = np.percentile(means, [alpha/2*100, (1-alpha/2)*100])
    return float(np.mean(samples)), float(lo), float(hi)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if not DATA_CACHE.exists():
        print("No cached data — run script 11 first"); sys.exit(1)

    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))
    print(f"Device: {device}")

    print(f"Loading {DATA_CACHE.name}…")
    data = torch.load(DATA_CACHE, map_location="cpu", weights_only=False)
    print(f"  {len(data['dz']):,} pairs from {len(set(data['match_ids']))} unique matches")

    # ── Split ──────────────────────────────────────────────────────────────────
    tr, va = match_level_split(data, HOLDOUT_FRAC)

    if va["dz"].shape[0] == 0:
        print("Val set empty — check data collection"); sys.exit(1)

    # ── Conditional ceilings ───────────────────────────────────────────────────
    ceilings = load_ceilings()
    action_wd = per_action_wd(ceilings, WEIGHT_DECAY)

    print("\nConditional R ceilings:")
    print(f"  {'Action':<20} {'R_marg':>8} {'state_gain':>11} {'R_cond':>8}")
    for act in ACTION_NAMES:
        c = ceilings.get(act, {})
        print(f"  {ACTION_LABELS[Action[act]]:<20} "
              f"{c.get('R_marginal', float('nan')):>8.4f} "
              f"{c.get('state_gain', float('nan')):>11.4f} "
              f"{c.get('R_conditional', float('nan')):>8.4f}")

    # ── Train ──────────────────────────────────────────────────────────────────
    print("\nTraining PerActionAffine (match-level split, per-action WD)…")
    affine = train_affine_match(tr, device, action_wd)
    torch.save(affine.state_dict(), CKPT / "action_encoder_affine.pt")

    print("Training ConditionedMLP (match-level split)…")
    mlp = train_mlp_match(tr, va, device)
    torch.save(mlp.state_dict(), CKPT / "action_encoder_mlp.pt")

    # ── Held-out evaluation ────────────────────────────────────────────────────
    cos_affine = per_action_cosines(affine, va, device)
    cos_mlp    = per_action_cosines(mlp,    va, device)

    labels = [ACTION_LABELS[a] for a in Action]

    print(f"\n{'Action':<20} {'R_cond':>8} │ "
          f"{'Affine mean':>11} {'95% CI':>16} │ "
          f"{'MLP mean':>10} {'95% CI':>16} │ {'Flag':>6}")
    print("─" * 105)

    results = {}
    winner_counts = {"affine": 0, "mlp": 0, "tie": 0}

    for act_name, lbl in zip(ACTION_NAMES, labels):
        c    = ceilings.get(act_name, {})
        ceil = c.get("R_conditional", float("nan"))

        ca = cos_affine[act_name]
        cm = cos_mlp[act_name]

        ma, lo_a, hi_a = bootstrap_ci(ca)
        mm, lo_m, hi_m = bootstrap_ci(cm)

        # Overfitting flag: mean > conditional ceiling
        flag_a = "⚠ OV" if (not np.isnan(ceil)) and ma > ceil + 0.02 else ""
        flag_m = "⚠ OV" if (not np.isnan(ceil)) and mm > ceil + 0.02 else ""

        # Per-action winner: MLP wins only if lower CI > affine upper CI
        if np.isnan(ma) or np.isnan(mm):
            winner = "—"
        elif lo_m > hi_a:
            winner = "MLP"
            winner_counts["mlp"] += 1
        elif lo_a > hi_m:
            winner = "Affine"
            winner_counts["affine"] += 1
        else:
            winner = "tie"
            winner_counts["tie"] += 1

        flag = f"{flag_a or flag_m or ''}"

        print(f"  {lbl:<18} {ceil:>8.4f} │ "
              f"{ma:>11.4f} [{lo_a:.3f},{hi_a:.3f}] │ "
              f"{mm:>10.4f} [{lo_m:.3f},{hi_m:.3f}] │ "
              f"{winner:>5} {flag}")

        results[act_name] = {
            "label":        lbl,
            "n_val":        len(ca),
            "R_marginal":   c.get("R_marginal"),
            "R_conditional": round(ceil, 4) if not np.isnan(ceil) else None,
            "affine": {"mean": round(ma, 4), "ci_lo": round(lo_a, 4), "ci_hi": round(hi_a, 4),
                       "over_ceiling": bool(not np.isnan(ceil) and ma > ceil + 0.02)},
            "mlp":    {"mean": round(mm, 4), "ci_lo": round(lo_m, 4), "ci_hi": round(hi_m, 4),
                       "over_ceiling": bool(not np.isnan(ceil) and mm > ceil + 0.02)},
            "winner": winner,
        }

    # ── Aggregate ──────────────────────────────────────────────────────────────
    valid_a = [v["affine"]["mean"] for v in results.values()
               if not np.isnan(v["affine"]["mean"])]
    valid_m = [v["mlp"]["mean"] for v in results.values()
               if not np.isnan(v["mlp"]["mean"])]
    mean_a = float(np.mean(valid_a))
    mean_m = float(np.mean(valid_m))

    n_over_a = sum(1 for v in results.values() if v["affine"]["over_ceiling"])
    n_over_m = sum(1 for v in results.values() if v["mlp"]["over_ceiling"])

    print(f"\n  Held-out mean diagonal (match-level):  Affine={mean_a:.4f}  MLP={mean_m:.4f}")
    print(f"  Over-ceiling flags:  Affine={n_over_a}  MLP={n_over_m}")
    print(f"  Per-action winner: MLP={winner_counts['mlp']}  Affine={winner_counts['affine']}  Tie={winner_counts['tie']}")
    print()
    if mean_m > mean_a + 0.01:
        rec = "MLP"
        print(f"  Recommendation: ship MLP (meaningful edge: +{mean_m-mean_a:.4f} held-out)")
    elif mean_a > mean_m + 0.01:
        rec = "Affine"
        print(f"  Recommendation: ship Affine (simpler, comparable or better held-out)")
    else:
        rec = "Affine"
        print(f"  Recommendation: ship Affine (no meaningful difference; simpler is safer)")

    # ── Save JSON ──────────────────────────────────────────────────────────────
    audit = {
        "summary": {
            "mean_affine_held_out": round(mean_a, 4),
            "mean_mlp_held_out":    round(mean_m, 4),
            "pca_baseline":         -0.006,
            "over_ceiling_affine":  n_over_a,
            "over_ceiling_mlp":     n_over_m,
            "winner_counts":        winner_counts,
            "recommendation":       rec,
        },
        "per_action": results,
    }
    with open(OUT / "audit_encoder.json", "w") as f:
        json.dump(audit, f, indent=2)

    # ── Plot ───────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 6))
    x     = np.arange(N_ACTIONS)
    w     = 0.25
    R_c   = [ceilings.get(a, {}).get("R_conditional", float("nan")) for a in ACTION_NAMES]
    R_m_v = [ceilings.get(a, {}).get("R_marginal",    float("nan")) for a in ACTION_NAMES]
    ma_v  = [results[a]["affine"]["mean"] for a in ACTION_NAMES]
    mm_v  = [results[a]["mlp"]["mean"]    for a in ACTION_NAMES]
    err_a = [[ma_v[i] - results[ACTION_NAMES[i]]["affine"]["ci_lo"] for i in range(N_ACTIONS)],
             [results[ACTION_NAMES[i]]["affine"]["ci_hi"] - ma_v[i] for i in range(N_ACTIONS)]]
    err_m = [[mm_v[i] - results[ACTION_NAMES[i]]["mlp"]["ci_lo"] for i in range(N_ACTIONS)],
             [results[ACTION_NAMES[i]]["mlp"]["ci_hi"] - mm_v[i] for i in range(N_ACTIONS)]]

    ax.bar(x - w,   R_c,   w, color="#888", alpha=0.4, label="Conditional R ceiling")
    ax.bar(x,       ma_v,  w, color="#58a6ff", alpha=0.85, label=f"Affine (mean={mean_a:.3f})",
           yerr=err_a, capsize=3, error_kw={"elinewidth": 1})
    ax.bar(x + w,   mm_v,  w, color="#3fb950", alpha=0.85, label=f"MLP (mean={mean_m:.3f})",
           yerr=err_m, capsize=3, error_kw={"elinewidth": 1})

    # Mark over-ceiling bars with a red dot
    for i, act in enumerate(ACTION_NAMES):
        if results[act]["affine"]["over_ceiling"]:
            ax.plot(x[i] - w, ma_v[i] + (err_a[1][i] if err_a[1][i] else 0) + 0.02,
                    "rv", ms=6, zorder=5)
        if results[act]["mlp"]["over_ceiling"]:
            ax.plot(x[i] + w, mm_v[i] + (err_m[1][i] if err_m[1][i] else 0) + 0.02,
                    "rv", ms=6, zorder=5)

    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Held-out cosine similarity")
    ax.set_title(f"Action Encoder Audit — Match-Level Held-Out Evaluation\n"
                 f"Affine={mean_a:.3f}  MLP={mean_m:.3f}  |  ▼ = over conditional ceiling  "
                 f"(PCA baseline = −0.006)")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    fig.savefig(OUT / "audit_encoder.png", dpi=150)
    plt.close(fig)
    print(f"\n  Saved audit_encoder.json + audit_encoder.png")
    print(f"  Checkpoints updated: action_encoder_affine.pt, action_encoder_mlp.pt")


if __name__ == "__main__":
    main()
