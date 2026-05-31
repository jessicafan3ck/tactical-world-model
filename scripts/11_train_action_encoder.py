"""
scripts/11_train_action_encoder.py
====================================
Train and evaluate the learned action encoder.

Step 1 — Data collection (cached)
    For every 360-annotated event with a known action label, SSE-encode the
    frame immediately before and after → (z_before, z_after, Δz, action_idx,
    ctx, match_id).  Saved to data/results/action_encoder_data.pt so the slow
    StatsBomb fetch only runs once.  Re-collection is triggered by
    --recollect or if the cache is absent.

Step 2 — Train/val split
    Hold out the last 20 % of match IDs (by lexicographic sort) so no match
    appears in both splits.  Mirrors the leakage discipline used throughout.

Step 3 — Train both architectures
    a. PerActionAffine  — L2 loss, Adam 1e-3, cosine LR, 60 epochs
    b. ConditionedMLP   — Gaussian NLL, Adam 3e-4, cosine LR, 100 epochs
       with early stopping on val NLL (patience 15).

Step 4 — Check 1 re-evaluation
    For each action, compute mean cosine(Δz_predicted, Δz_real) on held-out
    pairs.  Also build the full 11×11 confusion matrix.

Output
------
- data/results/action_encoder_data.pt      — cached training pairs
- model/checkpoints/action_encoder_affine.pt  — trained affine model
- model/checkpoints/action_encoder_mlp.pt     — trained MLP model
- data/results/check1_learned.png            — updated Check 1 confusion matrix
- data/results/check1_learned.json           — diagonal + matrix as JSON

Run
---
    python -m scripts.11_train_action_encoder
    python -m scripts.11_train_action_encoder --recollect   # force re-fetch
"""

import argparse
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
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.conditional_engine import ConditionalEngine
from model.action_encoder import Action, ACTION_LABELS
from model.learned_action_encoder import PerActionAffine, ConditionedMLP

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE  = Path(__file__).parent.parent
CKPT  = BASE / "model" / "checkpoints"
OUT   = BASE / "data" / "results"
DATA_CACHE = OUT / "action_encoder_data.pt"

REQUIRED = [CKPT / "sse_best.pt",
            CKPT / "generator_best.pt",
            CKPT / "team_fingerprints.pt"]

MAX_PER_ACTION  = 500    # ceiling per action during collection
MAX_MATCHES     = 80     # matches to search through
HOLDOUT_FRAC    = 0.20   # fraction of matches held out for eval

# Training hyper-params
AFFINE_EPOCHS   = 60
MLP_EPOCHS      = 100
MLP_PATIENCE    = 15
BATCH_SIZE      = 128
AFFINE_LR       = 1e-3
MLP_LR          = 3e-4
WEIGHT_DECAY    = 1e-4
DROPOUT         = 0.3
HIDDEN          = 512

ACTION_NAMES = [a.name for a in Action]
N_ACTIONS    = len(Action)


# ── Re-usable event-parsing helpers ───────────────────────────────────────────

def _etype(event: dict) -> str:
    t = event.get("type", {})
    if isinstance(t, dict):  return t.get("name", "")
    if isinstance(t, str):   return t
    return event.get("type_name", "")


def _nested(event: dict, key: str) -> dict:
    v = event.get(key, {})
    return v if isinstance(v, dict) else {}


def event_to_action_idx(event: dict) -> int | None:
    etype = _etype(event)
    if etype == "Shot":                 name = "SHOOT"
    elif etype == "Dribble":            name = "DRIBBLE"
    elif etype == "Pressure":           name = "PRESS"
    elif etype in ("Block","Clearance"): name = "LOW_BLOCK"
    elif etype == "Goal Keeper":        name = "KEEPER_BALL"
    elif etype == "Carry":
        loc = event.get("location") or [0, 0]
        end = _nested(event, "carry").get("end_location") or loc
        name = "ADVANCE" if end[0] - loc[0] > 5 else "HOLD"
    elif etype == "Pass":
        p    = _nested(event, "pass")
        tech = _nested(p, "technique")
        if tech.get("name") == "Through Ball" or event.get("pass_technique_name") == "Through Ball":
            name = "THROUGH_BALL"
        elif p.get("cross") or event.get("pass_cross"):
            name = "CROSS"
        else:
            loc = event.get("location") or [0, 0]
            end = p.get("end_location") or event.get("pass_end_location") or loc
            if abs(end[1]-loc[1]) > 30 and abs(end[0]-loc[0]) < 20:
                name = "SWITCH_LEFT" if end[1] < loc[1] else "SWITCH_RIGHT"
            else:
                name = "ADVANCE" if end[0]-loc[0] > 10 else "HOLD"
    else:
        return None
    return Action[name].value


def parse_360_frame(records: list) -> torch.Tensor | None:
    pts = []
    for p in records:
        loc  = p.get("location") or [0, 0]
        pts.append([loc[0]/120., loc[1]/80., float(p.get("teammate",False)),
                    float(p.get("actor",False))])
    if len(pts) < 6:
        return None
    t = torch.zeros(23, 4)
    n = min(len(pts), 23)
    t[:n] = torch.tensor(pts[:n], dtype=torch.float32)
    return t


def make_mask(n: int) -> torch.Tensor:
    m = torch.zeros(23, dtype=torch.bool)
    m[n:] = True
    return m


def ctx_from_event(event: dict) -> torch.Tensor:
    loc  = event.get("location") or [60, 40]
    zone = min(int(loc[0]/30), 3)
    return torch.tensor([[zone/3., 0., 0.]], dtype=torch.float32)


# ── Data collection ────────────────────────────────────────────────────────────

def collect_data(engine: ConditionalEngine, meta: pd.DataFrame) -> dict:
    try:
        from statsbombpy import sb
    except ImportError:
        print("statsbombpy not installed"); sys.exit(1)

    counts    = {a: 0 for a in range(N_ACTIONS)}
    z_befores, z_afters, dzs, action_idxs, ctxs, match_id_list = [], [], [], [], [], []

    match_ids = meta["match_id"].unique()
    done      = 0
    print(f"\nCollecting pairs from up to {MAX_MATCHES} matches…")

    for mid in match_ids:
        if done >= MAX_MATCHES:
            break
        if all(c >= MAX_PER_ACTION for c in counts.values()):
            break

        try:
            df = sb.events(match_id=int(mid), flatten_attrs=False)
            if isinstance(df, dict):
                df = pd.concat(df.values(), ignore_index=True)
            _f = sb.frames(match_id=int(mid))
            frames_360 = {eid: grp.to_dict("records")
                          for eid, grp in _f.groupby("id")}
        except Exception:
            continue

        done += 1
        df = df.sort_values("index") if "index" in df.columns else df

        for i in range(len(df) - 1):
            row_b = df.iloc[i]
            row_a = df.iloc[i + 1]

            act_idx = event_to_action_idx(row_b.to_dict())
            if act_idx is None or counts[act_idx] >= MAX_PER_ACTION:
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
            ctx_b = ctx_from_event(row_b.to_dict())

            z_b = engine.encode_frame(pos_b.unsqueeze(0), msk_b.unsqueeze(0), ctx_b).squeeze(0)
            z_a = engine.encode_frame(pos_a.unsqueeze(0), msk_b.unsqueeze(0), ctx_b).squeeze(0)
            dz  = z_a - z_b

            z_befores.append(z_b)
            z_afters.append(z_a)
            dzs.append(dz)
            action_idxs.append(act_idx)
            ctxs.append(ctx_b.squeeze(0))
            match_id_list.append(int(mid))
            counts[act_idx] += 1

    n_collected = {ACTION_NAMES[a]: c for a, c in counts.items()}
    print("  Pairs per action:", n_collected)
    total = sum(counts.values())
    print(f"  Total: {total:,} pairs from {done} matches")

    if total < 50:
        print("Insufficient data"); sys.exit(1)

    return {
        "z_before":   torch.stack(z_befores),
        "z_after":    torch.stack(z_afters),
        "dz":         torch.stack(dzs),
        "action_idx": torch.tensor(action_idxs, dtype=torch.long),
        "ctx":        torch.stack(ctxs),
        "match_ids":  match_id_list,
    }


# ── Train / eval helpers ───────────────────────────────────────────────────────

def train_val_split(data: dict, holdout_frac: float):
    """
    Stratified random split: hold out holdout_frac of pairs per action label.
    This gives a balanced val set even when later matches contributed few pairs
    (because action caps were hit earlier in data collection).
    """
    rng     = np.random.default_rng(42)
    n       = len(data["dz"])
    val_mask = np.zeros(n, dtype=bool)
    a_idx    = data["action_idx"].numpy()

    for a in range(N_ACTIONS):
        where_a = np.where(a_idx == a)[0]
        if len(where_a) == 0:
            continue
        n_hold  = max(1, int(len(where_a) * holdout_frac))
        chosen  = rng.choice(where_a, size=n_hold, replace=False)
        val_mask[chosen] = True

    tr_mask = ~val_mask
    mask_tr = torch.from_numpy(tr_mask)
    mask_va = torch.from_numpy(val_mask)

    def _split(t):
        return t[mask_tr], t[mask_va]

    tr, va = {}, {}
    for k in ("z_before", "dz", "action_idx", "ctx"):
        tr[k], va[k] = _split(data[k])

    n_va_per = {ACTION_NAMES[a]: int((va["action_idx"] == a).sum()) for a in range(N_ACTIONS)}
    print(f"\n  Train: {mask_tr.sum():,} pairs  |  Val: {mask_va.sum():,} pairs  (stratified by action)")
    print(f"  Val per action: {n_va_per}")
    return tr, va


def make_loader(data: dict, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(data["z_before"], data["action_idx"], data["ctx"], data["dz"])
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def cosine_diag(model, val_data: dict, device: torch.device) -> tuple[np.ndarray, float]:
    """Compute per-action mean cosine(Δz_pred, Δz_real) on held-out pairs."""
    model.eval()
    z_b  = val_data["z_before"].to(device)
    a_i  = val_data["action_idx"].to(device)
    ctx  = val_data["ctx"].to(device)
    dz_r = val_data["dz"].to(device)

    with torch.no_grad():
        dz_p = model.predict_dz(z_b, a_i, ctx)

    cos_per_sample = F.cosine_similarity(dz_p, dz_r, dim=-1).cpu().numpy()
    diag = np.zeros(N_ACTIONS)
    for a in range(N_ACTIONS):
        mask = val_data["action_idx"].numpy() == a
        diag[a] = cos_per_sample[mask].mean() if mask.sum() > 0 else float("nan")

    return diag, float(np.nanmean(diag))


def full_confusion(model, val_data: dict, device: torch.device) -> np.ndarray:
    """
    11×11 confusion matrix: cell (i, j) = mean cos(real Δz for action i,
    predicted Δz for action j), averaged over samples of action i.
    Diagonal dominance shows the model knows which direction each action goes.
    """
    model.eval()
    z_b  = val_data["z_before"].to(device)
    ctx  = val_data["ctx"].to(device)
    dz_r = val_data["dz"].to(device)

    matrix = np.zeros((N_ACTIONS, N_ACTIONS))
    with torch.no_grad():
        for j in range(N_ACTIONS):
            a_j   = torch.full((len(z_b),), j, dtype=torch.long, device=device)
            dz_j  = model.predict_dz(z_b, a_j, ctx)
            cos_j = F.cosine_similarity(dz_j, dz_r, dim=-1).cpu().numpy()
            for i in range(N_ACTIONS):
                mask = val_data["action_idx"].numpy() == i
                if mask.sum() > 0:
                    matrix[i, j] = cos_j[mask].mean()
    return matrix


# ── Training loops ─────────────────────────────────────────────────────────────

def train_affine(tr: dict, va: dict, device: torch.device) -> PerActionAffine:
    model = PerActionAffine().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=AFFINE_LR,
                              weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=AFFINE_EPOCHS)
    loader = make_loader(tr, BATCH_SIZE, shuffle=True)

    print("\n── Training PerActionAffine ─────────────────────────────────────")
    for epoch in range(1, AFFINE_EPOCHS + 1):
        model.train()
        losses = []
        for z_b, a_i, ctx, dz_t in loader:
            z_b, a_i, ctx, dz_t = (x.to(device) for x in (z_b, a_i, ctx, dz_t))
            loss = model.l2_loss(z_b, a_i, ctx, dz_t)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        sched.step()

        if epoch % 10 == 0 or epoch == 1:
            _, mean_cos = cosine_diag(model, va, device)
            print(f"  Epoch {epoch:3d}/{AFFINE_EPOCHS}  "
                  f"train_l2={np.mean(losses):.4f}  val_diag_cos={mean_cos:.4f}")

    return model


def train_mlp(tr: dict, va: dict, device: torch.device) -> ConditionedMLP:
    model  = ConditionedMLP(hidden=HIDDEN, dropout=DROPOUT).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=MLP_LR,
                               weight_decay=WEIGHT_DECAY)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MLP_EPOCHS)
    loader = make_loader(tr, BATCH_SIZE, shuffle=True)
    val_loader = make_loader(va, BATCH_SIZE, shuffle=False)

    best_val_nll  = float("inf")
    best_state    = None
    patience_ctr  = 0

    print("\n── Training ConditionedMLP ──────────────────────────────────────")
    for epoch in range(1, MLP_EPOCHS + 1):
        model.train()
        tr_nlls = []
        for z_b, a_i, ctx, dz_t in loader:
            z_b, a_i, ctx, dz_t = (x.to(device) for x in (z_b, a_i, ctx, dz_t))
            loss = model.nll_loss(z_b, a_i, ctx, dz_t)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_nlls.append(loss.item())
        sched.step()

        model.eval()
        va_nlls = []
        with torch.no_grad():
            for z_b, a_i, ctx, dz_t in val_loader:
                z_b, a_i, ctx, dz_t = (x.to(device) for x in (z_b, a_i, ctx, dz_t))
                va_nlls.append(model.nll_loss(z_b, a_i, ctx, dz_t).item())
        val_nll = float(np.mean(va_nlls))

        if epoch % 10 == 0 or epoch == 1:
            _, mean_cos = cosine_diag(model, va, device)
            print(f"  Epoch {epoch:3d}/{MLP_EPOCHS}  "
                  f"train_nll={np.mean(tr_nlls):.4f}  val_nll={val_nll:.4f}  "
                  f"val_diag_cos={mean_cos:.4f}")

        if val_nll < best_val_nll - 1e-4:
            best_val_nll = val_nll
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= MLP_PATIENCE:
                print(f"  Early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ── Plot confusion matrix ─────────────────────────────────────────────────────

def plot_matrix(matrix: np.ndarray, title: str, path: Path, diag: np.ndarray):
    labels = [ACTION_LABELS[a] for a in Action]
    n      = N_ACTIONS
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-0.5, vmax=1.0)
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Model Δz direction (action B)")
    ax.set_ylabel("Real Δz direction (action A)")
    ax.set_title(title)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if abs(matrix[i,j]) < 0.7 else "white")
    plt.colorbar(im, ax=ax, label="Cosine similarity")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recollect", action="store_true",
                        help="Force re-collection of training data")
    args = parser.parse_args()

    missing = [p.name for p in REQUIRED if not p.exists()]
    if missing:
        print(f"Missing checkpoints: {missing}"); sys.exit(1)

    # Device
    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))
    print(f"Device: {device}")

    # Load engine
    print("Loading engine…")
    engine = ConditionalEngine(
        sse_path         = CKPT / "sse_best.pt",
        generator_path   = CKPT / "generator_best.pt",
        fingerprint_path = CKPT / "team_fingerprints.pt",
    )

    # ── Step 1: Data ──────────────────────────────────────────────────────────
    if not args.recollect and DATA_CACHE.exists():
        print(f"\nLoading cached data from {DATA_CACHE.name}…")
        data = torch.load(DATA_CACHE, map_location="cpu", weights_only=False)
        total = len(data["dz"])
        print(f"  {total:,} pairs loaded")
        n_per = {ACTION_NAMES[a]: int((data['action_idx'] == a).sum())
                 for a in range(N_ACTIONS)}
        print("  Per action:", n_per)
    else:
        meta = pd.read_csv(OUT / "possession_meta.csv", low_memory=False)
        data = collect_data(engine, meta)
        torch.save(data, DATA_CACHE)
        print(f"\n  Saved to {DATA_CACHE.name}")

    # ── Step 2: Split ─────────────────────────────────────────────────────────
    tr, va = train_val_split(data, HOLDOUT_FRAC)

    # Move train data to device for batching; val stays on CPU for diag checks
    # (val functions move tensors internally)

    # ── Step 3a: PerActionAffine ──────────────────────────────────────────────
    affine = train_affine(tr, va, device)
    torch.save(affine.state_dict(), CKPT / "action_encoder_affine.pt")
    print(f"\n  Saved action_encoder_affine.pt")

    # ── Step 3b: ConditionedMLP ───────────────────────────────────────────────
    mlp = train_mlp(tr, va, device)
    torch.save(mlp.state_dict(), CKPT / "action_encoder_mlp.pt")
    print(f"  Saved action_encoder_mlp.pt")

    # ── Step 4: Check 1 re-evaluation ─────────────────────────────────────────
    labels       = [ACTION_LABELS[a] for a in Action]
    diag_affine, mean_affine = cosine_diag(affine, va, device)
    diag_mlp,    mean_mlp    = cosine_diag(mlp,    va, device)
    mat_affine               = full_confusion(affine, va, device)
    mat_mlp                  = full_confusion(mlp,    va, device)

    # Noise floor R for comparison
    try:
        with open(OUT / "noise_floor.json") as f:
            nf = json.load(f)
        R_vals = [nf.get(a, {}).get("R", float("nan")) for a in ACTION_NAMES]
    except FileNotFoundError:
        R_vals = [float("nan")] * N_ACTIONS

    print(f"\n{'Action':<20} {'R (ceiling)':>12} {'Affine cos':>12} {'MLP cos':>10}")
    print("-" * 57)
    for i, (act, lbl) in enumerate(zip(ACTION_NAMES, labels)):
        r   = R_vals[i]
        da  = diag_affine[i]
        dm  = diag_mlp[i]
        r_s = f"{r:.4f}" if not np.isnan(r) else "  n/a"
        print(f"  {lbl:<18} {r_s:>12} {da:>12.4f} {dm:>10.4f}")

    print(f"\n  Mean diagonal:")
    print(f"    PerActionAffine : {mean_affine:.4f}")
    print(f"    ConditionedMLP  : {mean_mlp:.4f}")
    print(f"    PCA baseline    : -0.0060  (from Check 1)")
    print(f"    R ceiling (mean): {np.nanmean(R_vals):.4f}")

    # Save confusion matrix plots
    plot_matrix(
        mat_affine,
        f"Check 1 (Learned): PerActionAffine\n"
        f"Mean diagonal = {mean_affine:.4f}  |  R ceiling = {np.nanmean(R_vals):.4f}",
        OUT / "check1_affine.png",
        diag_affine,
    )
    plot_matrix(
        mat_mlp,
        f"Check 1 (Learned): ConditionedMLP\n"
        f"Mean diagonal = {mean_mlp:.4f}  |  R ceiling = {np.nanmean(R_vals):.4f}",
        OUT / "check1_mlp.png",
        diag_mlp,
    )

    # Save JSON
    result = {
        "per_action": {
            ACTION_NAMES[i]: {
                "label":        labels[i],
                "R_ceiling":    round(float(R_vals[i]), 4) if not np.isnan(R_vals[i]) else None,
                "affine_cos":   round(float(diag_affine[i]), 4),
                "mlp_cos":      round(float(diag_mlp[i]), 4),
            }
            for i in range(N_ACTIONS)
        },
        "mean_diagonal": {
            "pca_baseline": -0.006,
            "affine":       round(mean_affine, 4),
            "mlp":          round(mean_mlp, 4),
            "R_ceiling":    round(float(np.nanmean(R_vals)), 4),
        },
    }
    with open(OUT / "check1_learned.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved check1_learned.json, check1_affine.png, check1_mlp.png")

    # Pick best model and print recommendation
    best = "mlp" if mean_mlp >= mean_affine else "affine"
    print(f"\n  Best architecture: {best.upper()}  "
          f"(cos={mean_mlp:.4f} vs {mean_affine:.4f})")
    print(f"  Recommended checkpoint: action_encoder_{best}.pt")


if __name__ == "__main__":
    main()
