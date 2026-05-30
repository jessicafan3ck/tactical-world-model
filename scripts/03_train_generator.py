"""
03_train_generator.py
=====================
Three-phase training pipeline for the Tactical World Model generator.

Phase 1 — Train SSE (TacticalPredictor)
    Trains the Set Spatial Encoder to predict possession outcomes
    from freeze-frame player configurations. Establishes the latent
    tactical space z ∈ R^256.

Phase 2 — Compute Team Fingerprints
    Runs the frozen SSE over all spatial samples, mean-pools z per
    team → stable fingerprint vectors used to condition the generator.
    Resolves opponent fingerprints from match co-occurrence.

Phase 3 — Train TacticalGenerator (Flow Matching)
    Trains the conditional flow matching model to generate realistic
    22-player configurations conditioned on team fingerprints + game state.
    Validates with MMD (Maximum Mean Discrepancy) on held-out matches.

Outputs:
    model/checkpoints/sse_best.pt
    model/checkpoints/team_fingerprints.pt      {team_id: tensor(256,)}
    model/checkpoints/generator_best.pt
    data/results/training_curves.csv
    data/results/mmd_validation.csv
"""

import sys
import json
import math
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.sse import build_predictor, masked_bce_loss, count_parameters
from model.flow_matching import build_generator, TacticalGenerator

# ── Config ────────────────────────────────────────────────────────────────────

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "mps"
                         if torch.backends.mps.is_available() else "cpu")
Z_DIM     = 256
FP_DIM    = 256         # fingerprint dim (= z_dim)
VAL_FRAC  = 0.1
SEED      = 42

SSE_EPOCHS  = 60
SSE_BS      = 512
SSE_LR      = 1e-3

GEN_EPOCHS  = 100
GEN_BS      = 256
GEN_LR      = 5e-4

CKPT_DIR    = Path("model/checkpoints")
RESULTS_DIR = Path("data/results")
DATA_PATH   = RESULTS_DIR / "spatial_dataset.pt"
META_PATH   = RESULTS_DIR / "possession_meta.csv"

CKPT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Dataset ───────────────────────────────────────────────────────────────────

class SpatialDataset(Dataset):
    def __init__(self, data: dict):
        self.positions = data["positions"]   # (N, P, 4)
        self.masks     = data["masks"]       # (N, P)
        self.contexts  = data["contexts"]    # (N, 3)
        self.outcomes  = data["outcomes"]    # (N, 3)
        self.match_ids = data["match_ids"]
        self.team_ids  = data["team_ids"]

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        return {
            "positions": self.positions[idx],
            "mask":      self.masks[idx],
            "context":   self.contexts[idx],
            "outcomes":  self.outcomes[idx],
            "match_id":  self.match_ids[idx],
            "team_id":   self.team_ids[idx],
        }


def match_split(dataset: SpatialDataset,
                val_frac: float = 0.1,
                seed: int = 42) -> tuple[list, list]:
    """Split by match_id so val matches never appear in train."""
    rng        = np.random.default_rng(seed)
    match_ids  = np.array(dataset.match_ids)
    unique     = np.unique(match_ids)
    rng.shuffle(unique)
    n_val      = max(1, int(len(unique) * val_frac))
    val_set    = set(unique[:n_val])

    train_idx = [i for i, m in enumerate(match_ids) if m not in val_set]
    val_idx   = [i for i, m in enumerate(match_ids) if m in val_set]
    return train_idx, val_idx


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_sse_auc(model: nn.Module,
                    loader: DataLoader,
                    device: torch.device) -> dict[str, float]:
    """OOF AUC for each of the 3 targets on a held-out loader."""
    model.eval()
    all_preds  = [[] for _ in range(3)]
    all_labels = [[] for _ in range(3)]
    with torch.no_grad():
        for batch in loader:
            pos = batch["positions"].to(device)
            msk = batch["mask"].to(device)
            ctx = batch["context"].to(device)
            tgt = batch["outcomes"]

            _, logits = model(pos, msk, ctx)
            probs = torch.sigmoid(logits).cpu()

            for t in range(3):
                valid = ~torch.isnan(tgt[:, t])
                if valid.sum() > 0:
                    all_preds[t].extend(probs[valid, t].tolist())
                    all_labels[t].extend(tgt[valid, t].tolist())

    names = ["reached_s2", "reached_s3", "reached_shot"]
    aucs  = {}
    for t, name in enumerate(names):
        if len(set(all_labels[t])) == 2 and len(all_labels[t]) > 0:
            aucs[name] = roc_auc_score(all_labels[t], all_preds[t])
        else:
            aucs[name] = float("nan")
    return aucs


def rbf_mmd(x: torch.Tensor, y: torch.Tensor,
            sigmas: list[float] = [0.5, 1.0, 2.0, 5.0]) -> float:
    """
    Maximum Mean Discrepancy with RBF kernel.
    x, y : (N, D)  — flattened position tensors
    Lower = generated distribution closer to real.
    """
    x, y = x.float(), y.float()
    n, m = x.shape[0], y.shape[0]

    def K(a, b):
        diff = a.unsqueeze(1) - b.unsqueeze(0)     # (n, m, D)
        sq   = (diff ** 2).sum(-1)                  # (n, m)
        return sum(torch.exp(-sq / (2 * s**2)) for s in sigmas) / len(sigmas)

    kxx = K(x, x).sum() / (n * n)
    kyy = K(y, y).sum() / (m * m)
    kxy = K(x, y).sum() / (n * m)
    return (kxx + kyy - 2 * kxy).item()


# ── Phase 1: Train SSE ────────────────────────────────────────────────────────

def train_sse(dataset: SpatialDataset) -> nn.Module:
    print("\n" + "=" * 55)
    print("PHASE 1 — Training SSE (TacticalPredictor)")
    print("=" * 55)

    model = build_predictor(z_dim=Z_DIM).to(DEVICE)
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Device: {DEVICE}")

    train_idx, val_idx = match_split(dataset, VAL_FRAC, SEED)
    print(f"Train samples: {len(train_idx):,}  |  Val samples: {len(val_idx):,}")

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=SSE_BS, shuffle=True, num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=SSE_BS, shuffle=False, num_workers=0,
    )

    opt   = torch.optim.AdamW(model.parameters(), lr=SSE_LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=SSE_EPOCHS)

    best_val_loss = float("inf")
    log_rows = []

    for epoch in range(1, SSE_EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            pos = batch["positions"].to(DEVICE)
            msk = batch["mask"].to(DEVICE)
            ctx = batch["context"].to(DEVICE)
            tgt = batch["outcomes"].to(DEVICE)

            _, logits = model(pos, msk, ctx)
            loss = sum(masked_bce_loss(logits[:, t], tgt[:, t]) for t in range(3))

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()

        sched.step()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                pos = batch["positions"].to(DEVICE)
                msk = batch["mask"].to(DEVICE)
                ctx = batch["context"].to(DEVICE)
                tgt = batch["outcomes"].to(DEVICE)
                _, logits = model(pos, msk, ctx)
                val_loss += sum(
                    masked_bce_loss(logits[:, t], tgt[:, t]) for t in range(3)
                ).item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CKPT_DIR / "sse_best.pt")

        if epoch % 10 == 0 or epoch == 1:
            aucs = compute_sse_auc(model, val_loader, DEVICE)
            print(f"  Epoch {epoch:3d}/{SSE_EPOCHS}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"s2={aucs['reached_s2']:.3f}  "
                  f"s3={aucs['reached_s3']:.3f}  "
                  f"shot={aucs['reached_shot']:.3f}")
        else:
            aucs = {"reached_s2": float("nan"),
                    "reached_s3": float("nan"),
                    "reached_shot": float("nan")}

        log_rows.append({
            "phase": "sse", "epoch": epoch,
            "train_loss": train_loss, "val_loss": val_loss,
            **aucs,
        })

    # Final AUC on best checkpoint
    model.load_state_dict(torch.load(CKPT_DIR / "sse_best.pt",
                                      map_location=DEVICE))
    aucs = compute_sse_auc(model, val_loader, DEVICE)
    print(f"\nBest SSE — s2={aucs['reached_s2']:.4f}  "
          f"s3={aucs['reached_s3']:.4f}  "
          f"shot={aucs['reached_shot']:.4f}")

    pd.DataFrame(log_rows).to_csv(
        RESULTS_DIR / "sse_training_log.csv", index=False
    )
    return model, log_rows


# ── Phase 2: Compute Team Fingerprints ────────────────────────────────────────

def compute_fingerprints(sse_model: nn.Module,
                          dataset:   SpatialDataset) -> dict[int, torch.Tensor]:
    """
    Mean-pools z over all possessions per team → team fingerprint.
    Also resolves opponent fingerprints from match co-occurrence.
    Returns {team_id: tensor(FP_DIM,)}.
    """
    print("\n" + "=" * 55)
    print("PHASE 2 — Computing Team Fingerprints")
    print("=" * 55)

    sse_model.eval()
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)

    team_z: dict[int, list] = {}
    with torch.no_grad():
        for batch in tqdm(loader, desc="  Encoding possessions"):
            pos  = batch["positions"].to(DEVICE)
            msk  = batch["mask"].to(DEVICE)
            z    = sse_model.encoder(pos, msk).cpu()
            tids = batch["team_id"]
            for i, tid in enumerate(tids):
                tid = int(tid)
                team_z.setdefault(tid, []).append(z[i])

    fingerprints = {
        tid: torch.stack(zs).mean(dim=0)
        for tid, zs in team_z.items()
    }
    print(f"  Team fingerprints computed: {len(fingerprints)} teams")

    # Infer opponent mapping from match co-occurrence
    match_teams: dict[int, set] = {}
    for i in range(len(dataset)):
        mid = dataset.match_ids[i]
        tid = int(dataset.team_ids[i])
        match_teams.setdefault(mid, set()).add(tid)

    opponent_map: dict[int, list] = {}
    for mid, teams in match_teams.items():
        teams = list(teams)
        if len(teams) == 2:
            opponent_map.setdefault(teams[0], []).append(teams[1])
            opponent_map.setdefault(teams[1], []).append(teams[0])

    # Mean opponent fingerprint per team (used when no specific opponent given)
    mean_fp = torch.stack(list(fingerprints.values())).mean(0)
    opp_fingerprints: dict[int, torch.Tensor] = {}
    for tid, opp_ids in opponent_map.items():
        opp_fps = [fingerprints[o] for o in opp_ids if o in fingerprints]
        opp_fingerprints[tid] = (
            torch.stack(opp_fps).mean(0) if opp_fps else mean_fp
        )

    save_dict = {
        "team_fingerprints":     fingerprints,
        "opponent_fingerprints": opp_fingerprints,
        "mean_fingerprint":      mean_fp,
    }
    torch.save(save_dict, CKPT_DIR / "team_fingerprints.pt")
    print(f"  Saved: {CKPT_DIR}/team_fingerprints.pt")

    # Summary
    fp_mat = torch.stack(list(fingerprints.values()))
    print(f"  Fingerprint stats — mean norm: {fp_mat.norm(dim=1).mean():.3f}  "
          f"std: {fp_mat.std(dim=0).mean():.3f}")

    return fingerprints, opp_fingerprints, mean_fp


# ── Phase 3: Train Flow Matching Generator ────────────────────────────────────

class GeneratorDataset(Dataset):
    """
    Wraps SpatialDataset with resolved team fingerprints.
    Yields (x_1, roles, c) for flow matching training.
    """

    def __init__(self,
                 spatial_ds:         SpatialDataset,
                 fingerprints:       dict[int, torch.Tensor],
                 opp_fingerprints:   dict[int, torch.Tensor],
                 mean_fp:            torch.Tensor):
        self.ds       = spatial_ds
        self.fp       = fingerprints
        self.opp_fp   = opp_fingerprints
        self.mean_fp  = mean_fp

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item    = self.ds[idx]
        pos     = item["positions"]          # (P, 4)
        mask    = item["mask"]               # (P,)
        ctx     = item["context"]            # (3,)  [zone/3, is_counter, is_sp]
        tid     = int(item["team_id"])

        # Split positions into (x,y) and roles
        xy    = pos[:, :2].clone()           # (P, 2) — noisy during training
        roles = pos[:, 2:].clone()           # (P, 2) — fixed [is_teammate, is_actor]

        # Team fingerprints
        z_A = self.fp.get(tid, self.mean_fp)
        z_B = self.opp_fp.get(tid, self.mean_fp)

        # Game state conditioning
        zone_idx     = int(round(float(ctx[0]) * 3))
        zone_onehot  = torch.zeros(4)
        zone_onehot[zone_idx] = 1.0

        is_counter  = float(ctx[1])
        is_sp       = float(ctx[2])
        phase_onehot = torch.zeros(4)
        if is_counter > 0.5:
            phase_onehot[1] = 1.0
        elif is_sp > 0.5:
            phase_onehot[2] = 1.0
        else:
            phase_onehot[0] = 1.0

        return {
            "xy":           xy,
            "roles":        roles,
            "mask":         mask,
            "z_A":          z_A,
            "z_B":          z_B,
            "score_diff":   torch.zeros(1),
            "minute_norm":  torch.full((1,), 0.5),
            "phase_onehot": phase_onehot,
            "zone_onehot":  zone_onehot,
        }


def train_generator(generator:       TacticalGenerator,
                    spatial_ds:       SpatialDataset,
                    fingerprints:     dict,
                    opp_fingerprints: dict,
                    mean_fp:          torch.Tensor) -> list:
    print("\n" + "=" * 55)
    print("PHASE 3 — Training Flow Matching Generator")
    print("=" * 55)
    print(f"Parameters: {count_parameters(generator):,}")

    gen_ds = GeneratorDataset(
        spatial_ds, fingerprints, opp_fingerprints, mean_fp
    )

    train_idx, val_idx = match_split(spatial_ds, VAL_FRAC, SEED)
    train_loader = DataLoader(
        Subset(gen_ds, train_idx),
        batch_size=GEN_BS, shuffle=True, num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        Subset(gen_ds, val_idx),
        batch_size=GEN_BS, shuffle=False, num_workers=0,
    )

    opt   = torch.optim.AdamW(
        generator.parameters(), lr=GEN_LR, weight_decay=1e-4
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=GEN_EPOCHS
    )

    best_val_loss = float("inf")
    log_rows      = []

    # Collect real val positions for MMD
    val_real_xy = torch.cat(
        [Subset(gen_ds, val_idx)[i]["xy"].unsqueeze(0) for i in range(min(500, len(val_idx)))],
        dim=0
    )  # (N, P, 2)

    for epoch in range(1, GEN_EPOCHS + 1):
        generator.train()
        train_loss = 0.0

        for batch in train_loader:
            xy    = batch["xy"].to(DEVICE)
            roles = batch["roles"].to(DEVICE)
            mask  = batch["mask"].to(DEVICE)
            z_A   = batch["z_A"].to(DEVICE)
            z_B   = batch["z_B"].to(DEVICE)
            sd    = batch["score_diff"].to(DEVICE)
            mn    = batch["minute_norm"].to(DEVICE)
            ph    = batch["phase_onehot"].to(DEVICE)
            zo    = batch["zone_onehot"].to(DEVICE)

            c    = generator.encode_condition(z_A, z_B, sd, mn, ph, zo)
            loss = generator.flow_matching_loss(xy, roles, c, mask)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()

        sched.step()
        train_loss /= len(train_loader)

        # Val loss
        generator.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                xy    = batch["xy"].to(DEVICE)
                roles = batch["roles"].to(DEVICE)
                mask  = batch["mask"].to(DEVICE)
                z_A   = batch["z_A"].to(DEVICE)
                z_B   = batch["z_B"].to(DEVICE)
                sd    = batch["score_diff"].to(DEVICE)
                mn    = batch["minute_norm"].to(DEVICE)
                ph    = batch["phase_onehot"].to(DEVICE)
                zo    = batch["zone_onehot"].to(DEVICE)
                c     = generator.encode_condition(z_A, z_B, sd, mn, ph, zo)
                val_loss += generator.flow_matching_loss(xy, roles, c, mask).item()
        val_loss /= max(len(val_loader), 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(generator.state_dict(),
                       CKPT_DIR / "generator_best.pt")

        # MMD every 10 epochs
        mmd = float("nan")
        if epoch % 10 == 0 or epoch == 1:
            with torch.no_grad():
                sample_batch = next(iter(val_loader))
                roles_s = sample_batch["roles"].to(DEVICE)
                mask_s  = sample_batch["mask"].to(DEVICE)
                z_A_s   = sample_batch["z_A"].to(DEVICE)
                z_B_s   = sample_batch["z_B"].to(DEVICE)
                sd_s    = sample_batch["score_diff"].to(DEVICE)
                mn_s    = sample_batch["minute_norm"].to(DEVICE)
                ph_s    = sample_batch["phase_onehot"].to(DEVICE)
                zo_s    = sample_batch["zone_onehot"].to(DEVICE)
                c_s     = generator.encode_condition(
                    z_A_s, z_B_s, sd_s, mn_s, ph_s, zo_s
                )
                gen_xy = generator.generate(roles_s, c_s, mask_s, n_steps=20)

            n = min(GEN_BS, val_real_xy.shape[0])
            real_flat = val_real_xy[:n].reshape(n, -1)
            gen_flat  = gen_xy.cpu()[:n].reshape(n, -1)
            mmd       = rbf_mmd(real_flat, gen_flat)
            print(f"  Epoch {epoch:3d}/{GEN_EPOCHS}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"MMD={mmd:.5f}")
        else:
            print(f"  Epoch {epoch:3d}/{GEN_EPOCHS}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}")

        log_rows.append({
            "phase": "generator", "epoch": epoch,
            "train_loss": train_loss, "val_loss": val_loss,
            "mmd": mmd,
        })

    print(f"\nBest generator val loss: {best_val_loss:.4f}")
    pd.DataFrame(log_rows).to_csv(
        RESULTS_DIR / "generator_training_log.csv", index=False
    )
    return log_rows


# ── Training curves plot ──────────────────────────────────────────────────────

def plot_curves(sse_log: list, gen_log: list) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Tactical World Model — Training Curves", fontweight="bold")

    # SSE loss
    sse_df = pd.DataFrame(sse_log)
    axes[0].plot(sse_df["epoch"], sse_df["train_loss"], label="Train")
    axes[0].plot(sse_df["epoch"], sse_df["val_loss"],   label="Val")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("BCE Loss")
    axes[0].set_title("SSE — Outcome Prediction Loss")
    axes[0].legend()

    # SSE AUC (epochs where it was computed)
    auc_df = sse_df.dropna(subset=["reached_s2"])
    for col, label in [("reached_s2","s2"), ("reached_s3","s3"),
                        ("reached_shot","shot")]:
        axes[1].plot(auc_df["epoch"], auc_df[col], marker="o", label=label)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("AUC")
    axes[1].set_title("SSE — Validation AUC")
    axes[1].legend()

    # Generator loss + MMD
    gen_df = pd.DataFrame(gen_log)
    ax2 = axes[2]
    ax2.plot(gen_df["epoch"], gen_df["train_loss"], label="Train loss", color="blue")
    ax2.plot(gen_df["epoch"], gen_df["val_loss"],   label="Val loss",   color="orange")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("CFM Loss")
    ax3 = ax2.twinx()
    mmd_df = gen_df.dropna(subset=["mmd"])
    ax3.plot(mmd_df["epoch"], mmd_df["mmd"],
             label="MMD", color="red", linestyle="--", marker="s")
    ax3.set_ylabel("MMD (↓ better)")
    ax2.set_title("Generator — Flow Matching Loss + MMD")
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {RESULTS_DIR}/training_curves.png")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("Tactical World Model — Generator Training")
    print(f"Device: {DEVICE}")

    if not DATA_PATH.exists():
        print(f"ERROR: {DATA_PATH} not found.")
        print("Run scripts/02_build_dataset.py first.")
        return

    print(f"\nLoading dataset: {DATA_PATH}")
    raw     = torch.load(DATA_PATH, map_location="cpu", weights_only=True)
    dataset = SpatialDataset(raw)
    print(f"  Samples: {len(dataset):,}")
    print(f"  Position tensor: {dataset.positions.shape}")

    # Phase 1
    sse_model, sse_log = train_sse(dataset)

    # Phase 2
    fingerprints, opp_fingerprints, mean_fp = compute_fingerprints(
        sse_model, dataset
    )

    # Phase 3
    generator = build_generator(fingerprint_dim=FP_DIM).to(DEVICE)
    gen_log   = train_generator(
        generator, dataset, fingerprints, opp_fingerprints, mean_fp
    )

    # Save combined curves
    plot_curves(sse_log, gen_log)

    all_log = pd.concat([
        pd.DataFrame(sse_log), pd.DataFrame(gen_log)
    ], ignore_index=True)
    all_log.to_csv(RESULTS_DIR / "training_curves.csv", index=False)
    print(f"\nAll done. Checkpoints in {CKPT_DIR}/")


if __name__ == "__main__":
    main()
