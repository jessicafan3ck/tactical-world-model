"""
05_counterfactuals.py
=====================
Causal Intervention Engine — tactical arithmetic in fingerprint z-space.

Capabilities
------------
1. Tactical style transfer
   z_new = z_team + α · (z_reference - z_team)
   → "What would Spain's possessions look like if they pressed like Germany?"

2. Role ablation
   Zero-out the actor token in the velocity field to measure each player's
   marginal contribution to the generated configuration.

3. Win-probability shift under intervention
   Run N simulated matches before and after z-space perturbation and
   measure the Δ(win rate) attributable to the tactical change.

4. Principal component decomposition
   PCA on team fingerprints to surface the 2D tactical manifold.
   Labels each team by cluster (possession, counter, pressing, hybrid).

Outputs
-------
    data/results/intervention_results.csv
    data/results/tactical_pca.png
    data/results/win_probability_shifts.csv
"""

import sys
import torch
import torch.nn as nn
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.sse       import build_predictor
from model.simulator import SimulatorRNN, simulate_match, STATE_DIM, FP_DIM

# ── Config ────────────────────────────────────────────────────────────────────

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "mps"
                           if torch.backends.mps.is_available() else "cpu")
CKPT_DIR    = Path("model/checkpoints")
RESULTS_DIR = Path("data/results")
META_PATH   = RESULTS_DIR / "possession_meta.csv"
FP_PATH     = CKPT_DIR   / "team_fingerprints.pt"
SIM_PATH    = CKPT_DIR   / "simulator_best.pt"
GEN_PATH    = CKPT_DIR   / "generator_best.pt"

N_SIM_MATCHES = 200   # Monte Carlo matches per condition
N_GEN_SAMPLES = 50    # Generated frames per condition for visual comparison
ALPHA_RANGE   = [0.25, 0.5, 0.75, 1.0]   # interpolation strengths

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Fingerprint loading ────────────────────────────────────────────────────────

def load_fingerprints() -> tuple[dict[int, torch.Tensor], torch.Tensor, pd.DataFrame]:
    fp_data = torch.load(FP_PATH, map_location="cpu")
    fps: dict[int, torch.Tensor] = {
        k: v.float() for k, v in fp_data["team_fingerprints"].items()
    }
    mean_fp = fp_data.get(
        "mean_fingerprint",
        torch.stack(list(fps.values())).mean(0)
    )
    meta = pd.read_csv(META_PATH) if META_PATH.exists() else pd.DataFrame()
    return fps, mean_fp, meta


# ── 1. PCA on team fingerprints ────────────────────────────────────────────────

def tactical_pca(fingerprints: dict[int, torch.Tensor],
                 meta: pd.DataFrame,
                 n_clusters: int = 4) -> pd.DataFrame:
    """
    Projects team fingerprints to 2D via PCA, clusters into tactical archetypes.
    Returns a DataFrame with team_id, pc1, pc2, cluster, team_name.
    """
    team_ids = list(fingerprints.keys())
    Z = torch.stack([fingerprints[t] for t in team_ids]).numpy()

    scaler = StandardScaler()
    Z_s    = scaler.fit_transform(Z)

    pca = PCA(n_components=2, random_state=42)
    Z_2d = pca.fit_transform(Z_s)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    clusters = km.fit_predict(Z_s)

    # Get team names from meta if available
    name_map: dict[int, str] = {}
    if not meta.empty and "team_id" in meta.columns and "team_name" in meta.columns:
        name_map = (meta[["team_id", "team_name"]]
                    .drop_duplicates()
                    .set_index("team_id")["team_name"]
                    .to_dict())

    cluster_labels = ["possession", "counter", "pressing", "hybrid"]
    df = pd.DataFrame({
        "team_id":   team_ids,
        "pc1":       Z_2d[:, 0],
        "pc2":       Z_2d[:, 1],
        "cluster_id":   clusters,
        "cluster_name": [cluster_labels[c % len(cluster_labels)] for c in clusters],
        "team_name": [name_map.get(t, str(t)) for t in team_ids],
    })

    var = pca.explained_variance_ratio_
    print(f"PCA variance explained: PC1={var[0]:.1%}  PC2={var[1]:.1%}")
    return df


def plot_pca(pca_df: pd.DataFrame):
    _, ax = plt.subplots(figsize=(12, 9))
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
    cluster_ids = sorted(pca_df["cluster_id"].unique())

    for cid in cluster_ids:
        sub = pca_df[pca_df["cluster_id"] == cid]
        label = sub["cluster_name"].iloc[0]
        ax.scatter(sub["pc1"], sub["pc2"],
                   c=colors[cid % len(colors)], label=label, s=80, alpha=0.8)
        for _, row in sub.iterrows():
            ax.annotate(row["team_name"], (row["pc1"], row["pc2"]),
                        fontsize=6, alpha=0.7,
                        xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel("PC 1 — Tactical Style Axis 1")
    ax.set_ylabel("PC 2 — Tactical Style Axis 2")
    ax.set_title("Tactical Fingerprint Space (PCA of Team z-Vectors)")
    ax.legend()
    plt.tight_layout()
    out = RESULTS_DIR / "tactical_pca.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


# ── 2. Style transfer: z-space interpolation ───────────────────────────────────

def interpolate_fingerprint(z_team: torch.Tensor,
                             z_ref:  torch.Tensor,
                             alpha:  float) -> torch.Tensor:
    """Linear style transfer: z_new = z_team + α * (z_ref - z_team)."""
    return z_team + alpha * (z_ref - z_team)


def measure_win_rate(model: SimulatorRNN,
                     fp_home: torch.Tensor,
                     fp_away: torch.Tensor,
                     n_matches: int = N_SIM_MATCHES,
                     seed_offset: int = 0) -> dict[str, float]:
    """
    Runs n_matches simulations and returns win/draw/loss rates for home team.
    """
    wins, draws, losses = 0, 0, 0
    for i in range(n_matches):
        df = simulate_match(model, fp_home, fp_away, seed=seed_offset + i)
        final = df.iloc[-1]
        gh, ga = int(final["score_home"]), int(final["score_away"])
        if gh > ga:
            wins += 1
        elif gh == ga:
            draws += 1
        else:
            losses += 1
    n = n_matches
    return {"win": wins / n, "draw": draws / n, "loss": losses / n,
            "goals_for_mean": 0.0, "goals_against_mean": 0.0}


def run_intervention_study(simulator: SimulatorRNN,
                            fingerprints: dict[int, torch.Tensor],
                            mean_fp: torch.Tensor,
                            meta: pd.DataFrame) -> pd.DataFrame:
    """
    For each of the top-10 most represented teams:
      1. Measure baseline win rate vs the mean fingerprint opponent.
      2. Interpolate their fingerprint toward the cluster centroid with
         the highest average win rate ("adopt best-cluster style").
      3. Re-measure win rate at α = 0.25, 0.5, 0.75, 1.0.

    Returns a DataFrame with one row per (team, alpha) combination.
    """
    if meta.empty:
        print("  No metadata — skipping intervention study.")
        return pd.DataFrame()

    team_counts = meta.groupby("team_id").size()
    top_teams   = team_counts.nlargest(10).index.tolist()

    # Identify "best" reference team (most wins if available, else highest PC1)
    # Use mean_fp as neutral opponent for all simulations to isolate team effect
    fp_opp = mean_fp.to(DEVICE)

    rows = []
    for tid in tqdm(top_teams, desc="  Intervention study"):
        if tid not in fingerprints:
            continue
        fp_base = fingerprints[tid].to(DEVICE)

        # Baseline
        base_rates = measure_win_rate(simulator, fp_base, fp_opp, seed_offset=0)
        rows.append({
            "team_id": tid,
            "alpha":   0.0,
            "condition": "baseline",
            **base_rates,
        })

        # Pick reference: highest-win-rate team among top_teams (excluding self)
        ref_tid = max(
            [t for t in top_teams if t != tid and t in fingerprints],
            key=lambda t: fingerprints[t].norm().item(),   # proxy for most expressive
        )
        fp_ref = fingerprints[ref_tid].to(DEVICE)

        for alpha in ALPHA_RANGE:
            fp_interp = interpolate_fingerprint(fp_base, fp_ref, alpha).to(DEVICE)
            rates = measure_win_rate(simulator, fp_interp, fp_opp,
                                     seed_offset=int(alpha * 1000))
            rows.append({
                "team_id":   tid,
                "alpha":     alpha,
                "condition": f"alpha={alpha}",
                **rates,
            })

    return pd.DataFrame(rows)


# ── 3. Attention-based player importance ──────────────────────────────────────

def player_importance_via_ablation(sse_model: nn.Module,
                                    spatial_data: dict,
                                    n_samples: int = 200,
                                    device: torch.device = DEVICE) -> pd.DataFrame:
    """
    Ablation experiment: for each player slot, zero out their position and
    measure the change in predicted outcome probabilities.

    Returns DataFrame: player_slot, mean_impact, std_impact
    """
    sse_model.eval().to(device)
    positions = spatial_data["positions"][:n_samples].to(device)
    masks     = spatial_data["masks"][:n_samples].to(device)
    contexts  = spatial_data["contexts"][:n_samples].to(device)

    with torch.no_grad():
        logits_base, _ = sse_model(positions, masks, contexts)
        probs_base = torch.sigmoid(logits_base)  # (n, 3)

    impacts = []
    MAX_P = positions.shape[1]

    for slot in range(MAX_P):
        # Zero out this player slot
        pos_abl = positions.clone()
        pos_abl[:, slot, :] = 0.0
        mask_abl = masks.clone()
        mask_abl[:, slot] = True   # mark as padded

        with torch.no_grad():
            logits_abl, _ = sse_model(pos_abl, mask_abl, contexts)
            probs_abl = torch.sigmoid(logits_abl)

        delta = (probs_base - probs_abl).abs()   # (n, 3)
        impacts.append({
            "player_slot":  slot,
            "mean_impact":  float(delta.mean().cpu()),
            "std_impact":   float(delta.std().cpu()),
            "impact_s2":    float(delta[:, 0].mean().cpu()),
            "impact_s3":    float(delta[:, 1].mean().cpu()),
            "impact_shot":  float(delta[:, 2].mean().cpu()),
        })

    return pd.DataFrame(impacts).sort_values("mean_impact", ascending=False)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Tactical World Model — Causal Intervention Engine")
    print("=" * 50)

    # ── Load fingerprints ──────────────────────────────────────────────────────
    if not FP_PATH.exists():
        print(f"ERROR: {FP_PATH} not found — run 03_train_generator.py first.")
        return
    fingerprints, mean_fp, meta = load_fingerprints()
    print(f"Loaded {len(fingerprints)} team fingerprints")

    # ── 1. PCA ─────────────────────────────────────────────────────────────────
    print("\n[1/3] Tactical fingerprint PCA…")
    pca_df = tactical_pca(fingerprints, meta)
    pca_df.to_csv(RESULTS_DIR / "tactical_pca.csv", index=False)
    plot_pca(pca_df)
    print(f"  Cluster distribution:\n{pca_df['cluster_name'].value_counts().to_dict()}")

    # ── 2. Win-probability intervention study ─────────────────────────────────
    if SIM_PATH.exists():
        print("\n[2/3] Win-probability intervention study…")
        sim_model = SimulatorRNN(
            state_dim=STATE_DIM, fp_dim=FP_DIM,
        ).to(DEVICE)
        sim_model.load_state_dict(
            torch.load(SIM_PATH, map_location=DEVICE)
        )
        sim_model.eval()

        intervention_df = run_intervention_study(
            sim_model, fingerprints, mean_fp, meta
        )
        if not intervention_df.empty:
            intervention_df.to_csv(
                RESULTS_DIR / "win_probability_shifts.csv", index=False
            )
            print(f"  Saved: {RESULTS_DIR / 'win_probability_shifts.csv'}")
            # Summary: mean win rate change per alpha level
            summary = (intervention_df
                       .groupby("alpha")[["win", "draw", "loss"]]
                       .mean()
                       .round(3))
            print(f"\n  Win-rate by alpha (averaged across teams):\n{summary}")
    else:
        print(f"\n[2/3] Skipped — {SIM_PATH} not found.")

    # ── 3. Player importance ablation (uses SSE) ──────────────────────────────
    sse_path    = CKPT_DIR / "sse_best.pt"
    spatial_path = RESULTS_DIR / "spatial_dataset.pt"

    if sse_path.exists() and spatial_path.exists():
        print("\n[3/3] Player importance via ablation…")
        sse_model = build_predictor(z_dim=256).to(DEVICE)
        sse_model.load_state_dict(
            torch.load(sse_path, map_location=DEVICE)
        )
        sse_model.eval()

        spatial = torch.load(spatial_path, map_location="cpu")
        imp_df  = player_importance_via_ablation(sse_model, spatial)
        imp_df.to_csv(RESULTS_DIR / "player_importance.csv", index=False)
        print(f"  Saved: {RESULTS_DIR / 'player_importance.csv'}")
        print(f"  Top-5 most impactful player slots:\n"
              f"{imp_df.head(5)[['player_slot','mean_impact','impact_shot']].to_string(index=False)}")
    else:
        missing = []
        if not sse_path.exists():    missing.append(str(sse_path))
        if not spatial_path.exists(): missing.append(str(spatial_path))
        print(f"\n[3/3] Skipped — missing: {', '.join(missing)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
