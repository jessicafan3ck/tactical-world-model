"""
scripts/13_debias_generator.py
================================
Generator debias: fit and validate per-zone x-position corrections.

The Check 3 audit showed the generator places teammates ~5pp too far
forward (territory_zone: 0.525 generated vs 0.477 real).  This inflates
p_advance predictions uniformly, compressing the Check 2 discriminative
range.

Strategy: post-hoc affine correction to generated teammate x-positions.
  For each (zone, phase) cell:
    1. Sample real positions from spatial_dataset.pt.
    2. Generate positions with matching conditioning.
    3. Fit Δx = real_mean_x(teammates) - gen_mean_x(teammates).
  At inference:
    x_corrected = clamp(x_gen + Δx(zone, phase), 0, 1)

This is applied *before* SSE sees the frame, so SSE receives calibrated
positions rather than the raw biased ones.

Outputs
-------
- data/results/generator_debias.json   correction table (zone × phase)
- data/results/generator_debias.png    real vs gen before/after per zone

Run
---
    python -m scripts.13_debias_generator
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.conditional_engine import ConditionalEngine
from model.action_encoder import MatchContext

BASE = Path(__file__).parent.parent
CKPT = BASE / "model" / "checkpoints"
OUT  = BASE / "data" / "results"

REQUIRED = [CKPT / "sse_best.pt",
            CKPT / "generator_best.pt",
            CKPT / "team_fingerprints.pt"]

N_PER_CELL = 300   # samples per (zone, phase) cell
GEN_STEPS  = 20


def teammate_mean_x(positions: torch.Tensor, mask: torch.Tensor) -> float:
    """Mean x of valid teammate positions. positions: (N, 4), mask: (N,) bool."""
    valid = positions[~mask]          # (K, 4)
    mates = valid[valid[:, 2] > 0.5]  # is_teammate
    return float(mates[:, 0].mean()) if len(mates) else 0.5


def territory_zone_frac(positions: torch.Tensor, mask: torch.Tensor) -> float:
    """Fraction of teammates in attacking half (x > 0.5)."""
    valid = positions[~mask]
    mates = valid[valid[:, 2] > 0.5]
    return float((mates[:, 0] > 0.5).float().mean()) if len(mates) else 0.5


def main():
    missing = [p.name for p in REQUIRED if not p.exists()]
    if missing:
        print(f"Missing: {missing}"); sys.exit(1)

    print("Loading engine and dataset…")
    engine = ConditionalEngine(
        sse_path         = CKPT / "sse_best.pt",
        generator_path   = CKPT / "generator_best.pt",
        fingerprint_path = CKPT / "team_fingerprints.pt",
    )

    dataset = torch.load(OUT / "spatial_dataset.pt", map_location="cpu")
    positions = dataset["positions"]   # (N, 23, 4) — last col may be padding
    masks     = dataset["masks"]       # (N, 23)
    contexts  = dataset["contexts"]    # (N, 3): [zone/3, phase1, phase_gt1]

    # Team fingerprint pool for conditioning
    team_ids  = list(engine.fingerprints.keys())
    fp_list   = [engine.fingerprints[t] for t in team_ids]
    fp_stack  = torch.stack(fp_list)   # (T, 256)

    # Decode zone/phase from context vector
    zones_all  = (contexts[:, 0] * 3).round().long()   # 0-3
    # phase: 0=open, 1=counter, 2=set_piece/restart (encoded as phase1 or phase_gt1)
    phases_all = torch.zeros(len(contexts), dtype=torch.long)
    phases_all[contexts[:, 1] > 0.5] = 1
    phases_all[contexts[:, 2] > 0.5] = 2

    print(f"\nFitting debias corrections over {N_PER_CELL} samples per (zone, phase) cell…")

    corrections = {}  # (zone, phase) → {"dx": float, "real_tz": float, "gen_tz": float}

    all_zones  = sorted(zones_all.unique().tolist())
    all_phases = sorted(phases_all.unique().tolist())

    for zone in all_zones:
        for phase in all_phases:
            cell_mask = (zones_all == zone) & (phases_all == phase)
            n_avail   = int(cell_mask.sum())
            if n_avail < 10:
                continue

            # Sample real frames
            idx = cell_mask.nonzero(as_tuple=True)[0]
            chosen = idx[torch.randperm(len(idx))[:N_PER_CELL]]

            real_pos  = positions[chosen]   # (K, 23, 4)
            real_mask = masks[chosen]       # (K, 23)

            real_tz = [territory_zone_frac(real_pos[i], real_mask[i])
                       for i in range(len(chosen))]
            real_mx = [teammate_mean_x(real_pos[i], real_mask[i])
                       for i in range(len(chosen))]

            # Generate frames with matching (zone, phase) conditioning
            # Use random team pairs from the fingerprint pool
            rng_idx = torch.randint(len(team_ids), (len(chosen),))
            opp_idx = torch.randint(len(team_ids), (len(chosen),))

            gen_tz_list = []
            gen_mx_list = []
            engine.generator.eval()

            # Batch to avoid OOM — generate in chunks of 64
            chunk = 64
            gen_all = []
            for start in range(0, len(chosen), chunk):
                end    = min(start + chunk, len(chosen))
                bs     = end - start
                z_A    = fp_stack[rng_idx[start:end]].to(engine.device)  # (bs, 256)
                z_B    = fp_stack[opp_idx[start:end]].to(engine.device)

                sd   = torch.zeros(bs, 1, device=engine.device)
                mn   = torch.full((bs, 1), 0.5, device=engine.device)
                p_oh = torch.zeros(bs, 4, device=engine.device)
                p_oh[:, min(phase, 3)] = 1.0
                z_oh = torch.zeros(bs, 4, device=engine.device)
                z_oh[:, min(zone, 3)]  = 1.0

                with torch.no_grad():
                    c = engine.generator.encode_condition(z_A, z_B, sd, mn, p_oh, z_oh)
                    roles = engine.roles.expand(bs, -1, -1)
                    mask_b = engine.mask.expand(bs, -1)
                    gen_xy = engine.generator.generate(
                        roles, c, mask_b, n_steps=GEN_STEPS
                    )  # (bs, 22, 2)

                # Build full (x, y, is_teammate, is_actor) tensor
                roles_full = roles.cpu()  # (bs, 22, 2)
                xy_cpu     = gen_xy.cpu()
                pos_full   = torch.cat([xy_cpu, roles_full], dim=-1)  # (bs, 22, 4)
                mask_full  = mask_b.cpu()

                gen_all.append((pos_full, mask_full))

            for pos_b, msk_b in gen_all:
                for i in range(len(pos_b)):
                    gen_tz_list.append(territory_zone_frac(pos_b[i], msk_b[i]))
                    gen_mx_list.append(teammate_mean_x(pos_b[i], msk_b[i]))

            real_tz_mean = float(np.mean(real_tz))
            gen_tz_mean  = float(np.mean(gen_tz_list))
            real_mx_mean = float(np.mean(real_mx))
            gen_mx_mean  = float(np.mean(gen_mx_list))
            dx           = real_mx_mean - gen_mx_mean   # shift to apply to gen x

            corrections[(int(zone), int(phase))] = {
                "dx":        round(dx, 6),
                "real_tz":   round(real_tz_mean, 4),
                "gen_tz":    round(gen_tz_mean, 4),
                "real_mx":   round(real_mx_mean, 4),
                "gen_mx":    round(gen_mx_mean, 4),
                "n_real":    int(n_avail),
                "n_sampled": len(gen_tz_list),
            }
            print(f"  Zone {zone} Phase {phase}: "
                  f"real_tz={real_tz_mean:.3f}  gen_tz={gen_tz_mean:.3f}  "
                  f"Δx={dx:+.4f}  (n={int(n_avail)})")

    # Build the JSON-serializable version with string keys
    json_corrections = {f"{z}_{p}": v for (z, p), v in corrections.items()}

    # Summary statistics
    all_dx  = [v["dx"]      for v in corrections.values()]
    all_rtz = [v["real_tz"] for v in corrections.values()]
    all_gtz = [v["gen_tz"]  for v in corrections.values()]
    print(f"\n  Mean Δx applied: {np.mean(all_dx):+.4f}  "
          f"(range [{min(all_dx):+.4f}, {max(all_dx):+.4f}])")
    print(f"  Mean real_tz: {np.mean(all_rtz):.4f}   "
          f"Mean gen_tz:  {np.mean(all_gtz):.4f}")

    # ── Visualise ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(all_zones), figsize=(4 * len(all_zones), 4),
                             sharey=True)
    if len(all_zones) == 1:
        axes = [axes]

    for ax, zone in zip(axes, all_zones):
        phase_labels, real_tzs, gen_tzs = [], [], []
        for phase in all_phases:
            key = (int(zone), int(phase))
            if key not in corrections:
                continue
            c = corrections[key]
            phase_labels.append(f"Ph{phase}")
            real_tzs.append(c["real_tz"])
            gen_tzs.append(c["gen_tz"])

        x_ = np.arange(len(phase_labels))
        ax.bar(x_ - 0.2, real_tzs, 0.35, label="Real", color="steelblue", alpha=0.8)
        ax.bar(x_ + 0.2, gen_tzs,  0.35, label="Generated", color="tomato",  alpha=0.8)
        ax.set_xticks(x_); ax.set_xticklabels(phase_labels)
        ax.set_title(f"Zone {zone}")
        ax.set_ylabel("territory_zone (frac forward)") if zone == all_zones[0] else None
        ax.legend(fontsize=7)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Generator Debias: Real vs Generated territory_zone\n"
                 "(before correction — Δx to be applied at inference)",
                 fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT / "generator_debias.png", dpi=150)
    plt.close(fig)
    print("  Saved generator_debias.png")

    # Save corrections
    with open(OUT / "generator_debias.json", "w") as f:
        json.dump({"corrections": json_corrections,
                   "meta": {"n_per_cell": N_PER_CELL, "gen_steps": GEN_STEPS,
                             "global_mean_dx": round(float(np.mean(all_dx)), 6)}},
                  f, indent=2)
    print("  Saved generator_debias.json")


if __name__ == "__main__":
    main()
