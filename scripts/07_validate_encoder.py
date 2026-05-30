"""
scripts/07_validate_encoder.py
================================
Check 1: Action-encoder directional validity.

For each of the 11 actions, find real StatsBomb possessions where that
action type occurred, SSE-encode the entry frame (z_before) and the next
available frame (z_after), compute the observed delta Δz_real = z_after −
z_before, and compare it to the model's predicted Δz_model =
action_encoder(z_before, A) − z_before via cosine similarity.

Output
------
- 11×11 cosine-similarity confusion matrix (model direction vs. real delta)
  saved as data/results/encoder_confusion.png + .csv
- Per-action mean cosine similarity on the diagonal (the key number)
- Console summary

Interpretation
--------------
Diagonal dominance means the PCA-seeded action directions align with how
real actions actually shift the latent space.  An off-diagonal entry
(A, B) being large means action A in the model moves z in a direction
more consistent with how action B actually shifts the space — a specific,
actionable misalignment.

Run
---
    python -m scripts.07_validate_encoder

Requirements: checkpoints must exist in model/checkpoints/.
StatsBomb open data is fetched live via statsbombpy.
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

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.conditional_engine import ConditionalEngine
from model.action_encoder import Action, ACTION_LABELS, MatchContext


# ── Paths ──────────────────────────────────────────────────────────────────────

BASE  = Path(__file__).parent.parent
CKPT  = BASE / "model" / "checkpoints"
OUT   = BASE / "data" / "results"

REQUIRED = [CKPT / "sse_best.pt",
            CKPT / "generator_best.pt",
            CKPT / "team_fingerprints.pt"]


# ── StatsBomb event → Action mapping ──────────────────────────────────────────

def _etype(event: dict) -> str:
    """Extract event type name from either nested or flattened StatsBomb format."""
    t = event.get("type", {})
    if isinstance(t, dict):
        return t.get("name", "")
    if isinstance(t, str):
        return t
    return event.get("type_name", "")


def _nested(event: dict, key: str) -> dict:
    """Return nested sub-dict (e.g. 'pass', 'carry') or {} if missing/string."""
    v = event.get(key, {})
    return v if isinstance(v, dict) else {}


def event_to_action(event: dict) -> str | None:
    """Map a StatsBomb event dict to one of the 11 Action names, or None."""
    etype = _etype(event)

    if etype == "Shot":
        return "SHOOT"
    if etype == "Dribble":
        return "DRIBBLE"
    if etype == "Pressure":
        return "PRESS"
    if etype in ("Block", "Clearance"):
        return "LOW_BLOCK"
    if etype == "Goal Keeper":
        return "KEEPER_BALL"

    if etype == "Carry":
        loc = event.get("location") or [0, 0]
        end = _nested(event, "carry").get("end_location") or loc
        return "ADVANCE" if end[0] - loc[0] > 5 else "HOLD"

    if etype == "Pass":
        p   = _nested(event, "pass")
        tech = _nested(p, "technique")
        if tech.get("name") == "Through Ball" or event.get("pass_technique_name") == "Through Ball":
            return "THROUGH_BALL"
        if p.get("cross") or event.get("pass_cross"):
            return "CROSS"
        loc = event.get("location") or [0, 0]
        end = p.get("end_location") or event.get("pass_end_location") or loc
        y_delta = abs(end[1] - loc[1])
        x_delta = end[0] - loc[0]
        if y_delta > 30 and abs(x_delta) < 20:
            return "SWITCH_LEFT" if end[1] < loc[1] else "SWITCH_RIGHT"
        return "ADVANCE" if x_delta > 10 else "HOLD"

    return None


def parse_360_frame(frame_data: list) -> torch.Tensor | None:
    """
    Convert StatsBomb freeze-frame list to (N, 4) position tensor [x,y,mate,actor].
    Returns None if fewer than 6 visible players.
    """
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


def make_mask(n_visible: int, max_players: int = 23) -> torch.Tensor:
    mask = torch.zeros(max_players, dtype=torch.bool)
    mask[n_visible:] = True
    return mask


def ctx_from_event(event: dict) -> torch.Tensor:
    """Extract a 3-dim context tensor [zone/3, phase_open, phase_counter]."""
    loc  = event.get("location") or [60, 40]
    zone = min(int(loc[0] / 30), 3)
    return torch.tensor([[zone / 3.0, 0.0, 0.0]], dtype=torch.float32)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    missing = [p.name for p in REQUIRED if not p.exists()]
    if missing:
        print(f"Missing checkpoints: {missing}\nRun 03_train_generator.py first.")
        sys.exit(1)

    print("Loading engine…")
    engine = ConditionalEngine(
        sse_path         = CKPT / "sse_best.pt",
        generator_path   = CKPT / "generator_best.pt",
        fingerprint_path = CKPT / "team_fingerprints.pt",
    )

    # Load possession meta to find match IDs
    meta = pd.read_csv(OUT / "possession_meta.csv", low_memory=False)
    match_ids = meta["match_id"].unique()
    print(f"  {len(match_ids)} matches available in metadata")

    # ── Collect z_before / z_after pairs per action type ──────────────────────

    try:
        from statsbombpy import sb
    except ImportError:
        print("statsbombpy not installed — pip install statsbombpy")
        sys.exit(1)

    ACTION_NAMES = [a.name for a in Action]
    pairs: dict[str, list[tuple]] = {a: [] for a in ACTION_NAMES}
    MAX_PER_ACTION = 200
    MAX_MATCHES    = 30

    print(f"\nCollecting z-before/after pairs from up to {MAX_MATCHES} matches…")
    matches_done = 0
    for mid in match_ids:
        if matches_done >= MAX_MATCHES:
            break
        if all(len(v) >= MAX_PER_ACTION for v in pairs.values()):
            break

        try:
            df = sb.events(match_id=int(mid), flatten_attrs=False)
            if isinstance(df, dict):
                df = pd.concat(df.values(), ignore_index=True)
            # Also fetch 360 freeze frames if available
            # sb.frames() returns a flat DataFrame (one row per player per event).
            # Group by event id to build {event_uuid: [player_dict, ...]} lookup.
            try:
                _f360 = sb.frames(match_id=int(mid))
                frames_360 = {
                    eid: grp.to_dict("records")
                    for eid, grp in _f360.groupby("id")
                }
                has_360 = bool(frames_360)
            except Exception:
                frames_360 = {}
                has_360 = False
        except Exception as e:
            continue

        matches_done += 1
        df = df.sort_values("index") if "index" in df.columns else df

        for i in range(len(df) - 1):
            row_before = df.iloc[i]
            row_after  = df.iloc[i + 1]

            action_name = event_to_action(row_before.to_dict())
            if action_name is None:
                continue
            if len(pairs[action_name]) >= MAX_PER_ACTION:
                continue

            # Get 360 frames for both events
            if not has_360:
                continue

            eid_before = row_before.get("id")
            eid_after  = row_after.get("id")
            if eid_before is None or eid_after is None:
                continue

            ff_before = frames_360.get(eid_before)
            ff_after  = frames_360.get(eid_after)
            if not ff_before or not ff_after:
                continue

            pos_before = parse_360_frame(ff_before)
            pos_after  = parse_360_frame(ff_after)
            if pos_before is None or pos_after is None:
                continue

            mask_b = make_mask(min(len(ff_before), 23))
            mask_a = make_mask(min(len(ff_after),  23))
            ctx_b  = ctx_from_event(row_before.to_dict())

            pairs[action_name].append((
                pos_before.unsqueeze(0),
                mask_b.unsqueeze(0),
                pos_after.unsqueeze(0),
                mask_a.unsqueeze(0),
                ctx_b,
            ))

    # Fallback: if 360 frames are empty, use team fingerprints directly
    n_collected = {a: len(v) for a, v in pairs.items()}
    print("  Pairs collected:", {a: n for a, n in n_collected.items()})

    if sum(n_collected.values()) < 11:
        print("\nInsufficient 360-frame pairs. Falling back to fingerprint-space check.")
        print("Comparing Δz_model direction consistency across actions using")
        print("  cosine(Δz_model_A, Δz_model_B) for all A, B pairs.")
        _fingerprint_space_check(engine)
        return

    # ── Compute Δz vectors ────────────────────────────────────────────────────

    print("\nComputing Δz vectors…")
    from model.action_encoder import apply_action

    # For model Δz: use mean fingerprint as representative z_before
    z_mean = engine.mean_fp.to(engine.device)
    ctx_default = MatchContext(score_diff=0, minute=45, zone=1, phase=0, poss_team=0)

    model_deltas: dict[str, torch.Tensor] = {}
    for act in Action:
        z_mod = apply_action(engine.action_encoder, z_mean, act, ctx_default, 1.0)
        model_deltas[act.name] = F.normalize((z_mod - z_mean).cpu().unsqueeze(0), dim=-1)

    real_deltas: dict[str, torch.Tensor] = {}
    for action_name, plist in pairs.items():
        if not plist:
            continue
        deltas = []
        for pos_b, mask_b, pos_a, mask_a, ctx_t in plist:
            ctx_3 = ctx_t  # (1, 3)
            z_b = engine.encode_frame(pos_b, mask_b, ctx_3)   # (1, 256)
            z_a = engine.encode_frame(pos_a, mask_a, ctx_3)
            delta = F.normalize((z_a - z_b), dim=-1)
            deltas.append(delta)
        real_deltas[action_name] = torch.cat(deltas, dim=0).mean(0, keepdim=True)   # (1, 256)

    # ── 11×11 cosine similarity matrix ────────────────────────────────────────

    labels = [ACTION_LABELS[a] for a in Action]
    n = len(Action)
    matrix = np.zeros((n, n))

    for i, act_i in enumerate(Action):
        if act_i.name not in real_deltas:
            continue
        real_d = real_deltas[act_i.name]          # (1, 256)
        for j, act_j in enumerate(Action):
            model_d = model_deltas[act_j.name]    # (1, 256)
            cos = F.cosine_similarity(real_d, model_d, dim=-1).item()
            matrix[i, j] = cos

    # Save CSV
    df_mat = pd.DataFrame(matrix, index=labels, columns=labels)
    df_mat.to_csv(OUT / "encoder_confusion.csv")
    print(f"  Saved encoder_confusion.csv")

    # ── Plot ──────────────────────────────────────────────────────────────────

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-0.5, vmax=1.0)
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Model Δz direction (action B)")
    ax.set_ylabel("Real Δz direction (action A)")
    ax.set_title("Action Encoder Directional Validity\n"
                 "Cell (A,B) = cos(real Δz for A, model Δz for B)\n"
                 "Diagonal dominance → PCA-seeded directions track real action semantics")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if abs(matrix[i,j]) < 0.7 else "white")
    plt.colorbar(im, ax=ax, label="Cosine similarity")
    plt.tight_layout()
    fig.savefig(OUT / "encoder_confusion.png", dpi=150)
    plt.close(fig)
    print(f"  Saved encoder_confusion.png")

    diag = matrix.diagonal()
    print(f"\nDiagonal (model-matches-real cosine similarity):")
    for i, act in enumerate(Action):
        print(f"  {ACTION_LABELS[act]:18s}  {diag[i]:+.3f}")
    print(f"\n  Mean diagonal: {diag.mean():.3f}  (>0 = model directions are aligned with real)")


def _fingerprint_space_check(engine: ConditionalEngine):
    """
    Fallback when 360 pairs aren't available.
    Checks internal consistency: are the model's 11 action directions
    mutually distinguishable, and do they point in geometrically sensible
    directions relative to each other?
    """
    from model.action_encoder import apply_action

    z_mean = engine.mean_fp.to(engine.device)
    ctx    = MatchContext(score_diff=0, minute=45, zone=1, phase=0, poss_team=0)
    labels = [ACTION_LABELS[a] for a in Action]
    n      = len(Action)

    # Compute normalised Δz for every action
    deltas = []
    for act in Action:
        z_mod = apply_action(engine.action_encoder, z_mean, act, ctx, 1.0)
        deltas.append(F.normalize((z_mod - z_mean).cpu(), dim=-1))
    deltas = torch.stack(deltas)   # (11, 256)

    # Pairwise cosine similarity between model action directions
    matrix = (deltas @ deltas.T).numpy()

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Action Direction Pairwise Cosine Similarity (model-internal)\n"
                 "Off-diagonal near 0 → actions are distinguishable in z-space")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if abs(matrix[i,j]) < 0.7 else "white")
    plt.colorbar(im, ax=ax, label="Cosine similarity")
    plt.tight_layout()
    fig.savefig(OUT / "encoder_self_similarity.png", dpi=150)
    plt.close(fig)
    print(f"  Saved encoder_self_similarity.png")

    off_diag = matrix[~np.eye(n, dtype=bool)]
    print(f"\n  Mean off-diagonal: {off_diag.mean():.3f}  (near 0 → action directions are distinguishable)")
    print(f"  Max off-diagonal:  {off_diag.max():.3f}  (< 0.7 suggests good separation)")


if __name__ == "__main__":
    main()
