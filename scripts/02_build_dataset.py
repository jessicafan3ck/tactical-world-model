"""
02_build_dataset.py
===================
Processes raw StatsBomb JSON into the spatial dataset used to train
the Tactical World Model.

Two outputs:
  A) spatial_dataset.pt   — freeze-frame tensors (for SSE + flow matching)
  B) possession_meta.csv  — one row per possession (for match simulator)

For each possession with an available 360 freeze frame, extracts:
  positions  : (MAX_PLAYERS, 4)  [x_norm, y_norm, is_teammate, is_actor]
  mask       : (MAX_PLAYERS,)    True where padded
  context    : (3,)              [entry_state/3, is_counter, is_set_piece]
  outcomes   : (3,)              [reached_s2, reached_s3, reached_shot]
               NaN where start-zone filter makes the target undefined
  probe_labels: dict             handcrafted spatial features for probing
  team_id    : int               for team fingerprint computation

Competitions processed:
  Priority (360 available): WWC 2023, UEFA Women's Euro 2025/2022
  Secondary (no 360):       WWC 2019, WSL, NWSL, Liga F, Frauen BL, Serie A W
  Secondary possessions have outcomes + meta but no spatial tensors.
"""

import json
import sys
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.statsbomb_utils import (
    load_events, load_matches, load_freeze_frames,
    iter_possessions, loc_to_zone, event_phase,
    get_match_ids_with_360, PHASE_NAMES, parse_match_outcome,
)

# ── Config ────────────────────────────────────────────────────────────────────

MAX_PLAYERS  = 23
UNGUARDED_D  = 5.0
OUT_DIR      = Path("data/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

COMPETITIONS = [
    # (comp_id, season_id)
    (72,  107),   # Women's World Cup 2023    — 360 ✓
    (53,  315),   # UEFA Women's Euro 2025    — 360 ✓
    (53,  106),   # UEFA Women's Euro 2022    — 360 ✓
    (72,   30),   # Women's World Cup 2019
    (37,  281),   # FA Women's Super League 2023/24
    (37,   90),   # FA Women's Super League 2020/21
    (37,   42),   # FA Women's Super League 2019/20
    (37,    4),   # FA Women's Super League 2018/19
    (49,  107),   # NWSL 2023
    (49,    3),   # NWSL 2018
    (182, 281),   # Liga F 2023/24
    (135, 281),   # Frauen Bundesliga 2023/24
    (131, 281),   # Serie A Women 2023/24
]


# ── Spatial feature extraction ────────────────────────────────────────────────

def _dist(a, b) -> float:
    return float(np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2))


def frame_to_sample(ff: list[dict],
                    entry_state: int,
                    phase_int:   int) -> dict | None:
    """
    Convert a freeze frame to padded position tensor + probe labels.
    Returns None if the frame is too sparse to be useful.
    """
    actors    = [e["location"] for e in ff if e.get("actor")]
    teammates = [e["location"] for e in ff if e.get("teammate") and not e.get("actor")]
    defenders = [e["location"] for e in ff if not e.get("teammate")]

    if not actors or not teammates or len(defenders) < 2:
        return None

    actor = actors[0]
    ax, ay = float(actor[0]), float(actor[1])

    players = [[ax/120.0, ay/80.0, 1.0, 1.0]]
    for t in teammates:
        players.append([float(t[0])/120.0, float(t[1])/80.0, 1.0, 0.0])
    for d in defenders:
        players.append([float(d[0])/120.0, float(d[1])/80.0, 0.0, 0.0])

    n_real = min(len(players), MAX_PLAYERS)
    players = players[:MAX_PLAYERS]
    while len(players) < MAX_PLAYERS:
        players.append([0.0, 0.0, 0.0, 0.0])

    positions = np.array(players, dtype=np.float32)
    mask      = np.zeros(MAX_PLAYERS, dtype=bool)
    mask[n_real:] = True

    # ── Probe labels (NOT inputs — used to test latent space geometry) ──
    ball_left  = ay < 40
    weak_count = sum(1 for t in teammates if (float(t[1]) < 40) != ball_left)
    weak_side  = weak_count / max(len(teammates), 1)

    pressure   = sum(1.0 / max(_dist(actor, d), 1.0)**2
                     for d in defenders if _dist(actor, d) <= 20)

    left_t     = sum(1 for t in teammates if float(t[1]) < 40)
    width_asym = abs(left_t - (len(teammates) - left_t)) / max(len(teammates), 1)

    vertical   = sum(1 for t in teammates if float(t[0]) > ax) / max(len(teammates), 1)

    open_mates   = [t for t in teammates
                    if all(_dist(t, d) >= UNGUARDED_D for d in defenders)]
    switch_avail = max((_dist(actor, t) for t in open_mates), default=0.0)

    return {
        "positions":      positions,
        "mask":           mask,
        "probe": {
            "s_weak_side":    float(weak_side),
            "s_pressure":     float(pressure),
            "s_switch_avail": float(switch_avail),
            "s_vert_support": float(vertical),
            "s_width_asym":   float(width_asym),
            "territory_zone": int(entry_state),
            "phase_int":      int(phase_int),
            "n_visible":      int(n_real),
        },
    }


# ── Outcome computation ───────────────────────────────────────────────────────

def compute_outcomes(events: list[dict],
                     entry_state: int,
                     shot_in_window: bool) -> np.ndarray:
    """Returns [reached_s2, reached_s3, reached_shot] with NaN where filtered."""
    all_zones = [
        loc_to_zone(ev.get("location"), ev.get("type", {}).get("name", ""))
        for ev in events
    ]
    max_zone = max((z for z in all_zones if z >= 0), default=0)
    has_shot = any(ev.get("type", {}).get("name") == "Shot" for ev in events)

    reached_s2   = float(max_zone >= 2) if entry_state < 2 else float("nan")
    reached_s3   = float(max_zone >= 3) if entry_state < 3 else float("nan")
    reached_shot = float(has_shot)      if not shot_in_window else float("nan")

    return np.array([reached_s2, reached_s3, reached_shot], dtype=np.float32)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Tactical World Model — Build Spatial Dataset")
    print("=" * 50)

    match_ids_360 = get_match_ids_with_360()
    print(f"Matches with 360 data: {len(match_ids_360)}")

    # Storage
    all_positions, all_masks, all_contexts = [], [], []
    all_outcomes, all_probes = [], []
    meta_rows = []

    poss_counter = 0
    N_WINDOW     = 8    # max early events to search for a freeze frame

    for comp_id, season_id in COMPETITIONS:
        matches = load_matches(comp_id, season_id)
        if not matches:
            print(f"  ⚠ No matches found: comp={comp_id} season={season_id}")
            continue

        comp_name = matches[0].get("competition", {}).get("competition_name", f"{comp_id}")
        season    = matches[0].get("season",      {}).get("season_name",      f"{season_id}")
        gender    = matches[0].get("competition", {}).get("competition_gender", "")
        n_spatial = 0

        for match in tqdm(matches, desc=f"  {comp_name} {season}"):
            match_id = match["match_id"]
            events   = load_events(match_id)
            if not events:
                continue

            has_360  = match_id in match_ids_360
            ff_index = load_freeze_frames(match_id) if has_360 else {}
            outcomes_map = parse_match_outcome(match)

            for poss in iter_possessions(events, match_id, match):
                pev = poss["events"]
                if len(pev) < 2:
                    continue

                phase_int   = event_phase(pev[0])
                window      = pev[:N_WINDOW]
                entry_state = 0
                for ev in window:
                    z = loc_to_zone(ev.get("location"),
                                    ev.get("type", {}).get("name", ""))
                    if z >= 0:
                        entry_state = z
                        break

                shot_in_window = any(
                    ev.get("type", {}).get("name") == "Shot" for ev in window
                )

                outcomes = compute_outcomes(pev, entry_state, shot_in_window)
                if np.all(np.isnan(outcomes)):
                    continue

                # Base metadata row (all possessions)
                row = {
                    "possession_id":  poss_counter,
                    "match_id":       match_id,
                    "team_id":        poss["team_id"],
                    "team_name":      poss["team_name"],
                    "competition":    comp_name,
                    "season":         season,
                    "gender":         gender,
                    "entry_state":    entry_state,
                    "phase_int":      phase_int,
                    "phase_name":     PHASE_NAMES.get(phase_int, "open_play"),
                    "match_outcome":  outcomes_map.get(poss["team_id"], "unknown"),
                    "reached_s2":     outcomes[0],
                    "reached_s3":     outcomes[1],
                    "reached_shot":   outcomes[2],
                    "has_spatial":    False,
                }

                # Spatial sample (only for 360 matches)
                if has_360:
                    sample = None
                    for ev in window:
                        ff = ff_index.get(ev.get("id", ""), [])
                        if not ff:
                            continue
                        sample = frame_to_sample(ff, entry_state, phase_int)
                        if sample is not None:
                            break

                    if sample is not None:
                        context = np.array([
                            entry_state / 3.0,
                            float(phase_int == 1),   # is_counter
                            float(phase_int >= 2),   # is_set_piece
                        ], dtype=np.float32)

                        all_positions.append(sample["positions"])
                        all_masks.append(sample["mask"])
                        all_contexts.append(context)
                        all_outcomes.append(outcomes)
                        all_probes.append(sample["probe"])

                        row["has_spatial"] = True
                        row.update(sample["probe"])
                        n_spatial += 1

                meta_rows.append(row)
                poss_counter += 1

        print(f"     → {len(matches)} matches | "
              f"{sum(1 for r in meta_rows if r['competition']==comp_name)} poss | "
              f"{n_spatial} spatial")

    # ── Save spatial dataset ──────────────────────────────────────────────────
    n_spatial_total = len(all_positions)
    print(f"\nTotal possessions: {poss_counter:,}")
    print(f"Spatial samples:   {n_spatial_total:,}")

    if n_spatial_total > 0:
        positions_t = torch.tensor(np.stack(all_positions), dtype=torch.float32)
        masks_t     = torch.tensor(np.stack(all_masks),     dtype=torch.bool)
        contexts_t  = torch.tensor(np.stack(all_contexts),  dtype=torch.float32)
        outcomes_t  = torch.tensor(np.stack(all_outcomes),  dtype=torch.float32)

        spatial_meta = [r for r in meta_rows if r["has_spatial"]]

        dataset = {
            "positions":  positions_t,
            "masks":      masks_t,
            "contexts":   contexts_t,
            "outcomes":   outcomes_t,
            "match_ids":  [r["match_id"] for r in spatial_meta],
            "team_ids":   [r["team_id"]  for r in spatial_meta],
            "genders":    [r["gender"]   for r in spatial_meta],
        }

        out_pt = OUT_DIR / "spatial_dataset.pt"
        torch.save(dataset, out_pt)
        print(f"Saved: {out_pt}  ({out_pt.stat().st_size/1e6:.1f} MB)")

    # ── Save full metadata ────────────────────────────────────────────────────
    meta_df = pd.DataFrame(meta_rows)
    out_csv = OUT_DIR / "possession_meta.csv"
    meta_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}  ({len(meta_df):,} rows)")

    # ── Summary stats ─────────────────────────────────────────────────────────
    spatial_df = meta_df[meta_df["has_spatial"]]
    print("\n── Spatial dataset statistics ──")
    for col in ["reached_s2", "reached_s3", "reached_shot"]:
        valid = spatial_df[col].dropna()
        print(f"  {col:14}: {len(valid):,} valid  base_rate={valid.mean():.3f}")
    print(f"  competitions : {spatial_df['competition'].value_counts().to_dict()}")
    print(f"  gender split : {spatial_df['gender'].value_counts().to_dict()}")
    print(f"  phase dist   : {spatial_df['phase_name'].value_counts().to_dict()}")

    print("\nDone.")


if __name__ == "__main__":
    main()
