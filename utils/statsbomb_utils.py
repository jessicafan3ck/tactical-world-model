"""
StatsBomb Data Loading Utilities
=================================
Reads raw StatsBomb JSON files (events, lineups, 360 freeze frames)
and segments events into possessions.

Possession definition: consecutive events by the same team.
A possession ends on:
  - team change (ball won / lost)
  - shot taken
  - ball out of play (detected via play_pattern change or dead ball events)
  - half end

Phase mapping (StatsBomb play_pattern → internal int):
  0 = open_play    (Regular Play)
  1 = counter      (From Counter)
  2 = set_piece    (From Free Kick, Corner, Throw In, Keeper)
  3 = restart      (From Goal Kick, Kick Off)
"""

import json
from pathlib import Path
from typing import Iterator


RAW = Path("data/raw/statsbomb")

DEAD_BALL_TYPES = {
    "Half Start", "Half End", "Period End", "Period Start",
    "Referee Ball-Drop", "Substitution", "Tactical Shift",
    "Player On", "Player Off", "Injury Stoppage", "Error",
    "50/50",                           # contested — treat as break
    "Goal Keeper",                     # keeper holds = break
}

PHASE_MAP = {
    "Regular Play":   0,
    "From Counter":   1,
    "From Free Kick": 2,
    "From Corner":    2,
    "From Throw In":  2,
    "From Keeper":    2,
    "From Goal Kick": 3,
    "From Kick Off":  3,
}
PHASE_NAMES = {0: "open_play", 1: "counter", 2: "set_piece", 3: "restart"}


# ── JSON loaders ──────────────────────────────────────────────────────────────

def load_events(match_id: int) -> list[dict]:
    path = RAW / "events" / f"{match_id}.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def load_lineups(match_id: int) -> list[dict]:
    path = RAW / "lineups" / f"{match_id}.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def load_freeze_frames(match_id: int) -> dict[str, list]:
    """Returns {event_uuid: freeze_frame_list}."""
    path = RAW / "three-sixty" / f"{match_id}.json"
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        return {fr["event_uuid"]: fr.get("freeze_frame", []) for fr in data}
    except (json.JSONDecodeError, KeyError, OSError):
        return {}


def load_matches(comp_id: int, season_id: int) -> list[dict]:
    path = RAW / "matches" / f"{comp_id}_{season_id}.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


# ── Zone / phase helpers ───────────────────────────────────────────────────────

BOX_X, BOX_Y1, BOX_Y2 = 102, 18, 62


def loc_to_zone(loc, event_type: str = "") -> int:
    """Maps a [x, y] location to zone 0–4 (4 = shot)."""
    if event_type == "Shot":
        return 4
    if loc is None:
        return -1
    try:
        x, y = float(loc[0]), float(loc[1])
    except (TypeError, IndexError, ValueError):
        return -1
    if x > BOX_X and BOX_Y1 <= y <= BOX_Y2:
        return 3
    if x > 80:
        return 2
    if x > 40:
        return 1
    return 0


def event_phase(event: dict) -> int:
    name = event.get("play_pattern", {}).get("name", "Regular Play")
    return PHASE_MAP.get(name, 0)


# ── Possession segmentation ───────────────────────────────────────────────────

def iter_possessions(events: list[dict],
                     match_id: int,
                     match_meta: dict) -> Iterator[dict]:
    """
    Yields possession dicts from a sorted event list.

    Each possession dict:
      possession_id  : int (sequential within match)
      match_id       : int
      team_id        : int
      team_name      : str
      competition    : str
      season         : str
      gender         : str
      events         : list[dict]  (full event dicts including id, location, type, etc.)
    """
    if not events:
        return

    comp    = match_meta.get("competition", {}).get("competition_name", "")
    season  = match_meta.get("season", {}).get("season_name", "")
    gender  = match_meta.get("competition", {}).get("competition_gender", "")

    current_team_id   = None
    current_team_name = ""
    current_events: list[dict] = []
    poss_id = 0

    def flush():
        nonlocal poss_id, current_events
        if current_events and current_team_id is not None:
            yield {
                "possession_id": poss_id,
                "match_id":      match_id,
                "team_id":       current_team_id,
                "team_name":     current_team_name,
                "competition":   comp,
                "season":        season,
                "gender":        gender,
                "events":        current_events,
            }
            poss_id += 1
        current_events = []

    for ev in events:
        etype   = ev.get("type",  {}).get("name", "")
        team_id = ev.get("team",  {}).get("id")
        tname   = ev.get("team",  {}).get("name", "")

        # Skip bookkeeping events that don't belong to a team
        if etype in {"Starting XI", "Tactical Shift", "Half Start",
                     "Half End", "Period Start", "Period End",
                     "Referee Ball-Drop", "Substitution", "Error"}:
            yield from flush()
            current_team_id = None
            continue

        if team_id is None:
            continue

        # Team change = new possession
        if team_id != current_team_id and current_team_id is not None:
            yield from flush()

        current_team_id   = team_id
        current_team_name = tname
        current_events.append(ev)

        # Shot ends the possession
        if etype == "Shot":
            yield from flush()
            current_team_id = None

    yield from flush()


# ── Match metadata helper ─────────────────────────────────────────────────────

def parse_match_outcome(match: dict) -> dict:
    """Extracts outcome info from a StatsBomb match dict."""
    home_id    = match.get("home_team", {}).get("home_team_id")
    away_id    = match.get("away_team", {}).get("away_team_id")
    home_score = match.get("home_score", 0) or 0
    away_score = match.get("away_score", 0) or 0

    outcome = {}
    for tid, score, opp_score in [
        (home_id, home_score, away_score),
        (away_id, away_score, home_score),
    ]:
        if score > opp_score:
            outcome[tid] = "win"
        elif score < opp_score:
            outcome[tid] = "loss"
        else:
            outcome[tid] = "draw"
    return outcome


def get_match_ids_with_360() -> set[int]:
    return {int(p.stem) for p in (RAW / "three-sixty").glob("*.json")}
