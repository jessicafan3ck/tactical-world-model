"""
01_download_data.py
===================
Downloads all Women's football data from StatsBomb open data,
prioritising competitions with 360 freeze frames for the
Tactical World Model (WWC 2027).

Outputs (gitignored):
    data/raw/statsbomb/matches/        one JSON per competition-season
    data/raw/statsbomb/events/         one JSON per match
    data/raw/statsbomb/lineups/        one JSON per match
    data/raw/statsbomb/three-sixty/    one JSON per match (where available)
"""

import json
import time
import requests
from pathlib import Path
from tqdm import tqdm

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
OUT  = Path("data/raw/statsbomb")

# All Women's competitions in StatsBomb open data.
# 360 column: True = freeze frames available (confirmed from coverage audit).
WOMEN_COMPETITIONS = [
    # competition_id, season_id, label, has_360
    (72,  107, "Women's World Cup 2023",         True),
    (72,   30, "Women's World Cup 2019",         False),
    (53,  315, "UEFA Women's Euro 2025",         True),
    (53,  106, "UEFA Women's Euro 2022",         True),
    (37,  281, "FA Women's Super League 2023/24",False),
    (37,   90, "FA Women's Super League 2020/21",False),
    (37,   42, "FA Women's Super League 2019/20",False),
    (37,    4, "FA Women's Super League 2018/19",False),
    (49,  107, "NWSL 2023",                      False),
    (49,    3, "NWSL 2018",                      False),
    (182, 281, "Liga F 2023/24",                 False),
    (135, 281, "Frauen Bundesliga 2023/24",       False),
    (131, 281, "Serie A Women 2023/24",           False),
]

def fetch_json(url: str, retries: int = 3) -> list | dict | None:
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
        except Exception as e:
            if attempt == retries - 1:
                print(f"  ✗ failed: {url} — {e}")
                return None
            time.sleep(2 ** attempt)
    return None

def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)

def download_competition(comp_id: int, season_id: int,
                         label: str, has_360: bool) -> dict:
    stats = {"label": label, "matches": 0, "events": 0,
             "lineups": 0, "three_sixty": 0, "skipped": 0}

    # matches
    matches_path = OUT / "matches" / f"{comp_id}_{season_id}.json"
    if matches_path.exists():
        with open(matches_path) as f:
            matches = json.load(f)
    else:
        url = f"{BASE}/matches/{comp_id}/{season_id}.json"
        matches = fetch_json(url)
        if matches is None:
            print(f"  ✗ no matches found for {label}")
            return stats
        save_json(matches_path, matches)
    stats["matches"] = len(matches)

    match_ids = [m["match_id"] for m in matches]

    for mid in tqdm(match_ids, desc=f"  {label}", leave=False):
        # events
        ev_path = OUT / "events" / f"{mid}.json"
        if not ev_path.exists():
            events = fetch_json(f"{BASE}/events/{mid}.json")
            if events:
                save_json(ev_path, events)
                stats["events"] += 1
            else:
                stats["skipped"] += 1
        else:
            stats["events"] += 1

        # lineups
        lu_path = OUT / "lineups" / f"{mid}.json"
        if not lu_path.exists():
            lineups = fetch_json(f"{BASE}/lineups/{mid}.json")
            if lineups:
                save_json(lu_path, lineups)
                stats["lineups"] += 1
        else:
            stats["lineups"] += 1

        # 360 freeze frames
        if has_360:
            ff_path = OUT / "three-sixty" / f"{mid}.json"
            if not ff_path.exists():
                ff = fetch_json(f"{BASE}/three-sixty/{mid}.json")
                if ff:
                    save_json(ff_path, ff)
                    stats["three_sixty"] += 1
            else:
                stats["three_sixty"] += 1

    return stats

def main():
    print("Tactical World Model — StatsBomb Data Download")
    print("=" * 55)
    print(f"Downloading {len(WOMEN_COMPETITIONS)} Women's competitions\n")

    total_matches = 0
    total_360 = 0

    for comp_id, season_id, label, has_360 in WOMEN_COMPETITIONS:
        tag = "360 ✓" if has_360 else "     "
        print(f"[{tag}] {label}")
        stats = download_competition(comp_id, season_id, label, has_360)
        print(f"       → {stats['matches']} matches, "
              f"{stats['events']} events, "
              f"{stats['three_sixty']} freeze-frame files")
        total_matches += stats["matches"]
        total_360 += stats["three_sixty"]

    print(f"\nDone. Total matches: {total_matches} | "
          f"Matches with 360 data: {total_360}")
    print(f"Raw data saved to: {OUT.resolve()}")

if __name__ == "__main__":
    main()
