"""
server/app.py
=============
Part 3 of the interactive simulation layer.

Thin FastAPI backend. One ConditionalEngine is loaded at startup and
shared across all requests (thread-safe for inference-only use).

Endpoints:
    GET  /              → serves the frontend HTML
    GET  /api/teams     → list of available teams with names
    POST /api/step      → execute one tactical action, return freeze frame + probs
    GET  /api/health    → liveness check

Run:
    cd /Users/jessicafan/tactical-world-model
    uvicorn server.app:app --reload --port 8765

Testable without frontend:
    curl http://localhost:8765/api/health
    curl http://localhost:8765/api/teams
    curl -X POST http://localhost:8765/api/step \
         -H "Content-Type: application/json" \
         -d '{"action":"ADVANCE","team_id_a":2391,"team_id_b":16802,
              "context":{"score_diff":-1,"minute":72,"zone":2,"phase":0,"poss_team":0}}'
"""

import os
import sys
import time
import hashlib
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env from project root if present (never committed — see .gitignore)
load_dotenv(Path(__file__).parent.parent / ".env")

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import torch

from model.action_encoder     import Action, KEY_TO_ACTION, ACTION_LABELS, MatchContext
from model.conditional_engine import ConditionalEngine

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE  = Path(__file__).parent.parent
CKPT  = BASE / "model" / "checkpoints"
META  = BASE / "data" / "results" / "possession_meta.csv"

# ── Global engine (loaded once at startup) ────────────────────────────────────

_engine: ConditionalEngine | None = None
_team_names: dict[int, str] = {}
_commentary_cache: dict[str, str] = {}   # keyed by hash of inputs


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _team_names

    required = [CKPT / "sse_best.pt",
                CKPT / "generator_best.pt",
                CKPT / "team_fingerprints.pt"]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        print(f"WARNING: Missing checkpoints {missing} — engine not loaded.")
        print("Run 03_train_generator.py first, then restart the server.")
    else:
        _engine = ConditionalEngine(
            sse_path         = CKPT / "sse_best.pt",
            generator_path   = CKPT / "generator_best.pt",
            fingerprint_path = CKPT / "team_fingerprints.pt",
        )

    # Load team names from metadata if available
    if META.exists():
        import pandas as pd
        try:
            meta = pd.read_csv(META, usecols=["team_id", "team_name"],
                               low_memory=False)
            _team_names = (meta.drop_duplicates("team_id")
                               .set_index("team_id")["team_name"]
                               .to_dict())
        except Exception:
            pass

    yield   # server runs here


app = FastAPI(title="Tactical World Model", lifespan=lifespan)

# Serve static files (the frontend) from server/static/
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ── Request / response schemas ─────────────────────────────────────────────────

class ContextIn(BaseModel):
    score_diff: float = 0.0
    minute:     float = 45.0
    zone:       int   = 1
    phase:      int   = 0
    poss_team:  int   = 0


class FormationCounts(BaseModel):
    """Outfield tier counts derived from lineup (GK excluded)."""
    def_: int = 4   # defenders
    mid:  int = 3   # midfielders
    fwd:  int = 3   # forwards

    model_config = {"populate_by_name": True}

    @classmethod
    def from_dict(cls, d: dict | None) -> "FormationCounts | None":
        if d is None:
            return None
        return cls(def_=d.get("def", 4), mid=d.get("mid", 3), fwd=d.get("fwd", 3))

    def to_engine_dict(self) -> dict:
        return {"def": self.def_, "mid": self.mid, "fwd": self.fwd}


class StepRequest(BaseModel):
    action:         str         # Action name OR keyboard key (e.g. "ADVANCE" or "w")
    team_id_a:      int
    team_id_b:      int
    context:        ContextIn
    alpha:          float = 1.0
    formation_a:    dict | None = None   # {def, mid, fwd} from /api/squad
    formation_b:    dict | None = None
    continuity:     float = 0.15         # max per-player Δ per step in [0,1] space
    prev_positions: list | None = None   # [[x,y]×22] from previous frame; enables continuity
    actor_slot:     int | None  = None   # Team A slot index holding the ball; None = slot 0


class SequenceStep(BaseModel):
    action: str
    alpha:  float = 1.0


class SequenceRequest(BaseModel):
    team_id_a:       int
    team_id_b:       int
    context:         ContextIn
    sequence:        list[SequenceStep]
    minute_per_step: float = 0.5
    adversarial:     bool  = False
    n_samples:       int   = 1
    noise_std:       float = 0.05
    continuity:      float = 0.0
    formation_a:     dict | None = None   # {def, mid, fwd} from /api/squad
    formation_b:     dict | None = None


class SuggestRequest(BaseModel):
    team_id_a: int
    team_id_b: int
    context:   ContextIn


class OpeningStateRequest(BaseModel):
    team_id_a:   int
    team_id_b:   int
    context:     ContextIn
    formation_a: dict | None = None
    formation_b: dict | None = None


class OptimizeRequest(BaseModel):
    team_id_a:   int
    team_id_b:   int
    context:     ContextIn
    max_depth:   int  = 4    # sequence length to search
    beam_width:  int  = 3    # candidates kept per depth level
    adversarial: bool = False


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return JSONResponse({"message": "Tactical World Model API. "
                         "PUT index.html in server/static/ for the UI."})


@app.get("/api/health")
async def health():
    return {
        "status":        "ok",
        "engine_loaded": _engine is not None,
        "n_teams":       len(_engine.fingerprints) if _engine else 0,
        "device":        str(_engine.device) if _engine else "none",
    }


@app.get("/api/teams")
async def list_teams():
    if _engine is None:
        raise HTTPException(503, "Engine not loaded")
    teams = []
    for tid, fp in _engine.fingerprints.items():
        teams.append({
            "team_id":   tid,
            "team_name": _team_names.get(tid, str(tid)),
            "fp_norm":   round(float(fp.norm()), 3),
        })
    teams.sort(key=lambda t: t["team_name"])
    return teams


# StatsBomb position name → formation tier
_POSITION_TIER: dict[str, str] = {
    "Goalkeeper":         "gk",
    "Center Back":        "def",
    "Left Back":          "def",
    "Right Back":         "def",
    "Left Wing Back":     "def",
    "Right Wing Back":    "def",
    "Defensive Midfield": "mid",
    "Central Midfield":   "mid",
    "Left Midfield":      "mid",
    "Right Midfield":     "mid",
    "Attacking Midfield": "mid",
    "Left Wing":          "fwd",
    "Right Wing":         "fwd",
    "Center Forward":     "fwd",
    "Second Striker":     "fwd",
}


def _start_position_name(positions: list) -> str | None:
    """Extract the position name for the Starting XI entry."""
    if not isinstance(positions, list):
        return None
    for p in positions:
        if p.get("start_reason") == "Starting XI":
            return p.get("name")
    return None


@app.get("/api/squad/{team_id}")
async def squad(team_id: int):
    """Return the starting XI for a team with name, jersey, tier, and formation counts.

    Uses the first match in possession_meta where this team appears, then
    fetches the StatsBomb lineup.  Returns players sorted by jersey_number
    plus a formation_counts dict {def, mid, fwd} derived from lineup positions.
    """
    if not META.exists():
        raise HTTPException(404, "No possession metadata available")

    import pandas as pd
    from statsbombpy import sb as sbpy

    meta = pd.read_csv(META, usecols=["team_id", "match_id"], low_memory=False)
    rows = meta[meta["team_id"] == team_id]["match_id"].unique()
    if not len(rows):
        raise HTTPException(404, f"No matches found for team_id={team_id}")

    match_id = int(rows[0])
    team_name = _team_names.get(team_id, "")

    try:
        lineups = sbpy.lineups(match_id=match_id)
    except Exception as e:
        raise HTTPException(502, f"StatsBomb lineup fetch failed: {e}")

    lineup_df = lineups.get(team_name)
    if lineup_df is None:
        for tname, df in lineups.items():
            if team_name.lower() in tname.lower() or tname.lower() in team_name.lower():
                lineup_df = df
                break
    if lineup_df is None:
        raise HTTPException(404, f"Could not find lineup for '{team_name}' in match {match_id}")

    def _is_starter(positions):
        if not isinstance(positions, list):
            return False
        return any(p.get("start_reason") == "Starting XI" for p in positions)

    starters = lineup_df[lineup_df["positions"].apply(_is_starter)]
    starters = starters.sort_values("jersey_number").head(11)

    players = []
    tier_counts: dict[str, int] = {"def": 0, "mid": 0, "fwd": 0}
    for _, row in starters.iterrows():
        pos_name = _start_position_name(row["positions"])
        tier = _POSITION_TIER.get(pos_name or "", "mid")  # unknown → mid
        if tier in tier_counts:
            tier_counts[tier] += 1
        players.append({
            "jersey_number":    int(row["jersey_number"]),
            "name":             row["player_name"],
            "position":         pos_name or "",
            "positional_tier":  tier,
        })

    # Fallback: if tier counts don't sum to 10 outfield players, use 4-3-3
    if sum(tier_counts.values()) != 10:
        tier_counts = {"def": 4, "mid": 3, "fwd": 3}

    return {
        "team_id":          team_id,
        "team_name":        team_name,
        "squad":            players,
        "formation_counts": tier_counts,
    }


@app.get("/api/actions")
async def list_actions():
    return [
        {"key": k, "action": a.name, "label": ACTION_LABELS[a]}
        for k, a in KEY_TO_ACTION.items()
    ]


@app.post("/api/step")
async def step(req: StepRequest):
    if _engine is None:
        raise HTTPException(503, "Engine not loaded — run training first.")

    action = _resolve_action(req.action.strip())

    if req.team_id_a not in _engine.fingerprints:
        raise HTTPException(400, f"team_id_a={req.team_id_a} not in fingerprints")
    if req.team_id_b not in _engine.fingerprints:
        raise HTTPException(400, f"team_id_b={req.team_id_b} not in fingerprints")

    ctx = MatchContext(
        score_diff = req.context.score_diff,
        minute     = req.context.minute,
        zone       = req.context.zone,
        phase      = req.context.phase,
        poss_team  = req.context.poss_team,
    )

    t0 = time.perf_counter()
    result = _engine.step(
        action         = action,
        context        = ctx,
        team_id_a      = req.team_id_a,
        team_id_b      = req.team_id_b,
        alpha          = req.alpha,
        formation_a    = req.formation_a,
        formation_b    = req.formation_b,
        continuity     = req.continuity,
        prev_positions = req.prev_positions,
        actor_slot     = req.actor_slot,
        gen_steps      = 15,
    )
    step_ms = round((time.perf_counter() - t0) * 1000, 1)

    payload = result.to_json_safe()
    payload["_timing_ms"] = step_ms   # surfaced in console for latency audit
    return payload


def _resolve_action(action_str: str) -> "Action":
    """Resolve action name or keyboard key to Action enum, or raise 400."""
    if action_str in KEY_TO_ACTION:
        return KEY_TO_ACTION[action_str]
    try:
        return Action[action_str.upper()]
    except KeyError:
        raise HTTPException(
            400,
            f"Unknown action '{action_str}'. "
            f"Valid: {[a.name for a in Action]} or keys {list(KEY_TO_ACTION)}"
        )


@app.post("/api/simulate_sequence")
async def simulate_sequence(req: SequenceRequest):
    if _engine is None:
        raise HTTPException(503, "Engine not loaded — run training first.")
    if not req.sequence:
        raise HTTPException(400, "sequence is empty")
    if len(req.sequence) > 15:
        raise HTTPException(400, "sequence too long (max 15 steps)")

    if req.team_id_a not in _engine.fingerprints:
        raise HTTPException(400, f"team_id_a={req.team_id_a} not in fingerprints")
    if req.team_id_b not in _engine.fingerprints:
        raise HTTPException(400, f"team_id_b={req.team_id_b} not in fingerprints")

    resolved = [(
        _resolve_action(s.action.strip()),
        s.alpha,
    ) for s in req.sequence]

    ctx = MatchContext(
        score_diff = req.context.score_diff,
        minute     = req.context.minute,
        zone       = req.context.zone,
        phase      = req.context.phase,
        poss_team  = req.context.poss_team,
    )

    if req.n_samples < 1 or req.n_samples > 20:
        raise HTTPException(400, "n_samples must be between 1 and 20")

    frames = _engine.simulate_sequence(
        sequence        = resolved,
        context         = ctx,
        team_id_a       = req.team_id_a,
        team_id_b       = req.team_id_b,
        minute_per_step = req.minute_per_step,
        adversarial     = req.adversarial,
        n_samples       = req.n_samples,
        noise_std       = req.noise_std,
        continuity      = max(0.0, min(req.continuity, 1.0)),
        formation_a     = req.formation_a,
        formation_b     = req.formation_b,
    )

    return {"frames": [f.to_json_safe() for f in frames]}


@app.post("/api/optimize")
async def optimize(req: OptimizeRequest):
    if _engine is None:
        raise HTTPException(503, "Engine not loaded")
    if req.team_id_a not in _engine.fingerprints:
        raise HTTPException(400, f"team_id_a={req.team_id_a} not in fingerprints")
    if req.team_id_b not in _engine.fingerprints:
        raise HTTPException(400, f"team_id_b={req.team_id_b} not in fingerprints")
    if not (1 <= req.max_depth <= 6):
        raise HTTPException(400, "max_depth must be 1–6")
    if not (1 <= req.beam_width <= 5):
        raise HTTPException(400, "beam_width must be 1–5")

    ctx = MatchContext(
        score_diff = req.context.score_diff,
        minute     = req.context.minute,
        zone       = req.context.zone,
        phase      = req.context.phase,
        poss_team  = req.context.poss_team,
    )
    results = _engine.optimize_sequence(
        context     = ctx,
        team_id_a   = req.team_id_a,
        team_id_b   = req.team_id_b,
        max_depth   = req.max_depth,
        beam_width  = req.beam_width,
        adversarial = req.adversarial,
    )
    return {"sequences": results}


@app.post("/api/opening_state")
async def opening_state(req: OpeningStateRequest):
    """Return a generated frame from team fingerprints with no action transform.

    Stage 2 mode: encode → generate → debias. No action embedding is applied,
    so the frame reflects only team identity + context, not a chosen action.
    The response shape is identical to /api/step so the frontend can treat it
    the same way.
    """
    if _engine is None:
        raise HTTPException(503, "Engine not loaded — run training first.")
    if req.team_id_a not in _engine.fingerprints:
        raise HTTPException(400, f"team_id_a={req.team_id_a} not in fingerprints")
    if req.team_id_b not in _engine.fingerprints:
        raise HTTPException(400, f"team_id_b={req.team_id_b} not in fingerprints")

    ctx = MatchContext(
        score_diff = req.context.score_diff,
        minute     = req.context.minute,
        zone       = req.context.zone,
        phase      = req.context.phase,
        poss_team  = req.context.poss_team,
    )

    fa = FormationCounts.from_dict(req.formation_a).to_engine_dict() if req.formation_a else None
    fb = FormationCounts.from_dict(req.formation_b).to_engine_dict() if req.formation_b else None

    with torch.no_grad():
        fp_a = _engine.fingerprints[req.team_id_a].to(_engine.device)
        fp_b = _engine.fingerprints[req.team_id_b].to(_engine.device)

        # Stage-2 mode: encode condition from fingerprints, generate, debias.
        # No action transform — frame reflects team identity + context only.
        c      = _engine._encode_condition(fp_a, fp_b, ctx)
        gen_xy = _engine._debias_positions(
            _engine.generator.generate(_engine.roles, c, _engine.mask, n_steps=30),
            ctx, formation_a=fa, formation_b=fb,
        )

        probs = _engine._run_sse_probs(fp_a, fp_b, ctx)

    positions_np = gen_xy.squeeze(0).cpu().numpy()
    roles_np     = _engine.roles.squeeze(0).cpu().numpy()
    mask_np      = _engine.mask.squeeze(0).cpu().numpy()

    return {
        "action":    "OPENING",
        "positions": positions_np.tolist(),
        "roles":     roles_np.tolist(),
        "mask":      mask_np.tolist(),
        "probs":     probs.to_dict(),
        "deltas":    None,
        "context": {
            "score_diff":       ctx.score_diff,
            "minute":           ctx.minute,
            "zone":             ctx.zone,
            "phase":            ctx.phase,
            "poss_team":        ctx.poss_team,
            "defense_response": "",
        },
    }


@app.post("/api/suggest")
async def suggest(req: SuggestRequest):
    if _engine is None:
        raise HTTPException(503, "Engine not loaded")
    if req.team_id_a not in _engine.fingerprints:
        raise HTTPException(400, f"team_id_a={req.team_id_a} not in fingerprints")
    if req.team_id_b not in _engine.fingerprints:
        raise HTTPException(400, f"team_id_b={req.team_id_b} not in fingerprints")

    ctx = MatchContext(
        score_diff = req.context.score_diff,
        minute     = req.context.minute,
        zone       = req.context.zone,
        phase      = req.context.phase,
        poss_team  = req.context.poss_team,
    )
    return {"suggestions": _engine.suggest_action(ctx, req.team_id_a, req.team_id_b)}


# ── Commentary ─────────────────────────────────────────────────────────────────

class CommentaryRequest(BaseModel):
    action:             str
    defense_response:   str
    team_name_a:        str
    team_name_b:        str
    p_shot:             float
    p_shot_delta:       float
    p_advance:          float
    p_advance_delta:    float
    minute:             float
    zone:               int
    score_diff:         float
    step_num:           int
    total_steps:        int
    # Player-level context (populated from canvas shirt-number logic)
    actor_num:          int | None  = None   # shirt number of player on the ball
    actor_name:         str | None  = None   # real player name if squad loaded
    actor_position:     str         = ""     # e.g. "right flank, attacking third"
    nearest_defenders:  list[int]   = []     # shirt numbers of closest 2 opponents
    defender_names:     list[str]   = []     # real names of nearest defenders
    striker_num:        int | None  = None   # most advanced teammate (likely receiver)
    striker_name:       str | None  = None   # real name of striker
    second_runner:      int | None  = None   # second-most advanced teammate


@app.post("/api/commentary")
async def commentary(req: CommentaryRequest):
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return {"commentary": "", "available": False}

    # Cache key includes player numbers so different spatial situations
    # generate different commentary even for the same abstract action
    raw = (f"{req.action}|{req.defense_response}|{req.team_name_a}|{req.team_name_b}"
           f"|{req.p_shot:.3f}|{req.p_shot_delta:.3f}|{req.zone}|{req.minute:.0f}"
           f"|{req.score_diff:.0f}|{req.actor_num}|{req.actor_position}"
           f"|{req.nearest_defenders}|{req.striker_num}")
    cache_key = hashlib.md5(raw.encode()).hexdigest()
    if cache_key in _commentary_cache:
        return {"commentary": _commentary_cache[cache_key], "available": True}

    score_str = (
        "level" if req.score_diff == 0
        else f"up {int(abs(req.score_diff))}" if req.score_diff > 0
        else f"down {int(abs(req.score_diff))}"
    )
    zone_names = {0: "own half", 1: "midfield", 2: "attacking third", 3: "penalty area"}
    zone_str   = zone_names.get(req.zone, f"zone {req.zone}")
    delta_str  = (
        f"shot probability {'rose' if req.p_shot_delta >= 0 else 'fell'} "
        f"{abs(req.p_shot_delta)*100:.0f}pp to {req.p_shot*100:.0f}%"
    )

    # Build player-specific lines, preferring real names over shirt numbers
    def _player_ref(name: str | None, num: int | None) -> str:
        if name:
            return f"{name} (#{num})" if num else name
        if num:
            return f"#{num}"
        return "the ball carrier"

    player_lines = []
    if req.actor_num or req.actor_name:
        actor_ref = _player_ref(req.actor_name, req.actor_num)
        loc = f" ({req.actor_position})" if req.actor_position else ""
        player_lines.append(f"{actor_ref}{loc} is on the ball.")
    if req.nearest_defenders or req.defender_names:
        if req.defender_names:
            pairs = [_player_ref(n, req.nearest_defenders[i] if i < len(req.nearest_defenders) else None)
                     for i, n in enumerate(req.defender_names)]
        else:
            pairs = [f"#{n}" for n in req.nearest_defenders]
        player_lines.append(f"Nearest defenders: {' and '.join(pairs)}.")
    if req.striker_num or req.striker_name:
        runner_ref = _player_ref(req.striker_name, req.striker_num)
        player_lines.append(f"Runner ahead: {runner_ref}.")
    player_ctx = " ".join(player_lines)

    has_names = bool(req.actor_name or req.defender_names or req.striker_name)
    name_instruction = (
        "Use the players' real names (given above) — do not replace them with generic phrases."
        if has_names else
        "Reference the specific shirt numbers — do not use generic phrases like 'the attacker'."
    )

    prompt = (
        f"You are a football analyst providing live match commentary. "
        f"Write exactly one sentence (max 35 words). {name_instruction}\n\n"
        f"Match: {req.team_name_a} vs {req.team_name_b}, "
        f"minute {req.minute:.0f}, {score_str}.\n"
        f"Action (step {req.step_num}/{req.total_steps}): "
        f"{req.team_name_a} plays {req.action} in the {zone_str}.\n"
        f"{player_ctx}\n"
        f"Defense responds with: {req.defense_response}. {delta_str}.\n\n"
        f"Commentary:"
    )

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model    = "claude-haiku-4-5-20251001",
            max_tokens = 80,
            messages = [{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text.strip()
        _commentary_cache[cache_key] = text
        return {"commentary": text, "available": True}
    except Exception as e:
        return {"commentary": "", "available": False, "error": str(e)}


# ── Sequence analysis ──────────────────────────────────────────────────────────

class SequenceFrameSummary(BaseModel):
    action:           str
    zone:             int   = 1
    minute:           float = 45.0
    score_diff:       float = 0.0
    p_shot:           float = 0.0
    p_shot_delta:     float | None = None
    p_advance:        float | None = None
    defense_response: str   = ""


class AnalyzeSequenceRequest(BaseModel):
    team_name_a: str
    team_name_b: str
    frames:      list[SequenceFrameSummary]
    score_diff:  float = 0.0
    minute:      float = 45.0


@app.post("/api/analyze_sequence")
async def analyze_sequence(req: AnalyzeSequenceRequest):
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return {"analysis": "", "available": False}
    if not req.frames:
        raise HTTPException(400, "No frames to analyze")

    zone_names = {0: "own half", 1: "midfield", 2: "attacking third", 3: "penalty area"}
    action_labels = {
        "ADVANCE": "advance", "THROUGH_BALL": "through ball", "DRIBBLE": "dribble",
        "SHOOT": "shoot", "CROSS": "cross", "SWITCH_LEFT": "switch left",
        "SWITCH_RIGHT": "switch right", "HOLD": "keep possession",
        "KEEPER_BALL": "play out from back", "PRESS": "high press", "LOW_BLOCK": "low block",
    }

    steps = []
    for i, f in enumerate(req.frames, 1):
        z = zone_names.get(f.zone, f"zone {f.zone}")
        a = action_labels.get(f.action, f.action.lower())
        delta_str = ""
        if f.p_shot_delta is not None:
            sign = "+" if f.p_shot_delta >= 0 else ""
            delta_str = f" ({sign}{f.p_shot_delta*100:.0f}pp)"
        steps.append(f"  {i}. {a} in {z} → shot prob {f.p_shot*100:.0f}%{delta_str}")

    score_str = (
        "level" if req.score_diff == 0
        else f"up {int(abs(req.score_diff))}" if req.score_diff > 0
        else f"down {int(abs(req.score_diff))}"
    )

    final_p    = req.frames[-1].p_shot
    peak_frame = max(req.frames, key=lambda f: f.p_shot)
    peak_step  = req.frames.index(peak_frame) + 1

    prompt = (
        f"You are a football analyst. Write a 3-sentence tactical briefing — no bullet points, "
        f"no headers, no markdown. Be specific: name the actions, describe the build-up pattern, "
        f"and say whether the sequence was effective.\n\n"
        f"Match: {req.team_name_a} vs {req.team_name_b}, "
        f"minute {req.minute:.0f}, score {score_str}.\n"
        f"{req.team_name_a} played this sequence:\n"
        + "\n".join(steps) +
        f"\n\nFinal shot probability: {final_p*100:.0f}%. "
        f"Peak threat at step {peak_step} ({peak_frame.p_shot*100:.0f}%).\n\n"
        f"Briefing:"
    )

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model      = "claude-haiku-4-5-20251001",
            max_tokens = 220,
            messages   = [{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text.strip()
        return {"analysis": text, "available": True}
    except Exception as e:
        return {"analysis": "", "available": False, "error": str(e)}
