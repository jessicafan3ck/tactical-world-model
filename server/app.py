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

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

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


class StepRequest(BaseModel):
    action:    str        # Action name OR keyboard key (e.g. "ADVANCE" or "w")
    team_id_a: int
    team_id_b: int
    context:   ContextIn
    alpha:     float = 1.0


class SequenceStep(BaseModel):
    action: str
    alpha:  float = 1.0


class SequenceRequest(BaseModel):
    team_id_a:       int
    team_id_b:       int
    context:         ContextIn
    sequence:        list[SequenceStep]
    minute_per_step: float = 0.5


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

    result = _engine.step(
        action    = action,
        context   = ctx,
        team_id_a = req.team_id_a,
        team_id_b = req.team_id_b,
        alpha     = req.alpha,
    )

    return result.to_json_safe()


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

    frames = _engine.simulate_sequence(
        sequence        = resolved,
        context         = ctx,
        team_id_a       = req.team_id_a,
        team_id_b       = req.team_id_b,
        minute_per_step = req.minute_per_step,
    )

    return {"frames": [f.to_json_safe() for f in frames]}
