"""
Eye-test: suggest_action() from representative opening states.

Checks that action recommendations are football-sensible across a range of
contexts: zones, score differentials, phases, and a few different team fingerprints.

Run time: ~2 min (SSE only, no generation).
Usage: python scripts/99_eyetest_suggest_action.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch
from model.conditional_engine import ConditionalEngine
from model.action_encoder import MatchContext

# ── Engine ─────────────────────────────────────────────────────────────────────

CKPT = ROOT / "model" / "checkpoints"
engine = ConditionalEngine(
    sse_path         = CKPT / "sse_best.pt",
    generator_path   = CKPT / "generator_best.pt",
    fingerprint_path = CKPT / "team_fingerprints.pt",
)

# A few teams with distinct footballing identities (Women's World Cup dataset)
TEAMS = {
    863:  "Spain W",
    851:  "Netherlands W",
    1214: "USA W",
    857:  "Germany W",
    865:  "England W",
}

# Only use teams that have fingerprints
TEAMS = {tid: name for tid, name in TEAMS.items() if tid in engine.fingerprints}
# Fallback: pick any 3 from what's available
if len(TEAMS) < 3:
    extra = [tid for tid in engine.fingerprints if tid not in TEAMS][:3 - len(TEAMS)]
    TEAMS.update({tid: f"Team {tid}" for tid in extra})

TEAM_IDS = list(TEAMS.keys())
tid_A, tid_B = TEAM_IDS[0], TEAM_IDS[1]

# ── Scenarios ──────────────────────────────────────────────────────────────────
# Each: (label, MatchContext, expected_top_actions (soft guidance), known_bad_actions)

SCENARIOS = [
    (
        "Own half, open play, 0-0, 45'",
        MatchContext(score_diff=0.0, minute=45.0, zone=0, phase=0, poss_team=0),
        ["ADVANCE", "HOLD", "PRESS"],                      # sensible
        ["SHOOT", "CROSS"],                                # absurd from own half
    ),
    (
        "Middle third, open play, 0-0, 45'",
        MatchContext(score_diff=0.0, minute=45.0, zone=1, phase=0, poss_team=0),
        ["ADVANCE", "THROUGH_BALL", "PRESS"],
        ["SHOOT"],
    ),
    (
        "Attacking third, open play, 0-0, 65'",
        MatchContext(score_diff=0.0, minute=65.0, zone=2, phase=0, poss_team=0),
        ["SHOOT", "CROSS", "THROUGH_BALL"],
        ["KEEPER_BALL", "LOW_BLOCK"],
    ),
    (
        "Just outside box, open play, 0-0, 70'",
        MatchContext(score_diff=0.0, minute=70.0, zone=3, phase=0, poss_team=0),
        ["SHOOT", "CROSS", "THROUGH_BALL"],
        ["KEEPER_BALL", "LOW_BLOCK", "HOLD"],
    ),
    (
        "Losing 0-1, own half, 75'  [should show urgency]",
        MatchContext(score_diff=-1.0, minute=75.0, zone=0, phase=0, poss_team=0),
        ["ADVANCE", "PRESS"],
        ["LOW_BLOCK", "KEEPER_BALL"],
    ),
    (
        "Losing 0-1, attacking third, 80'  [should push forward]",
        MatchContext(score_diff=-1.0, minute=80.0, zone=2, phase=0, poss_team=0),
        ["SHOOT", "CROSS", "THROUGH_BALL"],
        ["HOLD", "LOW_BLOCK"],
    ),
    (
        "Winning 1-0, own half, 85'  [should protect lead]",
        MatchContext(score_diff=1.0, minute=85.0, zone=0, phase=0, poss_team=0),
        ["HOLD", "LOW_BLOCK", "KEEPER_BALL"],
        ["SHOOT", "CROSS"],
    ),
    (
        "Counter attack, middle third, 0-0, 55'",
        MatchContext(score_diff=0.0, minute=55.0, zone=1, phase=1, poss_team=0),
        ["ADVANCE", "THROUGH_BALL"],
        ["LOW_BLOCK", "KEEPER_BALL"],
    ),
    (
        "Set piece, attacking third, 0-0, 60'",
        MatchContext(score_diff=0.0, minute=60.0, zone=2, phase=2, poss_team=0),
        ["SHOOT", "CROSS", "THROUGH_BALL"],
        ["LOW_BLOCK"],
    ),
    (
        "Losing 0-2, own half, 88'  [desperation mode]",
        MatchContext(score_diff=-2.0, minute=88.0, zone=0, phase=0, poss_team=0),
        ["ADVANCE", "PRESS"],
        ["LOW_BLOCK", "KEEPER_BALL"],
    ),
]

# ── Run ────────────────────────────────────────────────────────────────────────

def top_n(suggestions, n=5):
    return suggestions[:n]

def fmt_row(r):
    sign = "+" if r["p_shot_delta"] >= 0 else ""
    return (f"  {r['label']:<22} p_shot={r['p_shot']:.3f}  "
            f"Δ={sign}{r['p_shot_delta']:.3f}  "
            f"p_adv={r['p_advance']:.3f}")

PASS = "\033[92m✓\033[0m"
WARN = "\033[93m!\033[0m"
FAIL = "\033[91m✗\033[0m"

issues = []

print("\n" + "═" * 72)
print("  SUGGEST_ACTION EYE-TEST")
print(f"  Team A: {TEAMS[tid_A]}  vs  Team B: {TEAMS[tid_B]}")
print("═" * 72)

for label, ctx, expected, bad in SCENARIOS:
    suggestions = engine.suggest_action(ctx, tid_A, tid_B)
    top5 = top_n(suggestions, 5)
    top1_name = top5[0]["action"]
    top5_names = [r["action"] for r in top5]

    print(f"\n{label}")
    print(f"  Context: zone={ctx.zone}  phase={ctx.phase}  score={ctx.score_diff:+.0f}  min={ctx.minute:.0f}")
    for r in top5:
        print(fmt_row(r))

    # Sanity checks
    for bad_action in bad:
        if bad_action == top1_name:
            msg = f"  {FAIL} TOP-1 is {bad_action} — absurd for this context: {label}"
            print(msg)
            issues.append(msg)
        elif bad_action in top5_names[:3]:
            msg = f"  {WARN} {bad_action} in top-3 — borderline: {label}"
            print(msg)
            issues.append(msg)

# ── Context sensitivity check ──────────────────────────────────────────────────

print("\n" + "─" * 72)
print("  CONTEXT SENSITIVITY CHECK (same zone, vary score_diff)")
print("─" * 72)

zone1_contexts = {
    "draw  (0-0), 60'": MatchContext(0.0,  60.0, 1, 0, 0),
    "lose  (0-1), 75'": MatchContext(-1.0, 75.0, 1, 0, 0),
    "win   (+1),  85'": MatchContext(1.0,  85.0, 1, 0, 0),
    "lose  (0-2), 88'": MatchContext(-2.0, 88.0, 1, 0, 0),
}

top1_by_ctx = {}
for ctx_label, ctx in zone1_contexts.items():
    sugg = engine.suggest_action(ctx, tid_A, tid_B)
    top1_by_ctx[ctx_label] = sugg[0]["action"]
    print(f"  {ctx_label}  →  top-1: {sugg[0]['label']:<22}  Δp_shot={sugg[0]['p_shot_delta']:+.3f}")

all_same = len(set(top1_by_ctx.values())) == 1
if all_same:
    msg = f"  {FAIL} Top-1 action identical across all score differentials — no context sensitivity"
    print(msg); issues.append(msg)
else:
    print(f"  {PASS} Recommendations vary with score differential")

# ── Team fingerprint differentiation ──────────────────────────────────────────

print("\n" + "─" * 72)
print("  FINGERPRINT DIFFERENTIATION (attacking third, 0-0, 65')")
print("─" * 72)

ctx_att = MatchContext(0.0, 65.0, 2, 0, 0)
team_top3 = {}
for tid, name in list(TEAMS.items())[:4]:
    opp = [t for t in TEAM_IDS if t != tid][0]
    sugg = engine.suggest_action(ctx_att, tid, opp)
    top3 = [r["label"] for r in sugg[:3]]
    team_top3[name] = top3
    print(f"  {name:<28}  top-3: {' > '.join(top3)}")

all_identical = len({tuple(v) for v in team_top3.values()}) == 1
if all_identical:
    msg = f"  {WARN} All teams produce identical top-3 — fingerprints may have no effect here"
    print(msg); issues.append(msg)
else:
    print(f"  {PASS} At least some differentiation across team fingerprints")

# ── Summary ────────────────────────────────────────────────────────────────────

print("\n" + "═" * 72)
if not issues:
    print(f"  {PASS} PASS — no obviously absurd recommendations found")
else:
    print(f"  Found {len(issues)} issue(s):")
    for i in issues:
        print(f"    {i.strip()}")
print("═" * 72 + "\n")
