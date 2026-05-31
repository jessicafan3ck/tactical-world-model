# Tactical World Model — Design Notes

## 1. Overview

An interactive football simulation built on a learned State Space Encoder (SSE) + Generator pair. Given a `MatchContext` (zone, phase, minute, score differential) and a tactical action chosen by the user, the engine advances the game state and returns new player positions, probability estimates, and action recommendations.

---

## 2. Model Architecture

- **SSE (State Space Encoder)**: encodes 22-player positions + roles + ball context into a latent state vector `z`
- **Generator**: decodes `z` conditioned on an encoded action into the next frame's player positions
- **Action Encoder (ConditionedMLP)**: maps a `MatchContext` + action label into a differentiable action embedding
- **Team Fingerprints**: per-team latent vectors loaded from `model/checkpoints/team_fingerprints.pt`, blended into the generator's conditioning

---

## 3. Training Data

Women's World Cup StatsBomb open data. 360° freeze-frame positions form the spatial observations; possession-level sequences provide action labels.

---

## 4. Validation Baselines (as of 2026-05-30)

| Check | Metric | Value |
|---|---|---|
| Check 1 — Action Encoder | ConditionedMLP held-out AUC | 0.579 |
| Check 2 — Forward AUC (honest) | s2 / s3 / shot | 0.49 / 0.52 / 0.633 |
| Check 2 — Null baseline (ADVANCE×4) | shot AUC | 0.673 |
| Check 3 — Generator realism | classifier AUC | 0.892 |
| Stage 2 decomposition | s2 AUC (generator step) | 0.506 |

Check 2's original numbers (0.699/0.745/0.753) were leakage artifacts — real StatsBomb sequences contain outcome-correlated action labels. Honest numbers use policy-generated action sequences with mean pooling.

**With formation tier clamps (re-measured 2026-05-31)**:

| Check | Baseline | With clamps | Δ |
|---|---|---|---|
| Check 3 — classifier AUC | 0.892 | **0.845** | −0.047 |
| Stage 2 — s2 AUC | 0.506 | **0.509** | +0.003 |

Check 3 improvement is claimable: clamped frames are more formation-realistic. Stage 2 moved negligibly, confirming the clamp improves legibility, not predictive fidelity.

**Check 4 — Comparative action effect-ordering (2026-05-31)**:

`scripts/15_validate_action_ordering.py` tests whether the simulator ranks the *relative* effect of actions correctly — the only empirical bar a counterfactual tool can uniquely clear.

Result: **47.1% global sign-agreement across 51 action pairs (3 context buckets) — essentially random, FAIL.**

Per-bucket Kendall's τ (reached_s3 as outcome):

| Bucket | τ | Sign-agreement |
|---|---|---|
| Middle ⅓, Open play | −0.476 | 29% |
| Middle ⅓, Set piece | +0.258 | 60% |
| Middle ⅓, Restart | −0.086 | 60% |

Root cause: model action deltas span only ±0.005 — the action transforms do not produce discriminative spatial signal in a single step, consistent with Stage 2 AUC ≈ random. SWITCH_LEFT has the highest empirical progression rate in open play but the model ranks it last.

**Conclusion**: the simulation's value is **communicative, not analytical**. LIM remains the empirically sound forecasting layer. The simulation is the amplification/explanation surface: it makes the 167K-parameter set transformer interactive, expresses team style visibly, and gives sponsors/analysts something they can drag actions through and watch danger move. That is a real product value even without analytical AUC claims. Stop claiming empirical soundness for the sim and lean entirely on LIM for that.

---

## 5. Generator Post-Processing Layers

The raw generator output is anatomically plausible but lacks structural realism. Three hand-crafted correction layers are applied in `ConditionalEngine._debias_positions()`. These are **plausibility/legibility corrections, not fidelity improvements**: they make frames look like real football, but do not change the model's learned policy or improve predictive AUC. The proper cure for each is a correspondingly conditioned generator; all three are post-pitch improvements.

### Layer 1 — Per-(zone, phase) x-debias

Source: `data/results/generator_debias.json`

The generator places teams too far forward (territory_zone +5pp overall; Zone 0 Phase 2 worst: Δx = −0.179). A lookup-table x-shift corrects the bias per zone/phase bucket. Improves territory_zone KS stat by 62% (0.223 → 0.084). AUC impact: < 0.003 (rank-invariant).

### Layer 2 — Goalkeeper pin

`_pin_goalkeepers()` clamps:
- Team A GK (lowest x in slots 0–10): x ∈ [0.02, 0.13], y ∈ [0.35, 0.65]
- Team B GK (highest x in slots 11–21): x ∈ [0.87, 0.98], y ∈ [0.35, 0.65]

The generator does not model the goalkeeper's structural role; without this, GKs routinely appear at midfield. This is a symptom mask, not a fix — a position-aware generator conditioned on role would handle it natively.

### Layer 3 — Formation tier clamps

`_clamp_outfield_positions()` assigns each outfield player a DEF/MID/FWD tier based on nominal formation counts (parsed from the StatsBomb lineup, not hardcoded to 4-3-3), then constrains generated x-positions to fixed bands:

| Tier | Team A band | Team B band (flipped) |
|---|---|---|
| DEF | [0.07, 0.42] | [0.58, 0.93] |
| MID | [0.30, 0.65] | [0.35, 0.70] |
| FWD | [0.54, 0.88] | [0.12, 0.46] |

Bands are **not zone-indexed** — the ball's position doesn't define where players structurally belong, and zone-indexing caused Team B's forwards to appear deep in Team A's half (fixed 2026-05-31). Midfields overlap in the center third (realistic); DEF and FWD don't cross between teams. Urgency (score × minute) shifts the whole block ±10pp.

**Urgency adjustment** (designer-set thresholds, not learned):
- Losing ≥0.5 goals after minute 72: all bands shift +10pp (push up)
- Winning ≥0.5 goals after minute 78: all bands shift −8pp (hold back)

These thresholds are heuristics in the same category as the band values themselves — reasonable, but not derived from the data.

The clamp uses actual StatsBomb lineup position strings (via `_POSITION_TIER` in `server/app.py`) so a 3-5-2 gets 3 defenders clamped, not 4. A 4-3-3 clamped as a 3-5-2 would look worse than no clamp.

---

## 6. Action Legality Mask

Certain actions are suppressed based on zone and phase via a per-context boolean mask applied before action ranking in `suggest_action()`. This is a third category of hand-crafted correction — masking obviously illegal recommendations (e.g., SHOOT from zone 0) that the SSE's learned probabilities do not naturally suppress.

---

## 7. API Surface

| Endpoint | Method | Description |
|---|---|---|
| `/api/teams` | GET | All team IDs + names |
| `/api/squad/{team_id}` | GET | Players with `positional_tier`, `formation_counts` |
| `/api/step` | POST | Single action step → frame + probs |
| `/api/simulate_sequence` | POST | Batch sequence → frames list |
| `/api/suggest` | POST | Ranked action suggestions for current context |
| `/api/analyze_sequence` | POST | LLM (Claude Haiku) 3-sentence tactical briefing |

---

## 8. Analytical Overlays

Three optional overlays are drawn on the pitch canvas, toggled via the legend:

- **Pressure** (20×13 heatmap): defender proximity — dark red = high pressure
- **Lanes** (passing channels from ball carrier): green / amber / red with arrowheads; arrow length ∝ lane openness
- **Space** (32×20 Voronoi): nearest-player ownership per cell; Team A blue, Team B red

---

## 9. Pitch Rendering and Player Identity Tracking

### Two-pass rendering

`drawPlayers()` makes two canvas passes: all non-ball-carriers first, the actor last. This guarantees the ball-carrier's label is never occluded.

### Identity tracking

The generator outputs anonymous position arrays (slot 0–10 = Team A, 11–21 = Team B). Naively re-sorting by x-depth each frame causes two visible bugs in playback:

1. **Name skitter**: the label "Larroquette" jumps from one dot to another between frames as the x-sort rank shuffles.
2. **Tier jitter**: a player near a rank boundary pops between DEF and MID bands on consecutive frames, making the formation clamp (§5 Layer 3) look broken.

The fix: assign identities once on the opening frame, then track them across the sequence.

**Opening frame** (`initIdentities`): sort generator slots by x-depth, sort the squad roster by positional tier + jersey number, zip them. Each slot gets a fixed `{slotIdx, jersey, name, tier}` record.

**Subsequent frames** (`propagateIdentities`): greedy nearest-neighbor match between each identity's previous position and all candidate slots in the next frame. Each slot is claimed at most once. The identity record is updated with the new `slotIdx`, but `jersey`, `name`, and `tier` are never changed.

**Snapshot/restore** (`_snapshotIdentities` / `_restoreIdentities`): after each frame is processed, its identity state is stored on `frame._identities`. `seek(n)` restores from the snapshot before drawing, so scrubbing backwards gives stable labels.

**Known limitation**: the opening-frame depth-sort will misassign identity if two players of the same tier happen to be out of position on the very first generated frame. This is rare and self-correcting — the nearest-neighbor propagation recovers within 1–2 steps. The GK assignment uses the depth heuristic (lowest/highest x), not the StatsBomb Goalkeeper position string; a sweeper-keeper stepped forward could fool it.

### State lifecycle

| Event | Identity action |
|---|---|
| `explainerRun` — first frame | `initIdentities(frame)` |
| `explainerRun` — subsequent | `propagateIdentities(prev, new)` |
| `runSim` — frame 0 | `initIdentities(frame)` |
| `runSim` — frames 1..n | `propagateIdentities(prev, new)` |
| All frames | `frame._identities = _snapshotIdentities()` |
| `seek(n)` | `_restoreIdentities(frames[n]._identities)` |
| `clearAll()` | `S.identitiesA = null; S.identitiesB = null` |
