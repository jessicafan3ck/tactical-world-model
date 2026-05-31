"""
model/conditional_engine.py
============================
Part 2 of the interactive simulation layer.

Takes a user action + current match state and returns:
  1. A generated freeze frame showing the resulting spatial configuration
  2. Updated possession outcome probabilities P(advance|action), P(shot|action)
  3. A delta showing how probabilities changed vs the baseline (no action)

This is the core conditional inference loop that the server calls on every
user input. It is completely stateless — the caller (server) tracks the
match state across turns.

Flow per user action:
    z_team_modified = action_encoder(z_team, action, context)
    c = generator.encode_condition(z_A_mod, z_B, ...)
    freeze_frame = generator.generate(roles, c)      ← spatial output
    outcome_probs = sse.predict(freeze_frame, context) ← probability output
    delta_probs = outcome_probs - baseline_probs

Testable standalone:
    python -m model.conditional_engine
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

# All imports are lazy inside functions so this module imports cleanly
# even if torch models aren't loaded yet (useful for the server cold-start).


# ── Output types ──────────────────────────────────────────────────────────────

@dataclass
class FreezeFrameResult:
    """Spatial output: player positions after the action is applied."""
    positions: np.ndarray        # (N, 2) normalised [0,1] x,y per player
    roles:     np.ndarray        # (N, 2) [is_teammate, is_actor]
    mask:      np.ndarray        # (N,) True where padded


@dataclass
class OutcomeProbabilities:
    """Model's estimated possession outcome probabilities."""
    p_advance:    float   # P(ball reaches next zone)
    p_shot:       float   # P(possession ends in shot attempt)
    p_final_third: float  # P(ball enters final third)

    def to_dict(self) -> dict:
        return {
            "p_advance":     round(self.p_advance,     3),
            "p_shot":        round(self.p_shot,        3),
            "p_final_third": round(self.p_final_third, 3),
        }


@dataclass
class ActionResult:
    """Everything the server sends back to the frontend for one user action."""
    action_name:    str
    freeze_frame:   FreezeFrameResult
    probs:          OutcomeProbabilities
    prob_deltas:    OutcomeProbabilities      # change vs baseline
    context_after:  dict                     # updated match state
    prob_stds:      OutcomeProbabilities | None = field(default=None)  # set when n_samples > 1

    def to_json_safe(self) -> dict:
        result = {
            "action":     self.action_name,
            "positions":  self.freeze_frame.positions.tolist(),
            "roles":      self.freeze_frame.roles.tolist(),
            "mask":       self.freeze_frame.mask.tolist(),
            "probs":      self.probs.to_dict(),
            "deltas":     self.prob_deltas.to_dict(),
            "context":    self.context_after,
        }
        if self.prob_stds is not None:
            result["stds"] = self.prob_stds.to_dict()
        return result


# ── Engine ─────────────────────────────────────────────────────────────────────

class ConditionalEngine:
    """
    Stateless inference engine. The server instantiates one of these at
    startup and calls step() on each user action.

    All heavy models are loaded once at __init__ time and kept in memory.
    step() is ~O(generator forward pass) ≈ 30 ms on MPS.
    """

    # Default lookup-table counter-tactics (used when adversarial=False)
    _RESPONSE_MAP: dict  # populated in __init__ after Action is importable

    def __init__(self,
                 sse_path:       Path,
                 generator_path: Path,
                 fingerprint_path: Path,
                 device:         torch.device | None = None):

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from model.sse           import build_predictor
        from model.flow_matching import build_generator
        from model.action_encoder import build_action_encoder, init_from_pca

        self.device = device or (
            torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cpu")
        )

        # Load SSE
        self.sse = build_predictor(z_dim=256).to(self.device)
        self.sse.load_state_dict(
            torch.load(sse_path, map_location=self.device)
        )
        self.sse.eval()

        # Load generator
        self.generator = build_generator(fingerprint_dim=256).to(self.device)
        self.generator.load_state_dict(
            torch.load(generator_path, map_location=self.device)
        )
        self.generator.eval()

        # Load fingerprints
        fp_data = torch.load(fingerprint_path, map_location="cpu")
        self.fingerprints: dict[int, torch.Tensor] = {
            k: v.float() for k, v in fp_data["team_fingerprints"].items()
        }
        self.mean_fp: torch.Tensor = fp_data.get(
            "mean_fingerprint",
            torch.stack(list(self.fingerprints.values())).mean(0)
        )

        # Build PCA-seeded action encoder (legacy fallback)
        self.action_encoder = build_action_encoder(z_dim=256).to(self.device)
        pca_path = fingerprint_path.parent.parent / "results" / "tactical_pca.csv"
        init_from_pca(self.action_encoder, self.fingerprints, pca_path)
        self.action_encoder.eval()

        # Try to load a learned encoder — MLP preferred over affine
        _ckpt_dir = fingerprint_path.parent
        self.learned_encoder = None
        for _enc_name in ("action_encoder_mlp.pt", "action_encoder_affine.pt"):
            _path = _ckpt_dir / _enc_name
            if _path.exists():
                from model.learned_action_encoder import ConditionedMLP, PerActionAffine
                _cls  = ConditionedMLP if "mlp" in _enc_name else PerActionAffine
                _enc  = _cls().to(self.device)
                _enc.load_state_dict(torch.load(_path, map_location=self.device))
                _enc.eval()
                self.learned_encoder = _enc
                print(f"  Learned action encoder loaded: {_enc_name}")
                break

        # Fixed role template (11 teammates + 11 opponents, player 0 = actor)
        N = self.generator.n_players
        roles = torch.zeros(1, N, 2)
        roles[0, :11, 0] = 1.0   # teammates
        roles[0, 0,   1] = 1.0   # actor
        self.roles = roles.to(self.device)
        self.mask  = torch.zeros(1, N, dtype=torch.bool, device=self.device)

        # Load per-zone/phase x-position debias corrections if available
        import json as _json
        _debias_path = Path(__file__).parent.parent / "data" / "results" / "generator_debias.json"
        self._debias: dict[tuple[int, int], float] = {}
        if _debias_path.exists():
            with open(_debias_path) as _f:
                _db = _json.load(_f)
            for _key, _val in _db.get("corrections", {}).items():
                _z, _p = (int(x) for x in _key.split("_"))
                self._debias[(_z, _p)] = float(_val["dx"])
            print(f"  Generator debias loaded: {len(self._debias)} (zone, phase) corrections")

        from model.action_encoder import Action
        self._RESPONSE_MAP = {
            Action.ADVANCE:      Action.LOW_BLOCK,
            Action.THROUGH_BALL: Action.LOW_BLOCK,
            Action.SHOOT:        Action.LOW_BLOCK,
            Action.CROSS:        Action.LOW_BLOCK,
            Action.DRIBBLE:      Action.LOW_BLOCK,
            Action.HOLD:         Action.ADVANCE,
            Action.LOW_BLOCK:    Action.ADVANCE,
            Action.PRESS:        Action.HOLD,
            Action.KEEPER_BALL:  Action.PRESS,
            Action.SWITCH_LEFT:  Action.SWITCH_RIGHT,
            Action.SWITCH_RIGHT: Action.SWITCH_LEFT,
        }

        print(f"ConditionalEngine ready on {self.device}")
        print(f"  {len(self.fingerprints)} team fingerprints loaded")

    # ── Core inference step ────────────────────────────────────────────────────

    @torch.no_grad()
    def step(self,
             action:      "Action",
             context:     "MatchContext",
             team_id_a:   int,
             team_id_b:   int,
             alpha:       float       = 1.0,
             gen_steps:   int         = 30,
             formation_a: dict | None = None,
             formation_b: dict | None = None) -> "ActionResult":
        """
        Execute one conditional inference step.

        Args:
            action     : user's chosen Action
            context    : current MatchContext
            team_id_a  : attacking team id (the one acting)
            team_id_b  : defending team id
            alpha      : action intensity [0, 1]
            gen_steps  : ODE integration steps (30 = fast, 50 = quality)

        Returns:
            ActionResult with freeze frame + probabilities + deltas
        """
        from model.action_encoder import ACTION_LABELS

        z_A_base = self.fingerprints.get(team_id_a, self.mean_fp).to(self.device)
        z_B      = self.fingerprints.get(team_id_b, self.mean_fp).to(self.device)

        # ── Baseline (no action) ───────────────────────────────────────────────
        baseline_probs = self._run_sse_probs(z_A_base, z_B, context)

        # ── Apply action → modified fingerprint ───────────────────────────────
        z_A_mod = self._apply_action(z_A_base, action, context, alpha).to(self.device)

        # ── Generate freeze frame under modified fingerprint ───────────────────
        c = self._encode_condition(z_A_mod, z_B, context)
        gen_xy = self._debias_positions(
            self.generator.generate(self.roles, c, self.mask, n_steps=gen_steps),
            context,
            formation_a=formation_a,
            formation_b=formation_b,
        )  # (1, N, 2)

        # ── Evaluate SSE on generated frame ───────────────────────────────────
        modified_probs = self._run_sse_probs(z_A_mod, z_B, context)

        # ── Package outputs ───────────────────────────────────────────────────
        N = self.generator.n_players
        positions_np = gen_xy.squeeze(0).cpu().numpy()          # (N, 2)
        roles_np     = self.roles.squeeze(0).cpu().numpy()      # (N, 2)
        mask_np      = self.mask.squeeze(0).cpu().numpy()       # (N,)

        delta_probs = OutcomeProbabilities(
            p_advance     = modified_probs.p_advance     - baseline_probs.p_advance,
            p_shot        = modified_probs.p_shot        - baseline_probs.p_shot,
            p_final_third = modified_probs.p_final_third - baseline_probs.p_final_third,
        )

        context_after = {
            "score_diff":  context.score_diff,
            "minute":      round(context.minute, 1),
            "zone":        min(context.zone + int(action.value == 0), 3),
            "phase":       context.phase,
            "poss_team":   context.poss_team,
        }

        return ActionResult(
            action_name  = ACTION_LABELS[action],
            freeze_frame = FreezeFrameResult(positions_np, roles_np, mask_np),
            probs        = modified_probs,
            prob_deltas  = delta_probs,
            context_after = context_after,
        )

    # ── Sequence simulation ────────────────────────────────────────────────────

    @torch.no_grad()
    def simulate_sequence(
        self,
        sequence:        list,
        context:         "MatchContext",
        team_id_a:       int,
        team_id_b:       int,
        gen_steps:       int         = 30,
        minute_per_step: float       = 0.5,
        adversarial:     bool        = False,
        n_samples:       int         = 1,
        noise_std:       float       = 0.05,
        continuity:      float       = 0.0,
        formation_a:     dict | None = None,
        formation_b:     dict | None = None,
    ) -> list:
        """
        Simulate a user-designed tactical sequence with opposition response.

        Args:
            sequence        : list of (Action, alpha) tuples, max 15 steps
            context         : starting MatchContext
            minute_per_step : simulated time per action (default 0.5 min)
            adversarial     : if True, Team B picks the action that minimises
                              your P(shot) rather than using the lookup table
            n_samples       : >1 enables stochastic mode — adds Gaussian noise
                              to z_A before each step and averages outputs
            noise_std       : noise magnitude as a fraction of ||z_A|| (0.05 = 5%)
        """
        from model.action_encoder import ACTION_LABELS, MatchContext

        z_A = self.fingerprints.get(team_id_a, self.mean_fp).clone().to(self.device)
        z_B = self.fingerprints.get(team_id_b, self.mean_fp).clone().to(self.device)
        ctx = context

        # Bound z-space drift by re-normalising to the team's original ||z||
        norm_A = z_A.norm().clamp(min=1e-8)
        norm_B = z_B.norm().clamp(min=1e-8)

        baseline = self._run_sse_probs(z_A, z_B, ctx)
        results  = []
        prev_xy: torch.Tensor | None = None   # (1, N, 2) on device, updated each step

        for action, alpha in sequence:
            z_A_mod = self._apply_action(z_A, action, ctx, alpha).to(self.device)
            z_A_mod = z_A_mod * (norm_A / z_A_mod.norm().clamp(min=1e-8))

            def_action = self._choose_defense(action, z_A_mod, z_B, norm_B, ctx, adversarial)
            z_B_mod    = self._apply_action(z_B, def_action, ctx, 0.5).to(self.device)
            z_B_mod    = z_B_mod * (norm_B / z_B_mod.norm().clamp(min=1e-8))

            new_zone   = min(ctx.zone + int(action.value == 0), 3)
            new_minute = min(ctx.minute + minute_per_step, 90.0)

            # Temporal prior: previous frame's positions (None on first step)
            x_prior     = prev_xy if (continuity > 0 and prev_xy is not None) else None
            max_delta   = continuity if continuity > 0 else None

            if n_samples > 1:
                # Stochastic mode: inject noise into z_A_mod across samples,
                # average spatial outputs and track probability std.
                # x_prior is shared across samples so continuity is anchored
                # to the previous deterministic mean rather than a noisy sample.
                all_xy, all_p = [], []
                for _ in range(n_samples):
                    noise   = torch.randn_like(z_A_mod) * noise_std * norm_A
                    z_noisy = (z_A_mod + noise) * (norm_A / (z_A_mod + noise).norm().clamp(min=1e-8))
                    c_n     = self._encode_condition(z_noisy, z_B_mod, ctx)
                    xy_n    = self._debias_positions(
                        self.generator.generate(
                            self.roles, c_n, self.mask, n_steps=gen_steps,
                            x_prior=x_prior, max_delta_per_step=max_delta,
                        ), ctx, formation_a=formation_a, formation_b=formation_b)
                    p_n     = self._run_sse_probs(z_noisy, z_B_mod, ctx)
                    all_xy.append(xy_n)
                    all_p.append([p_n.p_advance, p_n.p_final_third, p_n.p_shot])

                gen_xy  = torch.stack(all_xy).mean(0)
                arr     = np.array(all_p)   # (n_samples, 3)
                probs   = OutcomeProbabilities(
                    p_advance     = float(arr[:, 0].mean()),
                    p_final_third = float(arr[:, 1].mean()),
                    p_shot        = float(arr[:, 2].mean()),
                )
                stds = OutcomeProbabilities(
                    p_advance     = float(arr[:, 0].std()),
                    p_final_third = float(arr[:, 1].std()),
                    p_shot        = float(arr[:, 2].std()),
                )
            else:
                c      = self._encode_condition(z_A_mod, z_B_mod, ctx)
                gen_xy = self._debias_positions(
                    self.generator.generate(
                        self.roles, c, self.mask, n_steps=gen_steps,
                        x_prior=x_prior, max_delta_per_step=max_delta,
                    ), ctx, formation_a=formation_a, formation_b=formation_b)
                probs  = self._run_sse_probs(z_A_mod, z_B_mod, ctx)
                stds   = None

            prev_xy = gen_xy   # carry forward for next step's prior

            deltas = OutcomeProbabilities(
                p_advance     = probs.p_advance     - baseline.p_advance,
                p_shot        = probs.p_shot        - baseline.p_shot,
                p_final_third = probs.p_final_third - baseline.p_final_third,
            )

            results.append(ActionResult(
                action_name   = ACTION_LABELS[action],
                freeze_frame  = FreezeFrameResult(
                    gen_xy.squeeze(0).cpu().numpy(),
                    self.roles.squeeze(0).cpu().numpy(),
                    self.mask.squeeze(0).cpu().numpy(),
                ),
                probs         = probs,
                prob_deltas   = deltas,
                prob_stds     = stds,
                context_after = {
                    "score_diff":       ctx.score_diff,
                    "minute":           round(new_minute, 1),
                    "zone":             new_zone,
                    "phase":            ctx.phase,
                    "poss_team":        ctx.poss_team,
                    "defense_response": ACTION_LABELS[def_action],
                },
            ))

            z_A = z_A_mod
            z_B = z_B_mod
            ctx = MatchContext(
                score_diff = ctx.score_diff,
                minute     = new_minute,
                zone       = new_zone,
                phase      = ctx.phase,
                poss_team  = ctx.poss_team,
            )

        return results

    # ── Sequence optimizer ─────────────────────────────────────────────────────

    @torch.no_grad()
    def optimize_sequence(
        self,
        context:     "MatchContext",
        team_id_a:   int,
        team_id_b:   int,
        max_depth:   int  = 4,
        beam_width:  int  = 3,
        adversarial: bool = False,
    ) -> list[dict]:
        """
        Beam search over action sequences to find the highest-P(shot) play.

        At each depth level, every surviving beam node is expanded with all 11
        actions.  The top `beam_width` candidates (by P(shot)) survive to the
        next level.  Returns the final beam ranked by P(shot).

        Complexity: O(beam_width × |Actions| × max_depth) SSE evaluations
                    = 3 × 11 × 4 = 132 at default settings.
        """
        from model.action_encoder import ACTION_LABELS, Action, MatchContext

        z_A0 = self.fingerprints.get(team_id_a, self.mean_fp).clone().to(self.device)
        z_B0 = self.fingerprints.get(team_id_b, self.mean_fp).clone().to(self.device)
        norm_A = z_A0.norm().clamp(min=1e-8)
        norm_B = z_B0.norm().clamp(min=1e-8)

        # Beam entry: (p_shot, sequence_list, z_A, z_B, ctx)
        beam = [(
            self._run_sse_probs(z_A0, z_B0, context).p_shot,
            [],
            z_A0.clone(),
            z_B0.clone(),
            context,
        )]

        for _ in range(max_depth):
            candidates = []
            for _, seq, z_A, z_B, ctx in beam:
                for action in Action:
                    z_A_mod = self._apply_action(z_A, action, ctx, 1.0).to(self.device)
                    z_A_mod = z_A_mod * (norm_A / z_A_mod.norm().clamp(min=1e-8))

                    def_action = self._choose_defense(action, z_A_mod, z_B, norm_B, ctx, adversarial)
                    z_B_mod    = self._apply_action(z_B, def_action, ctx, 0.5).to(self.device)
                    z_B_mod    = z_B_mod * (norm_B / z_B_mod.norm().clamp(min=1e-8))

                    p        = self._run_sse_probs(z_A_mod, z_B_mod, ctx)
                    new_zone = min(ctx.zone + int(action.value == 0), 3)
                    new_ctx  = MatchContext(
                        score_diff = ctx.score_diff,
                        minute     = min(ctx.minute + 0.5, 90.0),
                        zone       = new_zone,
                        phase      = ctx.phase,
                        poss_team  = ctx.poss_team,
                    )
                    candidates.append((
                        p.p_shot,
                        seq + [{"action": action.name, "label": ACTION_LABELS[action], "alpha": 1.0,
                                "defense": ACTION_LABELS[def_action], "p_shot": round(float(p.p_shot), 3)}],
                        z_A_mod,
                        z_B_mod,
                        new_ctx,
                    ))

            candidates.sort(key=lambda x: x[0], reverse=True)
            beam = candidates[:beam_width]

        return [
            {"sequence": seq, "p_shot_final": round(float(p), 3)}
            for p, seq, *_ in beam
        ]

    # ── Action suggestion ──────────────────────────────────────────────────────

    @torch.no_grad()
    def get_legal_actions(self, context: "MatchContext") -> frozenset:
        """
        Return the set of Action members that are legal for the possessing team.

        The encoder learned z-transformations from real data without modelling
        action preconditions, so some high-Δz actions are illegal in certain
        contexts (e.g. PRESS when you have the ball; KEEPER_BALL from the
        attacking third).  This mask enforces the obvious preconditions.
        """
        from model.action_encoder import Action

        # Require the opponent to have the ball — illegal for the possessing team
        illegal = {Action.PRESS, Action.LOW_BLOCK}

        # Goalkeeper distribution: own half, controlled play, not losing
        if context.zone >= 2 or context.phase == 1 or context.score_diff < 0:
            illegal.add(Action.KEEPER_BALL)

        # Shot attempts: too far from goal in own half
        if context.zone == 0:
            illegal.add(Action.SHOOT)

        return frozenset(a for a in Action if a not in illegal)

    def suggest_action(self,
                       context:    "MatchContext",
                       team_id_a:  int,
                       team_id_b:  int) -> list[dict]:
        """
        Evaluate legal actions from the current state and return them ranked
        by ΔP(shot) — highest first.  Used by the frontend "What next?" panel
        and the pre-match sequence optimiser.

        Only contextually legal actions are included (see get_legal_actions).
        The result reflects what the encoder *expects* from each action, not a
        prescriptive recommendation — the encoder has no legality model, so
        illegal actions are filtered here, not in the learned operator.
        """
        from model.action_encoder import ACTION_LABELS, Action

        legal = self.get_legal_actions(context)

        z_A = self.fingerprints.get(team_id_a, self.mean_fp).to(self.device)
        z_B = self.fingerprints.get(team_id_b, self.mean_fp).to(self.device)
        norm_A = z_A.norm().clamp(min=1e-8)

        baseline = self._run_sse_probs(z_A, z_B, context)

        rows = []
        for action in Action:
            if action not in legal:
                continue
            z_mod = self._apply_action(z_A, action, context, 1.0).to(self.device)
            z_mod = z_mod * (norm_A / z_mod.norm().clamp(min=1e-8))
            p     = self._run_sse_probs(z_mod, z_B, context)
            rows.append({
                "action":          action.name,
                "label":           ACTION_LABELS[action],
                "p_shot":          round(float(p.p_shot),        3),
                "p_shot_delta":    round(float(p.p_shot        - baseline.p_shot),        3),
                "p_advance":       round(float(p.p_advance),     3),
                "p_advance_delta": round(float(p.p_advance      - baseline.p_advance),     3),
                "p_final_third":   round(float(p.p_final_third), 3),
                "p_final_third_delta": round(float(p.p_final_third - baseline.p_final_third), 3),
            })

        rows.sort(key=lambda r: r["p_shot_delta"], reverse=True)
        return rows

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _debias_positions(self, gen_xy: torch.Tensor,
                          context:     "MatchContext",
                          formation_a: dict | None = None,
                          formation_b: dict | None = None) -> torch.Tensor:
        """
        Apply per-(zone, phase) x-position correction to teammate positions,
        then pin goalkeepers and clamp outfield players to formation bands.

        gen_xy      : (1, N, 2) generated positions (normalised [0,1])
        formation_a : {def, mid, fwd} tier counts for Team A (None → 4-3-3)
        formation_b : {def, mid, fwd} tier counts for Team B (None → 4-3-3)
        """
        if not self._debias:
            return self._clamp_outfield_positions(
                self._pin_goalkeepers(gen_xy), context, formation_a, formation_b)
        phase_key = 2 if context.phase >= 2 else context.phase
        dx = self._debias.get((min(context.zone, 3), phase_key), 0.0)
        if abs(dx) < 1e-6:
            return self._clamp_outfield_positions(
                self._pin_goalkeepers(gen_xy), context, formation_a, formation_b)
        teammate = (self.roles[:, :, 0] > 0.5).unsqueeze(-1)
        x_shift  = torch.zeros_like(gen_xy)
        x_shift[:, :, 0] = dx
        corrected = gen_xy + teammate.float() * x_shift
        return self._clamp_outfield_positions(
            self._pin_goalkeepers(corrected.clamp(0.0, 1.0)),
            context, formation_a, formation_b)

    def _pin_goalkeepers(self, xy: torch.Tensor) -> torch.Tensor:
        """
        Clamp each team's goalkeeper to a realistic goal-side band.

        Convention: Team A (indices 0-10) attacks left → right.
                    Their GK (lowest x among them) guards the left end (x ≈ 0.05).
                    Team B (indices 11-21) attacks right → left.
                    Their GK (highest x among them) guards the right end (x ≈ 0.95).

        Both GKs are clamped vertically to the central corridor (y ∈ [0.35, 0.65])
        so they stay on or near the goal line rather than drifting to a flank.
        """
        N = xy.shape[1]
        if N < 22:
            return xy
        xy = xy.clone()

        # Team A GK: player in [0, 10] with smallest x
        gk_a = int(xy[0, :11, 0].argmin().item())
        xy[0, gk_a, 0] = xy[0, gk_a, 0].clamp(0.02, 0.13)
        xy[0, gk_a, 1] = xy[0, gk_a, 1].clamp(0.35, 0.65)

        # Team B GK: player in [11, 21] with largest x
        gk_b = 11 + int(xy[0, 11:22, 0].argmax().item())
        xy[0, gk_b, 0] = xy[0, gk_b, 0].clamp(0.87, 0.98)
        xy[0, gk_b, 1] = xy[0, gk_b, 1].clamp(0.35, 0.65)

        return xy

    # Default formation when no lineup is provided
    _DEFAULT_FORMATION: dict = {"def": 4, "mid": 3, "fwd": 3}

    # Per-tier x-ranges for Team A (left → right), indexed by zone 0–3.
    _TIER_X: dict = {
        #         zone 0          zone 1          zone 2          zone 3
        "def": [(0.09, 0.42), (0.11, 0.50), (0.16, 0.58), (0.20, 0.63)],
        "mid": [(0.20, 0.55), (0.26, 0.63), (0.32, 0.72), (0.37, 0.77)],
        "fwd": [(0.30, 0.68), (0.38, 0.76), (0.48, 0.87), (0.53, 0.91)],
    }

    def _clamp_outfield_positions(self, xy: torch.Tensor,
                                   context:     "MatchContext",
                                   formation_a: dict | None = None,
                                   formation_b: dict | None = None) -> torch.Tensor:
        """
        Clamp outfield players to formation-appropriate x-bands.

        After sorting each team by x-depth, players are assigned to tiers using
        the actual formation counts from the lineup (or 4-3-3 as default):
          rank 0              → GK (already pinned by _pin_goalkeepers)
          ranks 1 .. 1+n_def  → DEF
          ranks ..  ..+n_mid  → MID
          ranks ..  ..+n_fwd  → FWD

        Bands are zone-sensitive and shift with urgency (score × late minute).
        Team B ranges are the mirror image of Team A (1 − x).
        """
        if xy.shape[1] < 11:
            return xy

        zone = min(context.zone, 3)

        urgency = 0.0
        if context.score_diff < -0.5 and context.minute > 72:
            urgency = min(0.06 * (-context.score_diff), 0.10)
        elif context.score_diff > 0.5 and context.minute > 78:
            urgency = -min(0.05 * context.score_diff, 0.08)

        def _tier_ranks(fm: dict | None) -> list[tuple[str, int, int]]:
            f = fm or self._DEFAULT_FORMATION
            n_d = max(1, int(f.get("def", 4)))
            n_m = max(1, int(f.get("mid", 3)))
            n_f = max(1, int(f.get("fwd", 3)))
            r0 = 1
            return [
                ("def", r0,           r0 + n_d),
                ("mid", r0 + n_d,     r0 + n_d + n_m),
                ("fwd", r0 + n_d + n_m, r0 + n_d + n_m + n_f),
            ]

        xy = xy.clone()

        def _apply(sorted_idx: list, tier_ranks: list, flip: bool) -> None:
            for tier, r0, r1 in tier_ranks:
                lo, hi = self._TIER_X[tier][zone]
                lo = float(lo) + urgency
                hi = float(hi) + urgency
                if flip:
                    lo, hi = 1.0 - hi, 1.0 - lo
                lo = max(0.02, lo)
                hi = min(0.98, max(lo + 0.05, hi))
                for rank in range(r0, min(r1, len(sorted_idx))):
                    idx = sorted_idx[rank]
                    xy[0, idx, 0] = xy[0, idx, 0].clamp(lo, hi)

        # Team A: indices 0–10, ascending x (rank 0 = GK)
        n_A = min(11, xy.shape[1])
        sorted_a = xy[0, :n_A, 0].argsort().tolist()
        _apply(sorted_a, _tier_ranks(formation_a), flip=False)

        # Team B: indices 11–21, descending x (rank 0 = GK)
        n_B = min(11, xy.shape[1] - 11)
        if n_B > 0:
            rel = xy[0, 11:11+n_B, 0].argsort(descending=True).tolist()
            sorted_b = [11 + r for r in rel]
            _apply(sorted_b, _tier_ranks(formation_b), flip=True)

        return xy

    def _ctx_tensor(self, ctx: "MatchContext") -> torch.Tensor:
        """Build the (1, 3) context tensor [zone/3, phase_open, phase_counter]."""
        return torch.tensor(
            [[ctx.zone / 3.0, float(ctx.phase == 1), float(ctx.phase >= 2)]],
            dtype=torch.float32, device=self.device,
        )

    def _apply_action(self,
                      z:      torch.Tensor,
                      action: "Action",
                      ctx:    "MatchContext",
                      alpha:  float) -> torch.Tensor:
        """
        Apply an action transform to z, using the learned encoder when available
        and falling back to the PCA-seeded encoder otherwise.

        Alpha < 1 interpolates between z (no effect) and z' (full effect).
        """
        if self.learned_encoder is not None:
            z_in  = z.unsqueeze(0) if z.dim() == 1 else z   # (1, 256)
            a_idx = torch.tensor([action.value], dtype=torch.long, device=self.device)
            ctx_t = self._ctx_tensor(ctx)
            z_out = self.learned_encoder.apply(z_in, a_idx, ctx_t).squeeze(0)
            if alpha < 1.0 - 1e-6:
                z_out = z + alpha * (z_out - z)
            return z_out
        else:
            from model.action_encoder import apply_action
            return apply_action(self.action_encoder, z, action, ctx, alpha)

    def _choose_defense(self,
                        attacker_action: "Action",
                        z_A_mod:         torch.Tensor,
                        z_B:             torch.Tensor,
                        norm_B:          torch.Tensor,
                        ctx:             "MatchContext",
                        adversarial:     bool) -> "Action":
        """
        Choose Team B's response action.

        adversarial=False: lookup table (fast, one dict get).
        adversarial=True:  enumerate all 11 actions, pick the one that
                           minimises Team A's P(shot) — 11 SSE evaluations.
        """
        from model.action_encoder import Action

        if not adversarial:
            return self._RESPONSE_MAP.get(attacker_action, Action.HOLD)

        best_action = Action.HOLD
        best_p_shot = float("inf")
        for def_a in Action:
            z_B_try = self._apply_action(z_B, def_a, ctx, 0.5).to(self.device)
            z_B_try = z_B_try * (norm_B / z_B_try.norm().clamp(min=1e-8))
            p = self._run_sse_probs(z_A_mod, z_B_try, ctx)
            if p.p_shot < best_p_shot:
                best_p_shot = p.p_shot
                best_action = def_a
        return best_action

    def _encode_condition(self,
                          z_A:     torch.Tensor,
                          z_B:     torch.Tensor,
                          context: "MatchContext") -> torch.Tensor:
        sd   = torch.tensor([[context.score_diff / 3.0]],
                            device=self.device).clamp(-1, 1)
        mn   = torch.tensor([[context.minute / 90.0]], device=self.device)
        p_oh = torch.zeros(1, 4, device=self.device)
        p_oh[0, min(context.phase, 3)] = 1.0
        z_oh = torch.zeros(1, 4, device=self.device)
        z_oh[0, min(context.zone, 3)]  = 1.0

        return self.generator.encode_condition(
            z_A.unsqueeze(0), z_B.unsqueeze(0), sd, mn, p_oh, z_oh
        )

    def _run_sse_probs(self,
                       z_A:     torch.Tensor,
                       z_B:     torch.Tensor,
                       context: "MatchContext") -> OutcomeProbabilities:
        """
        Run a synthetic freeze frame through the SSE to get outcome probs.
        We generate a quick low-step frame so the SSE sees realistic positions.
        """
        c      = self._encode_condition(z_A, z_B, context)
        gen_xy = self._debias_positions(
            self.generator.generate(self.roles, c, self.mask, n_steps=10),
            context,
        )  # (1, N, 2)

        # Reconstruct full position tensor (x, y, is_teammate, is_actor)
        N        = self.generator.n_players
        pos_full = torch.cat([gen_xy, self.roles], dim=-1)  # (1, N, 4)

        ctx_t = torch.tensor(
            [context.zone / 3.0,
             float(context.phase == 1),
             float(context.phase >= 2)],
            dtype=torch.float32, device=self.device
        ).unsqueeze(0)   # (1, 3)

        _, logits = self.sse(pos_full, self.mask, ctx_t)   # TacticalPredictor returns (z, logits)
        probs           = torch.sigmoid(logits).squeeze(0).cpu()   # (3,)

        return OutcomeProbabilities(
            p_advance     = float(probs[0]),   # reached_s2
            p_final_third = float(probs[1]),   # reached_s3
            p_shot        = float(probs[2]),   # reached_shot
        )

    @torch.no_grad()
    def encode_frame(self,
                     positions: torch.Tensor,
                     mask:      torch.Tensor,
                     context:   torch.Tensor) -> torch.Tensor:
        """
        Encode a raw freeze-frame tensor through the SSE and return the
        256-dim z embedding.  Used by validation scripts.

        Args:
            positions : (B, N, 4)  [x, y, is_teammate, is_actor]
            mask      : (B, N) bool
            context   : (B, 3)     [zone/3, phase1, phase2_flag]

        Returns:
            z : (B, 256)
        """
        z, _ = self.sse(positions.to(self.device),
                        mask.to(self.device),
                        context.to(self.device))
        return z.cpu()

    def list_teams(self) -> list[dict]:
        """Returns team ids and fingerprint norms (for UI team picker)."""
        return [
            {"team_id": tid, "fp_norm": float(fp.norm())}
            for tid, fp in self.fingerprints.items()
        ]


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    BASE  = Path(__file__).parent.parent
    CKPT  = BASE / "model" / "checkpoints"

    required = [CKPT / "sse_best.pt",
                CKPT / "generator_best.pt",
                CKPT / "team_fingerprints.pt"]
    missing = [p for p in required if not p.exists()]
    if missing:
        print("Missing checkpoints:", [p.name for p in missing])
        print("Run 03_train_generator.py first.")
        sys.exit(1)

    from model.action_encoder import Action, MatchContext

    print("Loading ConditionalEngine…")
    engine = ConditionalEngine(
        sse_path        = CKPT / "sse_best.pt",
        generator_path  = CKPT / "generator_best.pt",
        fingerprint_path= CKPT / "team_fingerprints.pt",
    )

    teams   = engine.list_teams()
    tid_a   = teams[0]["team_id"]
    tid_b   = teams[1]["team_id"]
    context = MatchContext(score_diff=-1.0, minute=72.0,
                           zone=2, phase=0, poss_team=0)

    print(f"\nTeam A: {tid_a}  |  Team B: {tid_b}")
    print(f"Context: losing 0-1 at 72', zone 2, open play\n")

    print("Testing all actions:")
    for action in Action:
        result = engine.step(action, context, tid_a, tid_b, alpha=1.0)
        p = result.probs
        d = result.prob_deltas
        print(
            f"  {result.action_name:16s}  "
            f"P(shot)={p.p_shot:.3f} (Δ{d.p_shot:+.3f})  "
            f"P(adv)={p.p_advance:.3f} (Δ{d.p_advance:+.3f})"
        )

    print("\nJSON output sample (ADVANCE):")
    result = engine.step(Action.ADVANCE, context, tid_a, tid_b)
    j = result.to_json_safe()
    print(f"  Keys: {list(j.keys())}")
    print(f"  positions shape: {len(j['positions'])} × {len(j['positions'][0])}")
    print(f"  probs: {j['probs']}")
    print(f"  deltas: {j['deltas']}")
