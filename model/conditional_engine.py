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

    def to_json_safe(self) -> dict:
        return {
            "action":     self.action_name,
            "positions":  self.freeze_frame.positions.tolist(),
            "roles":      self.freeze_frame.roles.tolist(),
            "mask":       self.freeze_frame.mask.tolist(),
            "probs":      self.probs.to_dict(),
            "deltas":     self.prob_deltas.to_dict(),
            "context":    self.context_after,
        }


# ── Engine ─────────────────────────────────────────────────────────────────────

class ConditionalEngine:
    """
    Stateless inference engine. The server instantiates one of these at
    startup and calls step() on each user action.

    All heavy models are loaded once at __init__ time and kept in memory.
    step() is ~O(generator forward pass) ≈ 30 ms on MPS.
    """

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

        # Build action encoder and seed from PCA if available
        self.action_encoder = build_action_encoder(z_dim=256).to(self.device)
        pca_path = fingerprint_path.parent.parent / "results" / "tactical_pca.csv"
        init_from_pca(self.action_encoder, self.fingerprints, pca_path)
        self.action_encoder.eval()

        # Fixed role template (11 teammates + 11 opponents, player 0 = actor)
        N = self.generator.n_players
        roles = torch.zeros(1, N, 2)
        roles[0, :11, 0] = 1.0   # teammates
        roles[0, 0,   1] = 1.0   # actor
        self.roles = roles.to(self.device)
        self.mask  = torch.zeros(1, N, dtype=torch.bool, device=self.device)

        print(f"ConditionalEngine ready on {self.device}")
        print(f"  {len(self.fingerprints)} team fingerprints loaded")

    # ── Core inference step ────────────────────────────────────────────────────

    @torch.no_grad()
    def step(self,
             action:     "Action",
             context:    "MatchContext",
             team_id_a:  int,
             team_id_b:  int,
             alpha:      float = 1.0,
             gen_steps:  int   = 30) -> "ActionResult":
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
        from model.action_encoder import apply_action, ACTION_LABELS

        z_A_base = self.fingerprints.get(team_id_a, self.mean_fp).to(self.device)
        z_B      = self.fingerprints.get(team_id_b, self.mean_fp).to(self.device)

        # ── Baseline (no action) ───────────────────────────────────────────────
        baseline_probs = self._run_sse_probs(z_A_base, z_B, context)

        # ── Apply action → modified fingerprint ───────────────────────────────
        z_A_mod = apply_action(
            self.action_encoder, z_A_base, action, context, alpha
        ).to(self.device)

        # ── Generate freeze frame under modified fingerprint ───────────────────
        c = self._encode_condition(z_A_mod, z_B, context)
        gen_xy = self.generator.generate(
            self.roles, c, self.mask, n_steps=gen_steps
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

    # ── Helpers ────────────────────────────────────────────────────────────────

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
        gen_xy = self.generator.generate(
            self.roles, c, self.mask, n_steps=10   # fast, 10-step
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

        logits, _ = self.sse(pos_full, self.mask, ctx_t)
        probs     = torch.sigmoid(logits).squeeze(0).cpu()   # (3,)

        return OutcomeProbabilities(
            p_advance     = float(probs[0]),   # reached_s2
            p_final_third = float(probs[1]),   # reached_s3
            p_shot        = float(probs[2]),   # reached_shot
        )

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
