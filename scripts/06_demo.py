"""
06_demo.py
==========
Interactive Tactical World Model demo — intended for live presentation to FIFA.

Panels
------
A. Pitch Animator
   Animated matplotlib figure showing a generated freeze frame morphing
   from random noise (x_0) to the final configuration (x_1) over the
   ODE integration steps. Saves as animated GIF.

B. Match Timeline
   Minute-by-minute probability curves for a simulated match:
   shot probability, goal probability, possession retention.
   Annotates actual goals as vertical markers.

C. Tactical Fingerprint Radar
   Per-team radar chart derived from PCA-projected fingerprint components,
   mapped to interpretable axes: width, verticality, pressing, counter,
   set-piece, possession.

D. Style-Transfer Comparison
   Side-by-side pitch snapshots: team A's natural style vs team A after
   α-interpolation toward team B's fingerprint. Shows how tactical
   DNA reshapes spatial formations.

E. Win-Probability Timeline
   Real-time win/draw/loss probabilities across match minute for a
   simulated match, updated as goals are scored.

Outputs
-------
    data/results/demo/
        freeze_frame_animation.gif
        match_timeline.png
        fingerprint_radar.png
        style_transfer.png
        win_prob_timeline.png
        demo_summary.png          ← all panels combined (poster format)
"""

import sys
import math
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.simulator import SimulatorRNN, simulate_match, STATE_DIM, FP_DIM

# ── Config ────────────────────────────────────────────────────────────────────

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "mps"
                           if torch.backends.mps.is_available() else "cpu")
CKPT_DIR    = Path("model/checkpoints")
RESULTS_DIR = Path("data/results")
DEMO_DIR    = RESULTS_DIR / "demo"
FP_PATH     = CKPT_DIR / "team_fingerprints.pt"
GEN_PATH    = CKPT_DIR / "generator_best.pt"
SIM_PATH    = CKPT_DIR / "simulator_best.pt"
META_PATH   = RESULTS_DIR / "possession_meta.csv"
PCA_PATH    = RESULTS_DIR / "tactical_pca.csv"

DEMO_DIR.mkdir(parents=True, exist_ok=True)

PITCH_W, PITCH_H = 120.0, 80.0   # StatsBomb pitch dimensions

RADAR_AXES = [
    "Width", "Verticality", "Pressing",
    "Counter", "Set-piece", "Possession"
]


# ── Pitch drawing helpers ──────────────────────────────────────────────────────

def draw_pitch(ax, color="white", linecolor="#aaaaaa", alpha=0.9):
    """Draw a StatsBomb-dimensioned football pitch on ax."""
    ax.set_facecolor("#2d6a4f")
    ax.set_xlim(0, PITCH_W)
    ax.set_ylim(0, PITCH_H)
    ax.set_aspect("equal")
    ax.axis("off")

    lc = dict(color=linecolor, linewidth=0.8, alpha=alpha)

    # Pitch outline
    ax.plot([0, PITCH_W, PITCH_W, 0, 0],
            [0, 0, PITCH_H, PITCH_H, 0], **lc)

    # Halfway line
    ax.plot([PITCH_W / 2, PITCH_W / 2], [0, PITCH_H], **lc)

    # Centre circle
    circle = plt.Circle((PITCH_W / 2, PITCH_H / 2), 9.15,
                         fill=False, **lc)
    ax.add_patch(circle)

    # 6-yard boxes
    for x0 in [0, PITCH_W - 5.5]:
        ax.plot([x0, x0 + (5.5 if x0 == 0 else -5.5),
                 x0 + (5.5 if x0 == 0 else -5.5), x0],
                [PITCH_H / 2 - 9.16, PITCH_H / 2 - 9.16,
                 PITCH_H / 2 + 9.16, PITCH_H / 2 + 9.16], **lc)

    # Penalty boxes
    for x0 in [0, PITCH_W - 16.5]:
        ax.plot([x0, x0 + (16.5 if x0 == 0 else -16.5),
                 x0 + (16.5 if x0 == 0 else -16.5), x0],
                [PITCH_H / 2 - 20.16, PITCH_H / 2 - 20.16,
                 PITCH_H / 2 + 20.16, PITCH_H / 2 + 20.16], **lc)

    # Penalty spots
    for px in [11, PITCH_W - 11]:
        ax.plot(px, PITCH_H / 2, "o", color=linecolor, markersize=2, alpha=alpha)

    return ax


def plot_positions(ax, positions: np.ndarray, mask: np.ndarray,
                   title: str = "", alpha: float = 0.9):
    """
    Scatter players on pitch.
    positions: (N, 4)  [x_norm, y_norm, is_teammate, is_actor]
    mask:      (N,)    True where padded
    """
    draw_pitch(ax)
    valid = ~mask

    for i in range(len(positions)):
        if mask[i]:
            continue
        x = float(positions[i, 0]) * PITCH_W
        y = float(positions[i, 1]) * PITCH_H
        is_tm  = bool(positions[i, 2] > 0.5)
        is_act = bool(positions[i, 3] > 0.5)

        if is_act:
            ax.plot(x, y, "o", color="#f9c74f", markersize=11, zorder=5)
            ax.plot(x, y, "o", color="black",   markersize=11,
                    fillstyle="none", linewidth=1.5, zorder=6)
        elif is_tm:
            ax.plot(x, y, "o", color="#90e0ef", markersize=9, zorder=4)
        else:
            ax.plot(x, y, "o", color="#e63946", markersize=9, zorder=4)

    if title:
        ax.set_title(title, color="white", fontsize=10, pad=4)

    legend = [
        mpatches.Patch(color="#f9c74f", label="Actor"),
        mpatches.Patch(color="#90e0ef", label="Teammates"),
        mpatches.Patch(color="#e63946", label="Opponents"),
    ]
    ax.legend(handles=legend, loc="lower right",
              fontsize=7, framealpha=0.4,
              labelcolor="white", facecolor="#1a1a2e")


# ── A. Freeze-frame animation ──────────────────────────────────────────────────

def make_freeze_frame_animation(generator, fingerprints: dict,
                                mean_fp: torch.Tensor,
                                n_steps: int = 30) -> None:
    """
    Animate one example freeze frame being generated via ODE integration.
    Saves freeze_frame_animation.gif.
    """
    try:
        import matplotlib.animation as anim
    except ImportError:
        print("  Animation requires matplotlib — skipping.")
        return

    generator.eval()

    # Pick a team pair
    team_ids = list(fingerprints.keys())
    if len(team_ids) < 2:
        print("  Not enough teams for animation — skipping.")
        return

    tid_a, tid_b = team_ids[0], team_ids[1]
    z_A  = fingerprints[tid_a].unsqueeze(0).to(DEVICE)
    z_B  = fingerprints[tid_b].unsqueeze(0).to(DEVICE)

    score_diff   = torch.zeros(1, 1, device=DEVICE)
    minute_norm  = torch.full((1, 1), 0.45, device=DEVICE)
    phase_oh     = torch.zeros(1, 4, device=DEVICE); phase_oh[0, 0] = 1.0
    zone_oh      = torch.zeros(1, 4, device=DEVICE); zone_oh[0, 1]  = 1.0

    with torch.no_grad():
        c = generator.encode_condition(
            z_A, z_B, score_diff, minute_norm, phase_oh, zone_oh
        )

    N = generator.n_players
    roles = torch.zeros(1, N, 2, device=DEVICE)
    roles[0, :11, 0] = 1.0   # teammates
    roles[0, 0,   1] = 1.0   # actor
    mask = torch.zeros(1, N, dtype=torch.bool, device=DEVICE)

    sigma = generator.sigma
    frames_data = []

    with torch.no_grad():
        x = torch.randn(1, N, 2, device=DEVICE) * sigma
        dt = 1.0 / n_steps
        for step in range(n_steps + 1):
            t = step / n_steps
            # Reconstruct position tensor with roles
            pos_np = torch.cat([
                x.squeeze(0).cpu(),
                roles.squeeze(0).cpu()
            ], dim=-1).numpy()
            frames_data.append(pos_np.copy())
            if step < n_steps:
                t_t = torch.full((1,), t, device=DEVICE)
                v   = generator.velocity_field(x, roles, t_t, c, mask)
                x   = x + dt * v

    fig, ax = plt.subplots(figsize=(10, 7), facecolor="#1a1a2e")
    draw_pitch(ax)
    scat_actor = ax.plot([], [], "o", color="#f9c74f",  markersize=11, zorder=5)[0]
    scat_team  = ax.plot([], [], "o", color="#90e0ef",  markersize=9,  zorder=4)[0]
    scat_opp   = ax.plot([], [], "o", color="#e63946",  markersize=9,  zorder=4)[0]
    title      = ax.set_title("", color="white", fontsize=11)
    fig.tight_layout()

    def update(frame_idx):
        pos = frames_data[frame_idx]
        t   = frame_idx / n_steps

        actor_xy = [(pos[i, 0] * PITCH_W, pos[i, 1] * PITCH_H)
                    for i in range(N) if pos[i, 3] > 0.5]
        team_xy  = [(pos[i, 0] * PITCH_W, pos[i, 1] * PITCH_H)
                    for i in range(N) if pos[i, 2] > 0.5 and pos[i, 3] < 0.5]
        opp_xy   = [(pos[i, 0] * PITCH_W, pos[i, 1] * PITCH_H)
                    for i in range(N) if pos[i, 2] < 0.5]

        scat_actor.set_data(*zip(*actor_xy) if actor_xy else ([], []))
        scat_team.set_data(*zip(*team_xy)  if team_xy  else ([], []))
        scat_opp.set_data(*zip(*opp_xy)   if opp_xy   else ([], []))
        title.set_text(f"ODE t = {t:.2f}  (step {frame_idx}/{n_steps})")
        return scat_actor, scat_team, scat_opp, title

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames_data),
        interval=120, blit=True,
    )
    out = DEMO_DIR / "freeze_frame_animation.gif"
    ani.save(str(out), writer="pillow", fps=12, dpi=100)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── B. Match timeline ──────────────────────────────────────────────────────────

def plot_match_timeline(sim_df: pd.DataFrame, team_name_a: str = "Team A",
                         team_name_b: str = "Team B") -> None:
    """
    Smoothed shot/goal probability and cumulative score over match minute.
    """
    df = sim_df.copy().sort_values("minute").reset_index(drop=True)

    window = 10
    df["shot_roll"]   = df["shot"].rolling(window, center=True, min_periods=1).mean()
    df["retain_roll"] = df["retain"].rolling(window, center=True, min_periods=1).mean()

    goal_rows = df[df["goal"] == 1]

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), facecolor="#1a1a2e",
                             gridspec_kw={"hspace": 0.45})

    for ax in axes:
        ax.set_facecolor("#1a1a2e")
        for spine in ax.spines.values():
            spine.set_color("#444")

    # Shot probability
    ax = axes[0]
    home_mask = df["poss_team"] == 0
    ax.plot(df.loc[home_mask, "minute"], df.loc[home_mask, "shot_roll"],
            color="#90e0ef", linewidth=1.5, label=team_name_a)
    ax.plot(df.loc[~home_mask, "minute"], df.loc[~home_mask, "shot_roll"],
            color="#e63946", linewidth=1.5, label=team_name_b)
    for _, row in goal_rows.iterrows():
        c = "#90e0ef" if row["poss_team"] == 0 else "#e63946"
        ax.axvline(row["minute"], color=c, alpha=0.6, linewidth=1, linestyle="--")
        ax.annotate("⚽", (row["minute"], ax.get_ylim()[1] * 0.8),
                    color=c, fontsize=9)
    ax.set_title("Shot Probability (rolling)", color="white", fontsize=10)
    ax.set_ylabel("P(shot)", color="#aaa")
    ax.tick_params(colors="#aaa")
    ax.legend(facecolor="#2a2a3e", labelcolor="white", fontsize=8)

    # Possession retention
    ax = axes[1]
    ax.plot(df.loc[home_mask,  "minute"], df.loc[home_mask,  "retain_roll"],
            color="#90e0ef", linewidth=1.5)
    ax.plot(df.loc[~home_mask, "minute"], df.loc[~home_mask, "retain_roll"],
            color="#e63946", linewidth=1.5)
    ax.axhline(0.5, color="#666", linestyle=":", linewidth=0.8)
    ax.set_title("Possession Retention Probability", color="white", fontsize=10)
    ax.set_ylabel("P(retain)", color="#aaa")
    ax.tick_params(colors="#aaa")

    # Cumulative score
    ax = axes[2]
    ax.step(df["minute"], df["score_home"], where="post",
            color="#90e0ef", linewidth=2, label=team_name_a)
    ax.step(df["minute"], df["score_away"], where="post",
            color="#e63946", linewidth=2, label=team_name_b)
    final_h = int(df.iloc[-1]["score_home"])
    final_a = int(df.iloc[-1]["score_away"])
    ax.set_title(f"Score: {team_name_a} {final_h} – {final_a} {team_name_b}",
                 color="white", fontsize=10)
    ax.set_ylabel("Goals", color="#aaa")
    ax.set_xlabel("Match minute", color="#aaa")
    ax.tick_params(colors="#aaa")
    ax.legend(facecolor="#2a2a3e", labelcolor="white", fontsize=8)

    out = DEMO_DIR / "match_timeline.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── C. Fingerprint radar chart ─────────────────────────────────────────────────

def fingerprint_to_radar_values(fp: torch.Tensor,
                                 pca_df: pd.DataFrame,
                                 team_id: int) -> list[float]:
    """
    Maps a team's fingerprint to 6 radar axis scores using PCA-derived
    components as proxies for interpretable tactical axes.
    """
    row = pca_df[pca_df["team_id"] == team_id]
    if row.empty:
        return [0.5] * 6

    pc1 = float(row.iloc[0]["pc1"])
    pc2 = float(row.iloc[0]["pc2"])
    fp_np = fp.numpy()

    # Proxy mappings from PCA dimensions + fingerprint statistics
    norm_pc1 = (pc1 + 3) / 6    # rough normalisation into [0,1]
    norm_pc2 = (pc2 + 3) / 6

    fp_mean = float(fp_np.mean())
    fp_std  = float(fp_np.std())

    return [
        float(np.clip(norm_pc1 * 1.2, 0, 1)),          # Width
        float(np.clip(norm_pc2 * 1.2, 0, 1)),          # Verticality
        float(np.clip(fp_std  * 2.5, 0, 1)),           # Pressing intensity
        float(np.clip(1 - norm_pc1, 0, 1)),             # Counter tendency
        float(np.clip((fp_mean + 0.5) / 1.0, 0, 1)),   # Set-piece proficiency
        float(np.clip((norm_pc1 + norm_pc2) / 2, 0, 1)), # Possession share
    ]


def plot_radar(ax, values: list[float], label: str, color: str,
               alpha: float = 0.35) -> None:
    N = len(RADAR_AXES)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]
    vals = values + values[:1]

    ax.plot(angles, vals, color=color, linewidth=2, label=label)
    ax.fill(angles, vals, color=color, alpha=alpha)


def make_fingerprint_radar(fingerprints: dict, pca_df: pd.DataFrame,
                            top_teams: list[int],
                            name_map: dict[int, str]) -> None:
    N = len(RADAR_AXES)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]

    colors = ["#90e0ef", "#e63946", "#f9c74f", "#a8dadc",
              "#457b9d", "#e9c46a"]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True),
                           facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(RADAR_AXES, color="white", fontsize=9)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["", "", "", ""], color="#888")
    ax.set_ylim(0, 1)
    ax.spines["polar"].set_color("#444")
    ax.grid(color="#444", linewidth=0.5)

    for i, tid in enumerate(top_teams[:6]):
        if tid not in fingerprints:
            continue
        vals = fingerprint_to_radar_values(fingerprints[tid], pca_df, tid)
        name = name_map.get(tid, str(tid))
        plot_radar(ax, vals, name, colors[i % len(colors)])

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
              facecolor="#2a2a3e", labelcolor="white", fontsize=8)
    ax.set_title("Tactical Fingerprint Radar", color="white", fontsize=12, pad=20)

    out = DEMO_DIR / "fingerprint_radar.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── D. Style-transfer comparison ───────────────────────────────────────────────

def make_style_transfer_plot(generator, fingerprints: dict,
                              tid_a: int, tid_b: int,
                              name_a: str, name_b: str,
                              alphas: list[float] = [0.0, 0.5, 1.0]) -> None:
    """
    Generates freeze frames for team A at α = 0, 0.5, 1.0 toward team B.
    """
    generator.eval()

    z_A_base = fingerprints.get(tid_a, torch.zeros(FP_DIM))
    z_B      = fingerprints.get(tid_b, torch.zeros(FP_DIM))

    fig, axes = plt.subplots(1, len(alphas), figsize=(5 * len(alphas), 5),
                             facecolor="#1a1a2e")

    N = generator.n_players
    roles = torch.zeros(1, N, 2, device=DEVICE)
    roles[0, :11, 0] = 1.0
    roles[0, 0,   1] = 1.0
    mask = torch.zeros(1, N, dtype=torch.bool, device=DEVICE)

    score_diff  = torch.zeros(1, 1, device=DEVICE)
    minute_norm = torch.full((1, 1), 0.45, device=DEVICE)
    phase_oh    = torch.zeros(1, 4, device=DEVICE); phase_oh[0, 0] = 1.0
    zone_oh     = torch.zeros(1, 4, device=DEVICE); zone_oh[0, 1]  = 1.0

    for ax_i, alpha in enumerate(alphas):
        z_A_interp = (z_A_base + alpha * (z_B - z_A_base)).unsqueeze(0).to(DEVICE)
        z_B_t      = z_B.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            c    = generator.encode_condition(
                z_A_interp, z_B_t, score_diff, minute_norm, phase_oh, zone_oh
            )
            gen  = generator.generate(roles, c, mask, n_steps=30)

        positions_np = torch.cat([
            gen.squeeze(0).cpu(),
            roles.squeeze(0).cpu()
        ], dim=-1).numpy()

        mask_np = np.zeros(N, dtype=bool)
        plot_positions(
            axes[ax_i], positions_np, mask_np,
            title=f"α = {alpha:.1f}  ({name_a}→{name_b})"
        )
        axes[ax_i].set_title(
            f"α = {alpha:.1f}\n{'Natural style' if alpha==0 else 'Style transfer'}",
            color="white", fontsize=9, pad=4
        )

    fig.suptitle(f"Tactical Style Transfer: {name_a} → {name_b}",
                 color="white", fontsize=13, y=1.01)
    out = DEMO_DIR / "style_transfer.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── E. Win-probability timeline ────────────────────────────────────────────────

def plot_win_prob_timeline(sim_df: pd.DataFrame,
                           name_a: str = "Team A",
                           name_b: str = "Team B") -> None:
    """
    Estimates rolling win probability from score differential using a
    logistic model. Not a trained model — a simple proxy for demo purposes.
    """
    df = sim_df.copy().sort_values("minute").reset_index(drop=True)

    def score_to_win_prob(score_diff: float, minute: float) -> tuple[float, float, float]:
        time_left = max(0, 90 - minute) / 90.0
        k = 1.5 * (1 + time_left)   # variance decreases as match ends
        p_home = 1 / (1 + math.exp(-k * score_diff))
        p_away = 1 / (1 + math.exp( k * score_diff))
        p_draw = 1 - p_home - p_away
        p_draw = max(0.0, p_draw)
        total  = p_home + p_away + p_draw
        return p_home / total, p_draw / total, p_away / total

    minutes, p_h, p_d, p_a = [], [], [], []
    for _, row in df.iterrows():
        m   = float(row["minute"])
        sd  = float(row["score_home"]) - float(row["score_away"])
        ph, pd_, pa = score_to_win_prob(sd, m)
        minutes.append(m)
        p_h.append(ph)
        p_d.append(pd_)
        p_a.append(pa)

    fig, ax = plt.subplots(figsize=(14, 5), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    for spine in ax.spines.values():
        spine.set_color("#444")

    ax.stackplot(minutes, p_h, p_d, p_a,
                 labels=[f"{name_a} win", "Draw", f"{name_b} win"],
                 colors=["#90e0ef", "#555577", "#e63946"],
                 alpha=0.75)

    goal_rows = df[df["goal"] == 1]
    for _, row in goal_rows.iterrows():
        c = "#90e0ef" if row["poss_team"] == 0 else "#e63946"
        ax.axvline(row["minute"], color=c, alpha=0.8, linewidth=1.5,
                   linestyle="--")

    ax.set_ylim(0, 1)
    ax.set_xlim(0, max(minutes) + 1)
    ax.set_xlabel("Match minute", color="#aaa")
    ax.set_ylabel("Win probability", color="#aaa")
    ax.tick_params(colors="#aaa")
    ax.set_title(f"Win Probability Timeline — {name_a} vs {name_b}",
                 color="white", fontsize=11)
    ax.legend(facecolor="#2a2a3e", labelcolor="white", fontsize=9,
              loc="upper left")

    out = DEMO_DIR / "win_prob_timeline.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Demo summary poster ────────────────────────────────────────────────────────

def make_demo_poster(sim_df: pd.DataFrame,
                     name_a: str, name_b: str) -> None:
    """
    Combines timeline and score into a single-page poster.
    """
    fig = plt.figure(figsize=(20, 12), facecolor="#1a1a2e")
    gs  = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # Load and embed saved sub-figures as images
    def embed(path: Path, ax):
        if path.exists():
            img = plt.imread(str(path))
            ax.imshow(img)
        ax.axis("off")

    embed(DEMO_DIR / "match_timeline.png",        fig.add_subplot(gs[0, :2]))
    embed(DEMO_DIR / "fingerprint_radar.png",     fig.add_subplot(gs[0, 2]))
    embed(DEMO_DIR / "style_transfer.png",        fig.add_subplot(gs[1, :2]))
    embed(DEMO_DIR / "win_prob_timeline.png",     fig.add_subplot(gs[1, 2]))

    fig.suptitle("Tactical World Model — FIFA Women's World Cup 2027 Demo",
                 color="white", fontsize=16, y=0.98, fontweight="bold")

    out = DEMO_DIR / "demo_summary.png"
    plt.savefig(out, dpi=120, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Tactical World Model — Demo Visualization")
    print("=" * 50)

    if not FP_PATH.exists():
        print(f"ERROR: {FP_PATH} not found — run 03_train_generator.py first.")
        return

    # ── Load shared assets ─────────────────────────────────────────────────────
    fp_data      = torch.load(FP_PATH, map_location="cpu")
    fingerprints = {k: v.float() for k, v in fp_data["team_fingerprints"].items()}
    mean_fp      = fp_data.get(
        "mean_fingerprint",
        torch.stack(list(fingerprints.values())).mean(0)
    )
    print(f"Loaded {len(fingerprints)} team fingerprints")

    meta = pd.read_csv(META_PATH) if META_PATH.exists() else pd.DataFrame()
    name_map: dict[int, str] = {}
    if not meta.empty and "team_name" in meta.columns:
        name_map = (meta[["team_id", "team_name"]]
                    .drop_duplicates()
                    .set_index("team_id")["team_name"]
                    .to_dict())

    team_counts = (meta.groupby("team_id").size()
                   if not meta.empty else {t: 0 for t in fingerprints})
    top_teams   = (team_counts.nlargest(10).index.tolist()
                   if hasattr(team_counts, "nlargest")
                   else list(fingerprints.keys())[:10])

    pca_df = pd.read_csv(PCA_PATH) if PCA_PATH.exists() else pd.DataFrame()

    # Pick demo teams
    tid_a = top_teams[0] if top_teams else list(fingerprints.keys())[0]
    tid_b = top_teams[1] if len(top_teams) > 1 else list(fingerprints.keys())[1]
    name_a = name_map.get(tid_a, f"Team {tid_a}")
    name_b = name_map.get(tid_b, f"Team {tid_b}")
    print(f"Demo teams: {name_a} vs {name_b}")

    # ── B. Match timeline ──────────────────────────────────────────────────────
    print("\n[1/5] Match timeline…")
    if SIM_PATH.exists():
        sim_model = SimulatorRNN(state_dim=STATE_DIM, fp_dim=FP_DIM).to(DEVICE)
        sim_model.load_state_dict(torch.load(SIM_PATH, map_location=DEVICE))
        sim_model.eval()

        fp_a = fingerprints.get(tid_a, mean_fp).to(DEVICE)
        fp_b = fingerprints.get(tid_b, mean_fp).to(DEVICE)
        sim_df = simulate_match(sim_model, fp_a, fp_b, device=DEVICE,
                                n_poss=500, seed=7)
        plot_match_timeline(sim_df, name_a, name_b)
        plot_win_prob_timeline(sim_df, name_a, name_b)
        print(f"  Score: {name_a} {int(sim_df.iloc[-1]['score_home'])}–"
              f"{int(sim_df.iloc[-1]['score_away'])} {name_b}")
    else:
        print(f"  Skipped — {SIM_PATH} not found.")
        sim_df = pd.DataFrame()

    # ── C. Radar chart ─────────────────────────────────────────────────────────
    print("\n[2/5] Fingerprint radar…")
    if not pca_df.empty:
        make_fingerprint_radar(fingerprints, pca_df, top_teams, name_map)
    else:
        print("  Skipped — tactical_pca.csv not found (run 05_counterfactuals.py).")

    # ── D. Style transfer ──────────────────────────────────────────────────────
    print("\n[3/5] Style transfer comparison…")
    if GEN_PATH.exists():
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from model.flow_matching import build_generator
        generator = build_generator(fingerprint_dim=FP_DIM).to(DEVICE)
        generator.load_state_dict(torch.load(GEN_PATH, map_location=DEVICE))
        make_style_transfer_plot(generator, fingerprints,
                                  tid_a, tid_b, name_a, name_b)
    else:
        print(f"  Skipped — {GEN_PATH} not found.")

    # ── A. Freeze-frame animation ──────────────────────────────────────────────
    print("\n[4/5] Freeze-frame ODE animation…")
    if GEN_PATH.exists():
        make_freeze_frame_animation(generator, fingerprints, mean_fp, n_steps=30)
    else:
        print(f"  Skipped — generator checkpoint not found.")

    # ── Poster ─────────────────────────────────────────────────────────────────
    print("\n[5/5] Demo summary poster…")
    if not sim_df.empty:
        make_demo_poster(sim_df, name_a, name_b)
    else:
        print("  Skipped — no simulation data.")

    print(f"\nAll outputs in: {DEMO_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
