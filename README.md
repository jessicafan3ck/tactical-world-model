# Tactical World Model

An interactive football simulation built on a learned State Space Encoder + Flow Matching Generator pair, trained on StatsBomb open data (Women's World Cup). Given a match context and a tactical action, the engine advances the game state and returns new player positions, outcome probabilities, and action suggestions.

See [DESIGN.md](DESIGN.md) for architecture details, validation baselines, and known limitations.

## Architecture

```
StatsBomb 360° freeze frames
        ↓
State Space Encoder (SSE — Set Transformer)
        ↓
Action Encoder (ConditionedMLP)   +   Team Fingerprints
        ↓
Flow Matching Generator
        ↓
ConditionalEngine  →  FastAPI server  →  browser UI
```

## Validation summary

| Check | Metric | Value |
|---|---|---|
| Check 1 — Action Encoder | ConditionedMLP held-out AUC | 0.579 |
| Check 2 — Forward AUC (honest) | s2 / s3 / shot | 0.49 / 0.52 / 0.633 |
| Check 3 — Generator realism (with clamps) | classifier AUC | 0.845 |
| Check 4 — Action effect-ordering | sign-agreement | 47% (FAIL) |

Check 4 failing means the engine does not reliably rank which actions improve outcomes — it should not be the basis for tactical decisions. For the primary use case (visualizing designed tactics, communicating how a shape evolves spatially, expressing team style interactively) Check 4 is not the relevant bar: football is too stochastic and continuous for any single model to be outcome-determinative, and analysts bring their own domain knowledge to the decision layer.

## Data

StatsBomb open data — Women's World Cup (360° freeze frames + possession sequences).

## Repository structure

```
tactical-world-model/
├── model/
│   ├── sse.py                    # State Space Encoder (Set Transformer)
│   ├── flow_matching.py          # Conditional flow matching generator
│   ├── action_encoder.py         # Affine action encoder
│   ├── learned_action_encoder.py # ConditionedMLP action encoder
│   ├── conditional_engine.py     # Inference engine (step, suggest, debias)
│   ├── simulator.py              # Autoregressive match simulator
│   └── checkpoints/              # Trained weights (gitignored)
├── server/
│   ├── app.py                    # FastAPI server + endpoints
│   └── static/index.html         # Browser UI
├── scripts/
│   ├── 01–05_*.py                # Data download, processing, training
│   ├── 07_validate_encoder.py
│   ├── 08_validate_forward_auc.py
│   ├── 09_validate_generator.py
│   ├── 10_noise_floor_diagnostic.py
│   ├── 11_train_action_encoder.py
│   ├── 12_audit_action_encoder.py
│   ├── 13_debias_generator.py
│   ├── 14_decompose_auc.py
│   └── 15_validate_action_ordering.py
├── data/
│   ├── raw/statsbomb/            # StatsBomb data (gitignored)
│   ├── processed/                # Processed datasets (gitignored)
│   └── results/                  # Evaluation outputs (CSVs, PNGs)
├── utils/
│   └── statsbomb_utils.py
└── DESIGN.md                     # Architecture, post-processing layers, validation
```

## Running the server

```bash
pip install -r requirements.txt
cp .env.example .env              # add ANTHROPIC_API_KEY for /api/analyze_sequence
uvicorn server.app:app --reload
```

Open `http://localhost:8000`.

## Citation

```
@software{fan2026tacticalworldmodel,
  title={Tactical World Model: Generative Football Simulation},
  author={Fan, Jessica},
  year={2026}
}
```
