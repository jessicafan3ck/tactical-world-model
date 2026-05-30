# Tactical World Model — WWC 2027

A neural football world model built for the FIFA Women's World Cup 2027.
Learns to simulate full matches from real possession data, conditioned on
team tactical fingerprints. Generates counterfactual scenarios ("what if
this team pressed higher?") with calibrated uncertainty.

## Architecture

```
Freeze Frame Generator (Flow Matching)
        ↓
Tactical State Encoder (Set Transformer / SSE)
        ↓
Possession Outcome Model
        ↓
Autoregressive Match Simulator
        ↓
Causal Intervention Engine
```

## Data

Base data: StatsBomb Open Data — Women's World Cup 2023 + 2019 (360 freeze frames),
UEFA Women's Euro 2022 + 2025, FA WSL, NWSL, Liga F, Frauen Bundesliga, Serie A Women.

## Repository Structure

```
tactical-world-model/
├── data/
│   ├── raw/statsbomb/         # Downloaded StatsBomb data (gitignored)
│   ├── processed/             # Processed datasets (gitignored)
│   └── results/               # Evaluation outputs (CSVs, PNGs)
├── model/
│   ├── sse.py                 # Set Spatial Encoder (from analytics project)
│   ├── flow_matching.py       # Conditional flow matching generator
│   ├── simulator.py           # Autoregressive match simulator
│   └── checkpoints/           # Trained weights (gitignored)
├── scripts/
│   ├── 01_download_data.py    # Pull StatsBomb WWC data
│   ├── 02_build_dataset.py    # Process freeze frames + possession labels
│   ├── 03_train_generator.py  # Train flow matching model
│   ├── 04_train_simulator.py  # Train + validate match simulator
│   └── 05_counterfactuals.py  # Causal intervention engine
├── utils/
│   └── statsbomb_utils.py     # Data loading helpers
└── notebooks/                 # Exploration and demo notebooks
```

## Citation

```
@software{fan2026tacticalworldmodel,
  title={Tactical World Model: Generative Football Simulation for WWC 2027},
  author={Fan, Jessica},
  year={2026}
}
```
