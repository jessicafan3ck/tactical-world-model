"""
Download model checkpoints from Hugging Face Hub if not present locally.
Called automatically at server startup when HF_REPO_ID is set.

Required env vars:
  HF_REPO_ID   — e.g. "jessicafan3ck/tactical-world-model"
  HF_TOKEN     — (optional) for private repos

To upload checkpoints (one-time, run locally):
  pip install huggingface_hub
  huggingface-cli login
  huggingface-cli upload jessicafan3ck/tactical-world-model \\
      model/checkpoints/ model/checkpoints/
"""

import os
from pathlib import Path

CKPT_DIR = Path(__file__).parent.parent / "model" / "checkpoints"
REQUIRED = ["sse_best.pt", "generator_best.pt", "team_fingerprints.pt"]


def checkpoints_present() -> bool:
    return all((CKPT_DIR / f).exists() for f in REQUIRED)


def download():
    repo_id = os.environ.get("HF_REPO_ID")
    if not repo_id:
        return

    if checkpoints_present():
        print("Checkpoints already present — skipping download.")
        return

    print(f"Downloading checkpoints from {repo_id} …")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=["model/checkpoints/*.pt"],
            local_dir=str(CKPT_DIR.parent.parent),
            token=os.environ.get("HF_TOKEN"),
        )
        if checkpoints_present():
            print("Checkpoints downloaded successfully.")
        else:
            print("WARNING: Download completed but some checkpoints still missing.")
    except Exception as e:
        print(f"WARNING: Checkpoint download failed — {e}")
        print("Server will start without the simulation engine.")
