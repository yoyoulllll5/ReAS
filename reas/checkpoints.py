"""Read legacy two-file checkpoints and self-contained portable exports."""

from pathlib import Path

import torch


def load_checkpoint(path, reconstruction=None):
    path = Path(path).resolve()
    state = torch.load(path, map_location="cpu", weights_only=True)
    if "reconstruction" not in state:
        reference = reconstruction or state.get("reconstruction_path")
        if not reference:
            raise ValueError("Missing reconstruction weights; pass --reconstruction")
        reference = Path(reference)
        if not reference.is_absolute():
            reference = path.parent / reference
        if not reference.is_file():
            raise FileNotFoundError(
                f"Reconstruction weights not found: {reference}; pass --reconstruction"
            )
        state["reconstruction"] = torch.load(reference, map_location="cpu", weights_only=True)
    for key in ["config", "segmentation", "reconstruction", "epoch"]:
        if key not in state:
            raise ValueError(f"Checkpoint is missing {key}")
    return state
