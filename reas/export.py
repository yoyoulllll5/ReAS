"""Bundle both networks and the split into a portable checkpoint."""

import argparse
import json
from pathlib import Path

import torch

from .checkpoints import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reconstruction", type=Path)
    parser.add_argument("--split-file", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    state = load_checkpoint(args.checkpoint, args.reconstruction)
    config = state["config"].copy()
    # A release checkpoint should not disclose the training workstation layout.
    config.update(data_root="data/mvtec/bottle", dtd_root="data/dtd/images")
    split_path = args.split_file or args.checkpoint.parent / "split.json"
    split = state.get("split") if not args.split_file else None
    if split is None:
        split = json.loads(split_path.read_text(encoding="utf-8-sig"))
    bundle = dict(
        format_version=1,
        config=config,
        epoch=state["epoch"],
        split=split,
        segmentation=state["segmentation"],
        reconstruction=state["reconstruction"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, args.output)
    print(f"Saved portable checkpoint: {args.output}")


if __name__ == "__main__":
    main()
