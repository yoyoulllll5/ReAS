"""Evaluate a saved checkpoint: python -m reas.evaluate --help."""

import argparse
import json
from pathlib import Path

import torch

from .checkpoints import load_checkpoint
from .config import validate_config
from .data import EvalDataset, validate_split
from .engine import dump, evaluate, loader, networks, seed_all


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--reconstruction", type=Path, help="Override a legacy reconstruction path")
    parser.add_argument(
        "--split-file",
        type=Path,
        help="Defaults to embedded split or checkpoint sibling split.json",
    )
    parser.add_argument("--split", choices=["validation", "test", "all"], default="test")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/evaluation"))
    parser.add_argument("--device", choices=["cpu", "cuda"])
    args = parser.parse_args()
    state = load_checkpoint(args.checkpoint, args.reconstruction)
    config = state["config"].copy()
    config["data_root"] = args.data_root
    validate_config(config)
    if args.split_file:
        split = json.loads(args.split_file.read_text(encoding="utf-8-sig"))
    elif "split" in state:
        split = state["split"]
    else:
        split = json.loads((args.checkpoint.parent / "split.json").read_text())
    validate_split(args.data_root, split)
    seed_all(config["seed"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    recon, seg = networks(config, device)
    recon.load_state_dict(state["reconstruction"], strict=True)
    seg.load_state_dict(state["segmentation"], strict=True)
    paths = split[args.split] if args.split != "all" else split["validation"] + split["test"]
    dataset = EvalDataset(args.data_root, paths, config["size"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = evaluate(
        recon,
        seg,
        loader(dataset, config),
        config,
        device,
        args.output_dir / f"{args.split}_predictions.json",
    )
    dump(args.output_dir / f"{args.split}_metrics.json", metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
