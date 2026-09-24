"""Train reconstruction and segmentation: python -m reas.train --help."""

import argparse
from pathlib import Path

from .config import load_config, validate_run_name
from .engine import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/extended_best.json"))
    parser.add_argument("--run", required=True)
    parser.add_argument(
        "--data-root", help="One MVTec category directory, containing train/test/ground_truth"
    )
    parser.add_argument(
        "--dtd-root", help="DTD images directory, containing texture subdirectories"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--split-file", type=Path, default=Path("configs/bottle_split.json"))
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--recon-epochs", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    validate_run_name(args.run)
    overrides = {
        k: getattr(args, k) for k in ["data_root", "dtd_root", "epochs", "recon_epochs", "seed"]
    }
    config = load_config(args.config, overrides)
    train(config, args.run, args.output_dir, args.split_file, args.device)


if __name__ == "__main__":
    main()
