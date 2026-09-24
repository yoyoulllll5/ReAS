"""Run the recorded search protocol sequentially on a single device."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from .config import load_config

TRIALS = [
    ("baseline", {}),
    ("lr_low", {"lr": 1e-4}),
    ("lr_high", {"lr": 1e-3}),
    ("batch4", {"batch_size": 4}),
    ("weight_decay", {"weight_decay": 1e-4}),
    ("cosine", {"scheduler": "cosine"}),
    ("dice", {"dice_weight": 0.5}),
    ("attention_depth2", {"attention_depth": 2}),
    ("attention_patch8", {"patch_size": 8, "attention_heads": 4}),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dtd-root", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/search"))
    parser.add_argument("--split-file", type=Path, default=Path("configs/bottle_split.json"))
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    configs = output / "_configs"
    configs.mkdir(exist_ok=True)
    base = load_config(overrides=dict(data_root=args.data_root, dtd_root=args.dtd_root))

    def execute(name, config):
        run = output / name
        if (run / "summary.json").is_file():
            result = json.loads((run / "summary.json").read_text())
            if result["config"] != config:
                raise ValueError(
                    f"{name} has a different saved config; choose another output directory"
                )
            saved_split = json.loads((run / "split.json").read_text())
            requested_split = json.loads(args.split_file.read_text(encoding="utf-8-sig"))
            if saved_split != requested_split:
                raise ValueError(f"{name} used a different split; choose another output directory")
            return result
        config_path = configs / f"{name}.json"
        config_path.write_text(json.dumps(config, indent=2))
        command = [
            sys.executable,
            "-u",
            "-m",
            "reas.train",
            "--config",
            str(config_path),
            "--run",
            name,
            "--output-dir",
            str(output),
            "--split-file",
            str(args.split_file),
        ]
        # Logs live outside the run because the trainer requires an empty run directory.
        with (output / f"{name}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads((run / "summary.json").read_text())
        print(f"Completed {name}: {result['validation']}", flush=True)
        return result

    summaries = [execute(name, {**base, **changes}) for name, changes in TRIALS]
    winner = max(summaries, key=lambda s: s["validation"]["selection_score"])
    summaries.append(
        execute("extended_best", {**winner["config"], "epochs": 12, "scheduler": "cosine"})
    )
    best = max(summaries, key=lambda s: s["validation"]["selection_score"])
    (output / "selection.json").write_text(
        json.dumps(
            dict(best_run=best["run"], config=best["config"], validation=best["validation"]),
            indent=2,
        )
    )
    for seed in [43, 44]:
        summaries.append(execute(f"best_seed{seed}", {**best["config"], "seed": seed}))
    rows = [
        dict(run=s["run"], best_epoch=s["best_epoch"], **s["config"], **s["validation"])
        for s in summaries
    ]
    with (output / "experiments.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for name in dict.fromkeys(["baseline", best["run"], "best_seed43", "best_seed44"]):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "reas.evaluate",
                "--checkpoint",
                str(output / name / "best.pt"),
                "--data-root",
                args.data_root,
                "--output-dir",
                str(output / name),
                "--split",
                "test",
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
