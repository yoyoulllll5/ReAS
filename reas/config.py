"""Configuration loading and early validation of supported experiment settings."""

import json
import math
from pathlib import Path

from .engine import DEFAULT


def load_config(path=None, overrides=None):
    config = DEFAULT.copy()
    if path:
        values = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if not isinstance(values, dict):
            raise ValueError("The configuration must be a JSON object")
        unknown = values.keys() - config.keys()
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        config.update(values)
    config.update({k: v for k, v in (overrides or {}).items() if v is not None})
    validate_config(config)
    return config


def validate_config(config):
    for key in [
        "batch_size",
        "epochs",
        "recon_epochs",
        "recon_width",
        "seg_width",
        "attention_depth",
        "attention_heads",
        "patch_size",
        "pool_kernel",
    ]:
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in [
        "lr",
        "recon_lr",
        "weight_decay",
        "ssim_weight",
        "mse_weight",
        "focal_weight",
        "dice_weight",
        "focal_gamma",
        "grad_clip",
    ]:
        value = config[key]
        if not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if config["lr"] == 0 or config["recon_lr"] == 0:
        raise ValueError("Learning rates must be positive")
    if config["mse_weight"] + config["ssim_weight"] == 0:
        raise ValueError("At least one reconstruction loss weight must be positive")
    if config["focal_weight"] + config["dice_weight"] == 0:
        raise ValueError("At least one segmentation loss weight must be positive")
    if config["size"] != 256 or 16 % config["patch_size"]:
        raise ValueError("Use size=256 and a patch_size dividing 16")
    if any(dim % config["attention_heads"] for dim in [64, 96, 128]):
        raise ValueError("attention_heads must divide 64, 96 and 128")
    if config["pool_kernel"] % 2 != 1:
        raise ValueError("pool_kernel must be odd")
    if config["scheduler"] not in ["multistep", "cosine", "none"]:
        raise ValueError("scheduler must be multistep, cosine or none")
    if config["num_workers"] != 0:
        raise ValueError("This reproduction protocol requires num_workers=0 for imgaug RNG order")
    if type(config["amp"]) is not bool:
        raise ValueError("amp must be a JSON boolean")
    for key in ["seed", "split_seed"]:
        if type(config[key]) is not int or not 0 <= config[key] < 2**32 - 1000:
            raise ValueError(f"{key} must be an integer in [0, 2**32-1000)")


def validate_run_name(name):
    if not name or name in [".", ".."] or "/" in name or "\\" in name or ":" in name:
        raise ValueError("run must be a single directory name")
