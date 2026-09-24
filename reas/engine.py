"""Reproducible training/evaluation of the existing ReNia ReAS networks."""

import hashlib
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import cv2
import imgaug
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from .data import EvalDataset, TrainDataset, make_split
from .losses import SSIM, FocalLoss
from .models import AttentionSubNetwork, ReconstructiveSubNetwork

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = dict(
    data_root="data/mvtec/bottle",
    dtd_root="data/dtd/images",
    size=256,
    seed=42,
    batch_size=8,
    lr=5e-4,
    weight_decay=0.0,
    scheduler="multistep",
    epochs=4,
    recon_epochs=3,
    recon_lr=5e-4,
    recon_width=128,
    seg_width=32,
    ssim_weight=1.0,
    mse_weight=1.0,
    focal_weight=1.0,
    dice_weight=0.0,
    focal_gamma=2.0,
    attention_depth=4,
    attention_heads=8,
    patch_size=16,
    amp=True,
    num_workers=0,
    pool_kernel=21,
    split_seed=2026,
    grad_clip=0.0,
)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    imgaug.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(4)
    cv2.setNumThreads(0)


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def loader(dataset, c, training=False):
    generator = torch.Generator().manual_seed(c["seed"] + 1000)
    return DataLoader(
        dataset,
        batch_size=c["batch_size"] if training else 4,
        shuffle=training,
        num_workers=c["num_workers"],
        pin_memory=True,
        generator=generator,
    )


def networks(c, device):
    recon = ReconstructiveSubNetwork(base_width=c["recon_width"]).to(device)
    seg = AttentionSubNetwork(
        base_width=c["seg_width"],
        attention_depth=c["attention_depth"],
        attention_heads=c["attention_heads"],
        patch_size=c["patch_size"],
    ).to(device)
    return recon, seg


def schedule(optimizer, kind, epochs):
    if kind == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    if kind == "none":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    # Integer milestones: original fractional milestones never fired for four epochs.
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer, sorted(set([max(1, int(epochs * 0.8)), max(1, int(epochs * 0.9))])), gamma=0.2
    )


@torch.inference_mode()
def evaluate(recon, seg, data, c, device, predictions=None):
    recon.eval()
    seg.eval()
    image_y, image_p, pixel_y, pixel_p, rows = [], [], [], [], []
    for batch in data:
        x = batch["image"].to(device)
        with torch.autocast(device_type=device.type, enabled=c["amp"] and device.type == "cuda"):
            reconstruction = recon(x)
            logits = seg(x, reconstruction)
        probability = logits.float().softmax(1)[:, 1:2]
        scores = (
            F.avg_pool2d(probability, c["pool_kernel"], stride=1, padding=c["pool_kernel"] // 2)
            .flatten(1)
            .max(1)
            .values.cpu()
            .numpy()
        )
        labels = batch["label"].numpy()
        image_y.extend(labels)
        image_p.extend(scores)
        pixel_y.append(batch["mask"].numpy().reshape(-1))
        pixel_p.append(probability.cpu().numpy().reshape(-1))
        rows.extend(
            dict(path=p, label=int(y), score=float(s))
            for p, y, s in zip(batch["path"], labels, scores)
        )
    y, p = np.concatenate(pixel_y), np.concatenate(pixel_p)
    metrics = dict(
        image_auroc=float(roc_auc_score(image_y, image_p)),
        image_ap=float(average_precision_score(image_y, image_p)),
        pixel_auroc=float(roc_auc_score(y, p)),
        pixel_ap=float(average_precision_score(y, p)),
        n_images=len(image_y),
    )
    # Predeclared selection criterion balances image detection and pixel localization.
    metrics["selection_score"] = (metrics["image_auroc"] + metrics["pixel_ap"]) / 2
    if predictions:
        dump(predictions, rows)
    return metrics


def train(c, run_name, output_dir="runs", split_file=None, device_name=None):
    output_dir = Path(output_dir).resolve()
    run = output_dir / run_name
    run.mkdir(parents=True, exist_ok=True)
    if any(run.iterdir()):
        raise FileExistsError(f"Completed run exists: {run}; use a new run name")
    dump(run / "config.json", c)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    dump(
        run / "environment.json",
        dict(
            torch=torch.__version__,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            numpy=np.__version__,
        ),
    )
    if c["size"] != 256 or 16 % c["patch_size"]:
        raise ValueError("Use size=256 and a patch size dividing 16 for all five attention scales")
    seed_all(c["seed"])
    split = make_split(
        c["data_root"],
        Path(split_file) if split_file else ROOT / "configs" / "bottle_split.json",
        c["split_seed"],
    )
    dump(run / "split.json", split)
    if split["seed"] != c["split_seed"]:
        raise ValueError("Requested split seed differs from locked split")
    trainset = TrainDataset(c["data_root"], c["dtd_root"], c["size"])
    valset = EvalDataset(c["data_root"], split["validation"], c["size"])
    dump(
        run / "data_manifest.json",
        dict(
            train=trainset.image_paths,
            dtd=trainset.anomaly_source_paths,
            validation=split["validation"],
            test=split["test"],
        ),
    )
    print(
        f"RUN {run_name} | train={len(trainset)} validation={len(valset)} test={len(split['test'])} | {device}",
        flush=True,
    )
    recon, seg = networks(c, device)
    scaler = torch.amp.GradScaler("cuda", enabled=c["amp"] and device.type == "cuda")
    # Only configurations with identical reconstruction training share this artifact.
    recon_keys = [
        "data_root",
        "dtd_root",
        "size",
        "seed",
        "batch_size",
        "recon_epochs",
        "recon_lr",
        "recon_width",
        "ssim_weight",
        "mse_weight",
        "amp",
    ]
    recon_config = {k: c[k] for k in recon_keys}
    source_hash = hashlib.sha256(
        b"".join(p.read_bytes() for p in sorted((ROOT / "reas").rglob("*.py")))
    ).hexdigest()
    recon_config["source_hash"] = source_hash
    key = hashlib.sha256(json.dumps(recon_config, sort_keys=True).encode()).hexdigest()[:16]
    cache = output_dir / "_reconstruction"
    cache.mkdir(exist_ok=True)
    recon_path = cache / f"{key}.pt"
    started = time.time()
    if recon_path.exists():
        recon.load_state_dict(torch.load(recon_path, map_location=device, weights_only=True))
        print(f"Reuse reconstruction {key}", flush=True)
    else:
        seed_all(c["seed"])
        recon.train()
        optimizer = torch.optim.Adam(recon.parameters(), lr=c["recon_lr"])
        ssim_loss = SSIM().to(device)
        for epoch in range(1, c["recon_epochs"] + 1):
            losses = []
            epoch_start = time.time()
            for batch in loader(trainset, c, True):
                x, target = batch["augmented_image"].to(device), batch["image"].to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type, enabled=c["amp"] and device.type == "cuda"
                ):
                    rec = recon(x)
                # SSIM moments require float32 to avoid half-precision cancellation.
                loss = c["mse_weight"] * F.mse_loss(rec.float(), target) + c[
                    "ssim_weight"
                ] * ssim_loss(rec.float(), target)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite reconstruction loss")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                losses.append(loss.item())
            row = dict(
                stage="reconstruction",
                epoch=epoch,
                loss=float(np.mean(losses)),
                seconds=time.time() - epoch_start,
            )
            with (run / "history.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        torch.save(recon.state_dict(), recon_path)
        dump(cache / f"{key}.json", recon_config)
        del optimizer, ssim_loss
    recon.eval().requires_grad_(False)
    # Reset segmentation initialization/augmentation independently of reconstruction-cache use.
    seed_all(c["seed"] + 1000)
    del seg
    seg = AttentionSubNetwork(
        base_width=c["seg_width"],
        attention_depth=c["attention_depth"],
        attention_heads=c["attention_heads"],
        patch_size=c["patch_size"],
    ).to(device)
    optimizer = torch.optim.Adam(seg.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    scheduler = schedule(optimizer, c["scheduler"], c["epochs"])
    focal = FocalLoss(gamma=c["focal_gamma"])
    best = -float("inf")
    best_epoch = None
    train_loader = loader(trainset, c, True)
    val_loader = loader(valset, c)
    for epoch in range(1, c["epochs"] + 1):
        seg.train()
        epoch_start, losses = time.time(), []
        lr = optimizer.param_groups[0]["lr"]
        for batch in train_loader:
            x, target = batch["augmented_image"].to(device), batch["anomaly_mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, enabled=c["amp"] and device.type == "cuda"
            ):
                with torch.no_grad():
                    rec = recon(x)
                logits = seg(x, rec)
            p = logits.float().softmax(1)
            focal_value = focal(p, target)
            p1 = p[:, 1:2]
            dice_value = (
                1
                - (
                    (2 * (p1 * target).sum((1, 2, 3)) + 1)
                    / (p1.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
                ).mean()
            )
            loss = c["focal_weight"] * focal_value + c["dice_weight"] * dice_value
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite segmentation loss")
            scaler.scale(loss).backward()
            if c["grad_clip"]:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(seg.parameters(), c["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
        scheduler.step()
        metrics = evaluate(recon, seg, val_loader, c, device)
        row = dict(
            stage="segmentation",
            epoch=epoch,
            lr=lr,
            loss=float(np.mean(losses)),
            seconds=time.time() - epoch_start,
            validation=metrics,
        )
        with (run / "history.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if metrics["selection_score"] > best:
            best, best_epoch, best_metrics = metrics["selection_score"], epoch, metrics
            torch.save(
                dict(
                    segmentation=seg.state_dict(),
                    reconstruction_path=Path(os.path.relpath(recon_path, run)).as_posix(),
                    config=c,
                    epoch=epoch,
                ),
                run / "best.pt",
            )
    summary = dict(
        run=run_name,
        config=c,
        best_epoch=best_epoch,
        validation=best_metrics,
        final_validation=metrics,
        seconds=time.time() - started,
        reconstruction_path=Path(os.path.relpath(recon_path, run)).as_posix(),
        selection_criterion="(validation image AUROC + validation pixel AP) / 2",
    )
    dump(run / "summary.json", summary)
    return summary
