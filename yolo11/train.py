"""
Ultralytics YOLO11 Classification Training Pipeline.
Trains yolo11s-cls with transfer learning on the existing local Parkinson's dataset
using a zero-copy symlink view, validation after every epoch, early stopping, and checkpointing.
"""

import os
import json
import time
import shutil
import argparse
from typing import Dict, Any
from ultralytics import YOLO

from yolo11.config import (
    YOLO11Config,
    WEIGHTS_CACHE_DIR,
    RUNS_OUTPUT_DIR,
    SAVED_WEIGHTS_DIR,
    METRICS_OUTPUT_DIR,
    detect_device,
    ensure_directories,
)
from yolo11.dataset_utils import prepare_symlink_dataset


def train_yolo11(cfg: YOLO11Config = None) -> Dict[str, Any]:
    """Train YOLO11 classification model and persist best/last checkpoints and metadata."""
    ensure_directories()
    if cfg is None:
        cfg = YOLO11Config()

    # 1. Prepare zero-copy symlink dataset view (train, val -> valid, test)
    dataset_view = prepare_symlink_dataset()

    # 2. Resolve pretrained weights path (local cache preferred)
    local_weights = os.path.join(WEIGHTS_CACHE_DIR, cfg.model_name)
    weights_source = local_weights if os.path.exists(local_weights) else cfg.model_name

    device = detect_device()
    print("=" * 68)
    print(f"YOLO11 CLASSIFICATION TRAINING ({cfg.model_name})")
    print(f"Device: {device} | Epochs: {cfg.epochs} | Batch: {cfg.batch_size} | ImgSz: {cfg.imgsz}")
    print(f"Dataset (symlink view): {dataset_view}")
    print("=" * 68)

    # 3. Initialize model with pretrained ImageNet weights
    model = YOLO(weights_source)

    # 4. Run transfer learning training (train + val splits only; test split untouched)
    t0 = time.time()
    model.train(
        data=dataset_view,
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch_size,
        patience=cfg.patience,
        optimizer=cfg.optimizer,
        lr0=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        seed=cfg.seed,
        device=device,
        workers=cfg.workers,
        project=RUNS_OUTPUT_DIR,
        name=cfg.run_name,
        exist_ok=True,
        pretrained=True,
        val=True,
        plots=True,
        save=True,
        # Drawing-safe augmentations (applied to training set only by Ultralytics)
        degrees=cfg.degrees,
        translate=cfg.translate,
        scale=cfg.scale,
        fliplr=cfg.fliplr,
        flipud=cfg.flipud,
        erasing=cfg.erasing,
        auto_augment=cfg.auto_augment,
        verbose=True,
    )
    train_time_sec = round(time.time() - t0, 2)

    # 5. Copy best and last checkpoints into yolo11/outputs/weights/
    run_dir = os.path.join(RUNS_OUTPUT_DIR, cfg.run_name)
    src_best = os.path.join(run_dir, "weights", "best.pt")
    src_last = os.path.join(run_dir, "weights", "last.pt")
    dst_best = os.path.join(SAVED_WEIGHTS_DIR, "best.pt")
    dst_last = os.path.join(SAVED_WEIGHTS_DIR, "last.pt")

    if os.path.exists(src_best):
        shutil.copy2(src_best, dst_best)
    if os.path.exists(src_last):
        shutil.copy2(src_last, dst_last)

    # 6. Collect model parameter count & file size
    best_model = YOLO(dst_best if os.path.exists(dst_best) else src_last)
    total_params = int(sum(p.numel() for p in best_model.model.parameters()))
    trainable_params = int(sum(p.numel() for p in best_model.model.parameters() if p.requires_grad))
    model_size_mb = round(os.path.getsize(dst_best) / (1024 * 1024), 2) if os.path.exists(dst_best) else 0.0

    train_meta = {
        "config": cfg.to_dict(),
        "device_used": device,
        "train_time_sec": train_time_sec,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": model_size_mb,
        "run_dir": run_dir,
        "best_weights": dst_best,
        "last_weights": dst_last,
        "results_csv": os.path.join(run_dir, "results.csv"),
    }

    with open(os.path.join(METRICS_OUTPUT_DIR, "yolo11_config.json"), "w") as f:
        json.dump(train_meta, f, indent=2)

    print(f"\n[Train] Completed in {train_time_sec:.2f}s | Params: {total_params:,} | Size: {model_size_mb} MB")
    print(f"[Train] Saved best weights -> {dst_best}")
    print(f"[Train] Saved last weights -> {dst_last}")
    return train_meta


def main():
    parser = argparse.ArgumentParser(description="Train YOLO11 Classification Model")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    cfg = YOLO11Config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        learning_rate=args.lr,
        patience=args.patience,
    )
    train_yolo11(cfg)


if __name__ == "__main__":
    main()
