"""
Centralized configuration for the isolated Ultralytics YOLO11 classification module.
Auto-detects hardware acceleration (CUDA -> Apple MPS -> CPU) and reuses the existing local dataset.
"""

import os
from dataclasses import dataclass, asdict
from typing import Dict, Any
import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YOLO11_ROOT = os.path.dirname(os.path.abspath(__file__))

# Existing local dataset path (NO external download / NO Roboflow API)
RAW_DATASET_DIR = os.path.join(
    PROJECT_ROOT, "ml", "data", "raw", "Parkinson Disease detection"
)

# Zero-copy symlink view mapping train -> train, val -> valid, test -> test for Ultralytics
DATASET_VIEW_DIR = os.path.join(YOLO11_ROOT, "dataset_view")

# Existing model evaluation results for side-by-side benchmarking (read-only)
EXISTING_COMPARISON_CSV = os.path.join(
    PROJECT_ROOT, "ml", "artifacts", "model_comparison.csv"
)
EXISTING_EVALUATION_JSON = os.path.join(
    PROJECT_ROOT, "ml", "artifacts", "models_evaluation.json"
)

# Isolated YOLO11 output directories
WEIGHTS_CACHE_DIR = os.path.join(YOLO11_ROOT, "weights")
OUTPUTS_DIR = os.path.join(YOLO11_ROOT, "outputs")
EDA_OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "eda")
RUNS_OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "runs")
SAVED_WEIGHTS_DIR = os.path.join(OUTPUTS_DIR, "weights")
EVAL_OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "evaluation")
METRICS_OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "metrics")


def detect_device() -> str:
    """Automatically select the best available compute device (CUDA -> MPS -> CPU)."""
    if torch.cuda.is_available():
        return "0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class YOLO11Config:
    model_name: str = "yolo11s-cls.pt"
    task: str = "classify"
    imgsz: int = 224
    epochs: int = 30
    batch_size: int = 16
    patience: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "AdamW"
    seed: int = 42
    workers: int = 0  # Safe default across macOS / sandbox / CPU
    # Conservative, drawing-safe training augmentations
    degrees: float = 10.0
    translate: float = 0.05
    scale: float = 0.10
    fliplr: float = 0.0  # Disabled to preserve directional motor tremor signatures
    flipud: float = 0.0
    erasing: float = 0.0
    auto_augment: str = ""
    run_name: str = "parkinson_yolo11s_cls"
    positive_class: str = "parkinson"
    negative_class: str = "healthy"
    decision_threshold: float = 0.50

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["resolved_device"] = detect_device()
        data["raw_dataset_dir"] = RAW_DATASET_DIR
        data["dataset_view_dir"] = DATASET_VIEW_DIR
        return data


def ensure_directories() -> None:
    """Create all isolated YOLO11 output directories."""
    for d in [
        WEIGHTS_CACHE_DIR,
        OUTPUTS_DIR,
        EDA_OUTPUT_DIR,
        RUNS_OUTPUT_DIR,
        SAVED_WEIGHTS_DIR,
        EVAL_OUTPUT_DIR,
        METRICS_OUTPUT_DIR,
    ]:
        os.makedirs(d, exist_ok=True)
