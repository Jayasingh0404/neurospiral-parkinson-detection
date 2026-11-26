"""
Zero-duplication dataset helper for Ultralytics YOLO11 image classification.
Reuses the existing local dataset by creating lightweight symlinks:
    yolo11/dataset_view/train -> ml/data/raw/Parkinson Disease detection/train
    yolo11/dataset_view/val   -> ml/data/raw/Parkinson Disease detection/valid
    yolo11/dataset_view/test  -> ml/data/raw/Parkinson Disease detection/test
"""

import os
from typing import Dict, List, Tuple
from yolo11.config import RAW_DATASET_DIR, DATASET_VIEW_DIR

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def prepare_symlink_dataset(
    raw_dir: str = RAW_DATASET_DIR, view_dir: str = DATASET_VIEW_DIR
) -> str:
    """
    Create a zero-copy symlink view of the existing local dataset so Ultralytics
    can locate 'train', 'val', and 'test' without duplicating any image files.
    """
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(
            f"Existing local dataset not found at: {raw_dir}"
        )

    os.makedirs(view_dir, exist_ok=True)
    split_map = {
        "train": os.path.join(raw_dir, "train"),
        "val": os.path.join(raw_dir, "valid"),
        "test": os.path.join(raw_dir, "test"),
    }

    for link_name, target_path in split_map.items():
        if not os.path.isdir(target_path):
            raise FileNotFoundError(f"Missing dataset split folder: {target_path}")
        link_path = os.path.join(view_dir, link_name)
        if os.path.islink(link_path):
            if os.readlink(link_path) != target_path:
                os.unlink(link_path)
                os.symlink(target_path, link_path)
        elif not os.path.exists(link_path):
            os.symlink(target_path, link_path)

    return view_dir


def list_split_images(
    dataset_root: str = RAW_DATASET_DIR,
) -> Dict[str, List[Tuple[str, str]]]:
    """
    Enumerate all valid image paths grouped by split ('train', 'valid', 'test').
    Returns: {split: [(image_path, class_label), ...]}
    """
    splits: Dict[str, List[Tuple[str, str]]] = {}
    for split_name in ["train", "valid", "test"]:
        split_dir = os.path.join(dataset_root, split_name)
        entries: List[Tuple[str, str]] = []
        if not os.path.isdir(split_dir):
            splits[split_name] = entries
            continue
        for class_name in sorted(os.listdir(split_dir)):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir) or class_name.startswith("."):
                continue
            for fname in sorted(os.listdir(class_dir)):
                if fname.startswith(".") or not fname.lower().endswith(VALID_EXTENSIONS):
                    continue
                entries.append((os.path.join(class_dir, fname), class_name.lower()))
        splits[split_name] = entries
    return splits
