"""
Data Audit and Quality Assurance Module for Parkinson's Disease Spiral Detection.
Implements:
- Dataset discovery and structural inspection
- Class balance and split ratio verification
- Image dimensional, color channel, and format analysis
- Corrupt image detection
- Exact MD5 duplicate detection and perceptual hash (dHash/pHash) analysis
- Subject-level extraction and Cross-Split Data Leakage analysis
- Subject-Independent Stratified Grouping for rigorous validation
"""

import os
import re
import hashlib
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from PIL import Image


def compute_md5(file_path: str) -> str:
    """Compute exact MD5 byte checksum of a file."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_phash(file_path: str, hash_size: int = 8) -> str:
    """Compute difference hash (dHash) for perceptual image comparison."""
    with Image.open(file_path) as img:
        img_gray = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        pixels = np.asarray(img_gray)
        # Compute differences between adjacent columns
        diff = pixels[:, 1:] > pixels[:, :-1]
        return hex(int("".join(["1" if b else "0" for b in diff.flatten()]), 2))[2:].zfill(hash_size * hash_size // 4)


def extract_subject_info(filename: str, fallback_class: str) -> Dict[str, str]:
    """
    Extract Subject ID, Exam Type, and Trial ID from filename.
    Standard HandPD / NewHandPD pattern:
    e.g., 'V08HE03_png.rf.42f3b33b...' ->
      Subject: 'V08'
      Class Code: 'H' (Healthy) or 'P' (Parkinson)
      Exam Code: 'E' (Spiral / Espiral) or 'O' (Meander / Outro)
      Trial: '03'
    """
    pattern = re.compile(r"^(V\d{2})([HP])([EO])(\d+)")
    m = pattern.match(filename)
    if m:
        subj = m.group(1)
        cls_code = "healthy" if m.group(2) == "H" else "parkinson"
        exam_type = "spiral" if m.group(3) == "E" else "meander"
        trial = m.group(4)
    else:
        # Fallback for slight variations like V03HE2 or V55HO13
        alt_pattern = re.compile(r"^(V\d{2})([HP])([A-Za-z0-9]+)")
        m2 = alt_pattern.match(filename)
        if m2:
            subj = m2.group(1)
            cls_code = "healthy" if m2.group(2) == "H" else "parkinson"
            task_str = m2.group(3)
            exam_type = "spiral" if "E" in task_str else ("meander" if "O" in task_str else "drawing")
            trial = "".join([c for c in task_str if c.isdigit()])
        else:
            subj = filename.split("_")[0][:3]
            cls_code = fallback_class
            exam_type = "spiral"
            trial = "1"
    
    unique_subject_id = f"{cls_code[:1].upper()}_{subj}"
    return {
        "subject_prefix": subj,
        "subject_id": unique_subject_id,
        "derived_class": cls_code,
        "exam_type": exam_type,
        "trial": trial,
    }


def audit_dataset(raw_dir: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Scans raw directory, validates all images, detects duplicates and cross-split leakage.
    Returns cleaned DataFrame and comprehensive metadata dictionary.
    """
    splits = ["train", "valid", "test"]
    classes = ["healthy", "parkinson"]
    
    records = []
    corrupt_files = []
    
    for split in splits:
        for cls_name in classes:
            folder = os.path.join(raw_dir, split, cls_name)
            if not os.path.isdir(folder):
                continue
            for fname in sorted(os.listdir(folder)):
                if fname.startswith("."):
                    continue
                file_path = os.path.join(folder, fname)
                
                # Check readability and corruption
                try:
                    with Image.open(file_path) as img:
                        img.verify()
                    # Reopen after verify() to inspect properties
                    with Image.open(file_path) as img:
                        width, height = img.size
                        channels = len(img.getbands())
                        img_format = img.format
                        mode = img.mode
                except Exception as e:
                    corrupt_files.append({"path": file_path, "error": str(e)})
                    continue
                
                md5_hash = compute_md5(file_path)
                phash_val = compute_phash(file_path)
                subj_meta = extract_subject_info(fname, cls_name)
                
                records.append({
                    "path": file_path,
                    "filename": fname,
                    "split": split,
                    "label": cls_name,
                    "width": width,
                    "height": height,
                    "aspect_ratio": round(width / height, 3),
                    "channels": channels,
                    "mode": mode,
                    "format": img_format,
                    "md5": md5_hash,
                    "phash": phash_val,
                    "subject_id": subj_meta["subject_id"],
                    "subject_prefix": subj_meta["subject_prefix"],
                    "exam_type": subj_meta["exam_type"],
                    "trial": subj_meta["trial"],
                })
                
    df = pd.DataFrame(records)
    
    # 1. Duplicates Detection
    md5_counts = df["md5"].value_counts()
    exact_duplicates = md5_counts[md5_counts > 1].to_dict()
    
    phash_counts = df["phash"].value_counts()
    near_duplicates = phash_counts[phash_counts > 1].to_dict()
    
    # 2. Cross-Split Subject Leakage Analysis
    subject_splits = df.groupby("subject_id")["split"].unique().to_dict()
    leaked_subjects = {subj: list(splits_arr) for subj, splits_arr in subject_splits.items() if len(splits_arr) > 1}
    clean_subjects = {subj: list(splits_arr) for subj, splits_arr in subject_splits.items() if len(splits_arr) == 1}
    
    # 3. Overall Statistics
    stats = {
        "total_images": len(df),
        "corrupt_images_count": len(corrupt_files),
        "corrupt_images": corrupt_files,
        "class_distribution": df["label"].value_counts().to_dict(),
        "split_distribution": df["split"].value_counts().to_dict(),
        "split_by_class": df.groupby(["split", "label"]).size().unstack(fill_value=0).to_dict(),
        "unique_resolutions": df.groupby(["width", "height"]).size().to_dict(),
        "channel_modes": df["mode"].value_counts().to_dict(),
        "formats": df["format"].value_counts().to_dict(),
        "exact_duplicates_count": len(exact_duplicates),
        "near_duplicates_count": len(near_duplicates),
        "total_unique_subjects": len(subject_splits),
        "subjects_with_leakage_count": len(leaked_subjects),
        "leakage_ratio": round(len(leaked_subjects) / len(subject_splits), 4) if subject_splits else 0,
        "leaked_subjects": leaked_subjects,
        "clean_subjects": clean_subjects,
        "exam_types": df["exam_type"].value_counts().to_dict(),
    }
    
    return df, stats

