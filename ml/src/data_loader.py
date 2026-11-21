"""
Data Loader and Batch Generator for Parkinson's Disease Drawing Classification.
Supports:
1. Original Image-Level Split (Train: 133, Val: 31, Test: 40)
2. Subject-Independent Group Folds (Stratified Group K-Fold)
3. Model-specific batching and real-time training augmentation
"""

import os
from typing import Tuple, List, Dict, Generator, Optional
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from ml.src.preprocessing import prepare_image, letterbox_image, preprocess_for_model, augment_image


class DrawingDataset:
    def __init__(self, manifest_csv: str = "/Users/jayasinghrajput/PARKINSON/ml/artifacts/dataset_manifest.csv"):
        self.manifest_path = manifest_csv
        if not os.path.exists(manifest_csv):
            raise FileNotFoundError(f"Manifest not found at {manifest_csv}. Run data audit first.")
        self.df = pd.read_csv(manifest_csv)
        self.label_map = {"healthy": 0, "parkinson": 1}
        self.classes = ["healthy", "parkinson"]
        
    def get_image_level_splits(self) -> Dict[str, pd.DataFrame]:
        """Returns standard train, valid, test subsets matching Roboflow directory splits."""
        return {
            "train": self.df[self.df["split"] == "train"].copy(),
            "valid": self.df[self.df["split"] == "valid"].copy(),
            "test": self.df[self.df["split"] == "test"].copy()
        }
        
    def get_subject_independent_folds(self, n_splits: int = 5, seed: int = 42) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Creates n-fold cross validation partitions with ZERO subject overlap across folds.
        Guarantees that all drawing exams from a given patient appear exclusively in either train or val.
        """
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        X = self.df.index.values
        y = self.df["label"].map(self.label_map).values
        groups = self.df["subject_id"].values
        
        folds = []
        for train_idx, val_idx in sgkf.split(X, y, groups):
            train_sub = self.df.iloc[train_idx].copy()
            val_sub = self.df.iloc[val_idx].copy()
            folds.append((train_sub, val_sub))
        return folds
    
    @staticmethod
    def load_split_arrays(sub_df: pd.DataFrame,
                          model_type: str = "cnn",
                          augment: bool = False,
                          target_size: Tuple[int, int] = (224, 224)) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Loads all images for a given DataFrame partition into memory as preprocessed NumPy arrays.
        Ideal for fast, reproducible training of 204 images.
        """
        images = []
        labels = []
        paths = []
        label_map = {"healthy": 0, "parkinson": 1}
        
        for _, row in sub_df.iterrows():
            img_path = row["path"]
            with Image.open(img_path) as img:
                rgb_img = img.convert("RGB")
                
            standardized = letterbox_image(rgb_img, target_size=target_size)
            if augment:
                standardized = augment_image(standardized)
                
            arr = np.asarray(standardized, dtype=np.float32)
            norm_arr = preprocess_for_model(arr, model_type=model_type)
            
            images.append(norm_arr)
            labels.append(label_map[row["label"]])
            paths.append(img_path)
            
        X = np.stack(images, axis=0)
        y = np.array(labels, dtype=np.float32)
        return X, y, paths

