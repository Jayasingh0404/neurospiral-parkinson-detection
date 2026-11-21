"""
Script to execute complete Dataset Audit and Exploratory Data Analysis (EDA).
Generates all figures required by Data_Preprocessing.pdf and the Capstone specification.
"""

import os
import json
import pandas as pd
from ml.src.data_audit import audit_dataset
from ml.src.visualization import (
    plot_class_distribution,
    plot_image_dimensions,
    plot_sample_grid,
    plot_pixel_intensity_distribution,
    plot_subject_leakage_map
)

def main():
    raw_dir = "/Users/jayasinghrajput/PARKINSON/ml/data/raw/Parkinson Disease detection"
    eda_output_dir = "/Users/jayasinghrajput/PARKINSON/ml/outputs/eda"
    artifacts_dir = "/Users/jayasinghrajput/PARKINSON/ml/artifacts"
    
    os.makedirs(eda_output_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)
    
    print("[1/5] Running comprehensive dataset audit...")
    df, stats = audit_dataset(raw_dir)
    
    # Save processed dataframe & audit JSON
    df_path = os.path.join(artifacts_dir, "dataset_manifest.csv")
    df.to_csv(df_path, index=False)
    print(f"Saved dataset manifest ({len(df)} rows) to: {df_path}")
    
    # Convert numpy types in stats for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, dict):
            return {str(k): convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        elif hasattr(obj, "item"):
            return obj.item()
        return obj

    clean_stats = convert_for_json(stats)
    audit_json_path = os.path.join(artifacts_dir, "dataset_audit.json")
    with open(audit_json_path, "w") as f:
        json.dump(clean_stats, f, indent=2)
    print(f"Saved dataset audit report to: {audit_json_path}")
    
    # Generate EDA plots
    print("[2/5] Generating class and split distribution plots...")
    plot_class_distribution(clean_stats, os.path.join(eda_output_dir, "01_class_distribution.png"))
    
    print("[3/5] Generating image dimensions and aspect ratio plots...")
    plot_image_dimensions(df, os.path.join(eda_output_dir, "02_image_dimensions.png"))
    
    print("[4/5] Generating sample grid...")
    plot_sample_grid(df, os.path.join(eda_output_dir, "03_sample_grid.png"), n_samples=8)
    
    print("[5/5] Generating pixel intensity distributions and channel stats...")
    plot_pixel_intensity_distribution(df, os.path.join(eda_output_dir, "04_pixel_intensity_distribution.png"))
    
    print("[Bonus] Generating Subject Cross-Split Leakage audit map...")
    plot_subject_leakage_map(df, os.path.join(eda_output_dir, "05_subject_leakage_map.png"))
    
    print("EDA Generation complete! All files saved in:", eda_output_dir)

if __name__ == "__main__":
    main()

