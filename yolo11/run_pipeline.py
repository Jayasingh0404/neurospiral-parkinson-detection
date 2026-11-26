"""
End-to-End Orchestrator for the Isolated YOLO11 Pipeline:
    1. Exploratory Data Analysis (EDA)
    2. YOLO11s-cls Transfer Learning Training & Validation
    3. Independent Test Evaluation & Existing Model Comparison
    4. Standalone Inference Verification on a Real Test Image
"""

import os
import json
from yolo11.eda import run_eda
from yolo11.train import train_yolo11
from yolo11.evaluate import evaluate_yolo11
from yolo11.predict import predict_image
from yolo11.config import RAW_DATASET_DIR, YOLO11Config
from yolo11.dataset_utils import list_split_images


def main():
    print("\n>>> STEP 1/4: Running Exploratory Data Analysis (EDA)...")
    run_eda()

    print("\n>>> STEP 2/4: Training YOLO11s-cls Transfer Learning Model...")
    cfg = YOLO11Config()
    train_yolo11(cfg)

    print("\n>>> STEP 3/4: Evaluating on Validation & Independent Test Sets + Comparing Models...")
    evaluate_yolo11()

    print("\n>>> STEP 4/4: Running Standalone Inference Smoke Test...")
    test_items = list_split_images(RAW_DATASET_DIR)["test"]
    sample_img, sample_label = test_items[0]
    pred = predict_image(sample_img)
    print(f"Sample image ({sample_label}): {os.path.basename(sample_img)}")
    print(json.dumps(pred, indent=2))


if __name__ == "__main__":
    main()
