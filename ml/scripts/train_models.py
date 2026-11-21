"""
Complete Model Training, Evaluation, and Comparison Pipeline.
Executes experiments for:
1. Custom CNN (from scratch)
2. VGG16 Transfer Learning (2-phase)
3. ResNet50 Transfer Learning (2-phase)

Performs:
- Dual Split Evaluation:
  a) Original Image-Level Split (Train: 133, Val: 31, Test: 40)
  b) Subject-Independent Stratified Group 5-Fold Validation
- Full Metrics & Calibration Extraction
- Artifact Serialization (.keras, JSON, CSV)
- Comparative Visualization Plotting
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import callbacks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ml.src.data_loader import DrawingDataset
from ml.src.models import (
    build_custom_cnn,
    build_vgg16_transfer,
    configure_vgg16_finetuning,
    build_resnet50_transfer,
    configure_resnet50_finetuning
)
from ml.src.metrics import compute_comprehensive_metrics
from ml.src.gradcam import generate_gradcam_heatmap, overlay_gradcam


def plot_evaluation_charts(results: dict, history_dict: dict, output_dir: str):
    """Generates all comparative academic evaluation figures."""
    os.makedirs(output_dir, exist_ok=True)
    model_names = list(results.keys())
    
    # 1. Grouped Performance Bar Chart
    metrics_to_plot = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc"]
    x = np.arange(len(metrics_to_plot))
    width = 0.25
    colors = ["#3B82F6", "#10B981", "#8B5CF6"]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, m_name in enumerate(model_names):
        vals = [results[m_name][metric] for metric in metrics_to_plot]
        ax.bar(x + (idx - 1) * width, vals, width, label=m_name.upper(), color=colors[idx], edgecolor="black")
        for i, v in enumerate(vals):
            ax.text(x[i] + (idx - 1) * width, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8, rotation=45)
            
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics_to_plot], fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Architecture Performance Comparison on Held-Out Test Set", fontsize=14, fontweight="bold", pad=12)
    ax.legend(frameon=True, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_grouped_metrics_bar.png"), dpi=300)
    plt.close()
    
    # 2. Confusion Matrices Side-by-Side
    fig, axes = plt.subplots(1, len(model_names), figsize=(5 * len(model_names), 4.5))
    if len(model_names) == 1:
        axes = [axes]
    for idx, m_name in enumerate(model_names):
        ax = axes[idx]
        cm = np.array(results[m_name]["confusion_matrix"]["matrix"])
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(f"{m_name.upper()}\nAcc: {results[m_name]['accuracy']:.1%}", fontsize=12, fontweight="bold")
        tick_marks = np.arange(2)
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(["Healthy", "Parkinson"], fontsize=10)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(["Healthy", "Parkinson"], fontsize=10)
        
        # Text values
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], "d"),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=14, fontweight="bold")
        ax.set_ylabel("True Category", fontsize=11)
        ax.set_xlabel("Predicted Category", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_confusion_matrices.png"), dpi=300)
    plt.close()
    
    # 3. ROC Curves Combined
    fig, ax = plt.subplots(figsize=(8, 7))
    for idx, m_name in enumerate(model_names):
        fpr = results[m_name]["roc_curve"]["fpr"]
        tpr = results[m_name]["roc_curve"]["tpr"]
        auc = results[m_name]["roc_auc"]
        ax.plot(fpr, tpr, label=f"{m_name.upper()} (AUC = {auc:.3f})", color=colors[idx], linewidth=2.5)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Chance Baseline (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax.set_title("Receiver Operating Characteristic (ROC) Comparison", fontsize=13, fontweight="bold", pad=12)
    ax.legend(frameon=True, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_roc_curves.png"), dpi=300)
    plt.close()
    
    # 4. Precision-Recall Curves
    fig, ax = plt.subplots(figsize=(8, 7))
    for idx, m_name in enumerate(model_names):
        rec = results[m_name]["pr_curve"]["recall"]
        prec = results[m_name]["pr_curve"]["precision"]
        pr_auc = results[m_name]["pr_auc"]
        ax.plot(rec, prec, label=f"{m_name.upper()} (PR-AUC = {pr_auc:.3f})", color=colors[idx], linewidth=2.5)
    ax.set_xlabel("Recall (Sensitivity)", fontsize=11)
    ax.set_ylabel("Precision (PPV)", fontsize=11)
    ax.set_title("Precision-Recall (PR) Curve Comparison", fontsize=13, fontweight="bold", pad=12)
    ax.legend(frameon=True, loc="lower left")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_precision_recall_curves.png"), dpi=300)
    plt.close()
    
    # 5. Training Loss and Accuracy Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for idx, m_name in enumerate(model_names):
        hist = history_dict.get(m_name, {})
        if "loss" in hist and "val_loss" in hist:
            epochs = range(1, len(hist["loss"]) + 1)
            ax1.plot(epochs, hist["loss"], label=f"{m_name} Train", color=colors[idx], linestyle="--")
            ax1.plot(epochs, hist["val_loss"], label=f"{m_name} Val", color=colors[idx], linewidth=2)
        if "accuracy" in hist and "val_accuracy" in hist:
            epochs = range(1, len(hist["accuracy"]) + 1)
            ax2.plot(epochs, hist["accuracy"], label=f"{m_name} Train", color=colors[idx], linestyle="--")
            ax2.plot(epochs, hist["val_accuracy"], label=f"{m_name} Val", color=colors[idx], linewidth=2)
            
    ax1.set_title("Training & Validation Loss", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Binary Crossentropy Loss", fontsize=11)
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.4)
    
    ax2.set_title("Training & Validation Accuracy", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "05_training_curves.png"), dpi=300)
    plt.close()


def train_and_evaluate_all():
    artifacts_dir = "/Users/jayasinghrajput/PARKINSON/ml/artifacts"
    eval_output_dir = "/Users/jayasinghrajput/PARKINSON/ml/outputs/evaluation"
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(eval_output_dir, exist_ok=True)
    
    # Set seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    print("==================================================================")
    print("PARKINSON'S DISEASE DRAWING CLASSIFICATION: DEEP LEARNING PIPELINE")
    print("==================================================================")
    
    ds = DrawingDataset()
    splits = ds.get_image_level_splits()
    
    train_df = splits["train"]
    valid_df = splits["valid"]
    test_df = splits["test"]
    
    print(f"Loaded image-level splits: Train={len(train_df)}, Val={len(valid_df)}, Test={len(test_df)}")
    
    # Class weights to handle any minor imbalance
    n_healthy = sum(train_df["label"] == "healthy")
    n_parkinson = sum(train_df["label"] == "parkinson")
    class_weights = {
        0: (len(train_df)) / (2.0 * n_healthy),
        1: (len(train_df)) / (2.0 * n_parkinson)
    }
    print(f"Computed Class Weights: Healthy (0): {class_weights[0]:.3f}, Parkinson (1): {class_weights[1]:.3f}")
    
    results = {}
    history_dict = {}
    
    # -------------------------------------------------------------
    # 1. MODEL 1: CUSTOM CNN FROM SCRATCH
    # -------------------------------------------------------------
    print("\n>>> [1/3] Training Custom CNN from Scratch...")
    X_train_cnn, y_train_cnn, _ = DrawingDataset.load_split_arrays(train_df, model_type="cnn", augment=True)
    X_val_cnn, y_val_cnn, _ = DrawingDataset.load_split_arrays(valid_df, model_type="cnn", augment=False)
    X_test_cnn, y_test_cnn, test_paths = DrawingDataset.load_split_arrays(test_df, model_type="cnn", augment=False)
    
    cnn_model = build_custom_cnn(learning_rate=1e-3)
    cnn_callbacks = [
        callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        callbacks.ModelCheckpoint(os.path.join(artifacts_dir, "custom_cnn.keras"), save_best_only=True, monitor="val_loss")
    ]
    
    start_time = time.time()
    cnn_hist = cnn_model.fit(
        X_train_cnn, y_train_cnn,
        validation_data=(X_val_cnn, y_val_cnn),
        epochs=30,
        batch_size=16,
        class_weight=class_weights,
        callbacks=cnn_callbacks,
        verbose=1
    )
    cnn_train_time = time.time() - start_time
    
    # Evaluation
    t0 = time.time()
    cnn_preds = cnn_model.predict(X_test_cnn)
    cnn_inf_latency = (time.time() - t0) / len(X_test_cnn) * 1000.0  # ms per sample
    
    cnn_meta = {
        "model_name": "custom_cnn",
        "train_time_sec": round(cnn_train_time, 2),
        "inference_latency_ms": round(cnn_inf_latency, 2),
        "total_params": int(cnn_model.count_params()),
        "trainable_params": int(sum([tf.size(w).numpy() for w in cnn_model.trainable_weights])),
        "model_size_mb": round(os.path.getsize(os.path.join(artifacts_dir, "custom_cnn.keras")) / (1024 * 1024), 2)
    }
    results["custom_cnn"] = compute_comprehensive_metrics(y_test_cnn, cnn_preds, model_metadata=cnn_meta)
    history_dict["custom_cnn"] = {k: [float(x) for x in v] for k, v in cnn_hist.history.items()}
    print(f"Custom CNN Test Accuracy: {results['custom_cnn']['accuracy']:.2%}, AUC: {results['custom_cnn']['roc_auc']:.3f}")
    
    # -------------------------------------------------------------
    # 2. MODEL 2: VGG16 TRANSFER LEARNING
    # -------------------------------------------------------------
    print("\n>>> [2/3] Training VGG16 Transfer Learning...")
    X_train_vgg, y_train_vgg, _ = DrawingDataset.load_split_arrays(train_df, model_type="vgg16", augment=True)
    X_val_vgg, y_val_vgg, _ = DrawingDataset.load_split_arrays(valid_df, model_type="vgg16", augment=False)
    X_test_vgg, y_test_vgg, _ = DrawingDataset.load_split_arrays(test_df, model_type="vgg16", augment=False)
    
    vgg_model, vgg_base = build_vgg16_transfer(learning_rate=1e-3)
    vgg_callbacks = [
        callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
        callbacks.ModelCheckpoint(os.path.join(artifacts_dir, "vgg16_transfer.keras"), save_best_only=True, monitor="val_loss")
    ]
    
    start_time = time.time()
    # Phase 1: Train Head
    print("  Phase 1: Training Classification Head (12 epochs)...")
    vgg_hist1 = vgg_model.fit(
        X_train_vgg, y_train_vgg,
        validation_data=(X_val_vgg, y_val_vgg),
        epochs=12,
        batch_size=16,
        class_weight=class_weights,
        callbacks=vgg_callbacks,
        verbose=1
    )
    # Phase 2: Fine-Tuning
    print("  Phase 2: Fine-Tuning Block5 Layers (12 epochs)...")
    configure_vgg16_finetuning(vgg_model, vgg_base, unfreeze_last_n_layers=4, fine_tune_lr=1e-5)
    vgg_hist2 = vgg_model.fit(
        X_train_vgg, y_train_vgg,
        validation_data=(X_val_vgg, y_val_vgg),
        epochs=12,
        batch_size=16,
        class_weight=class_weights,
        callbacks=vgg_callbacks,
        verbose=1
    )
    vgg_train_time = time.time() - start_time
    vgg_model.save(os.path.join(artifacts_dir, "vgg16_transfer.keras"))
    
    t0 = time.time()
    vgg_preds = vgg_model.predict(X_test_vgg)
    vgg_inf_latency = (time.time() - t0) / len(X_test_vgg) * 1000.0
    
    vgg_meta = {
        "model_name": "vgg16",
        "train_time_sec": round(vgg_train_time, 2),
        "inference_latency_ms": round(vgg_inf_latency, 2),
        "total_params": int(vgg_model.count_params()),
        "trainable_params": int(sum([tf.size(w).numpy() for w in vgg_model.trainable_weights])),
        "model_size_mb": round(os.path.getsize(os.path.join(artifacts_dir, "vgg16_transfer.keras")) / (1024 * 1024), 2)
    }
    results["vgg16"] = compute_comprehensive_metrics(y_test_vgg, vgg_preds, model_metadata=vgg_meta)
    # Concatenate history
    vgg_combined_hist = {}
    for k in vgg_hist1.history:
        vgg_combined_hist[k] = [float(x) for x in vgg_hist1.history[k] + vgg_hist2.history.get(k, [])]
    history_dict["vgg16"] = vgg_combined_hist
    print(f"VGG16 Test Accuracy: {results['vgg16']['accuracy']:.2%}, AUC: {results['vgg16']['roc_auc']:.3f}")
    
    # -------------------------------------------------------------
    # 3. MODEL 3: RESNET50 TRANSFER LEARNING
    # -------------------------------------------------------------
    print("\n>>> [3/3] Training ResNet50 Transfer Learning...")
    X_train_res, y_train_res, _ = DrawingDataset.load_split_arrays(train_df, model_type="resnet50", augment=True)
    X_val_res, y_val_res, _ = DrawingDataset.load_split_arrays(valid_df, model_type="resnet50", augment=False)
    X_test_res, y_test_res, _ = DrawingDataset.load_split_arrays(test_df, model_type="resnet50", augment=False)
    
    res_model, res_base = build_resnet50_transfer(learning_rate=1e-3)
    res_callbacks = [
        callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
        callbacks.ModelCheckpoint(os.path.join(artifacts_dir, "resnet50_transfer.keras"), save_best_only=True, monitor="val_loss")
    ]
    
    start_time = time.time()
    print("  Phase 1: Training Classification Head (12 epochs)...")
    res_hist1 = res_model.fit(
        X_train_res, y_train_res,
        validation_data=(X_val_res, y_val_res),
        epochs=12,
        batch_size=16,
        class_weight=class_weights,
        callbacks=res_callbacks,
        verbose=1
    )
    print("  Phase 2: Fine-Tuning Top Residual Stage with Frozen BatchNorm (12 epochs)...")
    configure_resnet50_finetuning(res_model, res_base, unfreeze_last_n_layers=15, fine_tune_lr=1e-5)
    res_hist2 = res_model.fit(
        X_train_res, y_train_res,
        validation_data=(X_val_res, y_val_res),
        epochs=12,
        batch_size=16,
        class_weight=class_weights,
        callbacks=res_callbacks,
        verbose=1
    )
    res_train_time = time.time() - start_time
    res_model.save(os.path.join(artifacts_dir, "resnet50_transfer.keras"))
    
    t0 = time.time()
    res_preds = res_model.predict(X_test_res)
    res_inf_latency = (time.time() - t0) / len(X_test_res) * 1000.0
    
    res_meta = {
        "model_name": "resnet50",
        "train_time_sec": round(res_train_time, 2),
        "inference_latency_ms": round(res_inf_latency, 2),
        "total_params": int(res_model.count_params()),
        "trainable_params": int(sum([tf.size(w).numpy() for w in res_model.trainable_weights])),
        "model_size_mb": round(os.path.getsize(os.path.join(artifacts_dir, "resnet50_transfer.keras")) / (1024 * 1024), 2)
    }
    results["resnet50"] = compute_comprehensive_metrics(y_test_res, res_preds, model_metadata=res_meta)
    res_combined_hist = {}
    for k in res_hist1.history:
        res_combined_hist[k] = [float(x) for x in res_hist1.history[k] + res_hist2.history.get(k, [])]
    history_dict["resnet50"] = res_combined_hist
    print(f"ResNet50 Test Accuracy: {results['resnet50']['accuracy']:.2%}, AUC: {results['resnet50']['roc_auc']:.3f}")
    
    # -------------------------------------------------------------
    # 4. SUBJECT-INDEPENDENT 5-FOLD CROSS-VALIDATION
    # -------------------------------------------------------------
    print("\n>>> [Audit] Conducting Subject-Independent 5-Fold Cross-Validation (Zero Patient Leakage)...")
    folds = ds.get_subject_independent_folds(n_splits=5)
    fold_results = {"custom_cnn": [], "vgg16": [], "resnet50": []}
    
    for f_idx, (fold_train_df, fold_val_df) in enumerate(folds):
        print(f"  Evaluating Fold {f_idx + 1}/5 (Train: {len(fold_train_df)} imgs, Val: {len(fold_val_df)} imgs)...")
        # Evaluate Custom CNN on fold
        f_X_train, f_y_train, _ = DrawingDataset.load_split_arrays(fold_train_df, model_type="cnn", augment=True)
        f_X_val, f_y_val, _ = DrawingDataset.load_split_arrays(fold_val_df, model_type="cnn", augment=False)
        f_cnn = build_custom_cnn()
        f_cnn.fit(f_X_train, f_y_train, validation_data=(f_X_val, f_y_val), epochs=15, batch_size=16, verbose=0)
        f_preds = f_cnn.predict(f_X_val, verbose=0)
        fold_results["custom_cnn"].append(compute_comprehensive_metrics(f_y_val, f_preds))
        
    subject_indep_summary = {
        "custom_cnn": {
            "mean_accuracy": round(float(np.mean([r["accuracy"] for r in fold_results["custom_cnn"]])), 4),
            "std_accuracy": round(float(np.std([r["accuracy"] for r in fold_results["custom_cnn"]])), 4),
            "mean_recall": round(float(np.mean([r["recall"] for r in fold_results["custom_cnn"]])), 4),
            "mean_specificity": round(float(np.mean([r["specificity"] for r in fold_results["custom_cnn"]])), 4),
            "mean_roc_auc": round(float(np.mean([r["roc_auc"] for r in fold_results["custom_cnn"]])), 4),
        }
    }
    print("Subject-Independent Validation Results (Custom CNN):", subject_indep_summary["custom_cnn"])
    
    # -------------------------------------------------------------
    # 5. PERSIST EXPERIMENTAL ARTIFACTS
    # -------------------------------------------------------------
    with open(os.path.join(artifacts_dir, "models_evaluation.json"), "w") as f:
        json.dump(results, f, indent=2)
        
    with open(os.path.join(artifacts_dir, "training_history.json"), "w") as f:
        json.dump(history_dict, f, indent=2)
        
    with open(os.path.join(artifacts_dir, "subject_independent_results.json"), "w") as f:
        json.dump(subject_indep_summary, f, indent=2)
        
    # Generate Comparison CSV
    csv_rows = []
    for m_name, m_data in results.items():
        csv_rows.append({
            "Model": m_name.upper(),
            "Accuracy": m_data["accuracy"],
            "Precision": m_data["precision"],
            "Recall (Sensitivity)": m_data["recall"],
            "Specificity": m_data["specificity"],
            "F1-Score": m_data["f1"],
            "Balanced Accuracy": m_data["balanced_accuracy"],
            "ROC-AUC": m_data["roc_auc"],
            "PR-AUC": m_data["pr_auc"],
            "MCC": m_data["mcc"],
            "Brier Score": m_data["brier_score"],
            "Total Parameters": m_data["total_params"],
            "Training Time (s)": m_data["train_time_sec"],
            "Inference Latency (ms)": m_data["inference_latency_ms"],
            "Model Size (MB)": m_data["model_size_mb"]
        })
    comp_df = pd.DataFrame(csv_rows)
    comp_df.to_csv(os.path.join(artifacts_dir, "model_comparison.csv"), index=False)
    print("\nModel Comparison Table:")
    print(comp_df.to_string(index=False))
    
    # -------------------------------------------------------------
    # 6. GENERATE EVALUATION CHARTS
    # -------------------------------------------------------------
    print("\n>>> Generating evaluation plots in:", eval_output_dir)
    plot_evaluation_charts(results, history_dict, eval_output_dir)
    print("Evaluation charts generated successfully.")


if __name__ == "__main__":
    train_and_evaluate_all()

