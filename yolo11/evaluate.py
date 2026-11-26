"""
Comprehensive Evaluation, Plotting, and Multi-Model Comparison for YOLO11.
Evaluates the trained YOLO11 weights on both Validation and Independent Test splits,
generates all required performance graphs, and compares against existing models.
"""

import os
import json
import time
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    brier_score_loss,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    classification_report,
)
from ultralytics import YOLO

from yolo11.config import (
    YOLO11Config,
    RAW_DATASET_DIR,
    SAVED_WEIGHTS_DIR,
    RUNS_OUTPUT_DIR,
    EVAL_OUTPUT_DIR,
    METRICS_OUTPUT_DIR,
    OUTPUTS_DIR,
    EXISTING_COMPARISON_CSV,
    detect_device,
    ensure_directories,
)
from yolo11.dataset_utils import list_split_images


def _evaluate_split(
    model: YOLO,
    items: List[Tuple[str, str]],
    imgsz: int,
    device: str,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """Run deterministic inference on a split and compute full classification metrics."""
    y_true = []
    y_prob = []
    latencies_ms = []

    for img_path, label in items:
        y_true.append(1 if label == "parkinson" else 0)
        t0 = time.time()
        res = model.predict(source=img_path, imgsz=imgsz, device=device, verbose=False)[0]
        latencies_ms.append((time.time() - t0) * 1000.0)

        names = {int(k): v.lower() for k, v in res.names.items()}
        probs = res.probs.data.cpu().numpy()
        p_idx = [idx for idx, name in names.items() if name == "parkinson"][0]
        y_prob.append(float(probs[p_idx]))

    y_true_arr = np.array(y_true, dtype=int)
    y_prob_arr = np.array(y_prob, dtype=float)
    y_pred_arr = (y_prob_arr >= threshold).astype(int)

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true_arr, y_pred_arr))
    roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))
    pr_auc = float(average_precision_score(y_true_arr, y_prob_arr))
    mcc = float(matthews_corrcoef(y_true_arr, y_pred_arr))
    brier = float(brier_score_loss(y_true_arr, y_prob_arr))
    mean_latency = float(np.mean(latencies_ms)) if latencies_ms else 0.0

    cls_report_text = classification_report(
        y_true_arr,
        y_pred_arr,
        target_names=["healthy", "parkinson"],
        digits=4,
        zero_division=0,
    )
    cls_report_dict = classification_report(
        y_true_arr,
        y_pred_arr,
        target_names=["healthy", "parkinson"],
        output_dict=True,
        zero_division=0,
    )

    fpr, tpr, _ = roc_curve(y_true_arr, y_prob_arr)
    pr_prec, pr_rec, _ = precision_recall_curve(y_true_arr, y_prob_arr)

    return {
        "sample_count": int(len(y_true_arr)),
        "threshold": threshold,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "sensitivity": round(rec, 4),
        "specificity": round(spec, 4),
        "f1_score": round(f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "mcc": round(mcc, 4),
        "brier_score": round(brier, 4),
        "inference_latency_ms": round(mean_latency, 2),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "matrix": cm.tolist()},
        "classification_report_text": cls_report_text,
        "classification_report_dict": cls_report_dict,
        "y_true": y_true_arr.tolist(),
        "y_prob": [round(float(p), 5) for p in y_prob_arr],
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "pr_curve": {"precision": pr_prec.tolist(), "recall": pr_rec.tolist()},
    }


def _plot_training_curves(results_csv: str, eval_dir: str) -> None:
    """Plot separate Training Loss, Validation Loss, and Accuracy curves from Ultralytics results.csv."""
    if not os.path.exists(results_csv):
        return
    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]
    epochs = df["epoch"] if "epoch" in df.columns else np.arange(1, len(df) + 1)

    # 1. Training Loss Graph
    if "train/loss" in df.columns:
        plt.figure(figsize=(8, 4.8))
        plt.plot(epochs, df["train/loss"], marker="o", linewidth=2, color="#0284C7", label="Training Loss")
        plt.title("YOLO11s-cls Training Loss Curve", fontsize=13, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(eval_dir, "01_training_loss.png"), dpi=220)
        plt.close()

    # 2. Validation Loss Graph
    if "val/loss" in df.columns:
        plt.figure(figsize=(8, 4.8))
        plt.plot(epochs, df["val/loss"], marker="s", linewidth=2, color="#DC2626", label="Validation Loss")
        plt.title("YOLO11s-cls Validation Loss Curve", fontsize=13, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(eval_dir, "02_validation_loss.png"), dpi=220)
        plt.close()

    # 3. Validation Accuracy Graph
    acc_col = "metrics/accuracy_top1" if "metrics/accuracy_top1" in df.columns else None
    if acc_col:
        plt.figure(figsize=(8, 4.8))
        plt.plot(epochs, df[acc_col] * 100.0, marker="o", linewidth=2, color="#10B981", label="Validation Top-1 Accuracy (%)")
        plt.title("YOLO11s-cls Validation Accuracy Curve", fontsize=13, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.ylim(0, 105)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(eval_dir, "03_accuracy_curve.png"), dpi=220)
        plt.close()


def _plot_eval_diagnostics(val_res: Dict[str, Any], test_res: Dict[str, Any], eval_dir: str) -> None:
    """Plot Precision-Recall, F1-Threshold, Confusion Matrix, and ROC-AUC curves."""
    # 4. Precision-Recall Curve (Independent Test Set)
    plt.figure(figsize=(7.5, 5.2))
    plt.plot(
        test_res["pr_curve"]["recall"],
        test_res["pr_curve"]["precision"],
        color="#6366F1",
        linewidth=2.4,
        label=f"YOLO11s-cls (PR-AUC = {test_res['pr_auc']:.3f})",
    )
    plt.title("YOLO11s-cls Precision-Recall Curve (Independent Test Set)", fontsize=13, fontweight="bold")
    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision")
    plt.xlim(0, 1.02)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, "04_precision_recall_curve.png"), dpi=220)
    plt.close()

    # 5. F1-Score, Precision & Recall vs Decision Threshold
    y_true = np.array(test_res["y_true"])
    y_prob = np.array(test_res["y_prob"])
    thresholds = np.linspace(0.05, 0.95, 37)
    f1s, precs, recs = [], [], []
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        f1s.append(f1_score(y_true, preds, zero_division=0))
        precs.append(precision_score(y_true, preds, zero_division=0))
        recs.append(recall_score(y_true, preds, zero_division=0))

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, f1s, linewidth=2.5, color="#10B981", label="F1-Score")
    plt.plot(thresholds, precs, linewidth=1.8, linestyle="--", color="#0284C7", label="Precision")
    plt.plot(thresholds, recs, linewidth=1.8, linestyle=":", color="#F59E0B", label="Recall")
    plt.axvline(0.50, color="#64748B", linestyle="-.", alpha=0.7, label="Default Threshold (0.50)")
    plt.title("YOLO11s-cls F1-Score, Precision & Recall Across Thresholds (Test Set)", fontsize=12.5, fontweight="bold")
    plt.xlabel("Decision Threshold")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, "05_f1_score_curve.png"), dpi=220)
    plt.close()

    # 6. Confusion Matrices (Validation + Independent Test)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, res, title in [
        (axes[0], val_res, f"Validation Set (N={val_res['sample_count']})"),
        (axes[1], test_res, f"Independent Test Set (N={test_res['sample_count']})"),
    ]:
        cm = np.array(res["confusion_matrix"]["matrix"])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"YOLO11s-cls Confusion Matrix\n{title}", fontsize=11.5, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred: Healthy", "Pred: Parkinson"])
        ax.set_yticklabels(["True: Healthy", "True: Parkinson"])
        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, "06_confusion_matrix.png"), dpi=220)
    plt.close()

    # 7. ROC-AUC Curve (Independent Test Set)
    plt.figure(figsize=(7.5, 5.2))
    plt.plot(
        test_res["roc_curve"]["fpr"],
        test_res["roc_curve"]["tpr"],
        color="#0284C7",
        linewidth=2.5,
        label=f"YOLO11s-cls (ROC-AUC = {test_res['roc_auc']:.3f})",
    )
    plt.plot([0, 1], [0, 1], linestyle="--", color="#94A3B8", label="Random Classifier (AUC = 0.500)")
    plt.title("YOLO11s-cls ROC Curve (Independent Test Set)", fontsize=13, fontweight="bold")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.xlim(0, 1.02)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, "07_roc_auc_curve.png"), dpi=220)
    plt.close()


def _compare_with_existing_models(
    test_res: Dict[str, Any],
    train_meta: Dict[str, Any],
    eval_dir: str,
    metrics_dir: str,
) -> pd.DataFrame:
    """Compare YOLO11 against existing models from ml/artifacts/model_comparison.csv without modifying it."""
    rows = []
    if os.path.exists(EXISTING_COMPARISON_CSV):
        existing_df = pd.read_csv(EXISTING_COMPARISON_CSV)
        for _, r in existing_df.iterrows():
            rows.append(
                {
                    "Model": str(r["Model"]),
                    "Accuracy": float(r["Accuracy"]),
                    "Precision": float(r["Precision"]),
                    "Recall": float(r["Recall (Sensitivity)"]),
                    "Specificity": float(r["Specificity"]),
                    "F1-score": float(r["F1-Score"]),
                    "ROC-AUC": float(r["ROC-AUC"]),
                    "Parameters": int(r["Total Parameters"]),
                    "Inference Latency (ms)": float(r["Inference Latency (ms)"]),
                }
            )

    rows.append(
        {
            "Model": "YOLO11s-cls",
            "Accuracy": test_res["accuracy"],
            "Precision": test_res["precision"],
            "Recall": test_res["recall"],
            "Specificity": test_res["specificity"],
            "F1-score": test_res["f1_score"],
            "ROC-AUC": test_res["roc_auc"],
            "Parameters": int(train_meta.get("total_params", 0)),
            "Inference Latency (ms)": test_res["inference_latency_ms"],
        }
    )

    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(os.path.join(metrics_dir, "model_comparison.csv"), index=False)
    comp_df.to_json(os.path.join(metrics_dir, "model_comparison.json"), orient="records", indent=2)

    # 8. Comparison Bar Graph
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    x = np.arange(len(metrics_to_plot))
    n_models = len(comp_df)
    width = 0.75 / max(n_models, 1)
    palette = ["#94A3B8", "#0284C7", "#6366F1", "#10B981"]

    fig, ax = plt.subplots(figsize=(11, 5.4))
    for i, (_, row) in enumerate(comp_df.iterrows()):
        vals = [row[m] for m in metrics_to_plot]
        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            vals,
            width,
            label=row["Model"],
            color=palette[i % len(palette)],
            edgecolor="#1E293B",
        )
        for b in bars:
            h = b.get_height()
            ax.text(
                b.get_x() + b.get_width() / 2,
                h + 0.015,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    ax.set_title("Independent Test Set Comparison: Existing Models vs YOLO11s-cls", fontsize=13.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot, fontsize=11)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.12)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, "08_model_comparison.png"), dpi=220)
    plt.close()

    return comp_df


def evaluate_yolo11() -> Dict[str, Any]:
    """Run full validation + independent test evaluation and generate comparison artifacts."""
    ensure_directories()
    cfg = YOLO11Config()
    device = detect_device()

    weights_path = os.path.join(SAVED_WEIGHTS_DIR, "best.pt")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Trained weights not found at '{weights_path}'. Run train.py first.")

    config_json = os.path.join(METRICS_OUTPUT_DIR, "yolo11_config.json")
    train_meta = json.load(open(config_json)) if os.path.exists(config_json) else {}

    model = YOLO(weights_path)
    splits = list_split_images(RAW_DATASET_DIR)

    print("[Eval] Evaluating YOLO11s-cls on Validation set...")
    val_res = _evaluate_split(model, splits["valid"], imgsz=cfg.imgsz, device=device, threshold=cfg.decision_threshold)

    print("[Eval] Evaluating YOLO11s-cls on Independent Test set...")
    test_res = _evaluate_split(model, splits["test"], imgsz=cfg.imgsz, device=device, threshold=cfg.decision_threshold)

    # Save classification reports
    with open(os.path.join(METRICS_OUTPUT_DIR, "classification_report_test.txt"), "w") as f:
        f.write("YOLO11s-cls Independent Test Set Classification Report\n")
        f.write("=" * 58 + "\n")
        f.write(test_res["classification_report_text"])

    # Save full metrics JSON + CSV
    metrics_payload = {
        "model_name": "YOLO11s-cls",
        "weights_path": weights_path,
        "device": device,
        "validation": {k: v for k, v in val_res.items() if k not in ("y_true", "y_prob", "roc_curve", "pr_curve")},
        "test": {k: v for k, v in test_res.items() if k not in ("y_true", "y_prob", "roc_curve", "pr_curve")},
    }
    with open(os.path.join(METRICS_OUTPUT_DIR, "yolo11_metrics.json"), "w") as f:
        json.dump(metrics_payload, f, indent=2)

    pd.DataFrame(
        [
            {"Split": "Validation", **{k: val_res[k] for k in ["sample_count", "accuracy", "precision", "recall", "specificity", "f1_score", "balanced_accuracy", "roc_auc", "pr_auc", "mcc", "brier_score", "inference_latency_ms"]}},
            {"Split": "Independent Test", **{k: test_res[k] for k in ["sample_count", "accuracy", "precision", "recall", "specificity", "f1_score", "balanced_accuracy", "roc_auc", "pr_auc", "mcc", "brier_score", "inference_latency_ms"]}},
        ]
    ).to_csv(os.path.join(METRICS_OUTPUT_DIR, "yolo11_metrics.csv"), index=False)

    # Generate all required graphs
    results_csv = os.path.join(RUNS_OUTPUT_DIR, cfg.run_name, "results.csv")
    _plot_training_curves(results_csv, EVAL_OUTPUT_DIR)
    _plot_eval_diagnostics(val_res, test_res, EVAL_OUTPUT_DIR)

    # Compare against existing models
    comp_df = _compare_with_existing_models(test_res, train_meta, EVAL_OUTPUT_DIR, METRICS_OUTPUT_DIR)

    best_by_acc = comp_df.loc[comp_df["Accuracy"].idxmax()]
    best_by_auc = comp_df.loc[comp_df["ROC-AUC"].idxmax()]

    # Write concise human-readable evaluation summary
    summary_md = f"""# YOLO11s-cls Evaluation Summary & Model Comparison

## 1. YOLO11s-cls Performance
- **Weights:** `yolo11/outputs/weights/best.pt`
- **Parameters:** {train_meta.get('total_params', 'N/A'):,} | **Model Size:** {train_meta.get('model_size_mb', 'N/A')} MB | **Device:** `{device}`

| Split | Samples | Accuracy | Precision | Recall (Sensitivity) | Specificity | F1-score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation | {val_res['sample_count']} | {val_res['accuracy']:.4f} | {val_res['precision']:.4f} | {val_res['recall']:.4f} | {val_res['specificity']:.4f} | {val_res['f1_score']:.4f} | {val_res['roc_auc']:.4f} | {val_res['pr_auc']:.4f} |
| **Independent Test** | **{test_res['sample_count']}** | **{test_res['accuracy']:.4f}** | **{test_res['precision']:.4f}** | **{test_res['recall']:.4f}** | **{test_res['specificity']:.4f}** | **{test_res['f1_score']:.4f}** | **{test_res['roc_auc']:.4f}** | **{test_res['pr_auc']:.4f}** |

### Independent Test Confusion Matrix
- TN (Healthy → Healthy): **{test_res['confusion_matrix']['tn']}**
- FP (Healthy → Parkinson): **{test_res['confusion_matrix']['fp']}**
- FN (Parkinson → Healthy): **{test_res['confusion_matrix']['fn']}**
- TP (Parkinson → Parkinson): **{test_res['confusion_matrix']['tp']}**

### Per-Class Classification Report (Test Set)
```text
{test_res['classification_report_text'].strip()}
```

## 2. Comparison With Existing Models (Independent Test Set, N=40)

| Model | Accuracy | Precision | Recall | Specificity | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
""" + "\n".join(
        f"| {r['Model']} | {r['Accuracy']:.4f} | {r['Precision']:.4f} | {r['Recall']:.4f} | {r['Specificity']:.4f} | {r['F1-score']:.4f} | {r['ROC-AUC']:.4f} |"
        for _, r in comp_df.iterrows()
    ) + f"""

- **Best Model by Accuracy:** **{best_by_acc['Model']}** ({best_by_acc['Accuracy']:.4f})
- **Best Model by ROC-AUC:** **{best_by_auc['Model']}** ({best_by_auc['ROC-AUC']:.4f})
"""

    with open(os.path.join(OUTPUTS_DIR, "evaluation_summary.md"), "w") as f:
        f.write(summary_md)

    print("\n" + comp_df[["Model", "Accuracy", "Precision", "Recall", "Specificity", "F1-score", "ROC-AUC"]].to_string(index=False))
    print(f"\n[Eval] All graphs and reports saved under: {OUTPUTS_DIR}")
    return {"validation": val_res, "test": test_res, "comparison": comp_df.to_dict(orient="records")}


if __name__ == "__main__":
    evaluate_yolo11()
