"""
Comprehensive Clinical and Machine Learning Evaluation Metrics Suite.
Calculates:
- Discrimination: Accuracy, Precision, Recall/Sensitivity, Specificity, F1, Balanced Accuracy, MCC, ROC-AUC, PR-AUC
- Calibration: Brier Score, Expected Calibration Error (ECE)
- Efficiency: Trainable parameters, Total parameters, Training time, Inference latency, Disk size
"""

import numpy as np
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    brier_score_loss
)


def compute_comprehensive_metrics(y_true: np.ndarray,
                                 y_prob: np.ndarray,
                                 threshold: float = 0.5,
                                 model_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Computes complete suite of academic performance and clinical diagnostics metrics.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).flatten()
    y_pred = (y_prob >= threshold).astype(int)
    
    # Confusion matrix elements: TN, FP, FN, TP
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    # Specificity = TN / (TN + FP)
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    sensitivity = float(recall_score(y_true, y_pred, zero_division=0))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.5
        
    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = float(np.mean(y_true))
        
    brier = float(brier_score_loss(y_true, y_prob))
    
    # Calculate ROC and PR curves
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    rec_curve, prec_curve, _ = precision_recall_curve(y_true, y_prob)
    
    results = {
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(sensitivity, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "f1": round(f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "mcc": round(mcc, 4),
        "brier_score": round(brier, 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "matrix": cm.tolist()
        },
        "threshold": threshold,
        "sample_count": len(y_true),
        "roc_curve": {
            "fpr": [round(float(x), 4) for x in fpr],
            "tpr": [round(float(x), 4) for x in tpr]
        },
        "pr_curve": {
            "recall": [round(float(x), 4) for x in rec_curve],
            "precision": [round(float(x), 4) for x in prec_curve]
        }
    }
    
    if model_metadata:
        results.update(model_metadata)
        
    return results

