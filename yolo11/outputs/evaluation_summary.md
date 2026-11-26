# YOLO11s-cls Evaluation Summary & Model Comparison

## 1. YOLO11s-cls Performance
- **Weights:** `yolo11/outputs/weights/best.pt`
- **Parameters:** 5,445,570 | **Model Size:** 10.52 MB | **Device:** `cpu`

| Split | Samples | Accuracy | Precision | Recall (Sensitivity) | Specificity | F1-score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation | 31 | 0.9677 | 0.9231 | 1.0000 | 0.9474 | 0.9600 | 1.0000 | 1.0000 |
| **Independent Test** | **40** | **0.9000** | **0.9474** | **0.8571** | **0.9474** | **0.9000** | **0.8922** | **0.9263** |

### Independent Test Confusion Matrix
- TN (Healthy → Healthy): **18**
- FP (Healthy → Parkinson): **1**
- FN (Parkinson → Healthy): **3**
- TP (Parkinson → Parkinson): **18**

### Per-Class Classification Report (Test Set)
```text
precision    recall  f1-score   support

     healthy     0.8571    0.9474    0.9000        19
   parkinson     0.9474    0.8571    0.9000        21

    accuracy                         0.9000        40
   macro avg     0.9023    0.9023    0.9000        40
weighted avg     0.9045    0.9000    0.9000        40
```

## 2. Comparison With Existing Models (Independent Test Set, N=40)

| Model | Accuracy | Precision | Recall | Specificity | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| CUSTOM_CNN | 0.4750 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.4762 |
| VGG16 | 0.8000 | 0.8095 | 0.8095 | 0.7895 | 0.8095 | 0.8947 |
| RESNET50 | 0.7750 | 0.8000 | 0.7619 | 0.7895 | 0.7805 | 0.8747 |
| YOLO11s-cls | 0.9000 | 0.9474 | 0.8571 | 0.9474 | 0.9000 | 0.8922 |

- **Best Model by Accuracy:** **YOLO11s-cls** (0.9000)
- **Best Model by ROC-AUC:** **VGG16** (0.8947)
