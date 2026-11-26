# Ultralytics YOLO11 Module — Parkinson's Disease Classification

This isolated module (`yolo11/`) adds **Ultralytics YOLO11 (`yolo11s-cls`)** image classification to the Parkinson's Disease Detection project without modifying any existing frontend, backend, API routes, existing models, or deployment configuration.

---

## 1. Task Type & Architecture Selection

The existing local dataset (`ml/data/raw/Parkinson Disease detection/`) is a **folder-structured binary image classification dataset** (`healthy` vs `parkinson`) with no bounding-box annotations. Accordingly, this module uses **`yolo11s-cls.pt`** (Ultralytics YOLO11 Small Classification, pretrained on ImageNet-1K) with transfer learning.

---

## 2. Installation & Dependencies

Activate the existing project virtual environment and install `ultralytics`:

```bash
cd /Users/jayasinghrajput/PARKINSON
source .venv/bin/activate
pip install --upgrade ultralytics
```

Core dependencies (already present in `.venv`):
- `ultralytics >= 8.3.0`
- `torch >= 2.0`, `torchvision`
- `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `pillow`, `opencv-python`

---

## 3. Dataset Path Configuration (Zero Duplication)

This module reuses your existing local dataset at:
```
ml/data/raw/Parkinson Disease detection/
├── train/  (133 images: 64 healthy, 69 parkinson)
├── valid/  (31 images: 19 healthy, 12 parkinson)
└── test/   (40 images: 19 healthy, 21 parkinson)
```

Because Ultralytics classification expects `val/` instead of `valid/`, `yolo11/dataset_utils.py` automatically creates a **zero-copy symlink directory** at `yolo11/dataset_view/` (`train -> train`, `val -> valid`, `test -> test`). No images are duplicated and no external or Roboflow API calls are made.

---

## 4. Running the Complete Pipeline

Run EDA, training, validation, independent test evaluation, existing-model comparison, and an inference smoke test in one command:

```bash
PYTHONPATH=. YOLO_CONFIG_DIR=/tmp/ultralytics_config MPLCONFIGDIR=/tmp/matplotlib_cache \
  .venv/bin/python -m yolo11.run_pipeline
```

### Running Individual Stages

1. **Exploratory Data Analysis (EDA):**
   ```bash
   PYTHONPATH=. .venv/bin/python -m yolo11.eda
   ```
   Outputs saved to `yolo11/outputs/eda/`.

2. **Training & Validation:**
   ```bash
   PYTHONPATH=. .venv/bin/python -m yolo11.train --epochs 30 --batch-size 16 --imgsz 224 --lr 0.001 --patience 10
   ```
   - Automatically detects hardware acceleration (`CUDA` → Apple `MPS` → `CPU`).
   - Applies conservative, drawing-safe augmentations (`degrees=10`, `translate=0.05`, `scale=0.10`, flips disabled) to the training split only.
   - Validates after every epoch with early stopping (`patience=10`) and saves `best.pt` and `last.pt` to `yolo11/outputs/weights/`.

3. **Evaluation & Multi-Model Comparison:**
   ```bash
   PYTHONPATH=. .venv/bin/python -m yolo11.evaluate
   ```
   Evaluates `best.pt` on both the validation set and the **independent test set** (40 images, untouched during training), saves metrics in CSV/JSON, generates all performance plots under `yolo11/outputs/evaluation/`, and produces a comparison against `CUSTOM_CNN`, `VGG16`, and `RESNET50`.

---

## 5. Loading the Saved Model & Running Inference

### CLI Inference
```bash
PYTHONPATH=. .venv/bin/python -m yolo11.predict \
  --image "ml/data/raw/Parkinson Disease detection/test/parkinson/V01PO01_png.rf.8dcf54d3d17c25b69468a13d9387ad7b.jpg" \
  --threshold 0.50
```

### Python API Inference
```python
from yolo11.predict import predict_image

result = predict_image(
    image_path="ml/data/raw/Parkinson Disease detection/test/parkinson/V01PO01_png.rf.8dcf54d3d17c25b69468a13d9387ad7b.jpg",
    weights_path="yolo11/outputs/weights/best.pt",
    threshold=0.50,
)
print(result)
```

---

## 6. Optional Future Integration With Existing FastAPI Backend

Your existing backend (`backend/app/`) was **not modified**. Because `yolo11.predict.predict_image` outputs the same binary class probabilities (`healthy` / `parkinson`), latency, and threshold structure as `ModelService.predict_image`, you can optionally add YOLO11 later by:
1. Adding `"yolo11"` to `ModelService.model_configs` in `backend/app/services/model_service.py` pointing to `yolo11/outputs/weights/best.pt`.
2. Loading it via `ultralytics.YOLO(weights_path)` when `model_name == "yolo11"` and returning the same `PredictionResponse` dictionary.
3. Adding `ultralytics` and `torch` to `requirements.txt` before redeploying.

---

## 7. Troubleshooting

| Issue | Cause & Resolution |
|---|---|
| `PermissionError` writing `~/.config/Ultralytics` or `~/.matplotlib` | Prefix commands with `YOLO_CONFIG_DIR=/tmp/ultralytics_config MPLCONFIGDIR=/tmp/matplotlib_cache`. |
| `FileNotFoundError: val/ not found` | Run `prepare_symlink_dataset()` (called automatically by `train.py`) to map `val -> valid`. |
| Slow CPU training or DataLoader hang on macOS | Keep `workers=0` (configured by default in `yolo11/config.py`) and reduce `--batch-size 8` if RAM is constrained. |
| Offline / no internet when loading `yolo11s-cls.pt` | Pretrained weights are cached locally at `yolo11/weights/yolo11s-cls.pt` and loaded automatically without network access. |
