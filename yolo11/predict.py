"""
Standalone YOLO11 Inference Utility for Parkinson's Disease Classification.
Loads the saved trained YOLO11 weights (best.pt) and predicts on a single image or folder.

Usage:
    python -m yolo11.predict --image "/path/to/spiral.jpg"
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, Any
from ultralytics import YOLO

from yolo11.config import SAVED_WEIGHTS_DIR, YOLO11Config, detect_device


def predict_image(
    image_path: str,
    weights_path: str = os.path.join(SAVED_WEIGHTS_DIR, "best.pt"),
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """Run YOLO11 classification inference on a single spiral/meander drawing image."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Trained YOLO11 weights not found at '{weights_path}'. Run 'python -m yolo11.train' first."
        )
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found at '{image_path}'.")

    device = detect_device()
    model = YOLO(weights_path)
    cfg = YOLO11Config()

    t0 = time.time()
    results = model.predict(
        source=image_path, imgsz=cfg.imgsz, device=device, verbose=False
    )
    latency_ms = round((time.time() - t0) * 1000.0, 2)

    res = results[0]
    names = {int(k): v.lower() for k, v in res.names.items()}
    probs_tensor = res.probs.data.cpu().numpy()
    class_probs = {names[i]: round(float(probs_tensor[i]), 4) for i in range(len(probs_tensor))}

    parkinson_prob = class_probs.get("parkinson", 0.0)
    healthy_prob = class_probs.get("healthy", round(1.0 - parkinson_prob, 4))
    predicted_class = "parkinson" if parkinson_prob >= threshold else "healthy"

    return {
        "model": "YOLO11s-cls",
        "weights_path": weights_path,
        "device": device,
        "image_path": image_path,
        "predicted_class": predicted_class,
        "confidence": round(max(parkinson_prob, healthy_prob), 4),
        "decision_threshold": threshold,
        "probabilities": {
            "healthy": healthy_prob,
            "parkinson": parkinson_prob,
        },
        "inference_latency_ms": latency_ms,
    }


def main():
    parser = argparse.ArgumentParser(description="Run YOLO11 Parkinson's Classification Inference")
    parser.add_argument("--image", required=True, help="Path to input spiral/meander image")
    parser.add_argument(
        "--weights",
        default=os.path.join(SAVED_WEIGHTS_DIR, "best.pt"),
        help="Path to trained YOLO11 weights (default: yolo11/outputs/weights/best.pt)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Binary classification threshold for Parkinson class (default: 0.50)",
    )
    args = parser.parse_args()
    output = predict_image(args.image, weights_path=args.weights, threshold=args.threshold)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
