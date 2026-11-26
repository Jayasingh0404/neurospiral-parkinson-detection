"""
Model Service managing model lifecycle, inference caching, and Grad-CAM generation.
"""

import os
import io
import time
import json
import base64
from typing import Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

from backend.app.core.config import settings
from ml.src.preprocessing import prepare_image, letterbox_image
from ml.src.gradcam import generate_gradcam_heatmap, overlay_gradcam, find_target_conv_layer


class ModelService:
    def __init__(self):
        self.loaded_models: Dict[str, keras.Model] = {}
        self.model_configs = {
            "custom_cnn": {
                "name": "Custom CNN (From Scratch)",
                "architecture": "3 Conv Blocks + BatchNorm + MaxPool + Dropout + Dense",
                "description": "Lightweight custom convolutional neural network trained directly on spiral drawing images without external pretraining.",
                "input_shape": [224, 224, 3],
                "artifact_file": "custom_cnn.keras",
                "model_type": "cnn"
            },
            "vgg16": {
                "name": "VGG16 (Transfer Learning)",
                "architecture": "VGG16 Backbone + GlobalAveragePooling + Dropout + Dense",
                "description": "ImageNet-pretrained VGG16 feature extractor fine-tuned on Parkinson spiral drawings with terminal block fine-tuning.",
                "input_shape": [224, 224, 3],
                "artifact_file": "vgg16_transfer.keras",
                "model_type": "vgg16"
            },
            "resnet50": {
                "name": "ResNet50 (Transfer Learning)",
                "architecture": "ResNet50 Residual Backbone + GlobalAveragePooling + Dropout + Dense",
                "description": "Deep residual neural network pretrained on ImageNet and fine-tuned with frozen BatchNormalization layers.",
                "input_shape": [224, 224, 3],
                "artifact_file": "resnet50_transfer.keras",
                "model_type": "resnet50"
            },
            "yolo11": {
                "name": "YOLO11s-cls (Transfer Learning)",
                "architecture": "Ultralytics YOLO11s-cls (C3k2 + C2PSA Backbone + Classification Head)",
                "description": "Ultralytics YOLO11 Small image classification network pretrained on ImageNet-1K and fine-tuned with AdamW on spiral drawings.",
                "input_shape": [224, 224, 3],
                "artifact_file": settings.YOLO11_WEIGHTS_PATH,
                "model_type": "yolo11"
            }
        }
        
    def _resolve_artifact_path(self, model_name: str) -> str:
        artifact = self.model_configs[model_name]["artifact_file"]
        return artifact if os.path.isabs(artifact) else os.path.join(settings.ARTIFACTS_DIR, artifact)

    def get_available_models(self) -> Dict[str, Any]:
        """Lists metadata and availability for all models."""
        models_info = []
        metrics_dict = self.get_evaluation_metrics().get("models", {})
        
        for m_id, cfg in self.model_configs.items():
            path = self._resolve_artifact_path(m_id)
            is_trained = os.path.exists(path)
            model_eval = metrics_dict.get(m_id, {})
            
            info = {
                "id": m_id,
                "name": cfg["name"],
                "architecture": cfg["architecture"],
                "description": cfg["description"],
                "input_shape": cfg["input_shape"],
                "is_trained": is_trained,
                "total_params": model_eval.get("total_params"),
                "trainable_params": model_eval.get("trainable_params"),
                "model_size_mb": model_eval.get("model_size_mb"),
                "accuracy": model_eval.get("accuracy"),
                "roc_auc": model_eval.get("roc_auc")
            }
            models_info.append(info)
            
        return {"models": models_info, "default_model": settings.DEFAULT_MODEL}
        
    def load_model(self, model_name: str) -> Any:
        """Loads and caches model in memory (Keras or Ultralytics YOLO11)."""
        model_name = model_name.lower()
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(self.model_configs.keys())}")
            
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
            
        artifact_path = self._resolve_artifact_path(model_name)
        if not os.path.exists(artifact_path):
            raise FileNotFoundError(
                f"Model file '{artifact_path}' not found."
            )
            
        print(f"Loading model '{model_name}' from: {artifact_path}")
        if self.model_configs[model_name]["model_type"] == "yolo11":
            os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics_config")
            from ultralytics import YOLO
            model = YOLO(artifact_path)
        else:
            model = keras.models.load_model(artifact_path, compile=False)
        self.loaded_models[model_name] = model
        return model
        
    def predict_image(self,
                      image_bytes: bytes,
                      model_name: str = "yolo11",
                      threshold: float = 0.5) -> Dict[str, Any]:
        """Runs validated inference on an uploaded image."""
        model_name = model_name.lower()
        model = self.load_model(model_name)
        model_type = self.model_configs[model_name]["model_type"]
        
        # Load PIL image from bytes
        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
            pil_img = pil_img.convert("RGB")
        except Exception as e:
            raise ValueError(f"Invalid or corrupted image format: {str(e)}")
            
        if model_type == "yolo11":
            letterboxed = letterbox_image(pil_img, target_size=(224, 224))
            t0 = time.time()
            res = model.predict(source=pil_img, imgsz=224, verbose=False)[0]
            inf_latency_ms = round((time.time() - t0) * 1000.0, 2)
            names = {int(k): v.lower() for k, v in res.names.items()}
            probs = res.probs.data.cpu().numpy()
            p_idx = [idx for idx, name in names.items() if name == "parkinson"][0]
            raw_prob = float(probs[p_idx])
        else:
            # Standardize and letterbox
            input_tensor, letterboxed = prepare_image(pil_img, target_size=(224, 224), model_type=model_type, augment=False)
            
            # Inference
            t0 = time.time()
            raw_prob = float(model.predict(input_tensor, verbose=0)[0][0])
            inf_latency_ms = round((time.time() - t0) * 1000.0, 2)
        
        predicted_class = "parkinson" if raw_prob >= threshold else "healthy"
        prediction_label = "Parkinson's Disease Pattern Detected" if predicted_class == "parkinson" else "Healthy Control Pattern Detected"
        
        # Convert letterboxed image to base64 preview
        buffered = io.BytesIO()
        letterboxed.save(buffered, format="JPEG", quality=85)
        preview_b64 = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return {
            "model_name": model_name,
            "predicted_class": predicted_class,
            "prediction_label": prediction_label,
            "confidence_score": round(raw_prob, 4),
            "decision_threshold": threshold,
            "probabilities": {
                "healthy": round(1.0 - raw_prob, 4),
                "parkinson": round(raw_prob, 4)
            },
            "inference_time_ms": inf_latency_ms,
            "letterboxed_preview_base64": preview_b64,
            "medical_disclaimer": settings.MEDICAL_DISCLAIMER
        }
        
    def explain_image(self, image_bytes: bytes, model_name: str = "custom_cnn") -> Dict[str, Any]:
        """Generates Grad-CAM visual explanation."""
        model_name = model_name.lower()
        model = self.load_model(model_name)
        model_type = self.model_configs[model_name]["model_type"]
        
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor, letterboxed = prepare_image(pil_img, target_size=(224, 224), model_type=model_type, augment=False)
        
        # Prediction
        raw_prob = float(model.predict(input_tensor, verbose=0)[0][0])
        predicted_class = "parkinson" if raw_prob >= settings.DECISION_THRESHOLD else "healthy"
        
        # Target layer
        target_layer = find_target_conv_layer(model, model_name)
        
        # Compute Grad-CAM
        heatmap = generate_gradcam_heatmap(input_tensor, model, last_conv_layer_name=target_layer)
        blended, colored_heat = overlay_gradcam(letterboxed, heatmap, alpha=0.5, colormap_name="jet")
        
        def to_b64(img: Image.Image) -> str:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
            
        interpretation = (
            f"Grad-CAM computes activation gradients of the terminal convolutional stage ('{target_layer}') "
            f"with respect to the predicted category ('{predicted_class.upper()}'). The intense warm regions (red/yellow) "
            f"highlight specific spatial segments of the drawing where pen tremor deviations or stroke irregularity "
            f"contributed most heavily to the model's decision."
        )
        
        limitations = (
            "Grad-CAM highlights correlation, not causation. Attention can be biased by background scanner contrast, "
            "line thickness variations, or letterboxing borders. In a small dataset (204 images), saliency maps must be "
            "audited by a clinician to verify whether focus corresponds to genuine kinematic motor tremors."
        )
        
        return {
            "model_name": model_name,
            "predicted_class": predicted_class,
            "confidence_score": round(raw_prob, 4),
            "target_layer_name": target_layer,
            "original_image_base64": to_b64(letterboxed),
            "heatmap_image_base64": to_b64(colored_heat),
            "overlay_image_base64": to_b64(blended),
            "interpretation": interpretation,
            "limitations": limitations,
            "medical_disclaimer": settings.MEDICAL_DISCLAIMER
        }
        
    def get_evaluation_metrics(self) -> Dict[str, Any]:
        """Loads and returns all persisted evaluation results and data audit."""
        models_eval_path = os.path.join(settings.ARTIFACTS_DIR, "models_evaluation.json")
        subj_indep_path = os.path.join(settings.ARTIFACTS_DIR, "subject_independent_results.json")
        audit_path = os.path.join(settings.ARTIFACTS_DIR, "dataset_audit.json")
        
        results = {"models": {}, "subject_independent_summary": {}, "dataset_audit": {}}
        
        if os.path.exists(models_eval_path):
            with open(models_eval_path) as f:
                results["models"] = json.load(f)

        # Merge YOLO11s-cls verified test metrics if available
        if os.path.exists(settings.YOLO11_METRICS_PATH):
            with open(settings.YOLO11_METRICS_PATH) as f:
                y_data = json.load(f)
            y_cfg = {}
            if os.path.exists(settings.YOLO11_CONFIG_PATH):
                with open(settings.YOLO11_CONFIG_PATH) as f:
                    y_cfg = json.load(f)
            y_test = y_data.get("test", {})
            results["models"]["yolo11"] = {
                "accuracy": y_test.get("accuracy", 0.0),
                "precision": y_test.get("precision", 0.0),
                "recall": y_test.get("recall", 0.0),
                "sensitivity": y_test.get("sensitivity", y_test.get("recall", 0.0)),
                "specificity": y_test.get("specificity", 0.0),
                "f1": y_test.get("f1_score", 0.0),
                "balanced_accuracy": y_test.get("balanced_accuracy", 0.0),
                "roc_auc": y_test.get("roc_auc", 0.0),
                "pr_auc": y_test.get("pr_auc", 0.0),
                "mcc": y_test.get("mcc", 0.0),
                "brier_score": y_test.get("brier_score", 0.0),
                "confusion_matrix": y_test.get("confusion_matrix", {}),
                "threshold": y_test.get("threshold", 0.5),
                "sample_count": y_test.get("sample_count", 40),
                "total_params": y_cfg.get("total_params"),
                "trainable_params": y_cfg.get("trainable_params"),
                "train_time_sec": y_cfg.get("train_time_sec"),
                "inference_latency_ms": y_test.get("inference_latency_ms"),
                "model_size_mb": y_cfg.get("model_size_mb"),
                "roc_curve": {"fpr": [], "tpr": []},
                "pr_curve": {"recall": [], "precision": []},
            }
                
        if os.path.exists(subj_indep_path):
            with open(subj_indep_path) as f:
                results["subject_independent_summary"] = json.load(f)
                
        if os.path.exists(audit_path):
            with open(audit_path) as f:
                results["dataset_audit"] = json.load(f)
                
        return results


model_service = ModelService()

