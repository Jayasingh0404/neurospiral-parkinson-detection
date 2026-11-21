"""
Direct Integration & Endpoint Test Runner for NeuroSpiral Backend.
Validates model inference, Grad-CAM saliency generation, and API responses directly.
"""

import os
import sys
import asyncio
from PIL import Image
import numpy as np

# Ensure workspace root is in path
workspace_dir = "/Users/jayasinghrajput/PARKINSON"
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)

from backend.app.services.model_service import model_service
from backend.app.api.endpoints import health_check, list_models, get_metrics


async def run_all_tests():
    print("=" * 60)
    print("NeuroSpiral Backend Integration & Validation Test Suite")
    print("=" * 60)

    # 1. Test Health Endpoint
    print("\n[Test 1] Health Check Endpoint...")
    health = await health_check()
    print("  Status:", health.status)
    print("  Models available:", health.models_available)
    assert health.status == "healthy", f"Expected healthy status, got {health.status}"
    assert health.dataset_manifest_exists is True, "Dataset manifest must exist"
    print("  ✓ Health Check Passed")

    # 2. Test Models Endpoint
    print("\n[Test 2] Models Inventory Endpoint...")
    models_res = await list_models()
    models_list = models_res["models"] if isinstance(models_res, dict) else models_res.models
    print(f"  Total models registered: {len(models_list)}")
    for m in models_list:
        m_name = m["name"] if isinstance(m, dict) else m.name
        m_id = m["id"] if isinstance(m, dict) else m.id
        m_arch = m["architecture"] if isinstance(m, dict) else m.architecture
        m_trained = m["is_trained"] if isinstance(m, dict) else m.is_trained
        print(f"   - {m_name} ({m_id}): {m_arch} | Trained: {m_trained}")
    model_ids = [m["id"] if isinstance(m, dict) else m.id for m in models_list]
    assert "custom_cnn" in model_ids
    assert "vgg16" in model_ids
    assert "resnet50" in model_ids
    print("  ✓ Models Inventory Passed")

    # 3. Test Metrics Endpoint
    print("\n[Test 3] Metrics & Cross-Validation Endpoint...")
    metrics_res = await get_metrics()
    assert "dataset_audit" in metrics_res
    print(f"  Total Audited Images: {metrics_res['dataset_audit']['total_images']}")
    total_subj = metrics_res['dataset_audit']['total_unique_subjects']
    leaking_subj = metrics_res['dataset_audit']['subjects_with_leakage_count']
    print(f"  Unique Subjects: {total_subj} (Leaking in static split: {leaking_subj})")
    print("  ✓ Metrics Endpoint Passed")

    # 4. Test Live Inference with Real Parkinson Spiral Image
    test_img_path = os.path.join(
        workspace_dir, 
        "ml/data/raw/Parkinson Disease detection/test/parkinson/V01PO01_png.rf.8dcf54d3d17c25b69468a13d9387ad7b.jpg"
    )
    assert os.path.exists(test_img_path), f"Test image not found at {test_img_path}"
    with open(test_img_path, "rb") as f:
        img_bytes = f.read()
    print(f"\n[Test 4] Inference on Real Test Image: {os.path.basename(test_img_path)} ({len(img_bytes)} bytes)...")

    for model_name in ["custom_cnn", "vgg16", "resnet50"]:
        pred = model_service.predict_image(img_bytes, model_name=model_name, threshold=0.50)
        print(f"  Model: {model_name.upper()}")
        print(f"    Predicted Class: {pred['predicted_class']} ({pred['prediction_label']})")
        print(f"    Parkinson Prob: {pred['probabilities']['parkinson']:.4f} | Confidence: {pred['confidence_score']:.4f}")
        print(f"    Inference Latency: {pred['inference_time_ms']:.1f} ms")
        assert pred["predicted_class"] in ["healthy", "parkinson"]
        assert 0.0 <= pred["confidence_score"] <= 1.0

    print("  ✓ Multi-Model Inference Tests Passed")

    # 5. Test Grad-CAM Explainability Synthesis
    print("\n[Test 5] Grad-CAM Explainability Generation...")
    for model_name in ["vgg16", "resnet50", "custom_cnn"]:
        print(f"  Generating Grad-CAM for {model_name}...")
        explanation = model_service.explain_image(img_bytes, model_name=model_name)
        print(f"    Target Layer: {explanation['target_layer_name']}")
        print(f"    Predicted Class: {explanation['predicted_class']}")
        print(f"    Heatmap Data URI length: {len(explanation['heatmap_image_base64'])} chars")
        print(f"    Overlay Data URI length: {len(explanation['overlay_image_base64'])} chars")
        assert explanation["heatmap_image_base64"].startswith("data:image/")
        assert explanation["overlay_image_base64"].startswith("data:image/")
    print("  ✓ Grad-CAM Explainability Tests Passed")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED WITH 100% SUCCESS")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_all_tests())

