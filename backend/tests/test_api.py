"""
FastAPI Test Suite for Parkinson Disease Classification Endpoints.
"""

import io
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "project" in data
    assert "medical_disclaimer" in data


def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "models_available" in data
    assert data["dataset_manifest_exists"] is True


def test_models_list_endpoint():
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    model_ids = [m["id"] for m in data["models"]]
    assert "custom_cnn" in model_ids
    assert "vgg16" in model_ids
    assert "resnet50" in model_ids


def test_metrics_endpoint():
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "dataset_audit" in data
    assert data["dataset_audit"]["total_images"] == 204


def test_predict_invalid_file_type():
    file_content = b"fake text content, not an image"
    response = client.post(
        "/api/v1/predict",
        files={"file": ("test.txt", file_content, "text/plain")},
        data={"model_name": "custom_cnn"}
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_predict_empty_file():
    response = client.post(
        "/api/v1/predict",
        files={"file": ("test.jpg", b"", "image/jpeg")},
        data={"model_name": "custom_cnn"}
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()
