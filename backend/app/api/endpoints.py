"""
FastAPI REST API Endpoints for Parkinson's Disease Drawing Classification.
"""

import os
from typing import Optional
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Query
from fastapi.responses import FileResponse

from backend.app.core.config import settings
from backend.app.schemas.schemas import (
    HealthResponse,
    ModelsListResponse,
    PredictionResponse,
    ExplanationResponse,
    MetricsResponse
)
from backend.app.services.model_service import model_service

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint reporting API status, loaded models, and disclaimers."""
    models_meta = model_service.get_available_models()
    trained_models = [m["id"] for m in models_meta["models"] if m["is_trained"]]
    manifest_exists = os.path.exists(os.path.join(settings.ARTIFACTS_DIR, "dataset_manifest.csv"))
    
    return HealthResponse(
        status="healthy",
        project=settings.PROJECT_NAME,
        version=settings.VERSION,
        models_available=trained_models,
        dataset_manifest_exists=manifest_exists,
        medical_disclaimer=settings.MEDICAL_DISCLAIMER
    )


@router.get("/models", response_model=ModelsListResponse)
async def list_models():
    """Returns metadata, status, parameters, and architectures for all deep learning models."""
    return model_service.get_available_models()


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Returns actual evaluated test metrics, 5-fold cross-validation scores, and data audit."""
    return model_service.get_evaluation_metrics()


@router.post("/predict", response_model=PredictionResponse)
async def predict_spiral(
    file: UploadFile = File(..., description="Uploaded spiral drawing image (JPEG/PNG)"),
    model_name: str = Form(default="custom_cnn", description="Architecture ID (custom_cnn, vgg16, resnet50)"),
    threshold: float = Form(default=0.5, description="Binary classification threshold")
):
    """
    Classifies an uploaded spiral drawing into 'Healthy' or 'Parkinson'.
    Applies aspect-ratio preserving letterbox padding and model-specific preprocessing.
    """
    # 1. Validate content type
    if file.content_type not in settings.ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Please upload JPEG or PNG images."
        )
        
    # 2. Read bytes and validate file size
    contents = await file.read()
    if len(contents) > settings.MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum allowed size of {settings.MAX_UPLOAD_SIZE_BYTES / (1024*1024)}MB."
        )
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        
    # 3. Perform prediction
    try:
        prediction_result = model_service.predict_image(
            image_bytes=contents,
            model_name=model_name,
            threshold=threshold
        )
        return prediction_result
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=str(fnf))
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@router.post("/explain", response_model=ExplanationResponse)
async def explain_spiral(
    file: UploadFile = File(..., description="Uploaded spiral drawing image"),
    model_name: str = Form(default="custom_cnn", description="Target model for Grad-CAM")
):
    """
    Computes and returns a Grad-CAM saliency map explaining model attention.
    """
    if file.content_type not in settings.ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{file.content_type}'.")
        
    contents = await file.read()
    if len(contents) > settings.MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="File too large.")
        
    try:
        explanation = model_service.explain_image(image_bytes=contents, model_name=model_name)
        return explanation
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=str(fnf))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explainability computation failed: {str(e)}")


@router.get("/plots/{category}/{filename}")
async def get_plot_image(category: str, filename: str):
    """Serves generated EDA and evaluation figures."""
    category_map = {
        "eda": settings.EDA_DIR,
        "evaluation": settings.EVAL_DIR,
        "explainability": settings.EXPLAIN_DIR
    }
    if category not in category_map:
        raise HTTPException(status_code=404, detail="Invalid plot category.")
        
    file_path = os.path.join(category_map[category], filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Plot not found.")
        
    return FileResponse(file_path, media_type="image/png")

