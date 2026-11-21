"""
Pydantic Request and Response Schemas for the Parkinson Detection API.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    project: str
    version: str
    models_available: List[str]
    dataset_manifest_exists: bool
    medical_disclaimer: str


class ModelInfo(BaseModel):
    id: str
    name: str
    architecture: str
    description: str
    input_shape: List[int]
    is_trained: bool
    total_params: Optional[int] = None
    trainable_params: Optional[int] = None
    model_size_mb: Optional[float] = None
    accuracy: Optional[float] = None
    roc_auc: Optional[float] = None


class ModelsListResponse(BaseModel):
    models: List[ModelInfo]
    default_model: str


class PredictionResponse(BaseModel):
    model_name: str
    predicted_class: str
    prediction_label: str
    confidence_score: float
    decision_threshold: float
    probabilities: Dict[str, float]
    inference_time_ms: float
    letterboxed_preview_base64: Optional[str] = None
    medical_disclaimer: str


class ExplanationResponse(BaseModel):
    model_name: str
    predicted_class: str
    confidence_score: float
    target_layer_name: str
    original_image_base64: str
    heatmap_image_base64: str
    overlay_image_base64: str
    interpretation: str
    limitations: str
    medical_disclaimer: str


class MetricsResponse(BaseModel):
    models: Dict[str, Any]
    subject_independent_summary: Optional[Dict[str, Any]] = None
    dataset_audit: Optional[Dict[str, Any]] = None

