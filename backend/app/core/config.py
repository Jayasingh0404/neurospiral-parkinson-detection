"""
Backend Application Configuration Settings.
"""

import os
from pydantic import BaseModel


class Settings(BaseModel):
    PROJECT_NAME: str = "Parkinson's Disease Spiral Classification & Explainability API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    BASE_DIR: str = "/Users/jayasinghrajput/PARKINSON"
    ARTIFACTS_DIR: str = "/Users/jayasinghrajput/PARKINSON/ml/artifacts"
    EDA_DIR: str = "/Users/jayasinghrajput/PARKINSON/ml/outputs/eda"
    EVAL_DIR: str = "/Users/jayasinghrajput/PARKINSON/ml/outputs/evaluation"
    EXPLAIN_DIR: str = "/Users/jayasinghrajput/PARKINSON/ml/outputs/explainability"
    
    ALLOWED_IMAGE_TYPES: list = ["image/jpeg", "image/png", "image/jpg"]
    MAX_UPLOAD_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB
    
    DEFAULT_MODEL: str = "custom_cnn"
    DECISION_THRESHOLD: float = 0.5
    
    MEDICAL_DISCLAIMER: str = (
        "RESEARCH PROTOTYPE NOTICE: This system is strictly an academic research and educational demonstration "
        "investigating deep learning architectures on digitized drawing patterns. It is NOT a cleared or approved "
        "medical diagnostic device and must NEVER be used for clinical diagnosis, treatment decisions, or patient triage. "
        "Clinical evaluation requires a board-certified neurologist."
    )


settings = Settings()

