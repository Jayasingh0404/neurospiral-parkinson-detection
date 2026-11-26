"""
Backend Application Configuration Settings.
"""

import os
from pydantic import BaseModel

_RESOLVED_BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


class Settings(BaseModel):
    PROJECT_NAME: str = "NeuroSpiral AI — Parkinson's Disease Classification API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    BASE_DIR: str = _RESOLVED_BASE_DIR
    ARTIFACTS_DIR: str = os.path.join(_RESOLVED_BASE_DIR, "ml", "artifacts")
    EDA_DIR: str = os.path.join(_RESOLVED_BASE_DIR, "ml", "outputs", "eda")
    EVAL_DIR: str = os.path.join(_RESOLVED_BASE_DIR, "ml", "outputs", "evaluation")
    EXPLAIN_DIR: str = os.path.join(_RESOLVED_BASE_DIR, "ml", "outputs", "explainability")
    YOLO11_WEIGHTS_PATH: str = os.path.join(_RESOLVED_BASE_DIR, "yolo11", "outputs", "weights", "best.pt")
    YOLO11_METRICS_PATH: str = os.path.join(_RESOLVED_BASE_DIR, "yolo11", "outputs", "metrics", "yolo11_metrics.json")
    YOLO11_CONFIG_PATH: str = os.path.join(_RESOLVED_BASE_DIR, "yolo11", "outputs", "metrics", "yolo11_config.json")
    YOLO11_EVAL_DIR: str = os.path.join(_RESOLVED_BASE_DIR, "yolo11", "outputs", "evaluation")

    ALLOWED_IMAGE_TYPES: list = ["image/jpeg", "image/png", "image/jpg"]
    MAX_UPLOAD_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB

    DEFAULT_MODEL: str = "yolo11"
    DECISION_THRESHOLD: float = 0.5

    MEDICAL_DISCLAIMER: str = (
        "INVESTIGATIONAL NOTICE: NeuroSpiral is a research prototype for computer vision "
        "pattern analysis on digitized drawing examinations. It is NOT a certified medical "
        "diagnostic device and must not be used for standalone clinical diagnosis."
    )


settings = Settings()
