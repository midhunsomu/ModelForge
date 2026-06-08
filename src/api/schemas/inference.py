"""
ModelMesh — Pydantic API Schemas
---------------------------------
All request/response models with full validation.
Pydantic v2 style — using model_config instead of inner Config class.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Feature Input ──────────────────────────────────────────────────────────────

class ChurnFeatures(BaseModel):
    """
    Feature schema for the churn-classification model.
    Matches the training dataset column names exactly.
    """

    model_config = {"json_schema_extra": {"example": {
        "tenure": 24,
        "monthly_charges": 65.5,
        "total_charges": 1572.0,
        "num_products": 2,
        "has_internet": 1,
        "contract_type": "Month-to-month",
        "payment_method": "Electronic check",
        "paperless_billing": 1,
        "tech_support": 0,
        "online_security": 0,
    }}}

    tenure: int = Field(..., ge=0, le=120, description="Customer tenure in months")
    monthly_charges: float = Field(..., ge=0.0, le=200.0)
    total_charges: float = Field(..., ge=0.0)
    num_products: int = Field(..., ge=1, le=10)
    has_internet: int = Field(..., ge=0, le=1)
    contract_type: str = Field(..., description="Month-to-month | One year | Two year")
    payment_method: str
    paperless_billing: int = Field(..., ge=0, le=1)
    tech_support: int = Field(..., ge=0, le=1)
    online_security: int = Field(..., ge=0, le=1)

    @field_validator("contract_type")
    @classmethod
    def validate_contract(cls, v: str) -> str:
        valid = {"Month-to-month", "One year", "Two year"}
        if v not in valid:
            raise ValueError(f"contract_type must be one of {valid}")
        return v


class BatchInferenceRequest(BaseModel):
    instances: List[ChurnFeatures] = Field(..., min_length=1, max_length=128)
    include_shap: bool = Field(default=False)
    request_id: Optional[str] = Field(default=None)


class SingleInferenceRequest(BaseModel):
    instance: ChurnFeatures
    include_shap: bool = Field(default=False)
    shadow_eligible: bool = Field(default=True)  # allow shadow routing


# ── Inference Response ─────────────────────────────────────────────────────────

class SHAPExplanation(BaseModel):
    feature_names: List[str]
    shap_values: List[float]
    base_value: float
    top_features: List[Dict[str, Any]]  # sorted by |shap| desc


class SinglePrediction(BaseModel):
    prediction: int                          # 0 = no churn, 1 = churn
    probability: float = Field(..., ge=0.0, le=1.0)
    model_name: str
    model_version: str
    model_stage: str
    trace_id: str
    latency_ms: float
    shap_explanation: Optional[SHAPExplanation] = None


class BatchPredictionResponse(BaseModel):
    predictions: List[SinglePrediction]
    total_latency_ms: float
    model_name: str
    model_version: str
    batch_size: int


# ── Feedback / Label Collection ────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    trace_id: str = Field(..., description="Trace ID returned by inference endpoint")
    actual_label: int = Field(..., ge=0, le=1)


class FeedbackResponse(BaseModel):
    success: bool
    message: str


# ── Health & Status ────────────────────────────────────────────────────────────

class ServiceStatus(BaseModel):
    name: str
    status: str        # healthy | degraded | unhealthy
    latency_ms: Optional[float] = None
    details: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    timestamp: datetime
    services: List[ServiceStatus]


class ReadinessResponse(BaseModel):
    ready: bool
    champion_model_loaded: bool
    challenger_model_loaded: bool
    database_connected: bool


# ── Drift & Monitoring ─────────────────────────────────────────────────────────

class DriftReportResponse(BaseModel):
    model_name: str
    model_version: str
    drift_detected: bool
    drift_share: Optional[float]
    feature_metrics: Optional[Dict[str, Any]]
    performance_metrics: Optional[Dict[str, Any]]
    created_at: datetime
    alert_triggered: bool


# ── Model Registry ─────────────────────────────────────────────────────────────

class ModelInfo(BaseModel):
    name: str
    version: str
    stage: str
    mlflow_run_id: str
    metrics: Optional[Dict[str, float]]
    created_at: Optional[datetime]


class ModelListResponse(BaseModel):
    models: List[ModelInfo]
    total: int


# ── Generic Responses ──────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    trace_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SuccessResponse(BaseModel):
    message: str
    data: Optional[Dict[str, Any]] = None
