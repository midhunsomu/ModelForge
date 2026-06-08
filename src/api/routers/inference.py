"""
ModelMesh — Inference Router
------------------------------
Endpoints:
  POST /v1/predict          — single prediction
  POST /v1/predict/batch    — batch prediction (up to 128)
  POST /v1/feedback         — submit ground-truth label
  GET  /v1/shadow/metrics   — live A/B shadow metrics

Full flow per request:
  1. Validate features via Pydantic
  2. Encode categorical features
  3. Run champion prediction (sync)
  4. Fire challenger shadow inference (async, non-blocking)
  5. Optionally compute SHAP values
  6. Log to PostgreSQL + Kafka (async, non-blocking)
  7. Return response with X-Model-Version header
"""

from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse

from config.logging_config import get_logger, set_model_version
from config.settings import get_settings
from src.api.middleware.auth import verify_api_key
from src.api.schemas.inference import (
    BatchInferenceRequest,
    BatchPredictionResponse,
    FeedbackRequest,
    FeedbackResponse,
    SHAPExplanation,
    SingleInferenceRequest,
    SinglePrediction,
)
from src.database.connection import get_db_session
from src.database.crud import create_prediction_log, update_prediction_feedback

logger = get_logger(__name__)
settings = get_settings()
router = APIRouter()


# ── Feature Encoding ──────────────────────────────────────────────────────────

CONTRACT_ENCODING = {"Month-to-month": 0, "One year": 1, "Two year": 2}
PAYMENT_ENCODING = {
    "Electronic check": 0,
    "Mailed check": 1,
    "Bank transfer (automatic)": 2,
    "Credit card (automatic)": 3,
}


def encode_features(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Convert categorical string features to integer codes used in training."""
    return {
        "tenure": raw["tenure"],
        "monthly_charges": raw["monthly_charges"],
        "total_charges": raw["total_charges"],
        "num_products": raw["num_products"],
        "has_internet": raw["has_internet"],
        "contract_type_encoded": CONTRACT_ENCODING.get(raw["contract_type"], 0),
        "payment_method_encoded": PAYMENT_ENCODING.get(raw["payment_method"], 0),
        "paperless_billing": raw["paperless_billing"],
        "tech_support": raw["tech_support"],
        "online_security": raw["online_security"],
    }


def hash_ip(ip: str) -> str:
    """One-way hash of client IP for privacy-safe logging."""
    return hashlib.sha256(ip.encode()).hexdigest()[:16]


# ── Single Prediction ─────────────────────────────────────────────────────────

@router.post(
    "/predict",
    response_model=SinglePrediction,
    summary="Single real-time prediction",
    description="Returns churn probability for one customer. Optionally includes SHAP explanation.",
)
async def predict_single(
    request_body: SingleInferenceRequest,
    http_request: Request,
    response: Response,
    _: str = Depends(verify_api_key),
) -> SinglePrediction:
    trace_id = http_request.headers.get("X-Trace-ID", str(uuid.uuid4()))
    model_loader = http_request.app.state.model_loader
    kafka_producer = http_request.app.state.kafka_producer

    # ── Encode features ───────────────────────────────────────────────────────
    raw_features = request_body.instance.model_dump()
    encoded_features = encode_features(raw_features)

    # ── Champion prediction ───────────────────────────────────────────────────
    proba, label, latency_ms = model_loader.shadow_router.predict_champion(
        encoded_features
    )
    champ_version = model_loader.champion_version
    set_model_version(champ_version)
    response.headers["X-Model-Version"] = champ_version

    # ── SHAP explanation (sampled) ────────────────────────────────────────────
    shap_result = None
    if (request_body.include_shap or model_loader.shap_explainer.should_explain()):
        shap_result = model_loader.shap_explainer.explain(encoded_features)

    shap_schema = None
    if shap_result:
        shap_schema = SHAPExplanation(**shap_result)

    # ── Shadow inference (fire and forget) ────────────────────────────────────
    if request_body.shadow_eligible:
        asyncio.create_task(
            model_loader.shadow_router.predict_shadow_async(
                features=encoded_features,
                champion_label=label,
            )
        )

    # ── Async logging (DB + Kafka) ─────────────────────────────────────────────
    client_ip = http_request.headers.get("X-Forwarded-For", "")
    if http_request.client:
        client_ip = client_ip or http_request.client.host

    asyncio.create_task(
        _log_prediction(
            trace_id=trace_id,
            model_version=champ_version,
            model_stage="champion",
            prediction=proba,
            label=label,
            features=encoded_features,
            latency_ms=latency_ms,
            shap_values=shap_result,
            client_ip=hash_ip(client_ip) if client_ip else None,
            kafka_producer=kafka_producer,
        )
    )

    logger.info(
        "prediction_served",
        label=label,
        proba=round(proba, 4),
        model_version=champ_version,
        latency_ms=round(latency_ms, 2),
    )

    return SinglePrediction(
        prediction=label,
        probability=round(proba, 5),
        model_name=settings.mlflow.model_name,
        model_version=champ_version,
        model_stage="champion",
        trace_id=trace_id,
        latency_ms=round(latency_ms, 2),
        shap_explanation=shap_schema,
    )


# ── Batch Prediction ──────────────────────────────────────────────────────────

@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Batch prediction (up to 128 instances)",
)
async def predict_batch(
    request_body: BatchInferenceRequest,
    http_request: Request,
    response: Response,
    _: str = Depends(verify_api_key),
) -> BatchPredictionResponse:
    t_total = time.perf_counter()
    model_loader = http_request.app.state.model_loader
    kafka_producer = http_request.app.state.kafka_producer

    encoded_list = [
        encode_features(inst.model_dump()) for inst in request_body.instances
    ]

    # Compute SHAP for whole batch if requested
    shap_results = None
    if request_body.include_shap:
        shap_results = model_loader.shap_explainer.explain_batch(encoded_list)

    predictions: List[SinglePrediction] = []
    for i, encoded in enumerate(encoded_list):
        trace_id = f"{request_body.request_id or uuid.uuid4()}-{i}"
        proba, label, latency_ms = model_loader.shadow_router.predict_champion(encoded)

        shap_schema = None
        if shap_results and shap_results[i]:
            shap_schema = SHAPExplanation(**shap_results[i])

        asyncio.create_task(
            _log_prediction(
                trace_id=trace_id,
                model_version=model_loader.champion_version,
                model_stage="champion",
                prediction=proba,
                label=label,
                features=encoded,
                latency_ms=latency_ms,
                shap_values=shap_results[i] if shap_results else None,
                kafka_producer=kafka_producer,
            )
        )

        predictions.append(
            SinglePrediction(
                prediction=label,
                probability=round(proba, 5),
                model_name=settings.mlflow.model_name,
                model_version=model_loader.champion_version,
                model_stage="champion",
                trace_id=trace_id,
                latency_ms=round(latency_ms, 2),
                shap_explanation=shap_schema,
            )
        )

    total_ms = (time.perf_counter() - t_total) * 1000
    response.headers["X-Model-Version"] = model_loader.champion_version

    return BatchPredictionResponse(
        predictions=predictions,
        total_latency_ms=round(total_ms, 2),
        model_name=settings.mlflow.model_name,
        model_version=model_loader.champion_version,
        batch_size=len(predictions),
    )


# ── Feedback Endpoint ─────────────────────────────────────────────────────────

@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    summary="Submit ground-truth label for a prediction",
)
async def submit_feedback(
    body: FeedbackRequest,
    _: str = Depends(verify_api_key),
) -> FeedbackResponse:
    async with get_db_session() as session:
        updated = await update_prediction_feedback(
            session,
            trace_id=body.trace_id,
            actual_label=body.actual_label,
        )
    if not updated:
        return FeedbackResponse(
            success=False,
            message=f"No prediction found for trace_id={body.trace_id}",
        )
    logger.info("feedback_received", trace_id=body.trace_id, label=body.actual_label)
    return FeedbackResponse(success=True, message="Feedback recorded")


# ── Shadow A/B Metrics ────────────────────────────────────────────────────────

@router.get(
    "/shadow/metrics",
    summary="Live champion vs challenger shadow metrics",
)
async def shadow_metrics(
    http_request: Request,
    _: str = Depends(verify_api_key),
) -> Dict:
    router_obj = http_request.app.state.model_loader.shadow_router
    return router_obj.get_shadow_metrics_snapshot()


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _log_prediction(
    *,
    trace_id: str,
    model_version: str,
    model_stage: str,
    prediction: float,
    label: int,
    features: Dict,
    latency_ms: float,
    shap_values: Any = None,
    client_ip: str = None,
    kafka_producer: Any = None,
) -> None:
    """Fire-and-forget: write prediction to DB and Kafka."""
    try:
        async with get_db_session() as session:
            await create_prediction_log(
                session,
                trace_id=trace_id,
                model_name=settings.mlflow.model_name,
                model_version=model_version,
                model_stage=model_stage,
                prediction=prediction,
                prediction_label=label,
                prediction_proba=prediction,
                features=features,
                latency_ms=latency_ms,
                shap_values=shap_values,
                client_ip=client_ip,
            )
    except Exception as exc:
        logger.error("prediction_db_log_failed", error=str(exc))

    if kafka_producer and settings.enable_kafka_streaming:
        try:
            kafka_producer.send_prediction(
                trace_id=trace_id,
                model_version=model_version,
                prediction=prediction,
                label=label,
                features=features,
            )
        except Exception as exc:
            logger.error("kafka_produce_failed", error=str(exc))
