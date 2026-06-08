"""
ModelMesh — Database CRUD Layer
--------------------------------
All database interactions go through this module.
Keeps raw SQL / ORM calls out of business logic.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import desc, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from config.logging_config import get_logger
from src.database.models import (
    ABTestResult,
    DriftReport,
    ModelMetadata,
    PredictionLog,
    RetrainEvent,
)

logger = get_logger(__name__)


# ── Prediction Logs ────────────────────────────────────────────────────────────

async def create_prediction_log(
    session: AsyncSession,
    *,
    trace_id: str,
    model_name: str,
    model_version: str,
    model_stage: str,
    prediction: float,
    prediction_label: Optional[int],
    prediction_proba: Optional[float],
    features: Dict[str, Any],
    latency_ms: float,
    shap_values: Optional[Dict] = None,
    client_ip: Optional[str] = None,
    cost_usd: Optional[float] = None,
) -> PredictionLog:
    log = PredictionLog(
        trace_id=trace_id,
        model_name=model_name,
        model_version=model_version,
        model_stage=model_stage,
        prediction=prediction,
        prediction_label=prediction_label,
        prediction_proba=prediction_proba,
        features=features,
        latency_ms=latency_ms,
        shap_values=shap_values,
        client_ip=client_ip,
        cost_usd=cost_usd,
    )
    session.add(log)
    await session.flush()
    return log


async def get_recent_predictions(
    session: AsyncSession,
    *,
    model_name: str,
    model_version: Optional[str] = None,
    limit: int = 1000,
    since: Optional[datetime] = None,
) -> List[PredictionLog]:
    stmt = (
        select(PredictionLog)
        .where(PredictionLog.model_name == model_name)
        .order_by(desc(PredictionLog.created_at))
        .limit(limit)
    )
    if model_version:
        stmt = stmt.where(PredictionLog.model_version == model_version)
    if since:
        stmt = stmt.where(PredictionLog.created_at >= since)

    result = await session.execute(stmt)
    return list(result.scalars().all())


async def get_prediction_features_for_drift(
    session: AsyncSession,
    *,
    model_name: str,
    start: datetime,
    end: datetime,
    limit: int = 5000,
) -> List[Dict[str, Any]]:
    """Return feature dicts for Evidently drift analysis."""
    stmt = (
        select(PredictionLog.features, PredictionLog.created_at)
        .where(
            PredictionLog.model_name == model_name,
            PredictionLog.created_at.between(start, end),
        )
        .order_by(PredictionLog.created_at)
        .limit(limit)
    )
    result = await session.execute(stmt)
    return [{"features": row.features, "ts": row.created_at} for row in result]


async def update_prediction_feedback(
    session: AsyncSession,
    *,
    trace_id: str,
    actual_label: int,
) -> bool:
    stmt = (
        update(PredictionLog)
        .where(PredictionLog.trace_id == trace_id)
        .values(
            actual_label=actual_label,
            feedback_received_at=datetime.utcnow(),
        )
    )
    result = await session.execute(stmt)
    return result.rowcount > 0


# ── Drift Reports ──────────────────────────────────────────────────────────────

async def create_drift_report(
    session: AsyncSession,
    *,
    model_name: str,
    model_version: str,
    report_type: str,
    dataset_drift_detected: bool,
    drift_share: Optional[float] = None,
    feature_metrics: Optional[Dict] = None,
    performance_metrics: Optional[Dict] = None,
    reference_window_start: Optional[datetime] = None,
    reference_window_end: Optional[datetime] = None,
    current_window_start: Optional[datetime] = None,
    current_window_end: Optional[datetime] = None,
    n_reference_samples: Optional[int] = None,
    n_current_samples: Optional[int] = None,
    alert_triggered: bool = False,
) -> DriftReport:
    report = DriftReport(
        model_name=model_name,
        model_version=model_version,
        report_type=report_type,
        dataset_drift_detected=dataset_drift_detected,
        drift_share=drift_share,
        feature_metrics=feature_metrics,
        performance_metrics=performance_metrics,
        reference_window_start=reference_window_start,
        reference_window_end=reference_window_end,
        current_window_start=current_window_start,
        current_window_end=current_window_end,
        n_reference_samples=n_reference_samples,
        n_current_samples=n_current_samples,
        alert_triggered=alert_triggered,
    )
    session.add(report)
    await session.flush()
    logger.info(
        "drift_report_created",
        model=model_name,
        drift_detected=dataset_drift_detected,
        drift_share=drift_share,
    )
    return report


async def get_latest_drift_report(
    session: AsyncSession, model_name: str
) -> Optional[DriftReport]:
    stmt = (
        select(DriftReport)
        .where(DriftReport.model_name == model_name)
        .order_by(desc(DriftReport.created_at))
        .limit(1)
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


# ── Retrain Events ─────────────────────────────────────────────────────────────

async def create_retrain_event(
    session: AsyncSession,
    *,
    trigger_type: str,
    model_name: str,
    drift_report_id: Optional[uuid.UUID] = None,
    previous_version: Optional[str] = None,
) -> RetrainEvent:
    event = RetrainEvent(
        trigger_type=trigger_type,
        model_name=model_name,
        drift_report_id=drift_report_id,
        previous_version=previous_version,
        status="pending",
    )
    session.add(event)
    await session.flush()
    return event


async def update_retrain_event(
    session: AsyncSession,
    event_id: uuid.UUID,
    **kwargs: Any,
) -> None:
    stmt = (
        update(RetrainEvent)
        .where(RetrainEvent.id == event_id)
        .values(**kwargs)
    )
    await session.execute(stmt)


async def count_retrains_today(
    session: AsyncSession, model_name: str
) -> int:
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    stmt = (
        select(func.count())
        .select_from(RetrainEvent)
        .where(
            RetrainEvent.model_name == model_name,
            RetrainEvent.started_at >= today_start,
        )
    )
    result = await session.execute(stmt)
    return result.scalar_one()


# ── A/B Test Results ───────────────────────────────────────────────────────────

async def create_ab_result(
    session: AsyncSession, **kwargs: Any
) -> ABTestResult:
    result = ABTestResult(**kwargs)
    session.add(result)
    await session.flush()
    return result


async def get_latest_ab_results(
    session: AsyncSession, model_name: str, limit: int = 10
) -> List[ABTestResult]:
    stmt = (
        select(ABTestResult)
        .where(ABTestResult.model_name == model_name)
        .order_by(desc(ABTestResult.created_at))
        .limit(limit)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


# ── Model Metadata ─────────────────────────────────────────────────────────────

async def upsert_model_metadata(
    session: AsyncSession, **kwargs: Any
) -> ModelMetadata:
    stmt = select(ModelMetadata).where(
        ModelMetadata.model_name == kwargs["model_name"],
        ModelMetadata.version == kwargs["version"],
    )
    result = await session.execute(stmt)
    existing = result.scalar_one_or_none()
    if existing:
        for k, v in kwargs.items():
            setattr(existing, k, v)
        return existing
    meta = ModelMetadata(**kwargs)
    session.add(meta)
    await session.flush()
    return meta


async def get_model_metadata(
    session: AsyncSession, model_name: str, version: str
) -> Optional[ModelMetadata]:
    stmt = select(ModelMetadata).where(
        ModelMetadata.model_name == model_name,
        ModelMetadata.version == version,
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()
