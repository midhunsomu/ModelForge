"""ModelMesh — Metrics & Monitoring Router"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List

from fastapi import APIRouter, Depends, Query

from src.api.middleware.auth import verify_api_key
from src.api.schemas.inference import DriftReportResponse
from src.database.connection import get_db_session
from src.database.crud import get_latest_ab_results, get_latest_drift_report
from config.settings import get_settings

settings = get_settings()
router = APIRouter()


@router.get("/drift/latest", response_model=DriftReportResponse)
async def get_drift_report(_: str = Depends(verify_api_key)) -> DriftReportResponse:
    async with get_db_session() as session:
        report = await get_latest_drift_report(
            session, settings.mlflow.model_name
        )
    if not report:
        return DriftReportResponse(
            model_name=settings.mlflow.model_name,
            model_version="unknown",
            drift_detected=False,
            drift_share=None,
            feature_metrics=None,
            performance_metrics=None,
            created_at=datetime.utcnow(),
            alert_triggered=False,
        )
    return DriftReportResponse(
        model_name=report.model_name,
        model_version=report.model_version,
        drift_detected=report.dataset_drift_detected,
        drift_share=report.drift_share,
        feature_metrics=report.feature_metrics,
        performance_metrics=report.performance_metrics,
        created_at=report.created_at,
        alert_triggered=report.alert_triggered,
    )


@router.get("/ab-tests/latest")
async def get_ab_test_results(
    limit: int = Query(default=10, ge=1, le=50),
    _: str = Depends(verify_api_key),
) -> List[dict]:
    async with get_db_session() as session:
        results = await get_latest_ab_results(
            session, settings.mlflow.model_name, limit=limit
        )
    return [
        {
            "champion_version": r.champion_version,
            "challenger_version": r.challenger_version,
            "window_start": r.window_start.isoformat(),
            "window_end": r.window_end.isoformat(),
            "champion_f1": r.champion_f1,
            "challenger_f1": r.challenger_f1,
            "champion_avg_latency_ms": r.champion_avg_latency_ms,
            "challenger_avg_latency_ms": r.challenger_avg_latency_ms,
            "winner": r.winner,
        }
        for r in results
    ]
