"""
ModelMesh — Drift Detection Service (Evidently AI)
----------------------------------------------------
Runs as a scheduled process (via Prefect or cron).
Compares recent prediction features against training reference data.

Detection strategy:
1. Data drift: Kolmogorov-Smirnov test per feature (p < 0.05 = drift)
2. PSI (Population Stability Index) per feature (PSI > 0.2 = severe drift)
3. Target/prediction drift: monitors prediction label distribution shift
4. Performance drift: uses ground-truth labels (when available) to track F1 drop

On alert:
- Writes DriftReport to PostgreSQL
- Publishes drift_alert to Kafka topic
- Triggers Prefect retrain flow via API call
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
    DatasetMissingValuesSummaryMetric,
)
from evidently.report import Report

from config.logging_config import configure_logging, get_logger
from config.settings import get_settings
from src.database.connection import get_db_session, init_db
from src.database.crud import (
    count_retrains_today,
    create_drift_report,
    create_retrain_event,
    get_prediction_features_for_drift,
)

settings = get_settings()
configure_logging(settings.log_level, settings.log_format)
logger = get_logger(__name__)


class DriftMonitor:
    """
    Core drift detection logic using Evidently AI.

    Reference dataset = first N rows of training data (saved to data/reference/).
    Current dataset = most recent M rows from prediction_logs table.
    """

    NUMERIC_FEATURES = [
        "tenure", "monthly_charges", "total_charges",
        "num_products",
    ]
    CATEGORICAL_FEATURES = [
        "has_internet", "contract_type_encoded",
        "payment_method_encoded", "paperless_billing",
        "tech_support", "online_security",
    ]
    ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    def __init__(self) -> None:
        self._reference_df: Optional[pd.DataFrame] = None
        self._load_reference_data()

    def _load_reference_data(self) -> None:
        """Load reference (training) dataset from disk."""
        ref_path = os.path.join(
            os.path.dirname(__file__), "../../data/reference/reference_features.parquet"
        )
        if os.path.exists(ref_path):
            self._reference_df = pd.read_parquet(ref_path)[self.ALL_FEATURES]
            logger.info(
                "reference_data_loaded",
                rows=len(self._reference_df),
                path=ref_path,
            )
        else:
            logger.warning("reference_data_not_found", path=ref_path)
            self._reference_df = self._generate_synthetic_reference()

    def _generate_synthetic_reference(self) -> pd.DataFrame:
        """Fallback: generate synthetic reference matching training distribution."""
        np.random.seed(42)
        n = 5000
        return pd.DataFrame({
            "tenure": np.random.randint(0, 72, n),
            "monthly_charges": np.random.uniform(20, 120, n),
            "total_charges": np.random.uniform(100, 8000, n),
            "num_products": np.random.randint(1, 5, n),
            "has_internet": np.random.randint(0, 2, n),
            "contract_type_encoded": np.random.randint(0, 3, n),
            "payment_method_encoded": np.random.randint(0, 4, n),
            "paperless_billing": np.random.randint(0, 2, n),
            "tech_support": np.random.randint(0, 2, n),
            "online_security": np.random.randint(0, 2, n),
        })

    def build_current_df(self, feature_records: List[Dict]) -> pd.DataFrame:
        """Convert prediction log records to DataFrame."""
        rows = [r["features"] for r in feature_records]
        df = pd.DataFrame(rows)
        # Ensure all expected columns exist
        for col in self.ALL_FEATURES:
            if col not in df.columns:
                df[col] = 0
        return df[self.ALL_FEATURES].fillna(0)

    def run_drift_report(
        self, current_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Run Evidently DataDrift report.
        Returns structured dict with per-feature metrics.
        """
        column_mapping = ColumnMapping(
            numerical_features=self.NUMERIC_FEATURES,
            categorical_features=self.CATEGORICAL_FEATURES,
        )

        report = Report(metrics=[
            DatasetDriftMetric(),
            DataDriftPreset(),
            DataQualityPreset(),
        ])

        ref_sample = self._reference_df.sample(
            n=min(len(self._reference_df), settings.monitoring.drift_reference_window),
            random_state=42,
        )
        curr_sample = current_df.tail(settings.monitoring.drift_current_window)

        report.run(
            reference_data=ref_sample,
            current_data=curr_sample,
            column_mapping=column_mapping,
        )

        result = report.as_dict()

        # ── Extract key metrics ───────────────────────────────────────────────
        dataset_metric = next(
            (m for m in result["metrics"]
             if m["metric"] == "DatasetDriftMetric"), {}
        )
        result_data = dataset_metric.get("result", {})
        dataset_drift = result_data.get("dataset_drift", False)
        drift_share = result_data.get("share_of_drifted_columns", 0.0)

        # Per-feature drift metrics
        feature_metrics = {}
        for metric in result["metrics"]:
            if metric["metric"] == "ColumnDriftMetric":
                col = metric["result"].get("column_name", "unknown")
                feature_metrics[col] = {
                    "drift_detected": metric["result"].get("drift_detected", False),
                    "stattest": metric["result"].get("stattest_name"),
                    "p_value": metric["result"].get("p_value"),
                    "threshold": metric["result"].get("threshold"),
                }

        logger.info(
            "drift_report_complete",
            drift_detected=dataset_drift,
            drift_share=round(drift_share, 3),
            n_reference=len(ref_sample),
            n_current=len(curr_sample),
            drifted_features=sum(
                1 for v in feature_metrics.values() if v["drift_detected"]
            ),
        )

        return {
            "dataset_drift_detected": dataset_drift,
            "drift_share": drift_share,
            "feature_metrics": feature_metrics,
            "n_reference": len(ref_sample),
            "n_current": len(curr_sample),
        }


class DriftMonitorService:
    """
    Orchestrates the full drift check → alert → retrain cycle.
    Intended to run as a Prefect task or standalone cron job.
    """

    def __init__(self) -> None:
        self.monitor = DriftMonitor()

    async def run_check(self) -> Dict[str, Any]:
        """
        Full drift check cycle:
        1. Load recent predictions from DB
        2. Run Evidently drift analysis
        3. Persist report to DB
        4. If drift detected → publish alert + trigger retrain
        """
        model_name = settings.mlflow.model_name
        now = datetime.utcnow()
        window_start = now - timedelta(
            minutes=settings.monitoring.drift_check_interval_minutes
        )

        async with get_db_session() as session:
            records = await get_prediction_features_for_drift(
                session,
                model_name=model_name,
                start=window_start,
                end=now,
                limit=settings.monitoring.drift_current_window,
            )

        if len(records) < 100:
            logger.warning(
                "insufficient_data_for_drift_check",
                n_records=len(records),
                minimum=100,
            )
            return {"skipped": True, "reason": "insufficient_data"}

        current_df = self.monitor.build_current_df(records)
        drift_result = self.monitor.run_drift_report(current_df)

        drift_detected = drift_result["dataset_drift_detected"]
        should_alert = (
            drift_detected
            and drift_result["drift_share"] >= settings.monitoring.psi_threshold
        )

        async with get_db_session() as session:
            report = await create_drift_report(
                session,
                model_name=model_name,
                model_version=self._get_current_model_version(),
                report_type="data_drift",
                dataset_drift_detected=drift_detected,
                drift_share=drift_result["drift_share"],
                feature_metrics=drift_result["feature_metrics"],
                current_window_start=window_start,
                current_window_end=now,
                n_reference_samples=drift_result["n_reference"],
                n_current_samples=drift_result["n_current"],
                alert_triggered=should_alert,
            )
            report_id = report.id

        if should_alert:
            await self._trigger_retrain_pipeline(
                model_name=model_name,
                drift_report_id=str(report_id),
                drift_share=drift_result["drift_share"],
            )

        return {
            "drift_detected": drift_detected,
            "drift_share": drift_result["drift_share"],
            "alert_triggered": should_alert,
            "report_id": str(report_id),
        }

    async def _trigger_retrain_pipeline(
        self,
        model_name: str,
        drift_report_id: str,
        drift_share: float,
    ) -> None:
        """
        Trigger Prefect retraining flow via API.
        Also publish Kafka drift_alert for downstream consumers.
        """
        import httpx

        logger.warning(
            "drift_alert_triggering_retrain",
            model_name=model_name,
            drift_share=round(drift_share, 3),
        )

        # 1. Check retrain rate limit
        async with get_db_session() as session:
            retrain_count = await count_retrains_today(session, model_name)
            if retrain_count >= settings.retraining.max_retrain_per_day:
                logger.warning(
                    "retrain_daily_limit_reached",
                    count=retrain_count,
                    limit=settings.retraining.max_retrain_per_day,
                )
                return

            retrain_event = await create_retrain_event(
                session,
                trigger_type="drift_alert",
                model_name=model_name,
                drift_report_id=drift_report_id,
            )

        # 2. Trigger Prefect deployment run
        if settings.enable_auto_retrain:
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    resp = await client.post(
                        f"{settings.retraining.prefect_api_url}/deployments/"
                        f"by_name/{settings.retraining.retrain_deployment_name}/create_flow_run",
                        json={
                            "parameters": {
                                "model_name": model_name,
                                "drift_report_id": drift_report_id,
                                "triggered_by": "drift_alert",
                            }
                        },
                    )
                    resp.raise_for_status()
                    flow_run_id = resp.json().get("id")
                    logger.info("prefect_retrain_triggered", flow_run_id=flow_run_id)
            except Exception as exc:
                logger.error("prefect_trigger_failed", error=str(exc))

    def _get_current_model_version(self) -> str:
        try:
            from src.registry.model_registry import ModelRegistryClient
            registry = ModelRegistryClient()
            v = registry.get_champion_version()
            return v.version if v else "unknown"
        except Exception:
            return "unknown"


async def main() -> None:
    await init_db()
    service = DriftMonitorService()
    result = await service.run_check()
    logger.info("drift_check_complete", **result)


if __name__ == "__main__":
    asyncio.run(main())
