"""
ModelMesh — Database Models (SQLAlchemy ORM)
--------------------------------------------
Stores prediction logs, drift reports, retrain events, and A/B test metrics.
Designed for time-series queries — uses BRIN indexes on timestamp columns
for efficient range scans over large prediction tables.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


# ── Prediction Log ─────────────────────────────────────────────────────────────

class PredictionLog(Base):
    """
    Every inference call is logged here for:
    - Drift detection (feature distribution over time)
    - Performance monitoring (label collection for re-eval)
    - Cost tracking
    - A/B test analysis
    """

    __tablename__ = "prediction_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trace_id = Column(String(64), nullable=False, index=True)
    model_name = Column(String(128), nullable=False)
    model_version = Column(String(32), nullable=False)
    model_stage = Column(String(32), nullable=False)  # champion | challenger
    prediction = Column(Float, nullable=False)
    prediction_label = Column(Integer, nullable=True)  # 0 or 1 for classification
    prediction_proba = Column(Float, nullable=True)
    features = Column(JSON, nullable=False)         # raw input features
    shap_values = Column(JSON, nullable=True)        # SHAP explanation (sampled)
    latency_ms = Column(Float, nullable=False)       # end-to-end inference time
    actual_label = Column(Integer, nullable=True)    # ground truth (filled later)
    feedback_received_at = Column(DateTime, nullable=True)
    cost_usd = Column(Float, nullable=True)          # inference cost
    created_at = Column(
        DateTime, nullable=False, server_default=func.now(), index=True
    )
    client_ip = Column(String(45), nullable=True)    # hashed for privacy

    __table_args__ = (
        # BRIN index: tiny on disk, fast for range scans on append-only logs
        Index("ix_prediction_logs_created_brin", "created_at", postgresql_using="brin"),
        Index("ix_prediction_logs_model_version", "model_name", "model_version"),
    )


# ── Drift Report ───────────────────────────────────────────────────────────────

class DriftReport(Base):
    """
    Each drift detection run produces one report row.
    Stores aggregate stats + per-feature drift metrics.
    """

    __tablename__ = "drift_reports"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name = Column(String(128), nullable=False)
    model_version = Column(String(32), nullable=False)
    report_type = Column(String(32), nullable=False)  # data_drift | target_drift | perf
    dataset_drift_detected = Column(Boolean, nullable=False, default=False)
    drift_share = Column(Float, nullable=True)        # fraction of features drifted
    feature_metrics = Column(JSON, nullable=True)     # per-feature PSI / KS stats
    performance_metrics = Column(JSON, nullable=True) # F1, AUC, precision, recall
    reference_window_start = Column(DateTime, nullable=True)
    reference_window_end = Column(DateTime, nullable=True)
    current_window_start = Column(DateTime, nullable=True)
    current_window_end = Column(DateTime, nullable=True)
    n_reference_samples = Column(Integer, nullable=True)
    n_current_samples = Column(Integer, nullable=True)
    alert_triggered = Column(Boolean, default=False)
    created_at = Column(DateTime, nullable=False, server_default=func.now())

    retrain_events = relationship(
        "RetrainEvent", back_populates="drift_report", lazy="select"
    )

    __table_args__ = (
        Index("ix_drift_reports_model_created", "model_name", "created_at"),
    )


# ── Retrain Event ──────────────────────────────────────────────────────────────

class RetrainEvent(Base):
    """
    Tracks every automated or manual retrain trigger, including outcome.
    """

    __tablename__ = "retrain_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trigger_type = Column(String(32), nullable=False)  # drift_alert | manual | scheduled
    drift_report_id = Column(
        UUID(as_uuid=True), ForeignKey("drift_reports.id"), nullable=True
    )
    model_name = Column(String(128), nullable=False)
    previous_version = Column(String(32), nullable=True)
    new_version = Column(String(32), nullable=True)
    champion_f1 = Column(Float, nullable=True)
    challenger_f1 = Column(Float, nullable=True)
    improvement_pct = Column(Float, nullable=True)
    promoted = Column(Boolean, default=False)
    promotion_reason = Column(String(256), nullable=True)
    prefect_flow_run_id = Column(String(64), nullable=True)
    status = Column(String(32), default="pending")  # pending | running | success | failed
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime, nullable=False, server_default=func.now())
    completed_at = Column(DateTime, nullable=True)

    drift_report = relationship("DriftReport", back_populates="retrain_events")

    __table_args__ = (
        Index("ix_retrain_events_model_started", "model_name", "started_at"),
    )


# ── A/B Test Result ────────────────────────────────────────────────────────────

class ABTestResult(Base):
    """
    Aggregated A/B test metrics between champion and challenger.
    Written by the shadow router after each batch analysis window.
    """

    __tablename__ = "ab_test_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name = Column(String(128), nullable=False)
    champion_version = Column(String(32), nullable=False)
    challenger_version = Column(String(32), nullable=False)
    window_start = Column(DateTime, nullable=False)
    window_end = Column(DateTime, nullable=False)
    champion_n_requests = Column(Integer, default=0)
    challenger_n_requests = Column(Integer, default=0)
    champion_avg_latency_ms = Column(Float, nullable=True)
    challenger_avg_latency_ms = Column(Float, nullable=True)
    champion_f1 = Column(Float, nullable=True)
    challenger_f1 = Column(Float, nullable=True)
    champion_avg_proba = Column(Float, nullable=True)
    challenger_avg_proba = Column(Float, nullable=True)
    statistical_significance = Column(Float, nullable=True)  # p-value
    winner = Column(String(32), nullable=True)               # champion | challenger
    created_at = Column(DateTime, nullable=False, server_default=func.now())


# ── Model Metadata ─────────────────────────────────────────────────────────────

class ModelMetadata(Base):
    """
    Mirrors MLflow model registry but with extra operational metadata.
    Useful for feature store mapping, data version tracking, etc.
    """

    __tablename__ = "model_metadata"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name = Column(String(128), nullable=False)
    version = Column(String(32), nullable=False)
    mlflow_run_id = Column(String(64), nullable=False)
    stage = Column(String(32), nullable=False)
    algorithm = Column(String(64), nullable=True)
    training_dataset_hash = Column(String(64), nullable=True)
    training_rows = Column(Integer, nullable=True)
    feature_names = Column(JSON, nullable=True)
    hyperparameters = Column(JSON, nullable=True)
    train_metrics = Column(JSON, nullable=True)
    val_metrics = Column(JSON, nullable=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    promoted_at = Column(DateTime, nullable=True)
    archived_at = Column(DateTime, nullable=True)

    __table_args__ = (
        UniqueConstraint("model_name", "version", name="uq_model_version"),
    )
