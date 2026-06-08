"""
ModelMesh — Centralized Configuration Management
-------------------------------------------------
All settings are driven by environment variables with sensible defaults.
This eliminates hardcoded values and enables 12-factor app compliance.
"""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from typing import List, Optional

from pydantic import AnyHttpUrl, Field, validator
from pydantic_settings import BaseSettings


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class ModelStage(str, Enum):
    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


class DatabaseSettings(BaseSettings):
    host: str = Field(default="localhost", env="POSTGRES_HOST")
    port: int = Field(default=5432, env="POSTGRES_PORT")
    user: str = Field(default="modelmesh", env="POSTGRES_USER")
    password: str = Field(default="modelmesh_secret", env="POSTGRES_PASSWORD")
    db: str = Field(default="modelmesh_db", env="POSTGRES_DB")
    pool_size: int = Field(default=10, env="DB_POOL_SIZE")
    max_overflow: int = Field(default=20, env="DB_MAX_OVERFLOW")

    @property
    def url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.db}"
        )

    @property
    def sync_url(self) -> str:
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.db}"
        )

    class Config:
        env_prefix = ""


class KafkaSettings(BaseSettings):
    bootstrap_servers: str = Field(
        default="localhost:9092", env="KAFKA_BOOTSTRAP_SERVERS"
    )
    prediction_topic: str = Field(
        default="modelmesh.predictions", env="KAFKA_PREDICTION_TOPIC"
    )
    drift_alert_topic: str = Field(
        default="modelmesh.drift_alerts", env="KAFKA_DRIFT_ALERT_TOPIC"
    )
    retrain_trigger_topic: str = Field(
        default="modelmesh.retrain_triggers", env="KAFKA_RETRAIN_TOPIC"
    )
    consumer_group: str = Field(
        default="modelmesh-consumers", env="KAFKA_CONSUMER_GROUP"
    )
    max_block_ms: int = Field(default=5000, env="KAFKA_MAX_BLOCK_MS")

    class Config:
        env_prefix = ""


class MLflowSettings(BaseSettings):
    tracking_uri: str = Field(
        default="http://localhost:5000", env="MLFLOW_TRACKING_URI"
    )
    registry_uri: str = Field(
        default="http://localhost:5000", env="MLFLOW_REGISTRY_URI"
    )
    experiment_name: str = Field(
        default="modelmesh-churn-detection", env="MLFLOW_EXPERIMENT_NAME"
    )
    model_name: str = Field(
        default="churn-classifier", env="MLFLOW_MODEL_NAME"
    )
    artifact_root: str = Field(
        default="./mlruns", env="MLFLOW_ARTIFACT_ROOT"
    )
    s3_bucket: Optional[str] = Field(default=None, env="MLFLOW_S3_BUCKET")

    class Config:
        env_prefix = ""


class ServingSettings(BaseSettings):
    champion_model_alias: str = Field(
        default="champion", env="CHAMPION_MODEL_ALIAS"
    )
    challenger_model_alias: str = Field(
        default="challenger", env="CHALLENGER_MODEL_ALIAS"
    )
    shadow_traffic_pct: float = Field(
        default=0.05, env="SHADOW_TRAFFIC_PCT"
    )  # 5% to challenger
    inference_timeout_secs: float = Field(
        default=2.0, env="INFERENCE_TIMEOUT_SECS"
    )
    max_batch_size: int = Field(default=128, env="MAX_BATCH_SIZE")
    enable_shap: bool = Field(default=True, env="ENABLE_SHAP")
    shap_sample_rate: float = Field(
        default=0.1, env="SHAP_SAMPLE_RATE"
    )  # 10% of requests get SHAP

    class Config:
        env_prefix = ""


class MonitoringSettings(BaseSettings):
    drift_check_interval_minutes: int = Field(
        default=60, env="DRIFT_CHECK_INTERVAL_MINUTES"
    )
    drift_reference_window: int = Field(
        default=5000, env="DRIFT_REFERENCE_WINDOW"
    )  # rows
    drift_current_window: int = Field(
        default=1000, env="DRIFT_CURRENT_WINDOW"
    )  # rows
    psi_threshold: float = Field(default=0.2, env="PSI_THRESHOLD")
    ks_p_value_threshold: float = Field(default=0.05, env="KS_P_VALUE_THRESHOLD")
    f1_degradation_threshold: float = Field(
        default=0.02, env="F1_DEGRADATION_THRESHOLD"
    )  # trigger retrain if F1 drops >2%
    prometheus_port: int = Field(default=8001, env="PROMETHEUS_PORT")

    class Config:
        env_prefix = ""


class RetrainingSettings(BaseSettings):
    min_improvement_pct: float = Field(
        default=0.02, env="MIN_IMPROVEMENT_PCT"
    )  # challenger must beat champion by 2% F1
    auto_promote: bool = Field(default=False, env="AUTO_PROMOTE")
    prefect_api_url: str = Field(
        default="http://localhost:4200/api", env="PREFECT_API_URL"
    )
    retrain_deployment_name: str = Field(
        default="modelmesh-retrain/production", env="RETRAIN_DEPLOYMENT_NAME"
    )
    max_retrain_per_day: int = Field(default=3, env="MAX_RETRAIN_PER_DAY")

    class Config:
        env_prefix = ""


class SecuritySettings(BaseSettings):
    api_key_header: str = Field(default="X-API-Key", env="API_KEY_HEADER")
    api_keys: List[str] = Field(
        default=["dev-key-1234", "test-key-5678"], env="API_KEYS"
    )
    rate_limit_per_minute: int = Field(default=1000, env="RATE_LIMIT_PER_MINUTE")
    cors_origins: List[AnyHttpUrl] = Field(
        default=["http://localhost:3000"], env="CORS_ORIGINS"
    )

    @validator("api_keys", pre=True)
    def parse_api_keys(cls, v):
        if isinstance(v, str):
            return [k.strip() for k in v.split(",")]
        return v

    class Config:
        env_prefix = ""


class Settings(BaseSettings):
    """
    Master settings object. All sub-configs are composed here.
    Access via: from config.settings import get_settings; s = get_settings()
    """

    # ── Meta ─────────────────────────────────────────────────────────────────
    app_name: str = "ModelMesh"
    app_version: str = "1.0.0"
    environment: Environment = Field(
        default=Environment.DEVELOPMENT, env="ENVIRONMENT"
    )
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")  # json | text

    # ── Sub-configs ───────────────────────────────────────────────────────────
    database: DatabaseSettings = DatabaseSettings()
    kafka: KafkaSettings = KafkaSettings()
    mlflow: MLflowSettings = MLflowSettings()
    serving: ServingSettings = ServingSettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    retraining: RetrainingSettings = RetrainingSettings()
    security: SecuritySettings = SecuritySettings()

    # ── Feature flags ─────────────────────────────────────────────────────────
    enable_shadow_deployment: bool = Field(
        default=True, env="ENABLE_SHADOW_DEPLOYMENT"
    )
    enable_auto_retrain: bool = Field(default=True, env="ENABLE_AUTO_RETRAIN")
    enable_kafka_streaming: bool = Field(default=True, env="ENABLE_KAFKA_STREAMING")
    enable_prometheus: bool = Field(default=True, env="ENABLE_PROMETHEUS")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns a cached singleton of Settings.
    Using lru_cache means we parse env vars exactly once per process.
    """
    return Settings()
