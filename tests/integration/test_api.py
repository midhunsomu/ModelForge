"""
ModelMesh — Integration Tests
------------------------------
Tests the full FastAPI request lifecycle against a real PostgreSQL DB.
Uses httpx.AsyncClient for async endpoint testing.
MLflow is mocked to avoid needing a running MLflow server.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# ── Fixtures ───────────────────────────────────────────────────────────────────

VALID_FEATURES = {
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
}

API_KEY = "dev-key-1234"
AUTH_HEADERS = {"X-API-Key": API_KEY}


def make_mock_model(proba: float = 0.72):
    import numpy as np
    model = MagicMock()
    model.predict_proba.return_value = np.array([[1 - proba, proba]])
    return model


def make_mock_shap_explainer():
    explainer = MagicMock()
    explainer.should_explain.return_value = False
    explainer.explain.return_value = None
    return explainer


def make_mock_model_loader():
    """Create a fully mocked ModelLoader that bypasses MLflow."""
    import numpy as np
    from src.serving.shadow_router import ShadowRouter

    champion = make_mock_model(proba=0.72)
    challenger = make_mock_model(proba=0.68)

    feature_names = [
        "tenure", "monthly_charges", "total_charges", "num_products",
        "has_internet", "contract_type_encoded", "payment_method_encoded",
        "paperless_billing", "tech_support", "online_security",
    ]

    shadow_router = ShadowRouter(
        champion_model=champion,
        champion_version="1",
        feature_names=feature_names,
        challenger_model=challenger,
        challenger_version="2",
    )

    loader = MagicMock()
    loader.champion_version = "1"
    loader.challenger_version = "2"
    loader.shadow_router = shadow_router
    loader.shap_explainer = make_mock_shap_explainer()
    loader.reload_champion = AsyncMock()
    loader.reload_challenger = AsyncMock()
    return loader


@pytest_asyncio.fixture
async def test_app():
    """
    Build FastAPI app with mocked ML models and Kafka.
    Uses real DB (set via env vars in CI) or SQLite fallback.
    """
    import os
    os.environ.setdefault("ENVIRONMENT", "development")
    os.environ.setdefault("ENABLE_KAFKA_STREAMING", "false")
    os.environ.setdefault("API_KEYS", "dev-key-1234")
    os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///./test_mlruns.db")

    # Patch model loading so we don't need MLflow running
    mock_loader = make_mock_model_loader()

    with patch("src.api.main.ModelLoader") as MockLoader:
        MockLoader.return_value = mock_loader
        mock_loader.load_all = AsyncMock()

        from src.api.main import create_app
        from src.database.connection import init_db

        app = create_app()

        # Override app state directly
        app.state.model_loader = mock_loader
        app.state.kafka_producer = None

        # Initialize DB schema
        with patch.object(app.state, "model_loader", mock_loader):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                yield client


# ── Health Endpoint Tests ──────────────────────────────────────────────────────

class TestHealthEndpoints:
    @pytest.mark.asyncio
    async def test_liveness_returns_200(self, test_app):
        resp = await test_app.get("/health/live")
        assert resp.status_code == 200
        assert resp.json()["status"] == "alive"

    @pytest.mark.asyncio
    async def test_readiness_structure(self, test_app):
        resp = await test_app.get("/health/ready")
        # May be 200 or 503 depending on DB state — just check structure
        data = resp.json()
        assert "ready" in data
        assert "champion_model_loaded" in data
        assert "database_connected" in data

    @pytest.mark.asyncio
    async def test_health_full_response_structure(self, test_app):
        resp = await test_app.get("/health/")
        data = resp.json()
        assert "status" in data
        assert "services" in data
        assert "version" in data


# ── Inference Endpoint Tests ───────────────────────────────────────────────────

class TestInferenceEndpoints:
    @pytest.mark.asyncio
    async def test_single_predict_success(self, test_app):
        payload = {
            "instance": VALID_FEATURES,
            "include_shap": False,
            "shadow_eligible": False,
        }
        resp = await test_app.post(
            "/v1/predict", json=payload, headers=AUTH_HEADERS
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "prediction" in data
        assert "probability" in data
        assert "trace_id" in data
        assert "model_version" in data
        assert data["prediction"] in (0, 1)
        assert 0.0 <= data["probability"] <= 1.0

    @pytest.mark.asyncio
    async def test_predict_returns_trace_id_header(self, test_app):
        payload = {"instance": VALID_FEATURES, "include_shap": False}
        resp = await test_app.post(
            "/v1/predict", json=payload, headers=AUTH_HEADERS
        )
        assert "x-trace-id" in resp.headers

    @pytest.mark.asyncio
    async def test_predict_forwards_trace_id(self, test_app):
        custom_trace = "test-trace-abc123"
        payload = {"instance": VALID_FEATURES, "include_shap": False}
        resp = await test_app.post(
            "/v1/predict",
            json=payload,
            headers={**AUTH_HEADERS, "X-Trace-ID": custom_trace},
        )
        assert resp.headers.get("x-trace-id") == custom_trace

    @pytest.mark.asyncio
    async def test_predict_requires_auth(self, test_app):
        payload = {"instance": VALID_FEATURES, "include_shap": False}
        resp = await test_app.post("/v1/predict", json=payload)
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_predict_invalid_features_rejected(self, test_app):
        invalid = {**VALID_FEATURES, "tenure": -5}  # negative tenure
        payload = {"instance": invalid, "include_shap": False}
        resp = await test_app.post(
            "/v1/predict", json=payload, headers=AUTH_HEADERS
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_predict_invalid_contract_rejected(self, test_app):
        invalid = {**VALID_FEATURES, "contract_type": "INVALID"}
        payload = {"instance": invalid, "include_shap": False}
        resp = await test_app.post(
            "/v1/predict", json=payload, headers=AUTH_HEADERS
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_batch_predict_success(self, test_app):
        payload = {
            "instances": [VALID_FEATURES, VALID_FEATURES, VALID_FEATURES],
            "include_shap": False,
        }
        resp = await test_app.post(
            "/v1/predict/batch", json=payload, headers=AUTH_HEADERS
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["batch_size"] == 3
        assert len(data["predictions"]) == 3
        assert "total_latency_ms" in data

    @pytest.mark.asyncio
    async def test_batch_too_large_rejected(self, test_app):
        payload = {
            "instances": [VALID_FEATURES] * 200,  # exceeds max of 128
            "include_shap": False,
        }
        resp = await test_app.post(
            "/v1/predict/batch", json=payload, headers=AUTH_HEADERS
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_batch_empty_rejected(self, test_app):
        payload = {"instances": [], "include_shap": False}
        resp = await test_app.post(
            "/v1/predict/batch", json=payload, headers=AUTH_HEADERS
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_shadow_metrics_endpoint(self, test_app):
        resp = await test_app.get(
            "/v1/shadow/metrics", headers=AUTH_HEADERS
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "champion_version" in data
        assert "shadow_pct" in data


# ── Rate Limiter Integration Tests ─────────────────────────────────────────────

class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_rate_limit_headers_present(self, test_app):
        payload = {"instance": VALID_FEATURES, "include_shap": False}
        resp = await test_app.post(
            "/v1/predict", json=payload, headers=AUTH_HEADERS
        )
        assert "x-ratelimit-limit" in resp.headers
        assert "x-ratelimit-remaining" in resp.headers


# ── Feedback Endpoint Tests ────────────────────────────────────────────────────

class TestFeedbackEndpoint:
    @pytest.mark.asyncio
    async def test_feedback_unknown_trace_returns_false(self, test_app):
        payload = {
            "trace_id": "nonexistent-trace-xyz",
            "actual_label": 1,
        }
        resp = await test_app.post(
            "/v1/feedback", json=payload, headers=AUTH_HEADERS
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False

    @pytest.mark.asyncio
    async def test_feedback_invalid_label_rejected(self, test_app):
        payload = {
            "trace_id": "some-trace",
            "actual_label": 5,  # only 0 or 1 valid
        }
        resp = await test_app.post(
            "/v1/feedback", json=payload, headers=AUTH_HEADERS
        )
        assert resp.status_code == 422


# ── Metrics Endpoint Tests ─────────────────────────────────────────────────────

class TestMetricsEndpoints:
    @pytest.mark.asyncio
    async def test_drift_report_returns_structure(self, test_app):
        resp = await test_app.get(
            "/v1/drift/latest", headers=AUTH_HEADERS
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "drift_detected" in data
        assert "model_name" in data

    @pytest.mark.asyncio
    async def test_prometheus_metrics_endpoint(self, test_app):
        resp = await test_app.get("/metrics")
        assert resp.status_code == 200
        assert b"modelmesh_requests_total" in resp.content
