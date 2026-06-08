"""
ModelMesh — Unit Tests
-----------------------
Tests for: shadow router, feature encoding, schema validation,
drift monitor logic, rate limiter.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np


# ── Test Fixtures ──────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "tenure", "monthly_charges", "total_charges", "num_products",
    "has_internet", "contract_type_encoded", "payment_method_encoded",
    "paperless_billing", "tech_support", "online_security",
]

SAMPLE_FEATURES = {
    "tenure": 24,
    "monthly_charges": 65.5,
    "total_charges": 1572.0,
    "num_products": 2,
    "has_internet": 1,
    "contract_type_encoded": 0,
    "payment_method_encoded": 0,
    "paperless_billing": 1,
    "tech_support": 0,
    "online_security": 0,
}


def make_mock_model(proba: float = 0.7):
    """Create a sklearn-compatible mock model."""
    model = MagicMock()
    model.predict_proba.return_value = np.array([[1 - proba, proba]])
    return model


# ── Shadow Router Tests ────────────────────────────────────────────────────────

class TestShadowRouter:
    def setup_method(self):
        from src.serving.shadow_router import ShadowRouter
        self.champion = make_mock_model(proba=0.8)
        self.challenger = make_mock_model(proba=0.75)
        self.router = ShadowRouter(
            champion_model=self.champion,
            champion_version="1",
            feature_names=FEATURE_NAMES,
            challenger_model=self.challenger,
            challenger_version="2",
        )

    def test_champion_prediction_returns_correct_proba(self):
        proba, label, latency = self.router.predict_champion(SAMPLE_FEATURES)
        assert abs(proba - 0.8) < 0.01
        assert label == 1
        assert latency > 0

    def test_champion_metrics_increment(self):
        initial = self.router.metrics.champion_requests
        self.router.predict_champion(SAMPLE_FEATURES)
        assert self.router.metrics.champion_requests == initial + 1

    def test_shadow_disabled_when_no_challenger(self):
        from src.serving.shadow_router import ShadowRouter
        router = ShadowRouter(
            champion_model=self.champion,
            champion_version="1",
            feature_names=FEATURE_NAMES,
            challenger_model=None,
            challenger_version=None,
        )
        assert not router.shadow_enabled

    def test_shadow_pct_update(self):
        self.router.set_shadow_pct(0.1)
        assert self.router._shadow_pct == 0.1

    def test_shadow_pct_invalid_raises(self):
        with pytest.raises(AssertionError):
            self.router.set_shadow_pct(1.5)

    def test_metrics_snapshot_structure(self):
        self.router.predict_champion(SAMPLE_FEATURES)
        snapshot = self.router.get_shadow_metrics_snapshot()
        assert "champion_version" in snapshot
        assert "challenger_version" in snapshot
        assert "champion_requests" in snapshot
        assert snapshot["champion_requests"] >= 1

    @pytest.mark.asyncio
    async def test_shadow_prediction_async_runs(self):
        # Should complete without raising
        await self.router.predict_shadow_async(
            features=SAMPLE_FEATURES,
            champion_label=1,
        )
        # Challenger may or may not run depending on sampling
        # Just verify it doesn't crash

    def test_challenger_reload(self):
        new_challenger = make_mock_model(proba=0.9)
        self.router.reload_challenger(new_challenger, "3")
        assert self.router.challenger_version == "3"
        assert self.router.metrics.challenger_requests == 0  # reset


# ── Feature Encoding Tests ────────────────────────────────────────────────────

class TestFeatureEncoding:
    def test_valid_encoding(self):
        from src.api.routers.inference import encode_features
        raw = {
            "tenure": 12,
            "monthly_charges": 50.0,
            "total_charges": 600.0,
            "num_products": 2,
            "has_internet": 1,
            "contract_type": "One year",
            "payment_method": "Mailed check",
            "paperless_billing": 0,
            "tech_support": 1,
            "online_security": 1,
        }
        encoded = encode_features(raw)
        assert encoded["contract_type_encoded"] == 1
        assert encoded["payment_method_encoded"] == 1
        assert "contract_type" not in encoded

    def test_unknown_contract_type_defaults_to_zero(self):
        from src.api.routers.inference import encode_features
        raw = {
            "tenure": 5,
            "monthly_charges": 30.0,
            "total_charges": 150.0,
            "num_products": 1,
            "has_internet": 0,
            "contract_type": "Unknown type",
            "payment_method": "Electronic check",
            "paperless_billing": 0,
            "tech_support": 0,
            "online_security": 0,
        }
        encoded = encode_features(raw)
        assert encoded["contract_type_encoded"] == 0


# ── Schema Validation Tests ────────────────────────────────────────────────────

class TestSchemas:
    def test_churn_features_valid(self):
        from src.api.schemas.inference import ChurnFeatures
        f = ChurnFeatures(
            tenure=24,
            monthly_charges=65.5,
            total_charges=1572.0,
            num_products=2,
            has_internet=1,
            contract_type="Month-to-month",
            payment_method="Electronic check",
            paperless_billing=1,
            tech_support=0,
            online_security=0,
        )
        assert f.tenure == 24

    def test_invalid_contract_type_rejected(self):
        from src.api.schemas.inference import ChurnFeatures
        from pydantic import ValidationError
        with pytest.raises(ValidationError) as exc_info:
            ChurnFeatures(
                tenure=24, monthly_charges=65.5, total_charges=1572.0,
                num_products=2, has_internet=1,
                contract_type="BadValue",
                payment_method="Electronic check",
                paperless_billing=1, tech_support=0, online_security=0,
            )
        assert "contract_type" in str(exc_info.value)

    def test_tenure_out_of_range_rejected(self):
        from src.api.schemas.inference import ChurnFeatures
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ChurnFeatures(
                tenure=-1, monthly_charges=65.5, total_charges=1572.0,
                num_products=2, has_internet=1,
                contract_type="One year",
                payment_method="Electronic check",
                paperless_billing=1, tech_support=0, online_security=0,
            )


# ── Rate Limiter Tests ─────────────────────────────────────────────────────────

class TestRateLimiter:
    def setup_method(self):
        # Import and instantiate without FastAPI app for unit testing
        from src.api.middleware.rate_limiter import RateLimitMiddleware
        self.middleware = RateLimitMiddleware(
            app=MagicMock(),
            max_requests=5,
            window_seconds=60,
        )

    def test_allows_requests_under_limit(self):
        for _ in range(5):
            allowed, remaining = self.middleware._is_allowed("test-key")
            assert allowed

    def test_blocks_request_over_limit(self):
        for _ in range(5):
            self.middleware._is_allowed("test-key-2")
        allowed, remaining = self.middleware._is_allowed("test-key-2")
        assert not allowed
        assert remaining == 0

    def test_different_keys_independent(self):
        for _ in range(5):
            self.middleware._is_allowed("key-A")
        # key-B should still be allowed
        allowed, _ = self.middleware._is_allowed("key-B")
        assert allowed

    def test_ip_key_extraction(self):
        request = MagicMock()
        request.headers = {"X-Forwarded-For": "10.0.0.1"}
        key = self.middleware._get_client_key(request)
        assert key == "ip:10.0.0.1"

    def test_api_key_header_takes_priority(self):
        request = MagicMock()
        request.headers = {
            "X-API-Key": "my-api-key",
            "X-Forwarded-For": "10.0.0.1",
        }
        key = self.middleware._get_client_key(request)
        assert key == "key:my-api-key"


# ── Training Data Tests ────────────────────────────────────────────────────────

class TestTrainingData:
    def test_synthetic_dataset_shape(self):
        from src.training.train import _generate_synthetic_dataset
        df = _generate_synthetic_dataset(n=100)
        assert len(df) == 100
        assert "Churn" in df.columns
        assert "tenure" in df.columns

    def test_feature_engineering_output_columns(self):
        from src.training.train import _generate_synthetic_dataset, engineer_features
        raw = _generate_synthetic_dataset(n=200)
        df = engineer_features(raw)
        expected_cols = {
            "tenure", "monthly_charges", "total_charges", "num_products",
            "has_internet", "contract_type_encoded", "payment_method_encoded",
            "paperless_billing", "tech_support", "online_security", "churn",
        }
        assert set(df.columns) == expected_cols

    def test_churn_rate_realistic(self):
        from src.training.train import _generate_synthetic_dataset, engineer_features
        raw = _generate_synthetic_dataset(n=1000)
        df = engineer_features(raw)
        churn_rate = df["churn"].mean()
        assert 0.15 <= churn_rate <= 0.45  # realistic range


# ── Drift Monitor Tests ────────────────────────────────────────────────────────

class TestDriftMonitor:
    def test_no_drift_on_same_distribution(self):
        from src.monitoring.drift_monitor import DriftMonitor
        import pandas as pd
        monitor = DriftMonitor()
        # Use same data as reference — should not detect drift
        current_df = monitor._reference_df.sample(500, random_state=1)
        result = monitor.run_drift_report(current_df)
        # Same distribution → drift_share should be very low
        assert result["drift_share"] < 0.5

    def test_drift_detected_on_shifted_distribution(self):
        from src.monitoring.drift_monitor import DriftMonitor
        import pandas as pd
        import numpy as np

        monitor = DriftMonitor()
        n = 1000
        # Create heavily shifted current data
        shifted = pd.DataFrame({
            "tenure": np.random.randint(60, 72, n),       # everyone is 5-6yr (was 0-6yr)
            "monthly_charges": np.random.uniform(100, 120, n),  # all high charges
            "total_charges": np.random.uniform(7000, 8000, n),
            "num_products": np.ones(n, dtype=int) * 4,
            "has_internet": np.ones(n, dtype=int),
            "contract_type_encoded": np.zeros(n, dtype=int),
            "payment_method_encoded": np.zeros(n, dtype=int),
            "paperless_billing": np.ones(n, dtype=int),
            "tech_support": np.zeros(n, dtype=int),
            "online_security": np.zeros(n, dtype=int),
        })
        result = monitor.run_drift_report(shifted)
        assert result["drift_share"] > 0.2
