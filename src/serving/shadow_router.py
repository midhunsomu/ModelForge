"""
ModelMesh — Shadow / A/B Deployment Router
-------------------------------------------
Routes a configurable percentage of production traffic to the challenger
model (shadow mode). Both responses are logged, but only the champion's
prediction is returned to the client.

This enables:
1. Safe challenger evaluation on real production traffic
2. Zero-latency challenger comparison (runs async, doesn't block response)
3. Statistical significance testing before promotion
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from config.logging_config import get_logger
from config.settings import get_settings

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class ShadowMetrics:
    """In-memory rolling counters for the shadow deployment window."""
    champion_requests: int = 0
    challenger_requests: int = 0
    champion_total_latency: float = 0.0
    challenger_total_latency: float = 0.0
    champion_total_proba: float = 0.0
    challenger_total_proba: float = 0.0
    agreement_count: int = 0   # both models predict same label


class ShadowRouter:
    """
    Manages champion/challenger model instances and routes inference traffic.

    Design choices:
    - Challenger inference is fire-and-forget (asyncio.create_task) so it
      never adds to the client-facing latency.
    - Uses a thread-safe in-memory counter for metrics (suitable for
      single-replica; use Redis for multi-replica).
    - Shadow pct is runtime-configurable without restart.
    """

    def __init__(
        self,
        champion_model: Any,
        champion_version: str,
        feature_names: list[str],
        challenger_model: Optional[Any] = None,
        challenger_version: Optional[str] = None,
    ) -> None:
        self.champion_model = champion_model
        self.champion_version = champion_version
        self.challenger_model = challenger_model
        self.challenger_version = challenger_version
        self.feature_names = feature_names
        self.metrics = ShadowMetrics()
        self._shadow_pct = settings.serving.shadow_traffic_pct

        logger.info(
            "shadow_router_initialized",
            champion_version=champion_version,
            challenger_version=challenger_version,
            shadow_pct=self._shadow_pct,
        )

    @property
    def shadow_enabled(self) -> bool:
        return (
            settings.enable_shadow_deployment
            and self.challenger_model is not None
            and self._shadow_pct > 0
        )

    def set_shadow_pct(self, pct: float) -> None:
        """Runtime adjustment of shadow traffic percentage (0.0 – 1.0)."""
        assert 0.0 <= pct <= 1.0, "pct must be between 0 and 1"
        self._shadow_pct = pct
        logger.info("shadow_pct_updated", new_pct=pct)

    def _features_to_array(self, features: Dict[str, Any]) -> list:
        """Convert feature dict to model input array in training column order."""
        return [features.get(name, 0) for name in self.feature_names]

    def _predict_single(
        self, model: Any, feature_array: list, model_version: str
    ) -> Tuple[float, int, float]:
        """
        Returns (proba, label, latency_ms).
        Assumes sklearn-compatible model with predict_proba.
        """
        t0 = time.perf_counter()
        import numpy as np
        X = np.array([feature_array])
        proba = float(model.predict_proba(X)[0][1])
        label = int(proba >= 0.5)
        latency = (time.perf_counter() - t0) * 1000
        return proba, label, latency

    def predict_champion(
        self, features: Dict[str, Any]
    ) -> Tuple[float, int, float]:
        """Champion prediction — always synchronous, always returned to client."""
        arr = self._features_to_array(features)
        proba, label, latency = self._predict_single(
            self.champion_model, arr, self.champion_version
        )
        self.metrics.champion_requests += 1
        self.metrics.champion_total_latency += latency
        self.metrics.champion_total_proba += proba
        return proba, label, latency

    async def predict_shadow_async(
        self,
        features: Dict[str, Any],
        champion_label: int,
        on_result_callback=None,
    ) -> None:
        """
        Challenger inference — runs in background, never blocks client response.
        Results are logged for A/B analysis but NOT returned to client.
        """
        if not self.shadow_enabled:
            return
        if random.random() > self._shadow_pct:
            return

        try:
            loop = asyncio.get_event_loop()
            arr = self._features_to_array(features)

            # Run CPU-bound inference in thread pool to avoid blocking event loop
            proba, label, latency = await loop.run_in_executor(
                None,
                self._predict_single,
                self.challenger_model,
                arr,
                self.challenger_version,
            )

            self.metrics.challenger_requests += 1
            self.metrics.challenger_total_latency += latency
            self.metrics.challenger_total_proba += proba

            if label == champion_label:
                self.metrics.agreement_count += 1

            logger.debug(
                "shadow_prediction_completed",
                challenger_version=self.challenger_version,
                challenger_proba=round(proba, 4),
                champion_label=champion_label,
                challenger_label=label,
                agreement=label == champion_label,
                latency_ms=round(latency, 2),
            )

            # Fire optional callback (e.g., write to Kafka, DB)
            if on_result_callback:
                asyncio.create_task(
                    on_result_callback(
                        challenger_version=self.challenger_version,
                        proba=proba,
                        label=label,
                        latency_ms=latency,
                    )
                )
        except Exception as exc:
            logger.error("shadow_prediction_failed", error=str(exc))

    def get_shadow_metrics_snapshot(self) -> Dict:
        """Return current in-memory A/B metrics."""
        m = self.metrics
        champ_n = m.champion_requests or 1
        chall_n = m.challenger_requests or 1
        return {
            "champion_version": self.champion_version,
            "challenger_version": self.challenger_version,
            "champion_requests": m.champion_requests,
            "challenger_requests": m.challenger_requests,
            "champion_avg_latency_ms": round(m.champion_total_latency / champ_n, 2),
            "challenger_avg_latency_ms": round(m.challenger_total_latency / chall_n, 2),
            "champion_avg_proba": round(m.champion_total_proba / champ_n, 4),
            "challenger_avg_proba": round(m.challenger_total_proba / chall_n, 4),
            "agreement_rate": round(m.agreement_count / max(m.challenger_requests, 1), 4),
            "shadow_pct": self._shadow_pct,
        }

    def reload_challenger(
        self, new_challenger: Any, new_version: str
    ) -> None:
        """Hot-swap the challenger model without downtime."""
        self.challenger_model = new_challenger
        self.challenger_version = new_version
        self.metrics = ShadowMetrics()   # reset counters for new challenger
        logger.info("challenger_model_reloaded", new_version=new_version)

    def reload_champion(
        self, new_champion: Any, new_version: str
    ) -> None:
        """Hot-swap the champion model (after promotion)."""
        self.champion_model = new_champion
        self.champion_version = new_version
        self.metrics = ShadowMetrics()
        logger.info("champion_model_reloaded", new_version=new_version)
