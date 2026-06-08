"""
ModelMesh — SHAP Explainability Layer
---------------------------------------
Computes SHAP values for model predictions.
Sampled at a configurable rate (default 10%) to control overhead.
Returns top-N features sorted by absolute SHAP impact.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import shap

from config.logging_config import get_logger
from config.settings import get_settings

logger = get_logger(__name__)
settings = get_settings()


class SHAPExplainer:
    """
    Wraps shap.TreeExplainer (for tree-based models like XGBoost/RF/LightGBM)
    or shap.LinearExplainer (for logistic regression).

    The explainer is built once and cached — computing it per-request would
    be prohibitively expensive.
    """

    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        model_type: str = "tree",        # tree | linear | kernel
        background_data: Optional[np.ndarray] = None,
    ) -> None:
        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type
        self._sample_rate = settings.serving.shap_sample_rate
        self._explainer = self._build_explainer(background_data)
        logger.info(
            "shap_explainer_built",
            model_type=model_type,
            n_features=len(feature_names),
        )

    def _build_explainer(
        self, background_data: Optional[np.ndarray]
    ) -> shap.Explainer:
        """
        Choose explainer type based on model class.
        TreeExplainer is O(n_features²) — fast for ≤100 features.
        For opaque models, use shap.KernelExplainer with small background.
        """
        try:
            return shap.TreeExplainer(
                self.model,
                feature_perturbation="interventional",
            )
        except Exception:
            logger.warning(
                "tree_explainer_failed_fallback_to_linear",
            )
            if background_data is not None:
                return shap.KernelExplainer(
                    self.model.predict_proba,
                    background_data[:50],   # small background for speed
                )
            # Last resort: use linear explainer
            return shap.LinearExplainer(
                self.model,
                masker=shap.maskers.Independent(
                    data=np.zeros((1, len(self.feature_names)))
                ),
            )

    def should_explain(self) -> bool:
        """Stochastic sampling to control SHAP overhead in production."""
        return random.random() < self._sample_rate

    def explain(
        self,
        features: Dict[str, Any],
        top_n: int = 5,
    ) -> Optional[Dict]:
        """
        Compute SHAP values for a single instance.
        Returns None if sampling decides not to explain this request.

        Args:
            features: feature dict matching training schema
            top_n: number of top features to return in explanation

        Returns:
            Dict with shap_values, base_value, feature_names, top_features
        """
        if not settings.serving.enable_shap:
            return None

        try:
            feature_array = np.array(
                [[features.get(name, 0) for name in self.feature_names]]
            )

            # TreeExplainer returns list of arrays for multi-class;
            # index [1] = positive class (churn=1)
            shap_output = self._explainer.shap_values(feature_array)
            if isinstance(shap_output, list):
                sv = shap_output[1][0].tolist()   # class 1, first instance
                base_val = float(self._explainer.expected_value[1])
            else:
                sv = shap_output[0].tolist()
                base_val = float(self._explainer.expected_value)

            # Build top features list sorted by absolute SHAP impact
            feature_importance = sorted(
                zip(self.feature_names, sv),
                key=lambda x: abs(x[1]),
                reverse=True,
            )

            top_features = [
                {
                    "feature": name,
                    "shap_value": round(val, 5),
                    "feature_value": features.get(name),
                    "direction": "increases_churn" if val > 0 else "decreases_churn",
                }
                for name, val in feature_importance[:top_n]
            ]

            return {
                "feature_names": self.feature_names,
                "shap_values": [round(v, 5) for v in sv],
                "base_value": round(base_val, 5),
                "top_features": top_features,
            }

        except Exception as exc:
            logger.error("shap_computation_failed", error=str(exc))
            return None

    def explain_batch(
        self,
        feature_list: List[Dict[str, Any]],
        top_n: int = 5,
    ) -> List[Optional[Dict]]:
        """Batch SHAP computation — more efficient for batch endpoints."""
        try:
            feature_matrix = np.array(
                [
                    [f.get(name, 0) for name in self.feature_names]
                    for f in feature_list
                ]
            )
            shap_output = self._explainer.shap_values(feature_matrix)
            if isinstance(shap_output, list):
                sv_matrix = shap_output[1]
                base_val = float(self._explainer.expected_value[1])
            else:
                sv_matrix = shap_output
                base_val = float(self._explainer.expected_value)

            results = []
            for i, sv in enumerate(sv_matrix):
                top_features = sorted(
                    zip(self.feature_names, sv.tolist()),
                    key=lambda x: abs(x[1]),
                    reverse=True,
                )[:top_n]
                results.append({
                    "feature_names": self.feature_names,
                    "shap_values": [round(v, 5) for v in sv.tolist()],
                    "base_value": round(base_val, 5),
                    "top_features": [
                        {
                            "feature": name,
                            "shap_value": round(val, 5),
                            "feature_value": feature_list[i].get(name),
                            "direction": "increases_churn" if val > 0 else "decreases_churn",
                        }
                        for name, val in top_features
                    ],
                })
            return results
        except Exception as exc:
            logger.error("batch_shap_failed", error=str(exc))
            return [None] * len(feature_list)
