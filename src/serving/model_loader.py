"""
ModelMesh — Model Loader Service
----------------------------------
Loads champion + challenger models from MLflow registry into memory.
Handles hot-reloading when new versions are promoted.
"""

from __future__ import annotations

import asyncio
from typing import Any, List, Optional, Tuple

import numpy as np

from config.logging_config import get_logger
from config.settings import get_settings
from src.explainability.shap_explainer import SHAPExplainer
from src.registry.model_registry import ModelRegistryClient
from src.serving.shadow_router import ShadowRouter

logger = get_logger(__name__)
settings = get_settings()

# Canonical feature order — must match training pipeline
FEATURE_NAMES: List[str] = [
    "tenure",
    "monthly_charges",
    "total_charges",
    "num_products",
    "has_internet",
    "contract_type_encoded",
    "payment_method_encoded",
    "paperless_billing",
    "tech_support",
    "online_security",
]


class ModelLoader:
    """
    Owns the in-process model objects and wires them together into a
    ShadowRouter + SHAPExplainer.

    Call load_all() once at startup. Call reload_champion() / reload_challenger()
    after a promotion event.
    """

    def __init__(self) -> None:
        self.registry = ModelRegistryClient()
        self.shadow_router: Optional[ShadowRouter] = None
        self.shap_explainer: Optional[SHAPExplainer] = None
        self._champion_version: Optional[str] = None
        self._challenger_version: Optional[str] = None

    async def load_all(self) -> None:
        """Load champion (required) and challenger (optional) at startup."""
        loop = asyncio.get_event_loop()

        # Run blocking MLflow calls in thread pool
        champion_model, champ_ver = await loop.run_in_executor(
            None, self.registry.load_champion
        )
        self._champion_version = champ_ver
        logger.info("champion_loaded", version=champ_ver)

        challenger_model, chall_ver = await loop.run_in_executor(
            None, self.registry.load_challenger
        )
        self._challenger_version = chall_ver
        if chall_ver:
            logger.info("challenger_loaded", version=chall_ver)

        # Build shadow router
        self.shadow_router = ShadowRouter(
            champion_model=champion_model,
            champion_version=champ_ver,
            feature_names=FEATURE_NAMES,
            challenger_model=challenger_model,
            challenger_version=chall_ver,
        )

        # Build SHAP explainer on champion
        background = self._generate_background_data()
        self.shap_explainer = SHAPExplainer(
            model=champion_model,
            feature_names=FEATURE_NAMES,
            background_data=background,
        )

    async def reload_champion(self) -> None:
        """Called after promotion — hot-swap champion without restart."""
        loop = asyncio.get_event_loop()
        model, version = await loop.run_in_executor(
            None, self.registry.load_champion
        )
        self._champion_version = version
        if self.shadow_router:
            self.shadow_router.reload_champion(model, version)

        # Rebuild SHAP for new champion
        background = self._generate_background_data()
        self.shap_explainer = SHAPExplainer(
            model=model,
            feature_names=FEATURE_NAMES,
            background_data=background,
        )
        logger.info("champion_hot_reloaded", version=version)

    async def reload_challenger(self) -> None:
        """Called after a new model is registered to staging."""
        loop = asyncio.get_event_loop()
        model, version = await loop.run_in_executor(
            None, self.registry.load_challenger
        )
        self._challenger_version = version
        if self.shadow_router and model:
            self.shadow_router.reload_challenger(model, version)
        logger.info("challenger_hot_reloaded", version=version)

    @property
    def champion_version(self) -> Optional[str]:
        return self._champion_version

    @property
    def challenger_version(self) -> Optional[str]:
        return self._challenger_version

    def _generate_background_data(self) -> np.ndarray:
        """
        Generate synthetic background data for SHAP KernelExplainer fallback.
        Uses representative median values from the churn dataset.
        """
        np.random.seed(42)
        n = 100
        data = np.column_stack([
            np.random.randint(0, 72, n),          # tenure
            np.random.uniform(20, 120, n),         # monthly_charges
            np.random.uniform(100, 8000, n),       # total_charges
            np.random.randint(1, 5, n),            # num_products
            np.random.randint(0, 2, n),            # has_internet
            np.random.randint(0, 3, n),            # contract_type_encoded
            np.random.randint(0, 4, n),            # payment_method_encoded
            np.random.randint(0, 2, n),            # paperless_billing
            np.random.randint(0, 2, n),            # tech_support
            np.random.randint(0, 2, n),            # online_security
        ])
        return data.astype(float)
