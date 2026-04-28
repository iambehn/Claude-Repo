"""
Lazy-loaded inference for the Learned Fusion Model.

Loads model.pkl from data/models/clip_judge/ on first call and caches it
for the process lifetime. Returns None gracefully when no model exists, so
the pipeline continues uninterrupted during the data-accumulation phase.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_MODEL_DIR = "data/models/clip_judge"


@lru_cache(maxsize=1)
def _load_pipeline(model_dir: str) -> Any:
    """Load the sklearn Pipeline from disk. Result cached for process lifetime."""
    model_path = Path(model_dir) / "model.pkl"
    if not model_path.exists():
        return None
    try:
        import joblib
        return joblib.load(model_path)
    except Exception as exc:
        logger.debug(f"[model_inference] Failed to load model: {exc}")
        return None


def predict_approval(meta: dict, config: dict | None = None) -> float | None:
    """Return predicted approval probability (0.0–1.0), or None if no model exists.

    Never raises — any failure returns None so callers can treat it as optional.
    """
    try:
        import numpy as np
        from utils.training_logger import _extract_features
        from utils.model_trainer import features_to_vector

        model_dir = (config or {}).get("model", {}).get("path", _DEFAULT_MODEL_DIR)
        pipeline = _load_pipeline(model_dir)
        if pipeline is None:
            return None

        features = _extract_features(meta)
        X = np.array([features_to_vector(features)])
        prob = pipeline.predict_proba(X)[0][1]
        return round(float(prob), 3)
    except Exception as exc:
        logger.debug(f"[model_inference] Prediction failed: {exc}")
        return None
