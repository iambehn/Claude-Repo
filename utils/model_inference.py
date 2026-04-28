"""
Lazy-loaded inference for the Learned Fusion Model.

Tries the game-specific model first (data/models/clip_judge/{game}/model.pkl),
then falls back to the global model (data/models/clip_judge/model.pkl).
Returns None gracefully when neither exists so the pipeline runs uninterrupted.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_MODEL_DIR = "data/models/clip_judge"


@lru_cache(maxsize=8)
def _load_pipeline(model_dir: str, game: str) -> Any:
    """Load the best available model for game. Cached per (model_dir, game)."""
    import joblib

    # Game-specific model takes precedence over global model
    candidates = []
    if game:
        candidates.append(Path(model_dir) / game / "model.pkl")
    candidates.append(Path(model_dir) / "model.pkl")

    for path in candidates:
        if path.exists():
            try:
                return joblib.load(path)
            except Exception as exc:
                logger.debug(f"[model_inference] Failed to load {path}: {exc}")

    return None


def predict_approval(
    meta: dict,
    config: dict | None = None,
    game: str | None = None,
) -> float | None:
    """Return predicted approval probability (0.0–1.0), or None if no model exists.

    Never raises — any failure returns None so callers can treat it as optional.
    """
    try:
        import numpy as np
        from utils.training_logger import _extract_features
        from utils.model_trainer import features_to_vector

        model_dir = (config or {}).get("model", {}).get("path", _DEFAULT_MODEL_DIR)
        pipeline = _load_pipeline(model_dir, game or "")
        if pipeline is None:
            return None

        features = _extract_features(meta)
        X = np.array([features_to_vector(features)])
        prob = pipeline.predict_proba(X)[0][1]
        return round(float(prob), 3)
    except Exception as exc:
        logger.debug(f"[model_inference] Prediction failed: {exc}")
        return None
