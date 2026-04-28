"""
Learned Fusion Model trainer.

Reads JSONL records from data/training_sets/clip_judge/, trains a logistic
regression on the 27-feature schema used by training_logger, and saves the
fitted pipeline to data/models/clip_judge/model.pkl.

Run via:  python run.py --train-model [GAME]

The model is display-only by default (model.blend_weight: 0.0 in config.yaml).
Set blend_weight > 0 to blend learned predictions into clip_judge decisions.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

from utils.logger import get_logger

logger = get_logger(__name__)

_MIN_SAMPLES_WARN = 50
_DEFAULT_MODEL_DIR = "data/models/clip_judge"

# Ordinal encoding maps for categorical features
_MOTION_MAP: dict[str, float] = {"low": 0.0, "medium": 0.5, "high": 1.0}
_ENERGY_MAP: dict[str, float] = {"low": 0.0, "medium": 0.33, "high": 0.67, "intense": 1.0}
_DECISION_MAP: dict[str, float] = {"reject": 0.0, "quarantine": 0.5, "accept": 1.0}

NUMERIC_FEATURES: list[str] = [
    "duration_seconds", "resolution_height", "fps", "silence_ratio",
    "scene_change_count", "keyword_count",
    "context_confidence", "hook_confidence", "postability_score", "final_score",
    "kill_count", "headshot_count", "sweat_score",
    "audio_spike_count", "weapon_confidence", "highlight_score",
    "hook_score", "max_gap_seconds", "roi_match_count",
]
BOOL_FEATURES: list[str] = [
    "multi_kill_detected", "kill_feed_passed", "weapon_detected",
    "early_hook_passed", "dead_air_risk",
]
ORDINAL_FEATURES: list[tuple[str, dict[str, float]]] = [
    ("motion_level", _MOTION_MAP),
    ("audio_energy", _ENERGY_MAP),
    ("system_decision", _DECISION_MAP),
]
ALL_FEATURE_NAMES: list[str] = (
    NUMERIC_FEATURES
    + BOOL_FEATURES
    + [name for name, _ in ORDINAL_FEATURES]
)


def features_to_vector(features: dict) -> list[float]:
    """Convert an extracted features dict to a flat float list (matches ALL_FEATURE_NAMES order)."""
    vec: list[float] = []
    for f in NUMERIC_FEATURES:
        v = features.get(f)
        vec.append(float(v) if v is not None else 0.0)
    for f in BOOL_FEATURES:
        v = features.get(f)
        vec.append(1.0 if v else 0.0)
    for f, mapping in ORDINAL_FEATURES:
        v = features.get(f)
        vec.append(mapping.get(str(v).lower() if v else "", 0.0))
    return vec


def train(game_filter: str | None = None, config: dict | None = None) -> dict[str, Any]:
    """Load JSONL records, train a logistic regression, save model, return metrics dict."""
    try:
        import joblib
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import classification_report
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        logger.error(f"[model_trainer] Missing dependency: {exc}. Run: pip install scikit-learn joblib")
        return {"ok": False, "error": str(exc)}

    cfg = (config or {}).get("training", {})
    data_dir = Path(cfg.get("output_dir", "data/training_sets")) / "clip_judge"
    model_dir = Path((config or {}).get("model", {}).get("path", _DEFAULT_MODEL_DIR))

    records = _load_records(data_dir, game_filter)

    if not records:
        logger.warning("[model_trainer] No labeled training records found — nothing to train.")
        return {"ok": False, "error": "no records"}

    n = len(records)
    if n < _MIN_SAMPLES_WARN:
        logger.warning(
            f"[model_trainer] Only {n} labeled records "
            f"(aim for {_MIN_SAMPLES_WARN}+ for reliable accuracy). "
            "Training anyway — treat predictions as experimental."
        )

    X = np.array([features_to_vector(r["features"]) for r in records])
    y = np.array([1 if r["label"]["decision"] == "accepted" else 0 for r in records])

    pos_count = int(y.sum())
    neg_count = n - pos_count
    logger.info(f"[model_trainer] {n} records: {pos_count} accepted, {neg_count} rejected")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline.fit(X, y)

    y_pred = pipeline.predict(X)
    report = classification_report(
        y, y_pred, target_names=["rejected", "accepted"], output_dict=True, zero_division=0
    )
    accuracy = round(float(report["accuracy"]), 3)

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.pkl"
    meta_out = model_dir / "metadata.json"

    joblib.dump(pipeline, model_path)

    with meta_out.open("w") as fh:
        json.dump({
            "feature_names": ALL_FEATURE_NAMES,
            "n_samples": n,
            "n_accepted": pos_count,
            "n_rejected": neg_count,
            "training_accuracy": accuracy,
            "game_filter": game_filter or "all",
        }, fh, indent=2)

    clf = pipeline.named_steps["clf"]
    coefs = clf.coef_[0]
    top = sorted(
        zip(ALL_FEATURE_NAMES, coefs.tolist()),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:10]

    logger.info("[model_trainer] Top feature weights (|coef|):")
    for name, coef in top:
        logger.info(f"  {name:<40} {coef:+.4f}")
    logger.info(f"[model_trainer] Saved to {model_path} (training accuracy={accuracy:.1%})")

    return {
        "ok": True,
        "n_samples": n,
        "accuracy": accuracy,
        "model_path": str(model_path),
        "top_features": top,
    }


def _load_records(data_dir: Path, game_filter: str | None) -> list[dict]:
    records: list[dict] = []
    if not data_dir.exists():
        return records

    for jsonl_path in sorted(data_dir.glob("*.jsonl")):
        with jsonl_path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Only binary labels — quarantine is ambiguous, skip it
                decision = (rec.get("label") or {}).get("decision")
                if decision not in ("accepted", "rejected"):
                    continue

                if game_filter and game_filter != "all":
                    if rec.get("game") != game_filter:
                        continue

                if not rec.get("features"):
                    continue

                records.append(rec)

    return records
