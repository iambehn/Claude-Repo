"""
ml/train_fusion.py — Window Fusion Model Trainer

Trains a logistic regression on window_v1 features collected by
pipeline/training_exporter.py. Mirrors the structure of utils/model_trainer.py
but targets the 14-feature window schema instead of the 27-feature clip-judge schema.

Prerequisites:
  - At least 10 labeled window records (human_label.decision != null)
  - Run `python run.py --scan-vod URL GAME` to collect window data
  - Populate human_label.decision via the auto-labeler (ml/label_windows.py, deferred)
    or by editing JSONL files directly

Usage:
  python run.py --train-window-model           # train global model
  python run.py --train-window-model marvel_rivals  # train game-specific model
  python run.py --window-training-stats        # dataset summary

Model saved to: data/models/window_fusion/model.pkl (global)
               data/models/window_fusion/{game}/model.pkl (per-game)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Feature contract — must match ml/feature_schema.yaml window_v1 exactly
# ---------------------------------------------------------------------------

WINDOW_FEATURE_NAMES: list[str] = [
    "window_duration",
    "proxy_score",
    "signal_count",
    "unique_signal_sources",
    "chat_spike_max_strength",
    "chat_spike_max_confidence",
    "chat_spike_count",
    "audio_spike_max_strength",
    "audio_spike_max_confidence",
    "audio_spike_count",
    "viewer_clip_count",
    "viewer_clip_max_strength",
    "stream_marker_present",
    "chat_audio_lag_seconds",
]

_POSITIVE_LABELS: set[str] = {"accept", "accepted", "download"}
_NEGATIVE_LABELS: set[str] = {"reject", "rejected", "skip"}
_DEFAULT_MODEL_DIR = "data/models/window_fusion"
_DEFAULT_TRAINING_DIR = "data/training_sets/windows"
_MIN_SAMPLES = 10


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def features_to_vector(features: dict) -> list[float]:
    """Convert a window_v1 features dict to a flat float vector.

    chat_audio_lag_seconds uses -1.0 as its own 'not-present' sentinel
    (set by training_exporter when chat and audio don't co-occur). All other
    null/missing features fall back to 0.0.
    """
    vec = []
    for name in WINDOW_FEATURE_NAMES:
        val = features.get(name)
        if val is None:
            vec.append(-1.0 if name == "chat_audio_lag_seconds" else 0.0)
        else:
            vec.append(float(val))
    return vec


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_records(
    training_dir: str,
    game_filter: str | None,
) -> list[dict]:
    """Load window JSONL records that have non-null human labels."""
    base = Path(training_dir)
    if not base.exists():
        return []

    records: list[dict] = []
    for path in sorted(base.glob("*.jsonl")):
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("human_label", {}).get("decision") is None:
                    continue
                if game_filter and game_filter != "all":
                    if row.get("game") != game_filter:
                        continue
                records.append(row)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"[train_fusion] Could not read {path.name}: {exc}")

    return records


def _load_all_records(
    training_dir: str,
    game_filter: str | None,
) -> list[dict]:
    """Load all window JSONL records, including unlabeled rows."""
    base = Path(training_dir)
    if not base.exists():
        return []

    records: list[dict] = []
    for path in sorted(base.glob("*.jsonl")):
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if game_filter and game_filter != "all":
                    if row.get("game") != game_filter:
                        continue
                records.append(row)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"[train_fusion] Could not read {path.name}: {exc}")

    return records


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    game_filter: str | None = None,
    config: dict | None = None,
) -> dict[str, Any]:
    """Train a logistic regression on labeled window_v1 records.

    Args:
        game_filter: None / "all" for global model; game name for per-game model.
        config:      Full config dict (reads window_model.path if present).

    Returns:
        Dict with ok, n_samples, cv_accuracy_mean/std, cv_f1_mean,
        train_accuracy, model_path, top_features — or ok=False + reason.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_validate
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        import joblib
        import numpy as np
    except ImportError as exc:
        return {"ok": False, "reason": f"scikit-learn / joblib not installed: {exc}"}

    cfg = config or {}
    training_dir = cfg.get("training", {}).get("output_dir", "data/training_sets")
    training_dir = str(Path(training_dir) / "windows")
    base_dir = Path(cfg.get("window_model", {}).get("path", _DEFAULT_MODEL_DIR))
    model_dir = (base_dir / game_filter) if (game_filter and game_filter != "all") else base_dir

    records = _load_records(training_dir, game_filter)

    if len(records) < _MIN_SAMPLES:
        return {
            "ok": False,
            "reason": (
                f"need at least {_MIN_SAMPLES} labeled windows, "
                f"found {len(records)}"
            ),
        }

    X_rows, y_rows = [], []
    for row in records:
        decision = (row.get("human_label") or {}).get("decision", "")
        if decision in _POSITIVE_LABELS:
            label = 1
        elif decision in _NEGATIVE_LABELS:
            label = 0
        else:
            continue
        X_rows.append(features_to_vector(row.get("features", {})))
        y_rows.append(label)

    n = len(X_rows)
    if n < _MIN_SAMPLES:
        return {
            "ok": False,
            "reason": f"only {n} binary-labeled windows after filtering (need {_MIN_SAMPLES})",
        }

    X = np.array(X_rows, dtype=float)
    y = np.array(y_rows, dtype=int)
    pos_count = int(y.sum())
    neg_count = n - pos_count
    logger.info(f"[train_fusion] {n} samples  pos={pos_count}  neg={neg_count}  game={game_filter or 'all'}")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])

    cv_accuracy_mean = cv_accuracy_std = cv_f1_mean = None
    n_splits = min(5, n)
    if n >= _MIN_SAMPLES:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_results = cross_validate(
            pipeline, X, y, cv=cv,
            scoring=["accuracy", "f1"],
            return_train_score=False,
        )
        cv_accuracy_mean = round(float(cv_results["test_accuracy"].mean()), 4)
        cv_accuracy_std  = round(float(cv_results["test_accuracy"].std()), 4)
        cv_f1_mean       = round(float(cv_results["test_f1"].mean()), 4)
        logger.info(
            f"[train_fusion] CV accuracy={cv_accuracy_mean:.3f}±{cv_accuracy_std:.3f}  "
            f"f1={cv_f1_mean:.3f}"
        )

    pipeline.fit(X, y)
    train_accuracy = round(float(pipeline.score(X, y)), 4)

    coefs = pipeline.named_steps["clf"].coef_[0]
    top = sorted(
        zip(WINDOW_FEATURE_NAMES, coefs.tolist()),
        key=lambda t: abs(t[1]),
        reverse=True,
    )[:10]

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.pkl"
    joblib.dump(pipeline, model_path)

    metadata = {
        "schema_version":    "window_v1",
        "feature_names":     WINDOW_FEATURE_NAMES,
        "n_samples":         n,
        "n_accepted":        pos_count,
        "n_rejected":        neg_count,
        "train_accuracy":    train_accuracy,
        "cv_accuracy_mean":  cv_accuracy_mean,
        "cv_accuracy_std":   cv_accuracy_std,
        "cv_f1_mean":        cv_f1_mean,
        "game_filter":       game_filter or "all",
        "trained_at":        datetime.now(timezone.utc).isoformat(),
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    logger.info(f"[train_fusion] Model saved → {model_path}")
    return {
        "ok":               True,
        "n_samples":        n,
        "cv_accuracy_mean": cv_accuracy_mean,
        "cv_accuracy_std":  cv_accuracy_std,
        "cv_f1_mean":       cv_f1_mean,
        "train_accuracy":   train_accuracy,
        "model_path":       str(model_path),
        "top_features":     top,
    }


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def stats(
    game_filter: str | None = None,
    config: dict | None = None,
) -> dict[str, Any]:
    """Summarise the window training dataset without training a model."""
    cfg = config or {}
    training_dir = cfg.get("training", {}).get("output_dir", "data/training_sets")
    training_dir = str(Path(training_dir) / "windows")

    all_records = _load_all_records(training_dir, game_filter)

    labeled = [r for r in all_records if (r.get("human_label") or {}).get("decision") is not None]
    unlabeled = [r for r in all_records if (r.get("human_label") or {}).get("decision") is None]
    positive = [r for r in labeled if (r.get("human_label") or {}).get("decision") in _POSITIVE_LABELS]
    negative = [r for r in labeled if (r.get("human_label") or {}).get("decision") in _NEGATIVE_LABELS]

    per_game: dict[str, dict] = {}
    for row in all_records:
        game = row.get("game", "unknown")
        if game not in per_game:
            per_game[game] = {"total": 0, "labeled": 0, "positive": 0, "negative": 0}
        per_game[game]["total"] += 1
        decision = (row.get("human_label") or {}).get("decision")
        if decision is not None:
            per_game[game]["labeled"] += 1
            if decision in _POSITIVE_LABELS:
                per_game[game]["positive"] += 1
            elif decision in _NEGATIVE_LABELS:
                per_game[game]["negative"] += 1

    # Feature null rates across labeled records only
    null_rates: dict[str, float] = {}
    if labeled:
        for feat in WINDOW_FEATURE_NAMES:
            null_count = sum(
                1 for r in labeled
                if r.get("features", {}).get(feat) is None
            )
            null_rates[feat] = round(null_count / len(labeled), 4)

    jsonl_files = sorted(Path(training_dir).glob("*.jsonl")) if Path(training_dir).exists() else []
    date_range = (
        f"{jsonl_files[0].stem} → {jsonl_files[-1].stem}"
        if jsonl_files else "no files"
    )

    return {
        "total_windows":    len(all_records),
        "labeled_windows":  len(labeled),
        "unlabeled_windows": len(unlabeled),
        "positive":         len(positive),
        "negative":         len(negative),
        "per_game":         per_game,
        "date_range":       date_range,
        "feature_null_rates": null_rates,
    }
