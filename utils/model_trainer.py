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
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import StratifiedKFold, cross_validate
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        logger.error(f"[model_trainer] Missing dependency: {exc}. Run: pip install scikit-learn joblib")
        return {"ok": False, "error": str(exc)}

    cfg = (config or {}).get("training", {})
    data_dir = Path(cfg.get("output_dir", "data/training_sets")) / "clip_judge"
    base_dir = Path((config or {}).get("model", {}).get("path", _DEFAULT_MODEL_DIR))
    # Game-specific training saves to {base}/{game}/; cross-game to {base}/
    model_dir = base_dir / game_filter if (game_filter and game_filter != "all") else base_dir

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

    # Cross-validation: honest held-out estimate. Fall back to train accuracy when
    # there aren't enough samples for a meaningful split (need ≥2 per fold per class).
    _MIN_CV_SAMPLES = 10

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if n >= _MIN_CV_SAMPLES:
            n_splits = min(5, n // 2)
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_results = cross_validate(
                pipeline, X, y, cv=cv,
                scoring=["accuracy", "f1", "precision", "recall"],
                return_train_score=False,
            )
            cv_accuracy_mean = round(float(cv_results["test_accuracy"].mean()), 3)
            cv_accuracy_std = round(float(cv_results["test_accuracy"].std()), 3)
            cv_f1_mean = round(float(cv_results["test_f1"].mean()), 3)
            cv_note = (
                f"{n_splits}-fold CV accuracy={cv_accuracy_mean:.1%} "
                f"±{cv_accuracy_std:.1%}  F1={cv_f1_mean:.1%}"
            )
        else:
            cv_accuracy_mean = cv_accuracy_std = cv_f1_mean = None
            cv_note = (
                f"CV skipped (need ≥{_MIN_CV_SAMPLES} samples, have {n}) — "
                "train accuracy shown below is overfit-prone"
            )
            logger.warning(f"[model_trainer] {cv_note}")

        # Fit final model on ALL data after CV evaluation
        pipeline.fit(X, y)

    train_accuracy = round(float(accuracy_score(y, pipeline.predict(X))), 3)

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
            "train_accuracy": train_accuracy,
            "cv_accuracy_mean": cv_accuracy_mean,
            "cv_accuracy_std": cv_accuracy_std,
            "cv_f1_mean": cv_f1_mean,
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
    logger.info(f"[model_trainer] Saved to {model_path} | {cv_note}")

    return {
        "ok": True,
        "n_samples": n,
        "cv_accuracy_mean": cv_accuracy_mean,
        "cv_accuracy_std": cv_accuracy_std,
        "cv_f1_mean": cv_f1_mean,
        "train_accuracy": train_accuracy,
        "model_path": str(model_path),
        "top_features": top,
    }


def stats(game_filter: str | None = None, config: dict | None = None) -> dict[str, Any]:
    """Return a summary dict describing the current training dataset.

    Reads all JSONL records (including quarantine — not filtered out here).
    Useful for checking data quality before running --train-model.
    """
    cfg = (config or {}).get("training", {})
    data_dir = Path(cfg.get("output_dir", "data/training_sets")) / "clip_judge"

    records: list[dict] = []
    if data_dir.exists():
        for jsonl_path in sorted(data_dir.glob("*.jsonl")):
            with jsonl_path.open() as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    if game_filter and game_filter != "all":
        records = [r for r in records if r.get("game") == game_filter]

    total = len(records)
    if total == 0:
        return {"total": 0, "data_dir": str(data_dir)}

    # Decision breakdown
    decisions: dict[str, int] = {}
    for r in records:
        d = (r.get("label") or {}).get("decision") or "unknown"
        decisions[d] = decisions.get(d, 0) + 1

    accepted = decisions.get("accepted", 0)
    rejected = decisions.get("rejected", 0)
    labeled = accepted + rejected
    approval_rate = round(accepted / labeled, 3) if labeled > 0 else None

    # Per-game breakdown
    by_game: dict[str, dict[str, int]] = {}
    for r in records:
        game = r.get("game") or "unknown"
        d = (r.get("label") or {}).get("decision") or "unknown"
        if game not in by_game:
            by_game[game] = {"total": 0, "accepted": 0, "rejected": 0}
        by_game[game]["total"] += 1
        if d in by_game[game]:
            by_game[game][d] += 1

    # Date range
    dates = sorted(r.get("created_at", "")[:10] for r in records if r.get("created_at"))
    date_range = (dates[0], dates[-1]) if dates else (None, None)

    # Feature null rates — which detector signals are most often missing
    feature_nonnull: dict[str, int] = {}
    for r in records:
        for k, v in (r.get("features") or {}).items():
            feature_nonnull[k] = feature_nonnull.get(k, 0) + (0 if v is None else 1)
    null_rates = {
        k: round(1.0 - feature_nonnull.get(k, 0) / total, 3)
        for k in ALL_FEATURE_NAMES
    }

    return {
        "total": total,
        "labeled": labeled,
        "accepted": accepted,
        "rejected": rejected,
        "approval_rate": approval_rate,
        "by_game": by_game,
        "date_range": date_range,
        "null_rates": null_rates,
        "data_dir": str(data_dir),
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
