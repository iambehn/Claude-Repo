"""
ml/evaluate_fusion.py — Window Fusion Model Evaluation Harness

Evaluates both the heuristic proxy score and the trained window fusion model
against all labeled window records in data/training_sets/windows/*.jsonl.

The heuristic is the permanent baseline (always available). The model column
appears when data/models/window_fusion/model.pkl exists. The delta column shows
model improvement over heuristic — a regression is any window where the
heuristic was correct but the model was wrong.

Usage:
    python run.py --evaluate-windows                # all games
    python run.py --evaluate-windows marvel_rivals  # one game
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.logger import get_logger

logger = get_logger(__name__)

LOGS_DIR = Path("logs")
HISTORY_FILE = LOGS_DIR / "eval_window_history.jsonl"

_POSITIVE: set[str] = {"accept", "accepted", "download"}
_NEGATIVE: set[str] = {"reject", "rejected", "skip", "never_downloaded"}
_DEFAULT_HEURISTIC_THRESHOLD = 0.30


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class WindowResult:
    window_id: str
    game: str
    expected: str                  # "accept" | "reject"
    heuristic_score: float
    heuristic_predicted: str
    model_score: float | None
    model_predicted: str | None
    elapsed_ms: float


@dataclass
class WindowMetrics:
    precision: float
    recall: float
    f1: float
    accuracy: float
    avg_process_ms: float
    n_total: int
    n_correct: int
    tp: int
    fp: int
    fn: int
    tn: int


@dataclass
class WindowEvalRun:
    run_id: str
    timestamp: str
    heuristic_metrics: WindowMetrics
    model_metrics: WindowMetrics | None   # None when no model file found
    results: list[WindowResult]
    regressions: list[WindowResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_window_evaluation(
    config: dict,
    game_filter: str | None = None,
) -> WindowEvalRun:
    """
    Load labeled window records, score with heuristic + model, compute metrics.

    Args:
        config:      Full config dict (reads window_model.path, proxy thresholds).
        game_filter: None / "all" → all games; slug → one game.

    Returns:
        WindowEvalRun with heuristic_metrics, model_metrics (or None), results,
        and regressions list.
    """
    from ml.train_fusion import _load_all_records, features_to_vector

    cfg = config or {}
    training_dir = str(
        Path(cfg.get("training", {}).get("output_dir", "data/training_sets")) / "windows"
    )
    game = None if (game_filter is None or game_filter == "all") else game_filter
    heuristic_threshold = float(
        cfg.get("proxy_scanner", {})
        .get("candidate_selection", {})
        .get("min_proxy_score", _DEFAULT_HEURISTIC_THRESHOLD)
    )

    all_records = _load_all_records(training_dir, game)
    labeled = [
        r for r in all_records
        if r.get("human_label", {}).get("decision") in _POSITIVE | _NEGATIVE
    ]

    if not labeled:
        logger.warning("[eval_windows] No labeled window records found.")
        empty = _empty_metrics()
        return WindowEvalRun(
            run_id=_run_id(),
            timestamp=_now(),
            heuristic_metrics=empty,
            model_metrics=None,
            results=[],
        )

    model = _load_model(cfg, game)
    if model is None:
        logger.info("[eval_windows] No trained model found — heuristic-only evaluation.")

    results: list[WindowResult] = []

    for rec in labeled:
        expected = (
            "accept"
            if rec["human_label"]["decision"] in _POSITIVE
            else "reject"
        )
        h_score = float(rec.get("scores", {}).get("heuristic_score", 0.0))
        h_predicted = "accept" if h_score >= heuristic_threshold else "reject"

        m_score: float | None = None
        m_predicted: str | None = None

        t0 = time.perf_counter()
        if model is not None:
            try:
                vec = features_to_vector(rec.get("features", {}))
                proba = model.predict_proba([vec])[0]
                # Class order from sklearn: assumes negative=0, positive=1
                # Use the class label array to find positive probability
                pos_idx = list(model.classes_).index(1) if 1 in model.classes_ else 1
                m_score = round(float(proba[pos_idx]), 4)
                m_predicted = "accept" if m_score >= 0.50 else "reject"
            except Exception as exc:
                logger.debug(f"[eval_windows] model inference failed: {exc}")
        elapsed_ms = (time.perf_counter() - t0) * 1000

        results.append(WindowResult(
            window_id=rec.get("window_id", "unknown"),
            game=rec.get("game", ""),
            expected=expected,
            heuristic_score=round(h_score, 4),
            heuristic_predicted=h_predicted,
            model_score=m_score,
            model_predicted=m_predicted,
            elapsed_ms=round(elapsed_ms, 2),
        ))

    h_metrics = _compute_metrics(results, use_model=False)
    m_metrics = _compute_metrics(results, use_model=True) if model is not None else None

    # Regressions: heuristic correct but model wrong (model degradation)
    regressions: list[WindowResult] = []
    if m_metrics is not None:
        regressions = [
            r for r in results
            if r.model_predicted is not None
            and r.heuristic_predicted == r.expected
            and r.model_predicted != r.expected
        ]

    return WindowEvalRun(
        run_id=_run_id(),
        timestamp=_now(),
        heuristic_metrics=h_metrics,
        model_metrics=m_metrics,
        results=results,
        regressions=regressions,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(results: list[WindowResult], use_model: bool) -> WindowMetrics:
    def predicted(r: WindowResult) -> str | None:
        return r.model_predicted if use_model else r.heuristic_predicted

    valid = [r for r in results if predicted(r) is not None]
    if not valid:
        return _empty_metrics()

    tp = sum(1 for r in valid if predicted(r) == "accept" and r.expected == "accept")
    fp = sum(1 for r in valid if predicted(r) == "accept" and r.expected != "accept")
    fn = sum(1 for r in valid if predicted(r) != "accept" and r.expected == "accept")
    tn = sum(1 for r in valid if predicted(r) != "accept" and r.expected != "accept")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    accuracy  = (tp + tn) / len(valid) if valid else 0.0
    avg_ms    = sum(r.elapsed_ms for r in valid) / len(valid)

    return WindowMetrics(
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        accuracy=round(accuracy, 4),
        avg_process_ms=round(avg_ms, 2),
        n_total=len(valid),
        n_correct=tp + tn,
        tp=tp, fp=fp, fn=fn, tn=tn,
    )


def _empty_metrics() -> WindowMetrics:
    return WindowMetrics(
        precision=0.0, recall=0.0, f1=0.0, accuracy=0.0,
        avg_process_ms=0.0, n_total=0, n_correct=0,
        tp=0, fp=0, fn=0, tn=0,
    )


# ---------------------------------------------------------------------------
# History persistence
# ---------------------------------------------------------------------------

def _persist_window_run(run: WindowEvalRun, game_filter: str | None) -> None:
    LOGS_DIR.mkdir(exist_ok=True)
    record = {
        "run_id":           run.run_id,
        "timestamp":        run.timestamp,
        "game_filter":      game_filter,
        "heuristic_metrics": asdict(run.heuristic_metrics),
        "model_metrics":    asdict(run.model_metrics) if run.model_metrics else None,
        "n_regressions":    len(run.regressions),
    }
    with HISTORY_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")

    detail_path = LOGS_DIR / f"eval_windows_{run.run_id}.json"
    detail_path.write_text(
        json.dumps(
            {
                **record,
                "results":     [asdict(r) for r in run.results],
                "regressions": [asdict(r) for r in run.regressions],
            },
            indent=2,
        )
    )
    logger.info(f"[eval_windows] Run saved → {detail_path}")


# ---------------------------------------------------------------------------
# Scorecard output
# ---------------------------------------------------------------------------

def print_window_scorecard(run: WindowEvalRun) -> None:
    h = run.heuristic_metrics
    m = run.model_metrics
    has_model = m is not None

    def _delta(curr: float, base: float) -> str:
        d = curr - base
        return f"{d:+.4f}"

    def _status(curr: float, base: float) -> str:
        d = curr - base
        if d > 0.005:  return "UP"
        if d < -0.005: return "DN"
        return "--"

    sep = "  " + "─" * 66

    print()
    print("  ╔══════════════════════════════════════════════════════════════╗")
    print(f"  ║  Window Fusion Scorecard  [{run.run_id}]")
    print("  ╚══════════════════════════════════════════════════════════════╝")
    print()

    if has_model:
        print(f"  {'Metric':<26} {'Heuristic':<12} {'Model':<12} {'Delta':<10} Status")
    else:
        print(f"  {'Metric':<26} {'Heuristic':<12}  (no model — run --train-window-model)")
    print(sep)

    def row(label: str, h_val: float, m_val: float | None, fmt: str = ".4f") -> None:
        h_s = f"{h_val:{fmt}}"
        if has_model and m_val is not None:
            m_s     = f"{m_val:{fmt}}"
            delta_s = _delta(m_val, h_val)
            stat_s  = _status(m_val, h_val)
            print(f"  {label:<26} {h_s:<12} {m_s:<12} {delta_s:<10} {stat_s}")
        else:
            print(f"  {label:<26} {h_s}")

    row("Precision",      h.precision,      m.precision      if m else None)
    row("Recall",         h.recall,         m.recall         if m else None)
    row("F1 Score",       h.f1,             m.f1             if m else None)
    row("Accuracy",       h.accuracy,       m.accuracy       if m else None)
    row("Avg time (ms)",  h.avg_process_ms, m.avg_process_ms if m else None, ".2f")
    print(sep)

    base = m if m else h
    print(f"  {'Windows evaluated':<26} {base.n_total}")
    print(f"  {'Correct decisions':<26} {base.n_correct}")
    ref = m if m else h
    print(f"  {'TP / FP / FN / TN':<26} {ref.tp} / {ref.fp} / {ref.fn} / {ref.tn}")
    print()

    if not run.regressions:
        if has_model:
            print("  No model regressions (no window where heuristic was right but model wrong).")
        else:
            print("  Heuristic-only run — train a model to see regressions.")
    else:
        print(f"  Model Regressions ({len(run.regressions)} window(s) where model degraded):")
        print("  " + "─" * 66)
        for r in run.regressions[:10]:
            print(f"  REGRESS  {r.window_id}")
            print(f"           Expected    : {r.expected}")
            print(f"           Heuristic   : {r.heuristic_predicted} "
                  f"(score={r.heuristic_score:.3f})")
            print(f"           Model       : {r.model_predicted} "
                  f"(score={r.model_score:.3f})")
            print()
        if len(run.regressions) > 10:
            print(f"  … and {len(run.regressions) - 10} more. "
                  f"See logs/eval_windows_{run.run_id}.json for full list.")

    if run.heuristic_metrics.n_total == 0:
        print()
        print("  No labeled window records found.")
        print("  Collect data:  python run.py --scan-vod URL GAME")
        print("  Review clips:  python run.py --game GAME (then approve/reject in UI)")
        print("  Label windows: python run.py --label-windows GAME")

    print()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model(config: dict, game_filter: str | None) -> Any:
    try:
        import joblib
    except ImportError:
        return None

    base = Path(
        config.get("window_model", {}).get("path", "data/models/window_fusion")
    )
    # Try game-specific model first, then fall back to global
    candidates = []
    if game_filter:
        candidates.append(base / game_filter / "model.pkl")
    candidates.append(base / "model.pkl")

    for path in candidates:
        if path.exists():
            try:
                model = joblib.load(path)
                logger.info(f"[eval_windows] Loaded model from {path}")
                return model
            except Exception as exc:
                logger.warning(f"[eval_windows] Could not load model {path}: {exc}")

    return None


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
