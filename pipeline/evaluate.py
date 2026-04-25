"""
pipeline/evaluate.py — Gold Set Evaluation + Stability Scorecard

Runs every clip in assets/gold_set/ through the clip_judge scoring logic
without touching any files, then compares each predicted decision against a
hand-labeled .truth.json to produce precision / recall / F1 and a regression
report.

Usage:
    python -m pipeline.evaluate
    python -m pipeline.evaluate --game marvel_rivals
    python -m pipeline.evaluate --since 2026-04-01
    python -m pipeline.evaluate --save          # write run to logs/

Gold set layout:
    assets/gold_set/
        <game>/
            <bucket>/           # e.g. easy_accept, easy_reject, hard_cases
                <stem>.meta.json    # frozen detector-output snapshot
                <stem>.truth.json   # hand-labeled ground truth

Truth file schema (see assets/gold_set/README below):
    {
        "clip_id": "clip_001",
        "game": "marvel_rivals",
        "expected_decision": "accept",   # accept | quarantine | reject
        "difficulty": "easy",            # easy | hard
        "labels": ["multi_kill", "strong_hook"],
        "notes": "3-kill sequence, clear hook",
        "added_at": "2026-04-25",
        "signal_truth": {                # optional — enables signal drift (Ds)
            "context_confidence": 0.78,
            "hook_confidence": 0.85,
            "postability_score": 0.72,
            "final_score": 0.76
        }
    }
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from pipeline import game_pack
from pipeline.clip_judge import (
    _clamp,
    _context_confidence,
    _decide,
    _hook_confidence,
    _postability_score,
)
from utils.logger import get_logger

logger = get_logger(__name__)

GOLD_ROOT = Path("assets/gold_set")
LOGS_DIR = Path("logs")
HISTORY_FILE = LOGS_DIR / "eval_history.jsonl"

# "Positive" class for precision/recall: clips the pipeline should accept.
_POSITIVE = "accept"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TruthEntry:
    clip_id: str
    game: str
    bucket: str
    meta_path: Path
    expected_decision: str
    difficulty: str
    labels: list[str]
    notes: str
    added_at: str
    signal_truth: dict[str, float]


@dataclass
class ClipResult:
    clip_id: str
    game: str
    bucket: str
    expected: str
    predicted: str
    quarantine_reason: str | None
    context_confidence: float
    hook_confidence: float
    postability_score: float
    final_score: float
    elapsed_ms: float
    signal_drift: float | None  # None when signal_truth absent


@dataclass
class RunMetrics:
    precision: float
    recall: float
    f1: float
    avg_signal_drift: float | None
    avg_process_ms: float
    n_total: int
    n_correct: int
    tp: int
    fp: int
    fn: int
    tn: int


@dataclass
class EvalRun:
    run_id: str
    timestamp: str
    config_summary: dict[str, Any]
    metrics: RunMetrics
    results: list[ClipResult]
    regressions: list[ClipResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Gold set loader
# ---------------------------------------------------------------------------


def _load_gold_set(game_filter: str | None, since: str | None) -> list[TruthEntry]:
    entries: list[TruthEntry] = []

    since_dt = datetime.fromisoformat(since) if since else None

    if not GOLD_ROOT.exists():
        logger.warning(f"[evaluate] Gold set directory not found: {GOLD_ROOT}")
        return entries

    games = (
        [GOLD_ROOT / game_filter] if game_filter else list(GOLD_ROOT.iterdir())
    )

    for game_dir in games:
        if not game_dir.is_dir():
            continue
        game = game_dir.name

        for bucket_dir in sorted(game_dir.iterdir()):
            if not bucket_dir.is_dir():
                continue
            bucket = bucket_dir.name

            for truth_path in sorted(bucket_dir.glob("*.truth.json")):
                meta_path = truth_path.with_suffix("").with_suffix(".meta.json")
                if not meta_path.exists():
                    logger.warning(
                        f"[evaluate] Missing meta.json for {truth_path.name} — skipping"
                    )
                    continue

                raw = json.loads(truth_path.read_text())

                added_at = raw.get("added_at", "")
                if since_dt and added_at:
                    try:
                        entry_dt = datetime.fromisoformat(added_at)
                        if entry_dt < since_dt:
                            continue
                    except ValueError:
                        pass

                entries.append(
                    TruthEntry(
                        clip_id=raw.get("clip_id", truth_path.stem),
                        game=raw.get("game", game),
                        bucket=bucket,
                        meta_path=meta_path,
                        expected_decision=raw.get("expected_decision", "reject"),
                        difficulty=raw.get("difficulty", "unknown"),
                        labels=raw.get("labels") or [],
                        notes=raw.get("notes", ""),
                        added_at=added_at,
                        signal_truth=raw.get("signal_truth") or {},
                    )
                )

    return entries


# ---------------------------------------------------------------------------
# Scoring (side-effect-free clone of clip_judge.evaluate)
# ---------------------------------------------------------------------------


def _score(meta: dict, pack: game_pack.GamePack, config: dict) -> dict:
    """Compute worthiness on an in-memory copy — no file writes."""
    context, _ = _context_confidence(meta, pack, config)
    hook, hook_signals, _ = _hook_confidence(meta, pack)
    postability, moment_mul, _ = _postability_score(meta, pack)

    w = pack.weights.composite
    final = _clamp(
        w["context_weight"] * context
        + w["hook_weight"] * hook
        + w["postability_weight"] * postability
    )

    th = pack.weights.thresholds
    decision, reason = _decide(
        context=context,
        hook=hook,
        postability=postability,
        accept_th=float(th["accept"]),
        reject_th=float(th["reject"]),
        min_context=float(th["min_context"]),
        hook_gate=float(th["hook_gate"]),
        reliable_conf=float(th["reliable_confidence"]),
    )

    return {
        "context_confidence": round(context, 3),
        "hook_confidence": round(hook, 3),
        "postability_score": round(postability, 3),
        "moment_multiplier": round(moment_mul, 2),
        "final_score": round(final, 3),
        "decision": decision,
        "quarantine_reason": reason,
    }


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


def run_evaluation(
    config: dict,
    game_filter: str | None = None,
    since: str | None = None,
) -> EvalRun:
    entries = _load_gold_set(game_filter, since)

    if not entries:
        logger.warning("[evaluate] No gold set entries found.")
        empty_metrics = RunMetrics(
            precision=0.0, recall=0.0, f1=0.0, avg_signal_drift=None,
            avg_process_ms=0.0, n_total=0, n_correct=0,
            tp=0, fp=0, fn=0, tn=0,
        )
        return EvalRun(
            run_id=_run_id(),
            timestamp=_now(),
            config_summary=_config_summary(config),
            metrics=empty_metrics,
            results=[],
        )

    # Pre-load packs (one per unique game in the gold set).
    packs: dict[str, game_pack.GamePack] = {}
    for entry in entries:
        if entry.game not in packs:
            try:
                packs[entry.game] = game_pack.load(entry.game)
            except (FileNotFoundError, game_pack.GamePackError) as exc:
                logger.error(f"[evaluate] Cannot load pack '{entry.game}': {exc}")

    results: list[ClipResult] = []

    for entry in entries:
        pack = packs.get(entry.game)
        if pack is None:
            continue

        meta = json.loads(entry.meta_path.read_text())
        # Strip any cached worthiness so the scoring is fresh.
        meta.pop("worthiness", None)

        eval_config = _make_eval_config(config, meta)

        t0 = time.perf_counter()
        scored = _score(meta, pack, eval_config)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        signal_drift = _compute_drift(scored, entry.signal_truth)

        results.append(
            ClipResult(
                clip_id=entry.clip_id,
                game=entry.game,
                bucket=entry.bucket,
                expected=entry.expected_decision,
                predicted=scored["decision"],
                quarantine_reason=scored.get("quarantine_reason"),
                context_confidence=scored["context_confidence"],
                hook_confidence=scored["hook_confidence"],
                postability_score=scored["postability_score"],
                final_score=scored["final_score"],
                elapsed_ms=round(elapsed_ms, 1),
                signal_drift=signal_drift,
            )
        )

    metrics = _compute_metrics(results)
    regressions = [r for r in results if r.expected != r.predicted]

    return EvalRun(
        run_id=_run_id(),
        timestamp=_now(),
        config_summary=_config_summary(config),
        metrics=metrics,
        results=results,
        regressions=regressions,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _compute_drift(scored: dict, signal_truth: dict) -> float | None:
    keys = ["context_confidence", "hook_confidence", "postability_score", "final_score"]
    pairs = [
        (scored.get(k, 0.0), signal_truth.get(k))
        for k in keys
        if signal_truth.get(k) is not None
    ]
    if not pairs:
        return None
    return round(sum(abs(pred - truth) for pred, truth in pairs) / len(pairs), 4)


def _compute_metrics(results: list[ClipResult]) -> RunMetrics:
    tp = sum(1 for r in results if r.predicted == _POSITIVE and r.expected == _POSITIVE)
    fp = sum(1 for r in results if r.predicted == _POSITIVE and r.expected != _POSITIVE)
    fn = sum(1 for r in results if r.predicted != _POSITIVE and r.expected == _POSITIVE)
    tn = sum(1 for r in results if r.predicted != _POSITIVE and r.expected != _POSITIVE)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    drifts = [r.signal_drift for r in results if r.signal_drift is not None]
    avg_drift = round(sum(drifts) / len(drifts), 4) if drifts else None

    avg_ms = round(sum(r.elapsed_ms for r in results) / len(results), 1) if results else 0.0
    n_correct = tp + tn

    return RunMetrics(
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        avg_signal_drift=avg_drift,
        avg_process_ms=avg_ms,
        n_total=len(results),
        n_correct=n_correct,
        tp=tp, fp=fp, fn=fn, tn=tn,
    )


# ---------------------------------------------------------------------------
# History + delta
# ---------------------------------------------------------------------------


def _load_last_run(game_filter: str | None) -> RunMetrics | None:
    if not HISTORY_FILE.exists():
        return None
    lines = HISTORY_FILE.read_text().strip().splitlines()
    for line in reversed(lines):
        try:
            entry = json.loads(line)
            if game_filter and entry.get("game_filter") != game_filter:
                continue
            m = entry.get("metrics", {})
            return RunMetrics(
                precision=m.get("precision", 0.0),
                recall=m.get("recall", 0.0),
                f1=m.get("f1", 0.0),
                avg_signal_drift=m.get("avg_signal_drift"),
                avg_process_ms=m.get("avg_process_ms", 0.0),
                n_total=m.get("n_total", 0),
                n_correct=m.get("n_correct", 0),
                tp=m.get("tp", 0), fp=m.get("fp", 0),
                fn=m.get("fn", 0), tn=m.get("tn", 0),
            )
        except (json.JSONDecodeError, KeyError):
            continue
    return None


def _persist_run(run: EvalRun, game_filter: str | None) -> None:
    LOGS_DIR.mkdir(exist_ok=True)
    record = {
        "run_id": run.run_id,
        "timestamp": run.timestamp,
        "game_filter": game_filter,
        "metrics": asdict(run.metrics),
        "config_summary": run.config_summary,
    }
    with HISTORY_FILE.open("a") as fh:
        fh.write(json.dumps(record) + "\n")

    detail_path = LOGS_DIR / f"eval_{run.run_id}.json"
    detail_path.write_text(
        json.dumps(
            {
                **record,
                "results": [asdict(r) for r in run.results],
                "regressions": [asdict(r) for r in run.regressions],
            },
            indent=2,
        )
    )
    logger.info(f"[evaluate] Run saved → {detail_path}")


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _delta_str(current: float | None, previous: float | None) -> str:
    if previous is None or current is None:
        return "  —  "
    diff = current - previous
    return f"{diff:+.4f}"


def _status(current: float | None, previous: float | None) -> str:
    if previous is None or current is None:
        return "  "
    diff = current - previous
    if diff > 0.005:
        return "UP"
    if diff < -0.005:
        return "DN"
    return "--"


def print_scorecard(run: EvalRun, prev: RunMetrics | None) -> None:
    m = run.metrics
    p = prev

    def row(label: str, curr: float | None, prev_val: float | None, fmt: str = ".4f") -> str:
        curr_s = f"{curr:{fmt}}" if curr is not None else "  —  "
        delta_s = _delta_str(curr, prev_val)
        status_s = _status(curr, prev_val)
        return f"  {label:<26} {curr_s:<10} {delta_s:<12} {status_s}"

    sep = "  " + "-" * 58
    print()
    print("  ╔══════════════════════════════════════════════════════╗")
    print(f"  ║  Stability Scorecard  [{run.run_id}]")
    print("  ╚══════════════════════════════════════════════════════╝")
    print()
    print(f"  {'Metric':<26} {'Current':<10} {'Delta':<12} Status")
    print(sep)
    print(row("Precision", m.precision, p.precision if p else None))
    print(row("Recall", m.recall, p.recall if p else None))
    print(row("F1 Score", m.f1, p.f1 if p else None))
    if m.avg_signal_drift is not None:
        print(row("Avg Signal Drift (Ds)", m.avg_signal_drift, p.avg_signal_drift if p else None))
    print(row("Avg Process Time (ms)", m.avg_process_ms, p.avg_process_ms if p else None, ".1f"))
    print(sep)
    print(f"  {'Clips evaluated':<26} {m.n_total}")
    print(f"  {'Correct decisions':<26} {m.n_correct}")
    print(f"  {'TP / FP / FN / TN':<26} {m.tp} / {m.fp} / {m.fn} / {m.tn}")
    print()

    if not run.regressions:
        print("  No regressions.")
    else:
        print(f"  Regression Report ({len(run.regressions)} clip(s)):")
        reg_sep = "  " + "-" * 58
        print(reg_sep)
        for r in run.regressions:
            qr = f" ({r.quarantine_reason})" if r.quarantine_reason else ""
            print(f"  FAIL  {r.clip_id}")
            print(f"        Expected : {r.expected}")
            print(f"        Actual   : {r.predicted}{qr}")
            print(
                f"        Scores   : ctx={r.context_confidence:.2f}  "
                f"hook={r.hook_confidence:.2f}  "
                f"post={r.postability_score:.2f}  "
                f"final={r.final_score:.2f}"
            )
            if r.signal_drift is not None:
                print(f"        Ds       : {r.signal_drift:.4f}")
            print()

    print()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_eval_config(config: dict, meta: dict) -> dict:
    """Return a config copy with detectors auto-enabled based on which data
    actually exists in the meta.json snapshot.

    Gold set clips were processed with detectors enabled.  If config.yaml has
    them disabled (the default for new installs), scoring returns 0.0 for every
    context signal — making the evaluation useless.  This function infers
    enablement from the data rather than the live config flag.
    """
    import copy

    cfg = copy.deepcopy(config)

    def _has(key: str) -> bool:
        v = meta.get(key)
        return bool(v) and v != {}

    if _has("kill_feed"):
        cfg.setdefault("kill_feed", {})["enabled"] = True
    if _has("weapon_detection"):
        cfg.setdefault("weapon_detector", {})["enabled"] = True
    if _has("audio_detector"):
        cfg.setdefault("audio_detector", {})["enabled"] = True
    if _has("roi_matches"):
        cfg.setdefault("roi_matcher", {})["enabled"] = True

    return cfg


def _config_summary(config: dict) -> dict:
    return {
        "kill_feed_enabled": bool(config.get("kill_feed", {}).get("enabled")),
        "weapon_detector_enabled": bool(config.get("weapon_detector", {}).get("enabled")),
        "audio_detector_enabled": bool(config.get("audio_detector", {}).get("enabled")),
        "roi_matcher_enabled": bool(config.get("roi_matcher", {}).get("enabled")),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gold set evaluation + stability scorecard")
    p.add_argument("--game", metavar="SLUG", help="Evaluate only this game slug")
    p.add_argument(
        "--since",
        metavar="DATE",
        help="Only include gold entries added on or after YYYY-MM-DD",
    )
    p.add_argument(
        "--save",
        action="store_true",
        help="Persist run metrics to logs/eval_history.jsonl",
    )
    p.add_argument(
        "--config",
        metavar="PATH",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[evaluate] Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    config = yaml.safe_load(config_path.read_text()) or {}

    prev = _load_last_run(args.game)
    run = run_evaluation(config, game_filter=args.game, since=args.since)

    print_scorecard(run, prev)

    if args.save:
        _persist_run(run, args.game)

    if run.regressions:
        sys.exit(1)


if __name__ == "__main__":
    main()
