"""
Training Logger — silently capture human review decisions as JSONL training records.

Every approve/reject in the review UI calls log_review_decision(), which extracts
all detector signals from meta.json and appends one JSON line to:
    data/training_sets/clip_judge/YYYY-MM-DD.jsonl

These records form the labeled dataset for a future Learned Fusion Model that will
replace the hand-tuned weights in clip_judge.py.

Schema version: 1
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_OUTPUT_DIR = "data/training_sets"
_SCHEMA_VERSION = "1"


def log_review_decision(meta: dict, clip_id: str, game: str, config: dict | None = None) -> bool:
    """Append one JSONL training record for a human review decision.

    Returns True on success, False on any error. Never raises — the review UI
    must not be interrupted by logging failures.
    """
    try:
        training_cfg = (config or {}).get("training", {})
        if not training_cfg.get("enabled", True):
            return False

        output_dir = Path(training_cfg.get("output_dir", _DEFAULT_OUTPUT_DIR)) / "clip_judge"
        output_dir.mkdir(parents=True, exist_ok=True)

        today = datetime.now().strftime("%Y-%m-%d")
        log_path = output_dir / f"{today}.jsonl"

        record = _build_record(meta, clip_id, game)
        with log_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")

        logger.debug(f"[training_logger] Logged {clip_id} ({meta.get('review_status')}) → {log_path}")
        return True

    except Exception as exc:
        logger.debug(f"[training_logger] Failed to log {clip_id}: {exc}")
        return False


def _build_record(meta: dict, clip_id: str, game: str) -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "id": clip_id,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "game": game,
        "label": {
            "decision": meta.get("review_status"),
            "reviewed_at": meta.get("reviewed_at"),
        },
        "features": _extract_features(meta),
    }


def _extract_features(meta: dict) -> dict[str, Any]:
    worthiness = meta.get("worthiness") or {}
    kill_feed = meta.get("kill_feed") or {}
    audio = meta.get("audio_events") or {}
    weapon = meta.get("weapon_detection") or {}
    scoring = meta.get("scoring") or {}
    hook = meta.get("hook_enforcer") or {}
    roi = meta.get("roi_matches") or {}

    hook_trim = (hook.get("trim_plan") or {})
    retention = (hook.get("retention_flags") or {})

    roi_matches = roi.get("matches") or []

    return {
        # Clip basics
        "duration_seconds": _f(meta.get("duration_seconds")),
        "resolution_height": _i(meta.get("resolution_height")),
        "fps": _f(meta.get("fps")),
        "motion_level": meta.get("motion_level"),
        "audio_energy": meta.get("audio_energy"),
        "silence_ratio": _f(meta.get("silence_ratio")),
        "scene_change_count": _i(meta.get("scene_change_count")),
        "keyword_count": len(meta.get("keywords") or []),
        # Clip judge composite scores
        "context_confidence": _f(worthiness.get("context_confidence")),
        "hook_confidence": _f(worthiness.get("hook_confidence")),
        "postability_score": _f(worthiness.get("postability_score")),
        "final_score": _f(worthiness.get("final_score")),
        "system_decision": worthiness.get("decision"),
        "quarantine_reason": worthiness.get("quarantine_reason"),
        # Kill feed signals
        "kill_count": _i(kill_feed.get("kill_count")),
        "headshot_count": _i(kill_feed.get("headshot_count")),
        "sweat_score": _f(kill_feed.get("sweat_score")),
        "kill_feed_passed": kill_feed.get("passed"),
        # Audio signals
        "audio_spike_count": _i(audio.get("spike_count")),
        "multi_kill_detected": audio.get("multi_kill_detected"),
        # Weapon detection
        "weapon_confidence": _f(weapon.get("confidence")),
        "weapon_detected": weapon.get("weapon_id") is not None,
        # AI scoring
        "highlight_score": _i(scoring.get("highlight_score")),
        "clip_type": scoring.get("clip_type"),
        # Hook enforcer
        "hook_score": _f(hook.get("hook_score")),
        "early_hook_passed": hook.get("early_hook_passed"),
        "dead_air_risk": retention.get("dead_air_risk"),
        "max_gap_seconds": _f(retention.get("max_gap_between_moments_seconds")),
        "trim_plan_strategy": hook_trim.get("strategy"),
        # ROI matcher
        "roi_match_count": len(roi_matches),
    }


def _f(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def _i(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
