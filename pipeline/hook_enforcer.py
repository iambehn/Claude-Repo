"""
Hook Enforcer — early engagement gate for gaming clips.

Reads kill_feed, audio_events, niceshot_detection, and yolo_detection from
meta.json and determines whether the clip has an anchorable hook moment within
the first `window_seconds` (default 1.5s). If not, it looks for a late hook
and proposes a hard_trim strategy. Writes results to meta["hook_enforcer"].

Config block (config.yaml → hook_enforcer):
    window_seconds: 1.5
    acceptance_threshold: 0.5
    pre_event_padding_seconds: 0.5
    minimum_remaining_seconds: 6.0
    dead_air_gap_seconds: 4.0
    signal_weights:
        kill_feed: 0.45
        niceshot: 0.25
        yolo: 0.20
        audio: 0.10
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULTS = {
    "window_seconds": 1.5,
    "acceptance_threshold": 0.5,
    "pre_event_padding_seconds": 0.5,
    "minimum_remaining_seconds": 6.0,
    "dead_air_gap_seconds": 4.0,
    "signal_weights": {
        "kill_feed": 0.45,
        "niceshot": 0.25,
        "yolo": 0.20,
        "audio": 0.10,
    },
}


def run_hook_enforcer(
    clip_path: str | Path,
    game: str,
    config: dict,
    game_pack: dict | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Create an explicit early-hook/remediation manifest for one clip."""
    clip = Path(clip_path)
    meta_path = clip.with_suffix(".meta.json")

    if not meta_path.exists():
        logger.error(f"[hook_enforcer] No meta.json found for {clip.name}")
        return _error_result("missing_meta")

    meta = json.loads(meta_path.read_text())
    if not force and meta.get("hook_enforcer"):
        logger.debug(f"[hook_enforcer] Already analysed: {clip.name}")
        return meta["hook_enforcer"]

    cfg = _hook_config(config, game_pack)
    window_seconds = float(cfg["window_seconds"])
    threshold = float(cfg["acceptance_threshold"])
    pre_event_padding = float(cfg["pre_event_padding_seconds"])
    min_remaining = float(cfg["minimum_remaining_seconds"])
    duration = _optional_float(meta.get("duration_seconds"))

    moments = _collect_moments(meta, cfg)
    early_moments = [m for m in moments if m["timestamp"] <= window_seconds]
    early_signals = _summarize_signals(early_moments, cfg)
    early_anchor = _best_anchorable(early_moments, threshold)
    early_hook_score = _score_moments(early_moments, cfg)
    early_hook_passed = bool(early_anchor) and early_hook_score >= threshold

    anchor_moment = early_anchor
    trim_plan = {
        "strategy": "none" if early_hook_passed else "unresolved",
        "trim_start_seconds": 0.0,
        "expected_hook_timestamp": early_anchor["timestamp"] if early_anchor else None,
        "pre_event_padding_seconds": pre_event_padding,
    }

    explanation: list[str] = []
    if early_hook_passed:
        explanation.append(
            f"Early hook found from {early_anchor['source']} at {early_anchor['timestamp']}s."
        )
    else:
        late_anchor = _first_late_anchor(moments, threshold, window_seconds)
        if late_anchor:
            trim_start = max(0.0, late_anchor["timestamp"] - pre_event_padding)
            remaining = None if duration is None else max(0.0, duration - trim_start)
            if remaining is None or remaining >= min_remaining:
                anchor_moment = late_anchor
                trim_plan = {
                    "strategy": "hard_trim",
                    "trim_start_seconds": round(trim_start, 3),
                    "expected_hook_timestamp": round(late_anchor["timestamp"] - trim_start, 3),
                    "pre_event_padding_seconds": pre_event_padding,
                }
                explanation.append(
                    f"Late hook from {late_anchor['source']} at {late_anchor['timestamp']}s "
                    f"can be aligned with hard trim."
                )
            else:
                anchor_moment = late_anchor
                explanation.append(
                    f"Late hook at {late_anchor['timestamp']}s would leave only "
                    f"{round(remaining, 3)}s, below minimum {min_remaining}s."
                )
        else:
            explanation.append("No anchorable hook moment met the configured threshold.")

    retention_flags = _retention_flags(moments, cfg)
    hook_score = early_hook_score if early_moments else _score_moments(
        [anchor_moment] if anchor_moment else [], cfg
    )

    result = {
        "status": "ok",
        "early_hook_passed": early_hook_passed,
        "hook_score": round(hook_score, 3),
        "window_seconds": window_seconds,
        "anchor_moment": anchor_moment,
        "early_signals": early_signals,
        "all_moments": moments,
        "retention_flags": retention_flags,
        "trim_plan": trim_plan,
        "explanation": explanation,
    }

    meta["hook_enforcer"] = result
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info(
        f"[hook_enforcer] {clip.name}: "
        f"{'early hook' if early_hook_passed else trim_plan['strategy']} "
        f"(score={result['hook_score']})"
    )
    return result


def _hook_config(config: dict, game_pack: dict | None) -> dict[str, Any]:
    cfg = {
        **_DEFAULTS,
        "signal_weights": dict(_DEFAULTS["signal_weights"]),
    }
    weights = ((game_pack or {}).get("weights") or {}).get("clip_judge") or {}
    hard_gates = weights.get("hard_gates") or {}
    hook_cfg = weights.get("hook_enforcer") or {}

    if "hook_window_seconds" in hard_gates and "window_seconds" not in hook_cfg:
        cfg["window_seconds"] = hard_gates["hook_window_seconds"]

    for key in (
        "window_seconds",
        "acceptance_threshold",
        "pre_event_padding_seconds",
        "minimum_remaining_seconds",
        "dead_air_gap_seconds",
    ):
        if key in hook_cfg:
            cfg[key] = hook_cfg[key]

    if isinstance(hook_cfg.get("signal_weights"), dict):
        cfg["signal_weights"].update(hook_cfg["signal_weights"])

    global_cfg = config.get("hook_enforcer") or {}
    for key in (
        "window_seconds",
        "acceptance_threshold",
        "pre_event_padding_seconds",
        "minimum_remaining_seconds",
        "dead_air_gap_seconds",
    ):
        if key in global_cfg:
            cfg[key] = global_cfg[key]
    if isinstance(global_cfg.get("signal_weights"), dict):
        cfg["signal_weights"].update(global_cfg["signal_weights"])

    return cfg


def _collect_moments(meta: dict, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    moments: list[dict[str, Any]] = []

    kill_feed = meta.get("kill_feed") or {}
    for ts in kill_feed.get("kill_timestamps") or []:
        moments.append(_moment(ts, "kill_feed", "kill", 0.78, True, "kill_feed", cfg))
    for ts in kill_feed.get("headshot_timestamps") or []:
        moments.append(_moment(ts, "kill_feed", "headshot", 0.92, True, "kill_feed", cfg))

    audio = meta.get("audio_events") or {}
    for ts in audio.get("spike_timestamps") or []:
        moments.append(_moment(ts, "audio_detector", "audio_spike", 0.55, False, "audio", cfg))

    niceshot = meta.get("niceshot_detection") or {}
    if niceshot.get("status") == "ok":
        for item in niceshot.get("moments") or []:
            if not isinstance(item, dict):
                continue
            moments.append(
                _moment(
                    item.get("timestamp", 0.0),
                    "niceshot",
                    str(item.get("kind") or "action_spike"),
                    _safe_float(item.get("confidence"),
                                _safe_float(niceshot.get("confidence"), 0.0)),
                    bool(item.get("hook_candidate", True)),
                    "niceshot",
                    cfg,
                )
            )

    yolo = meta.get("yolo_detection") or {}
    if yolo.get("status") == "ok":
        for item in yolo.get("event_candidates") or []:
            if not isinstance(item, dict):
                continue
            moments.append(
                _moment(
                    item.get("timestamp", 0.0),
                    "yolo_detector",
                    str(item.get("event_id") or item.get("label") or "visual_event"),
                    _safe_float(item.get("confidence"), 0.0),
                    True,
                    "yolo",
                    cfg,
                )
            )

    deduped: dict[tuple[float, str, str], dict[str, Any]] = {}
    for item in moments:
        key = (item["timestamp"], item["source"], item["kind"])
        existing = deduped.get(key)
        if existing is None or item["confidence"] > existing["confidence"]:
            deduped[key] = item
    return sorted(deduped.values(), key=lambda item: (item["timestamp"], -item["confidence"]))


def _moment(
    timestamp: Any,
    source: str,
    kind: str,
    confidence: float,
    anchorable: bool,
    signal_key: str,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    confidence = _clamp(confidence)
    signal_weight = _signal_weight(signal_key, cfg)
    return {
        "timestamp": round(_safe_float(timestamp, 0.0), 3),
        "source": source,
        "kind": kind,
        "confidence": confidence,
        "hook_candidate": anchorable,
        "signal": signal_key,
        "signal_weight": signal_weight,
        "moment_score": round(
            max(signal_weight * confidence, confidence if anchorable else 0.0), 3
        ),
    }


def _summarize_signals(
    moments: list[dict[str, Any]], cfg: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for signal in (cfg.get("signal_weights") or {}).keys():
        hits = [m for m in moments if m["signal"] == signal]
        summary[signal] = {
            "present": bool(hits),
            "count": len(hits),
            "best_timestamp": min((m["timestamp"] for m in hits), default=None),
            "best_confidence": round(max((m["confidence"] for m in hits), default=0.0), 3),
        }
    return summary


def _score_moments(moments: list[dict[str, Any]], cfg: dict[str, Any]) -> float:
    if not moments:
        return 0.0
    weighted = 0.0
    for signal, weight in (cfg.get("signal_weights") or {}).items():
        signal_confidence = max(
            (m["confidence"] for m in moments if m["signal"] == signal), default=0.0
        )
        weighted += float(weight) * signal_confidence
    best_anchor = max(
        (m["confidence"] for m in moments if m.get("hook_candidate")), default=0.0
    )
    return round(max(weighted, best_anchor), 3)


def _best_anchorable(
    moments: list[dict[str, Any]], threshold: float
) -> dict[str, Any] | None:
    eligible = [
        item for item in moments
        if item.get("hook_candidate") and item.get("moment_score", 0.0) >= threshold
    ]
    if not eligible:
        return None
    return sorted(eligible, key=lambda item: (-item["confidence"], item["timestamp"]))[0]


def _first_late_anchor(
    moments: list[dict[str, Any]], threshold: float, window_seconds: float
) -> dict[str, Any] | None:
    eligible = [
        item for item in moments
        if item["timestamp"] > window_seconds
        and item.get("hook_candidate")
        and item.get("moment_score", 0.0) >= threshold
    ]
    if not eligible:
        return None
    return sorted(eligible, key=lambda item: (item["timestamp"], -item["confidence"]))[0]


def _retention_flags(moments: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    anchorable = sorted(m["timestamp"] for m in moments if m.get("hook_candidate"))
    max_gap = 0.0
    if len(anchorable) >= 2:
        max_gap = max(b - a for a, b in zip(anchorable, anchorable[1:]))
    dead_air_threshold = float(cfg.get("dead_air_gap_seconds", 4.0))
    return {
        "max_gap_between_moments_seconds": round(max_gap, 3),
        "dead_air_risk": max_gap > dead_air_threshold,
    }


def _signal_weight(signal: str, cfg: dict[str, Any]) -> float:
    return float((cfg.get("signal_weights") or {}).get(signal, 0.0))


def _error_result(reason: str) -> dict[str, Any]:
    return {
        "status": "error",
        "early_hook_passed": False,
        "hook_score": 0.0,
        "window_seconds": _DEFAULTS["window_seconds"],
        "anchor_moment": None,
        "early_signals": {},
        "all_moments": [],
        "retention_flags": {"max_gap_between_moments_seconds": 0.0, "dead_air_risk": False},
        "trim_plan": {
            "strategy": "unresolved",
            "trim_start_seconds": 0.0,
            "expected_hook_timestamp": None,
        },
        "explanation": [reason],
    }


def _safe_float(value: Any, default: float | None = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0 if default is None else float(default)


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp(value: Any) -> float:
    try:
        return round(max(0.0, min(1.0, float(value))), 3)
    except (TypeError, ValueError):
        return 0.0
