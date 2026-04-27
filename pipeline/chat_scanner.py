"""
pipeline/chat_scanner.py — Twitch chat velocity scanner for proxy signal detection

Parses a Twitch chat log (pre-downloaded file) and returns normalized ProxySignal
objects representing chat velocity spikes — moments where chat activity surged
relative to the rolling baseline.

Supported log formats:
  [HH:MM:SS] username: message text        <- most common export format
  <seconds_offset> username: message text  <- raw IRC offset format

Usage:
    from pipeline.chat_scanner import scan_chat_log
    signals = scan_chat_log("chat.log", config={"burst_threshold": 3.0})
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math
import re
from pathlib import Path
from typing import Any

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ProxySignal:
    source: str
    timestamp: float
    strength: float
    confidence: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


KEYWORD_WEIGHTS: dict[str, float] = {
    "clip it": 2.5,
    "clip": 1.8,
    "wtf": 2.0,
    "no way": 2.0,
    "insane": 1.8,
    "omg": 1.8,
    "pogchamp": 1.5,
    "pog": 1.5,
    "lmao": 1.4,
    "lul": 1.2,
    "???": 1.2,
    "kekw": 1.2,
}

_BRACKETED_LOG_RE = re.compile(
    r"^\s*\[(?P<hours>\d{1,2}):(?P<minutes>\d{2}):(?P<seconds>\d{2})\]\s+[^:]+:\s*(?P<message>.*)\s*$"
)
_OFFSET_LOG_RE = re.compile(
    r"^\s*(?P<seconds>\d+(?:\.\d+)?)\s+[^:]+:\s*(?P<message>.*)\s*$"
)
_KEYWORD_PATTERNS = [
    (keyword, weight, re.compile(rf"(?<!\w){re.escape(keyword)}(?!\w)", re.IGNORECASE))
    for keyword, weight in sorted(KEYWORD_WEIGHTS.items(), key=lambda item: (-len(item[0]), item[0]))
]


def _message_weight(message: str) -> float:
    lowered = f" {message.lower()} "
    occupied: list[tuple[int, int]] = []
    total = 0.0

    for _keyword, weight, pattern in _KEYWORD_PATTERNS:
        matched_spans: list[tuple[int, int]] = []
        for match in pattern.finditer(lowered):
            start, end = match.span()
            if any(start < taken_end and end > taken_start for taken_start, taken_end in occupied):
                continue
            matched_spans.append((start, end))
        if matched_spans:
            occupied.extend(matched_spans)
            total += weight

    return round(total, 4)


def _parse_log(path: Path) -> list[dict[str, float]]:
    records: list[dict[str, float]] = []
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        bracketed = _BRACKETED_LOG_RE.match(line)
        if bracketed:
            timestamp = (
                int(bracketed.group("hours")) * 3600
                + int(bracketed.group("minutes")) * 60
                + int(bracketed.group("seconds"))
            )
            message = bracketed.group("message")
        else:
            offset = _OFFSET_LOG_RE.match(line)
            if not offset:
                logger.debug(f"[chat_scanner] Skipping unrecognized chat line {line_no}: {raw_line!r}")
                continue
            timestamp = float(offset.group("seconds"))
            message = offset.group("message")

        weight = _message_weight(message)
        if weight <= 0:
            continue
        records.append({"seconds": float(timestamp), "weight": weight})

    return records


def _compute_velocity_signals(records: list[dict[str, float]], config: dict[str, Any]) -> list[ProxySignal]:
    if not records:
        return []

    bucket_seconds = max(1, int(config.get("bucket_seconds", 5)))
    rolling_baseline_seconds = max(bucket_seconds, int(config.get("rolling_baseline_seconds", 300)))
    burst_threshold = float(config.get("burst_threshold", 3.0))
    confidence = float(config.get("confidence", 0.70))
    baseline_floor = float(config.get("baseline_floor", 0.1))
    max_score_velocity = float(config.get("max_score_velocity", 10.0))

    bucket_scores: dict[int, float] = {}
    max_bucket = 0
    for record in records:
        bucket = int(math.floor(float(record["seconds"]) / bucket_seconds))
        bucket_scores[bucket] = bucket_scores.get(bucket, 0.0) + float(record["weight"])
        max_bucket = max(max_bucket, bucket)

    rolling_window_size = max(1, int(math.ceil(rolling_baseline_seconds / bucket_seconds)))
    rolling_values: deque[float] = deque()
    rolling_sum = 0.0
    signals: list[ProxySignal] = []

    for bucket in range(0, max_bucket + 1):
        raw_score = bucket_scores.get(bucket, 0.0)
        rolling_values.append(raw_score)
        rolling_sum += raw_score
        if len(rolling_values) > rolling_window_size:
            rolling_sum -= rolling_values.popleft()

        baseline = rolling_sum / len(rolling_values)
        velocity = raw_score / (baseline + baseline_floor)
        if velocity < burst_threshold or raw_score <= 0:
            continue

        strength = min(1.0, velocity / max_score_velocity)
        timestamp = round(bucket * bucket_seconds, 3)
        signals.append(
            ProxySignal(
                source="chat_spike",
                timestamp=timestamp,
                strength=round(strength, 4),
                confidence=round(confidence, 4),
                reason=f"chat velocity {velocity:.1f}x baseline",
            )
        )

    return signals


def scan_chat_log(log_path: str | Path, config: dict[str, Any] | None = None) -> list[ProxySignal]:
    cfg = dict(config or {})
    path = Path(log_path)
    if not path.exists():
        raise FileNotFoundError(f"Chat log not found: {path}")

    records = _parse_log(path)
    signals = _compute_velocity_signals(records, cfg)
    logger.info(
        f"[chat_scanner] {path.name}: parsed_records={len(records)}, emitted_signals={len(signals)}"
    )
    return signals
