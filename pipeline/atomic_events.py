"""
pipeline/atomic_events.py — Atomic Event Mapper

Fuses per-frame detector outputs (kill_feed, audio_detector, roi_matcher)
into typed AtomicEvent objects with confidence scores and excitement weights.

Atomic events (single occurrences detected directly from signals):
    kill            — from kill_feed kill_timestamps
    headshot        — from kill_feed headshot_timestamps
    ultimate_used   — from roi_matcher match on ability_ultimate ROI
    medal_awarded   — from roi_matcher match on medal_popup ROI

Composite events (patterns of atomic events within a time window):
    double_kill     — 2 kills within multikill_window_seconds
    triple_kill     — 3 kills
    quad_kill       — 4 kills
    team_wipe       — 5+ kills

Written to meta.json["atomic_events"]. Read by clip_judge.py for an
optional small context_confidence bonus.

Run order: after roi_matcher, before hook_enforcer.

Config block (config.yaml → atomic_events):
    enabled: true
    multikill_window_seconds: 4.0
    audio_corroborate_window: 1.5
    confidence_base_kill: 0.70
    confidence_audio_bonus: 0.15
    confidence_headshot_bonus: 0.10
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_EXCITEMENT: dict[str, float] = {
    "kill":          1.0,
    "headshot":      2.0,
    "ultimate_used": 3.0,
    "medal_awarded": 2.0,
    "double_kill":   4.0,
    "triple_kill":   6.0,
    "quad_kill":     8.0,
    "team_wipe":    10.0,
}

# Maps ROI name (from hud.yaml) to the atomic event type it signals.
# Extend this dict when new ROI slots are added to game packs.
ROI_TO_EVENT_TYPE: dict[str, str] = {
    "ability_ultimate": "ultimate_used",
    "medal_popup":      "medal_awarded",
}

_DEFAULT_MULTIKILL_WINDOW = 4.0   # seconds — kills within this window form a multi-kill
_DEFAULT_AUDIO_WINDOW    = 1.5    # seconds — audio spike within this of a kill corroborates it
_DEFAULT_CONF_BASE       = 0.70
_DEFAULT_CONF_AUDIO      = 0.15
_DEFAULT_CONF_HEADSHOT   = 0.10


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AtomicEvent:
    event_id: str
    event_type: str             # "kill" | "headshot" | "ultimate_used" | …
    timestamp: float            # seconds into clip; best estimate
    confidence: float           # 0.0 – 1.0
    excitement_score: float     # base excitement weight
    contributing_signals: list[dict]  # raw signals that produced this event


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def map_atomic_events(clip_path: Path, game: str, config: dict) -> dict:
    """Read meta.json, fuse signals into AtomicEvents, write back to meta.json.

    Idempotent: skips if meta.json already contains an "atomic_events" key.

    Returns:
        The atomic_events dict (for testing / logging).
    """
    meta_path = clip_path.with_suffix(".meta.json")
    if not meta_path.exists():
        logger.debug(f"[atomic_events] No meta.json for {clip_path.name} — skipping.")
        return {}

    try:
        meta = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f"[atomic_events] Could not read meta for {clip_path.name}: {exc}")
        return {}

    if "atomic_events" in meta:
        logger.debug(f"[atomic_events] Already processed: {clip_path.name}")
        return meta["atomic_events"]

    ae_cfg = config.get("atomic_events", {})

    multikill_window = float(ae_cfg.get("multikill_window_seconds", _DEFAULT_MULTIKILL_WINDOW))
    audio_window     = float(ae_cfg.get("audio_corroborate_window", _DEFAULT_AUDIO_WINDOW))
    conf_base        = float(ae_cfg.get("confidence_base_kill",    _DEFAULT_CONF_BASE))
    conf_audio       = float(ae_cfg.get("confidence_audio_bonus",  _DEFAULT_CONF_AUDIO))
    conf_headshot    = float(ae_cfg.get("confidence_headshot_bonus", _DEFAULT_CONF_HEADSHOT))

    # --- Collect raw signals from meta.json ---
    kf  = meta.get("kill_feed", {}) or {}
    # audio_detector stage writes to either "audio_detector" or "audio_events" key
    ad  = meta.get("audio_detector", {}) or meta.get("audio_events", {}) or {}
    rm  = meta.get("roi_matches", {}) or {}

    kill_ts      = [float(t) for t in (kf.get("kill_timestamps")      or [])]
    headshot_ts  = [float(t) for t in (kf.get("headshot_timestamps")  or [])]
    audio_spikes = [float(t) for t in (ad.get("spike_timestamps")     or [])]
    roi_matches  = rm.get("matches") or []

    # --- Build atomic events ---
    kill_events = _build_kill_events(
        kill_ts, headshot_ts, audio_spikes,
        conf_base, conf_audio, conf_headshot, audio_window,
    )
    roi_events  = _build_roi_events(roi_matches)
    composites  = _build_composite_events(kill_events, multikill_window)

    all_events = kill_events + roi_events + composites

    total_excitement = round(sum(e.excitement_score for e in all_events), 2)
    cs = _composite_summary(kill_events, roi_events, composites)

    result = {
        "events":            [asdict(e) for e in all_events],
        "composite_summary": cs,
        "total_excitement":  total_excitement,
        "method":            "heuristic",
    }

    meta["atomic_events"] = result
    meta_path.write_text(json.dumps(meta, indent=2))

    logger.info(
        f"[atomic_events] {clip_path.name}: {len(all_events)} events  "
        f"excitement={total_excitement:.1f}  "
        f"kills={cs['kills']}  composites={cs['double_kills']+cs['triple_kills']+cs['quad_kills']+cs['team_wipes']}"
    )
    return result


# ---------------------------------------------------------------------------
# Event builders
# ---------------------------------------------------------------------------

def _build_kill_events(
    kill_ts: list[float],
    headshot_ts: list[float],
    audio_spikes: list[float],
    conf_base: float,
    conf_audio: float,
    conf_headshot: float,
    audio_window: float,
) -> list[AtomicEvent]:
    """One AtomicEvent per kill timestamp; headshots override with higher excitement."""
    headshot_set = set(round(t, 3) for t in headshot_ts)
    events: list[AtomicEvent] = []

    for ts in kill_ts:
        is_headshot = round(ts, 3) in headshot_set

        conf = conf_base
        signals: list[dict] = [{"source": "kill_feed", "timestamp": ts}]

        # Audio corroboration
        closest_spike = _closest(audio_spikes, ts)
        if closest_spike is not None and abs(closest_spike - ts) <= audio_window:
            conf += conf_audio
            signals.append({"source": "audio_detector", "timestamp": closest_spike})

        if is_headshot:
            conf += conf_headshot

        conf = round(min(1.0, conf), 4)
        event_type = "headshot" if is_headshot else "kill"
        eid = f"evt_{event_type}_{int(ts * 100)}"

        events.append(AtomicEvent(
            event_id=eid,
            event_type=event_type,
            timestamp=ts,
            confidence=conf,
            excitement_score=BASE_EXCITEMENT[event_type],
            contributing_signals=signals,
        ))

    return sorted(events, key=lambda e: e.timestamp)


def _build_roi_events(roi_matches: list[dict]) -> list[AtomicEvent]:
    """One AtomicEvent per ROI match whose roi_name maps to a known event type."""
    events: list[AtomicEvent] = []
    seen: set[str] = set()  # deduplicate per (roi_name, frame_bucket)

    for m in roi_matches:
        roi_name    = m.get("roi_name", "")
        event_type  = ROI_TO_EVENT_TYPE.get(roi_name)
        if event_type is None:
            continue

        ts   = float(m.get("frame_time", 0.0))
        conf = round(float(m.get("confidence", 0.75)), 4)

        # Deduplicate events of same type within a 1s bucket
        bucket = f"{event_type}_{int(ts)}"
        if bucket in seen:
            continue
        seen.add(bucket)

        eid = f"evt_{event_type}_{int(ts * 100)}"
        events.append(AtomicEvent(
            event_id=eid,
            event_type=event_type,
            timestamp=ts,
            confidence=conf,
            excitement_score=BASE_EXCITEMENT.get(event_type, 1.0),
            contributing_signals=[{
                "source": "roi_matcher",
                "roi_name": roi_name,
                "template_id": m.get("id", ""),
                "confidence": conf,
                "timestamp": ts,
            }],
        ))

    return sorted(events, key=lambda e: e.timestamp)


def _build_composite_events(
    kill_events: list[AtomicEvent],
    window_sec: float,
) -> list[AtomicEvent]:
    """Derive multi-kill composite events from a sorted list of kill AtomicEvents."""
    if len(kill_events) < 2:
        return []

    sorted_evts = sorted(kill_events, key=lambda e: e.timestamp)
    composites: list[AtomicEvent] = []
    used: set[int] = set()

    for i, anchor in enumerate(sorted_evts):
        if i in used:
            continue

        # Gather all kills within the window starting at anchor.timestamp
        cluster = [
            j for j in range(i, len(sorted_evts))
            if sorted_evts[j].timestamp - anchor.timestamp <= window_sec
        ]
        n = len(cluster)
        if n < 2:
            continue

        for j in cluster:
            used.add(j)

        event_type = {2: "double_kill", 3: "triple_kill", 4: "quad_kill"}.get(
            min(n, 4), "team_wipe"
        )
        conf = round(
            min(sorted_evts[j].confidence for j in cluster) * 0.90, 4
        )

        composites.append(AtomicEvent(
            event_id=f"evt_{event_type}_{int(anchor.timestamp * 100)}",
            event_type=event_type,
            timestamp=anchor.timestamp,
            confidence=conf,
            excitement_score=BASE_EXCITEMENT.get(event_type, 4.0),
            contributing_signals=[
                {"event_id": sorted_evts[j].event_id, "timestamp": sorted_evts[j].timestamp}
                for j in cluster
            ],
        ))

    return composites


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _composite_summary(
    kill_events: list[AtomicEvent],
    roi_events: list[AtomicEvent],
    composites: list[AtomicEvent],
) -> dict:
    return {
        "kills":        sum(1 for e in kill_events if e.event_type == "kill"),
        "headshots":    sum(1 for e in kill_events if e.event_type == "headshot"),
        "double_kills": sum(1 for e in composites  if e.event_type == "double_kill"),
        "triple_kills": sum(1 for e in composites  if e.event_type == "triple_kill"),
        "quad_kills":   sum(1 for e in composites  if e.event_type == "quad_kill"),
        "team_wipes":   sum(1 for e in composites  if e.event_type == "team_wipe"),
        "ultimates_used": sum(1 for e in roi_events if e.event_type == "ultimate_used"),
        "medals":       sum(1 for e in roi_events  if e.event_type == "medal_awarded"),
    }


def _closest(timestamps: list[float], target: float) -> float | None:
    if not timestamps:
        return None
    return min(timestamps, key=lambda t: abs(t - target))
