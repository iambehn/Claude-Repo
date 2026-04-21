"""
Composite Clip Judge — per-clip accept / reject / quarantine decision.

Runs between feature_extraction and decision_engine. Combines detector signals
(kill_feed, weapon_detector, audio_detector) and feature signals (sweat_score,
audio_energy, motion_level, keywords) into three weighted sub-scores:

    context_confidence  — how much detector evidence we have (0.0 - 1.0)
    hook_confidence     — does something interesting happen in the first
                          `hook_window_seconds` (default 1.5s)? (0.0 - 1.0)
    postability_score   — how likely this clip is to perform if posted (0.0 - 1.0)

Decision:
    accept:                     postability >= accept AND context >= min_context AND hook >= hook_gate
    quarantine(hook_not_resolved): same but hook < hook_gate
    reject:                     postability < reject AND context >= reliable_confidence
    quarantine(missing_context): context < min_context
    quarantine(low_confidence):  otherwise

Writes a `worthiness` block to the clip's .meta.json. Idempotent — re-running
on a clip that already has `worthiness.decision` returns the cached block.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pipeline import game_pack
from utils.file_utils import move_to_quarantine
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _energy_to_float(level: str | None) -> float:
    return {"low": 0.3, "medium": 0.6, "high": 1.0}.get((level or "").lower(), 0.0)


def _rescale(weights: dict[str, float], enabled_keys: set[str]) -> dict[str, float]:
    """Drop disabled keys and rescale the remaining weights to sum to 1.0."""
    active = {k: v for k, v in weights.items() if k in enabled_keys}
    total = sum(active.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in active.items()}


# ---------------------------------------------------------------------------
# Sub-scores
# ---------------------------------------------------------------------------


def _context_confidence(meta: dict, pack: game_pack.GamePack, config: dict) -> tuple[float, list[str]]:
    """Weighted mean of detector-signal confidences.

    Components whose detector is disabled drop out and the weights rescale.
    """
    weights = pack.weights.context_inputs
    explanation: list[str] = []

    # Which detectors are enabled (config-level switches; game-pack detectors
    # flag is informational only here).
    kf_enabled = bool(config.get("kill_feed", {}).get("enabled"))
    wd_enabled = bool(config.get("weapon_detector", {}).get("enabled"))
    ad_enabled = bool(config.get("audio_detector", {}).get("enabled"))
    rm_enabled = bool(config.get("roi_matcher", {}).get("enabled"))

    active: set[str] = set()
    if wd_enabled:
        active.add("weapon_confidence")
    if kf_enabled:
        active.add("kill_detection_saturation")
    if ad_enabled:
        active.add("audio_saturation")
    if rm_enabled:
        active.add("roi_match_count")

    if not active:
        explanation.append("no detectors enabled — context_confidence=0")
        return 0.0, explanation

    active_weights = _rescale(weights, active)

    components: dict[str, float] = {}

    if "weapon_confidence" in active:
        wd = meta.get("weapon_detection", {}) or {}
        components["weapon_confidence"] = _clamp(float(wd.get("confidence", 0.0)))

    if "kill_detection_saturation" in active:
        kf = meta.get("kill_feed", {}) or {}
        kills = float(kf.get("kill_count", 0))
        components["kill_detection_saturation"] = min(1.0, kills / 3.0)

    if "audio_saturation" in active:
        ad = meta.get("audio_detector", {}) or {}
        spikes = len(ad.get("spike_timestamps") or [])
        components["audio_saturation"] = min(1.0, spikes / 4.0)

    if "roi_match_count" in active:
        rm = meta.get("roi_matches", {}) or {}
        matches = rm.get("matches") or []
        components["roi_match_count"] = min(1.0, len(matches) / 2.0)

    score = sum(active_weights[k] * components[k] for k in components)
    for k, v in components.items():
        explanation.append(f"{k}={v:.2f} (weight={active_weights[k]:.2f})")
    return _clamp(score), explanation


def _hook_confidence(meta: dict, pack: game_pack.GamePack) -> tuple[float, dict[str, Any], list[str]]:
    """Did something interesting happen in the first hook_window_seconds?

    Formula: (0.35 if early_kill) + (0.50 if early_audio_spike) + (0.15 if early_speech)
    Capped at 1.0.
    """
    hook_window = float(pack.weights.thresholds.get("hook_window_seconds", 1.5))
    explanation: list[str] = []

    kf = meta.get("kill_feed", {}) or {}
    ad = meta.get("audio_detector", {}) or {}
    tr = meta.get("transcription", {}) or {}

    kill_ts = kf.get("kill_timestamps") or []
    spike_ts = ad.get("spike_timestamps") or []

    early_kill = any(t <= hook_window for t in kill_ts)
    early_spike = any(t <= hook_window for t in spike_ts)

    # Speech detection: prefer Whisper segments with start<hook_window. Fall
    # back to "any non-empty transcript" when segments aren't available.
    early_speech = False
    segments = tr.get("segments") or meta.get("segments") or []
    if segments:
        early_speech = any(float(s.get("start", 0.0)) <= hook_window for s in segments)
    else:
        # Fallback: we don't know when speech starts. Any non-empty text counts.
        text = tr.get("text") or meta.get("transcript", "") or ""
        early_speech = bool(text.strip())

    score = 0.0
    if early_kill:
        score += 0.35
        explanation.append(f"early_kill within {hook_window}s")
    if early_spike:
        score += 0.50
        explanation.append(f"early_audio_spike within {hook_window}s")
    if early_speech:
        score += 0.15
        explanation.append(f"early_speech within {hook_window}s")

    action_start: float | None = None
    candidates: list[float] = []
    if early_kill:
        candidates.extend(t for t in kill_ts if t <= hook_window)
    if early_spike:
        candidates.extend(t for t in spike_ts if t <= hook_window)
    if candidates:
        action_start = round(min(candidates), 3)

    signals = {
        "early_kill": early_kill,
        "early_audio_spike": early_spike,
        "early_speech": early_speech,
        "action_start_seconds": action_start,
        "hook_window_seconds": hook_window,
    }
    return _clamp(score), signals, explanation


def _postability_score(meta: dict, pack: game_pack.GamePack) -> tuple[float, float, list[str]]:
    """Weighted sum of normalised "will this pop" signals.

    Returns (postability_score, moment_multiplier, explanation).
    """
    weights = pack.weights.postability_inputs
    explanation: list[str] = []

    kf = meta.get("kill_feed", {}) or {}
    wd = meta.get("weapon_detection", {}) or {}
    ad = meta.get("audio_detector", {}) or {}

    # Normalise each raw signal to [0, 1].
    components = {
        "sweat_score": _clamp(float(kf.get("sweat_score", 0.0)) / 100.0),
        "audio_energy": _energy_to_float(meta.get("audio_energy")),
        "motion_level": _energy_to_float(meta.get("motion_level")),
        "keyword_count": _clamp(len(meta.get("keywords", []) or []) / 5.0),
        "weapon_confidence": _clamp(float(wd.get("confidence", 0.0))),
        "audio_spike_count": _clamp(len(ad.get("spike_timestamps") or []) / 6.0),
    }

    total_weight = sum(weights.get(k, 0.0) for k in components)
    if total_weight <= 0:
        return 0.0, 1.0, ["no postability weights configured"]
    score = sum(weights.get(k, 0.0) * v for k, v in components.items()) / total_weight

    for k, v in components.items():
        if weights.get(k, 0.0) > 0:
            explanation.append(f"{k}={v:.2f} (w={weights[k]:.2f})")

    # Moment multiplier — pick the highest matching moment's score_weight.
    multiplier = 1.0
    matched_moment_ids: list[str] = []
    for moment in pack.moments.moments:
        if _moment_matches(moment, meta):
            matched_moment_ids.append(moment.id)
            multiplier = max(multiplier, moment.score_weight)

    if matched_moment_ids:
        explanation.append(
            f"moments matched: {matched_moment_ids} -> x{multiplier:.2f}"
        )

    return _clamp(score * multiplier), multiplier, explanation


def _moment_matches(moment: "game_pack.Moment", meta: dict) -> bool:
    """Evaluate a moment's evidence rules against metadata."""
    ev = moment.evidence or {}
    kf = meta.get("kill_feed", {}) or {}
    ad = meta.get("audio_detector", {}) or {}

    kill_count = int(kf.get("kill_count", 0))
    headshot_count = int(kf.get("headshot_count", 0))
    kill_ts = kf.get("kill_timestamps") or []
    spike_ts = ad.get("spike_timestamps") or []
    duration = float(meta.get("duration_seconds") or 0.0)

    if "kill_count_min" in ev and kill_count < int(ev["kill_count_min"]):
        return False
    if "headshot_count_min" in ev and headshot_count < int(ev["headshot_count_min"]):
        return False

    if "within_window_seconds" in ev and kill_ts:
        window = float(ev["within_window_seconds"])
        needed = int(ev.get("kill_count_min", 0))
        if needed:
            # True if any sliding window of size `window` contains >= needed kills.
            ok = False
            for i, anchor in enumerate(kill_ts):
                nearby = [t for t in kill_ts[i:] if t - anchor <= window]
                if len(nearby) >= needed:
                    ok = True
                    break
            if not ok:
                return False

    if "kill_in_final_seconds" in ev and kill_ts and duration:
        tail = duration - float(ev["kill_in_final_seconds"])
        if not any(t >= tail for t in kill_ts):
            return False

    if "kill_within_seconds" in ev and kill_ts:
        if not any(t <= float(ev["kill_within_seconds"]) for t in kill_ts):
            return False

    if "audio_spike_after_lull" in ev and ev["audio_spike_after_lull"]:
        if len(spike_ts) < 1:
            return False
        # Simple proxy: at least one spike in the second half of the clip.
        if not any(t >= (duration / 2.0) for t in spike_ts):
            return False

    # ability_class is Phase 2 territory — we don't have ability detection yet.
    if "ability_class" in ev:
        return False

    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate(meta_path: Path, pack: game_pack.GamePack, config: dict) -> dict:
    """Compute worthiness block and write it to meta_path.

    Idempotent: returns the cached block if `worthiness.decision` already exists.
    """
    meta = json.loads(meta_path.read_text())
    cached = meta.get("worthiness") or {}
    if cached.get("decision"):
        return cached

    context, ctx_expl = _context_confidence(meta, pack, config)
    hook, hook_signals, hook_expl = _hook_confidence(meta, pack)
    postability, moment_mul, post_expl = _postability_score(meta, pack)

    w = pack.weights.composite
    final = (
        w["context_weight"] * context
        + w["hook_weight"] * hook
        + w["postability_weight"] * postability
    )

    th = pack.weights.thresholds
    accept_th = float(th["accept"])
    reject_th = float(th["reject"])
    min_context = float(th["min_context"])
    hook_gate = float(th["hook_gate"])
    reliable_conf = float(th["reliable_confidence"])

    decision, reason = _decide(
        context=context,
        hook=hook,
        postability=postability,
        accept_th=accept_th,
        reject_th=reject_th,
        min_context=min_context,
        hook_gate=hook_gate,
        reliable_conf=reliable_conf,
    )

    worthiness = {
        "context_confidence": round(context, 3),
        "hook_confidence": round(hook, 3),
        "postability_score": round(postability, 3),
        "moment_multiplier": round(moment_mul, 2),
        "final_score": round(final, 3),
        "decision": decision,
        "quarantine_reason": reason,
        "hook_satisfied": hook >= hook_gate,
        "action_start_seconds": hook_signals.get("action_start_seconds"),
        "explanation": {
            "context": ctx_expl,
            "hook": hook_expl,
            "postability": post_expl,
        },
    }

    meta["worthiness"] = worthiness
    meta_path.write_text(json.dumps(meta, indent=2))
    return worthiness


def _decide(
    *,
    context: float,
    hook: float,
    postability: float,
    accept_th: float,
    reject_th: float,
    min_context: float,
    hook_gate: float,
    reliable_conf: float,
) -> tuple[str, str | None]:
    if context < min_context:
        return "quarantine", "missing_context"

    if postability < reject_th and context >= reliable_conf:
        return "reject", None

    if postability >= accept_th and context >= min_context:
        if hook < hook_gate:
            return "quarantine", "hook_not_resolved"
        return "accept", None

    if context < reliable_conf:
        return "quarantine", "low_confidence"

    return "quarantine", "low_confidence"


def run_clip_judge(clip_path: str | Path, game: str, config: dict) -> dict:
    """Pipeline entry point.

    Loads the game pack, evaluates the clip, and routes to quarantine when the
    decision says so. Returns the worthiness dict.
    """
    clip = Path(clip_path)
    meta_path = clip.with_suffix(".meta.json")
    if not meta_path.exists():
        logger.warning(f"[clip_judge] No meta.json for {clip.name} — skipping.")
        return {"decision": "skip", "quarantine_reason": None}

    pack = game_pack.load(game)
    worthiness = evaluate(meta_path, pack, config)

    logger.info(
        f"[clip_judge] {clip.name}: decision={worthiness['decision']}"
        + (f" ({worthiness['quarantine_reason']})" if worthiness.get("quarantine_reason") else "")
        + f" | context={worthiness['context_confidence']:.2f}"
        + f" hook={worthiness['hook_confidence']:.2f}"
        + f" post={worthiness['postability_score']:.2f}"
        + f" final={worthiness['final_score']:.2f}"
    )

    if worthiness["decision"] == "quarantine":
        move_to_quarantine(clip, game, config, reason=worthiness["quarantine_reason"] or "low_confidence")

    return worthiness
