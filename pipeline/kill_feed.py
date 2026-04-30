"""
Kill-Feed Parser — Advanced Clip Intelligence, Stage 2.5

Analyses the kill-feed ROI of a downloaded clip using OpenCV to detect
kill and headshot events BEFORE committing to transcription or AI scoring.
Clips that score above the sweat threshold are promoted; below-threshold
clips still proceed through the pipeline but carry a lower kill_feed score.

Detection signal chain (highest-priority signal that fires wins):
    1. Color mask     — cv2.inRange() pixel-count spike in kill-feed-colored pixels
    2. Template match — cv2.matchTemplate() for kill/headshot icon assets (optional)
    3. MOG2           — background subtractor; detects new UI elements appearing in ROI
    4. Edge fallback  — cv2.Canny() edge-count spike

Sweat Score (sliding window):
    Kill event      = +10 pts  (configurable via kill_score)
    Headshot event  = +20 pts  (configurable via headshot_score)
    Peak window total > sweat_threshold (default 50) → passed = True

Resolution normalization:
    Every frame is resized to TARGET_WIDTH × TARGET_HEIGHT (1920×1080) before
    the ROI crop, so config coordinates are stable regardless of stream resolution
    (1440p, ultrawide, etc.).

Config block (config.yaml → kill_feed):
    enabled: false
    sweat_threshold: 50
    window_seconds: 5
    sample_fps: 5
    kill_score: 10
    headshot_score: 20
    color_spike_factor: 5.0   # current pixels must be > factor × baseline to trigger
    min_pixel_mass: 50        # ignore color hits below this raw pixel count
    template_dir: "assets/kill_feed_templates"

Per-game HUD configuration (rois.kill_feed + color bands) lives in:
    assets/games/<slug>/hud.yaml  →  rois.kill_feed

ROI calibration:
    To tune coordinates for a new game:
      1. Grab a screenshot at native resolution: python -c "
             import cv2, sys
             cap = cv2.VideoCapture(sys.argv[1])
             ok, frame = cap.read()
             cv2.imwrite('frame0.png', frame)"
      2. Open frame0.png in any image editor; note the pixel rectangle around the kill feed.
      3. Convert to pct coordinates (0.0-1.0):
             x_pct = x_raw / frame_width
             y_pct = y_raw / frame_height
      4. Update rois.kill_feed in assets/games/<slug>/hud.yaml.

Template assets:
    Place kill/headshot icon PNG files in:
        assets/kill_feed_templates/<game>/kill_*.png
        assets/kill_feed_templates/<game>/headshot_*.png
    If no templates are found for a game, template matching is skipped and the
    color mask + MOG2 chain handles detection.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import NamedTuple

from pipeline import game_pack
from utils.logger import get_logger

logger = get_logger(__name__)

try:
    import cv2
    import numpy as np
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    logger.warning(
        "OpenCV not installed — Kill-Feed Parser will be skipped.\n"
        "  Fix: pip install opencv-python-headless"
    )


# ---------------------------------------------------------------------------
# Constants (overridden by config values when provided)
# ---------------------------------------------------------------------------

TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080

_DEFAULT_SAMPLE_FPS = 5
_DEFAULT_WINDOW_SECONDS = 5
_DEFAULT_SWEAT_THRESHOLD = 50
_DEFAULT_KILL_SCORE = 10
_DEFAULT_HEADSHOT_SCORE = 20
_DEFAULT_SPIKE_FACTOR = 5.0   # 500% above baseline
_DEFAULT_MIN_PIXEL_MASS = 50
_DEFAULT_TEMPLATE_THRESHOLD = 0.80
_BASELINE_WINDOW = 30          # rolling baseline uses this many frames


class _Event(NamedTuple):
    timestamp: float    # seconds into clip
    kind: str           # "kill" | "headshot"
    method: str         # "color_mask" | "template_match" | "mog2" | "edge"
    confidence: float   # 0.0 – 1.0


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def run_kill_feed_parser(clip_path: Path, game: str, config: dict) -> dict:
    """Analyse kill-feed events in a clip and return a result manifest.

    Writes the result into the clip's .meta.json under the 'kill_feed' key
    so the stage is idempotent (re-running skips already-analysed clips).

    Args:
        clip_path: Path to the downloaded .mp4 file.
        game:      Config game key (e.g. 'deadlock').
        config:    Full parsed config.yaml dict.

    Returns:
        {
            'passed':             bool,
            'sweat_score':        float,   # peak 5-second window score
            'kill_count':         int,
            'headshot_count':     int,
            'kill_timestamps':    list[float],
            'headshot_timestamps':list[float],
            'reason':             str,
            'method':             str,     # dominant detection method
        }
    """
    kf_cfg = config.get("kill_feed", {})
    sweat_threshold = float(kf_cfg.get("sweat_threshold", _DEFAULT_SWEAT_THRESHOLD))

    # --- Idempotency: skip if already analysed ---
    meta_path = Path(str(clip_path).replace(".mp4", ".meta.json"))
    if not meta_path.suffix:
        meta_path = clip_path.with_suffix(".meta.json")
    if meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text())
            if "kill_feed" in existing:
                logger.debug(f"[kill_feed] Already analysed: {clip_path.name} — skipping.")
                return existing["kill_feed"]
        except (json.JSONDecodeError, OSError):
            pass

    if not _CV2_AVAILABLE:
        result = _disabled_result("opencv not installed")
        _write_kf_meta(meta_path, result)
        return result

    if not clip_path.exists():
        result = _disabled_result(f"clip not found: {clip_path}")
        return result

    game_cfg = _legacy_game_cfg(game)
    if not game_cfg:
        logger.debug(f"[kill_feed] No kill_feed ROI in game pack for '{game}' — skipping.")
        result = _disabled_result(f"no kill_feed roi in game pack '{game}'")
        _write_kf_meta(meta_path, result)
        return result

    template_dir = Path(kf_cfg.get("template_dir", "assets/kill_feed_templates")) / game
    templates = _load_templates(template_dir)

    # If audio_detector ran first, use its spike timestamps to focus frame sampling.
    audio_spike_timestamps: list[float] = []
    if meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text())
            audio_spike_timestamps = existing.get("audio_events", {}).get("spike_timestamps", [])
        except (json.JSONDecodeError, OSError):
            pass

    if audio_spike_timestamps:
        logger.info(
            f"[kill_feed] Using {len(audio_spike_timestamps)} audio spike(s) as frame hints "
            f"for {clip_path.name}"
        )

    neg_bank = None
    nb_threshold = 0.70
    if config.get("negative_bank", {}).get("enabled", False):
        from pipeline.negative_bank import NegativeBank
        nb_cfg = config["negative_bank"]
        neg_bank = NegativeBank(nb_cfg.get("bank_dir", "data/negative_bank"))
        nb_threshold = float(nb_cfg.get("check_threshold", 0.70))

    logger.info(f"[kill_feed] Analysing {clip_path.name} ({game})...")

    try:
        events = _analyse_clip(
            clip_path, game_cfg, kf_cfg, templates, audio_spike_timestamps,
            game=game, neg_bank=neg_bank, nb_threshold=nb_threshold,
        )
    except Exception as e:
        logger.error(f"[kill_feed] Analysis failed for {clip_path.name}: {e}")
        result = _disabled_result(f"analysis error: {e}")
        _write_kf_meta(meta_path, result)
        return result

    window_seconds = float(kf_cfg.get("window_seconds", _DEFAULT_WINDOW_SECONDS))
    kill_score = float(kf_cfg.get("kill_score", _DEFAULT_KILL_SCORE))
    headshot_score = float(kf_cfg.get("headshot_score", _DEFAULT_HEADSHOT_SCORE))

    peak_score, kill_ts, headshot_ts, dominant_method = _compute_sweat_score(
        events, window_seconds, kill_score, headshot_score
    )

    passed = peak_score >= sweat_threshold
    reason = (
        f"peak window score {peak_score:.1f} >= threshold {sweat_threshold}"
        if passed
        else f"peak window score {peak_score:.1f} < threshold {sweat_threshold}"
    )

    result = {
        "passed": passed,
        "sweat_score": round(peak_score, 2),
        "kill_count": len(kill_ts),
        "headshot_count": len(headshot_ts),
        "kill_timestamps": [round(t, 2) for t in kill_ts],
        "headshot_timestamps": [round(t, 2) for t in headshot_ts],
        "reason": reason,
        "method": dominant_method or "no_events",
    }

    _write_kf_meta(meta_path, result)

    level = logger.info if passed else logger.debug
    level(
        f"[kill_feed] {'PASS' if passed else 'LOW'} {clip_path.name} — "
        f"score={peak_score:.1f}, kills={len(kill_ts)}, headshots={len(headshot_ts)}"
    )

    return result


# ---------------------------------------------------------------------------
# Frame analysis
# ---------------------------------------------------------------------------

def _analyse_clip(
    clip_path: Path,
    game_cfg: dict,
    kf_cfg: dict,
    templates: dict,
    audio_spike_timestamps: list[float] | None = None,
    game: str = "",
    neg_bank: "NegativeBank | None" = None,  # noqa: F821
    nb_threshold: float = 0.70,
) -> list[_Event]:
    """Open the clip, sample frames, and return a list of detected events.

    When audio_spike_timestamps is provided (from audio_detector), only frames
    within ±audio_window_seconds of each spike are inspected. This cuts OpenCV
    work to a fraction of the full clip on long recordings.
    """
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise OSError(f"Could not open {clip_path}")

    _AUDIO_WINDOW = 2.0   # seconds either side of an audio spike to inspect

    try:
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        sample_fps = float(kf_cfg.get("sample_fps", _DEFAULT_SAMPLE_FPS))
        frame_interval = max(1, int(round(native_fps / sample_fps)))

        # Build an O(1) lookup set of frame indices to inspect when audio hints exist.
        audio_frame_set: set[int] | None = None
        if audio_spike_timestamps:
            audio_frame_set = set()
            for spike_t in audio_spike_timestamps:
                start_frame = max(0, int((spike_t - _AUDIO_WINDOW) * native_fps))
                end_frame = int((spike_t + _AUDIO_WINDOW) * native_fps)
                for f in range(start_frame, end_frame + 1, frame_interval):
                    audio_frame_set.add(f - (f % frame_interval))

        roi = game_cfg.get("roi", {"x": 1600, "y": 10, "w": 320, "h": 400})
        rx, ry, rw, rh = roi["x"], roi["y"], roi["w"], roi["h"]

        bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=25, detectShadows=False
        )

        spike_factor = float(kf_cfg.get("color_spike_factor", _DEFAULT_SPIKE_FACTOR))
        min_pixels = int(kf_cfg.get("min_pixel_mass", _DEFAULT_MIN_PIXEL_MASS))

        # Rolling baselines: deque of recent pixel counts (color mask and edge)
        color_baseline: deque[int] = deque(maxlen=_BASELINE_WINDOW)
        edge_baseline: deque[int] = deque(maxlen=_BASELINE_WINDOW)

        events: list[_Event] = []
        frame_idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue

            # When audio hints are available, skip frames outside spike windows.
            if audio_frame_set is not None and frame_idx not in audio_frame_set:
                frame_idx += 1
                continue

            timestamp = frame_idx / native_fps
            frame_idx += 1

            # Normalize to 1920×1080 before cropping
            norm = _normalize_frame(frame)

            if neg_bank is not None:
                is_neg, scene_type, nb_score = neg_bank.check_frame(norm, game, nb_threshold)
                if is_neg:
                    logger.debug(
                        f"[kill_feed] @{timestamp:.1f}s suppressed by bank: "
                        f"{scene_type} ({nb_score:.2f})"
                    )
                    continue

            # Crop to kill-feed ROI
            roi_bgr = norm[ry:ry + rh, rx:rx + rw]
            if roi_bgr.size == 0:
                continue

            event = _detect_event(
                roi_bgr=roi_bgr,
                timestamp=timestamp,
                game_cfg=game_cfg,
                templates=templates,
                bg_sub=bg_sub,
                color_baseline=color_baseline,
                edge_baseline=edge_baseline,
                spike_factor=spike_factor,
                min_pixels=min_pixels,
            )
            if event:
                events.append(event)

        return events
    finally:
        cap.release()


def _detect_event(
    roi_bgr: "np.ndarray",
    timestamp: float,
    game_cfg: dict,
    templates: dict,
    bg_sub: "cv2.BackgroundSubtractor",
    color_baseline: deque,
    edge_baseline: deque,
    spike_factor: float,
    min_pixels: int,
) -> _Event | None:
    """Run the detection chain on a single ROI frame. Returns first signal that fires."""

    # --- Signal 1: Color mask ---
    kill_ranges = game_cfg.get("kill_colors", [])
    headshot_ranges = game_cfg.get("headshot_colors", [])
    pixel_threshold = int(game_cfg.get("pixel_spike_threshold", min_pixels))

    event_kind, pixel_count = _color_spike(
        roi_bgr, kill_ranges, headshot_ranges, color_baseline,
        spike_factor, pixel_threshold
    )

    if event_kind:
        return _Event(timestamp, event_kind, "color_mask", min(1.0, pixel_count / 500.0))

    # Update color baseline with headshot pixel count (proxy for overall activity)
    hs_total = _total_color_pixels(roi_bgr, headshot_ranges)
    kill_total = _total_color_pixels(roi_bgr, kill_ranges)
    color_baseline.append(max(hs_total, kill_total))

    # --- Signal 2: Template match ---
    if templates:
        hit_kind = _template_hit(roi_bgr, templates)
        if hit_kind:
            return _Event(timestamp, hit_kind, "template_match", 0.9)

    # --- Signal 3: MOG2 background subtraction ---
    fg_mask = bg_sub.apply(roi_bgr)
    fg_count = int(np.count_nonzero(fg_mask))
    if len(color_baseline) >= 5:
        baseline_val = float(np.median(list(color_baseline))) or 1.0
        if fg_count > spike_factor * baseline_val and fg_count > pixel_threshold:
            return _Event(timestamp, "kill", "mog2", min(1.0, fg_count / 2000.0))

    # --- Signal 4: Edge-count fallback ---
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_count = int(np.count_nonzero(edges))

    if len(edge_baseline) >= 5:
        edge_base = float(np.median(list(edge_baseline))) or 1.0
        if edge_count > spike_factor * edge_base and edge_count > pixel_threshold * 4:
            edge_baseline.append(edge_count)
            return _Event(timestamp, "kill", "edge", min(1.0, edge_count / 3000.0))

    edge_baseline.append(edge_count)
    return None


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _color_spike(
    roi_bgr: "np.ndarray",
    kill_ranges: list[dict],
    headshot_ranges: list[dict],
    baseline: deque,
    spike_factor: float,
    min_pixels: int,
) -> tuple[str | None, int]:
    """Check for a significant pixel-count spike in kill or headshot color ranges.

    Returns (event_kind, pixel_count) or (None, 0).
    Headshot is checked first (higher score, more specific color).
    """
    base_val = float(np.median(list(baseline))) if len(baseline) >= 5 else 0.0

    # Headshot check first
    hs_count = _total_color_pixels(roi_bgr, headshot_ranges)
    if hs_count >= min_pixels and (base_val < 1.0 or hs_count > spike_factor * base_val):
        return "headshot", hs_count

    # Kill check
    kill_count = _total_color_pixels(roi_bgr, kill_ranges)
    if kill_count >= min_pixels and (base_val < 1.0 or kill_count > spike_factor * base_val):
        return "kill", kill_count

    return None, 0


def _total_color_pixels(roi_bgr: "np.ndarray", color_ranges: list[dict]) -> int:
    """Sum the non-zero pixels across all HSV color ranges for a list of ranges."""
    if not color_ranges:
        return 0
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for cr in color_ranges:
        lower = np.array(cr["lower"], dtype=np.uint8)
        upper = np.array(cr["upper"], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        combined = cv2.bitwise_or(combined, mask)
    return int(np.count_nonzero(combined))


def _template_hit(roi_bgr: "np.ndarray", templates: dict) -> str | None:
    """Run template matching for kill/headshot icons. Returns event kind or None."""
    threshold = _DEFAULT_TEMPLATE_THRESHOLD
    for kind in ("headshot", "kill"):
        for tmpl in templates.get(kind, []):
            if tmpl.shape[0] > roi_bgr.shape[0] or tmpl.shape[1] > roi_bgr.shape[1]:
                continue
            result = cv2.matchTemplate(roi_bgr, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if max_val >= threshold:
                return kind
    return None


# ---------------------------------------------------------------------------
# Sweat score (sliding window)
# ---------------------------------------------------------------------------

def _compute_sweat_score(
    events: list[_Event],
    window_seconds: float,
    kill_score: float,
    headshot_score: float,
) -> tuple[float, list[float], list[float], str | None]:
    """Slide a window over events and return the peak score within any window.

    Returns:
        (peak_score, kill_timestamps, headshot_timestamps, dominant_method)
    """
    if not events:
        return 0.0, [], [], None

    kill_ts = [e.timestamp for e in events if e.kind == "kill"]
    headshot_ts = [e.timestamp for e in events if e.kind == "headshot"]
    all_ts = sorted([e.timestamp for e in events])

    peak = 0.0
    for start in all_ts:
        end = start + window_seconds
        window_score = sum(
            (headshot_score if e.kind == "headshot" else kill_score)
            for e in events
            if start <= e.timestamp < end
        )
        if window_score > peak:
            peak = window_score

    method_counts: dict[str, int] = {}
    for e in events:
        method_counts[e.method] = method_counts.get(e.method, 0) + 1
    dominant_method = max(method_counts, key=method_counts.get) if method_counts else None

    return peak, kill_ts, headshot_ts, dominant_method


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _normalize_frame(frame: "np.ndarray") -> "np.ndarray":
    """Resize frame to TARGET_WIDTH × TARGET_HEIGHT (1920×1080)."""
    h, w = frame.shape[:2]
    if w == TARGET_WIDTH and h == TARGET_HEIGHT:
        return frame
    return cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)


def _load_templates(template_dir: Path) -> dict:
    """Load kill and headshot PNG/JPG templates from the game's template directory.

    Expected structure:
        template_dir/kill_*.png
        template_dir/headshot_*.png

    Returns {'kill': [ndarray, ...], 'headshot': [ndarray, ...]}
    """
    templates: dict[str, list] = {"kill": [], "headshot": []}
    if not template_dir.exists():
        return templates

    for kind in ("kill", "headshot"):
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            for path in sorted(template_dir.glob(f"{kind}_{ext}")):
                img = cv2.imread(str(path))
                if img is not None:
                    templates[kind].append(img)
                    logger.debug(f"[kill_feed] Loaded template: {path.name}")

    if not any(templates.values()):
        logger.debug(f"[kill_feed] No templates found in {template_dir} — template match skipped.")
    return templates


def _write_kf_meta(meta_path: Path, result: dict) -> None:
    """Merge kill_feed result into the existing .meta.json sidecar."""
    try:
        existing = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        existing["kill_feed"] = result
        meta_path.write_text(json.dumps(existing, indent=2))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"[kill_feed] Could not write meta: {meta_path}: {e}")


def _disabled_result(reason: str) -> dict:
    return {
        "passed": True,        # permissive: don't gate if we can't analyse
        "sweat_score": 0.0,
        "kill_count": 0,
        "headshot_count": 0,
        "kill_timestamps": [],
        "headshot_timestamps": [],
        "reason": reason,
        "method": "disabled",
    }


def _legacy_game_cfg(game: str) -> dict | None:
    """Return a dict shaped like the old config.yaml kill_feed.games.<slug> block.

    Synthesised from the game pack so the rest of this module can keep reading
    `game_cfg["roi"]`, `game_cfg["kill_colors"]`, etc. without change.
    """
    try:
        pack = game_pack.load(game)
    except (FileNotFoundError, game_pack.GamePackError) as exc:
        logger.warning(f"[kill_feed] Could not load game pack '{game}': {exc}")
        return None

    roi = pack.hud.get("kill_feed")
    if roi is None:
        return None

    return {
        "roi": roi,
        "kill_colors": pack.hud.colors("kill_feed", "kill"),
        "headshot_colors": pack.hud.colors("kill_feed", "headshot"),
        "pixel_spike_threshold": pack.hud.pixel_spike_threshold("kill_feed") or 50,
    }
