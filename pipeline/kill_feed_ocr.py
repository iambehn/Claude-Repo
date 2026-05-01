"""
pipeline/kill_feed_ocr.py — OCR Confirmation Layer for Kill-Feed Events

Samples frames at kill_timestamps (from kill_feed.py), crops the kill-feed
ROI, preprocesses for readability, and runs OCR to confirm that real text
is present. This cuts the false-positive rate of the pixel-spike and MOG2
detectors by requiring a legible string in the kill-feed region.

Results are written to meta.json["kill_feed_ocr"] and read by
pipeline/atomic_events.py to boost kill event confidence when OCR agrees.

Backends (ordered by accuracy; falls back automatically):
    easyocr   — best accuracy; requires: pip install easyocr
    tesseract — fast; requires: apt install tesseract-ocr + pip install pytesseract

Run order: after kill_feed, before weapon_detector / atomic_events.

Config block (config.yaml → kill_feed_ocr):
    enabled: false
    backend: "easyocr"
    confidence_threshold: 0.50
    kill_keywords:
      - "eliminated"
      - "killed"
      - "defeated"
      - "headshot"
    sample_window_seconds: 1.0    # frames within ±this of each kill_timestamp
    frames_per_kill: 3            # how many frames to sample per kill timestamp
    sample_fps_fallback: 2.0      # sample rate when no kill_timestamps available
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pipeline import game_pack
from utils.logger import get_logger

logger = get_logger(__name__)

try:
    import cv2
    import numpy as np
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

_TARGET_W, _TARGET_H = 1920, 1080
_OCR_MIN_HEIGHT = 150   # upscale ROI to at least this height before OCR
_DEFAULT_CONF_THRESHOLD = 0.50
_DEFAULT_SAMPLE_WINDOW  = 1.0
_DEFAULT_FRAMES_PER_KILL = 3
_DEFAULT_SAMPLE_FPS     = 2.0
_DEFAULT_KILL_KEYWORDS  = ["eliminated", "killed", "defeated", "headshot"]

# Module-level lazy reader (EasyOCR init is expensive — ~5–10s on first call)
_easyocr_reader: Any = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_kill_feed_ocr(clip_path: Path, game: str, config: dict) -> dict:
    """OCR-confirm kill events in the kill-feed ROI and write to meta.json.

    Idempotent: skips if meta.json already contains "kill_feed_ocr".

    Returns:
        {
            ocr_detections:            list of per-frame OCR results
            confirmed_kill_timestamps: kill_timestamps corroborated by OCR
            unconfirmed_kill_timestamps: kill_timestamps with no OCR text
            ocr_confirmation_rate:     confirmed / total (0.0 if no kills)
            method:                    "easyocr" | "tesseract" | "skipped"
            passed:                    True (OCR is supplementary, never blocks)
        }
    """
    meta_path = clip_path.with_suffix(".meta.json")

    # Idempotency
    if meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text())
            if "kill_feed_ocr" in existing:
                logger.debug(f"[kill_feed_ocr] Already processed: {clip_path.name}")
                return existing["kill_feed_ocr"]
        except (json.JSONDecodeError, OSError):
            pass

    ocr_cfg = config.get("kill_feed_ocr", {})

    if not ocr_cfg.get("enabled", False):
        return _write_result(meta_path, _skipped("kill_feed_ocr disabled"))

    if not _CV2_AVAILABLE:
        return _write_result(meta_path, _skipped("opencv not installed"))

    if not clip_path.exists():
        return _write_result(meta_path, _skipped(f"clip not found: {clip_path}"))

    # Load kill-feed ROI from game pack
    try:
        pack = game_pack.load(game)
    except (FileNotFoundError, game_pack.GamePackError) as exc:
        return _write_result(meta_path, _skipped(f"game pack error: {exc}"))

    roi_info = pack.hud.get("kill_feed")
    if roi_info is None:
        return _write_result(meta_path, _skipped(f"no kill_feed ROI in game pack '{game}'"))

    rx = roi_info["x"]; ry = roi_info["y"]
    rw = roi_info["w"]; rh = roi_info["h"]

    # Read kill_timestamps from existing meta (from kill_feed stage)
    kill_timestamps: list[float] = []
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            kf = meta.get("kill_feed", {}) or {}
            kill_timestamps = [float(t) for t in (kf.get("kill_timestamps") or [])]
        except (json.JSONDecodeError, OSError):
            pass

    # Determine which backend to use
    backend = _pick_backend(ocr_cfg.get("backend", "easyocr"))
    if backend is None:
        return _write_result(meta_path, _skipped(
            "no OCR backend available — install easyocr or pytesseract"
        ))

    conf_threshold   = float(ocr_cfg.get("confidence_threshold", _DEFAULT_CONF_THRESHOLD))
    sample_window    = float(ocr_cfg.get("sample_window_seconds", _DEFAULT_SAMPLE_WINDOW))
    frames_per_kill  = int(ocr_cfg.get("frames_per_kill", _DEFAULT_FRAMES_PER_KILL))
    sample_fps       = float(ocr_cfg.get("sample_fps_fallback", _DEFAULT_SAMPLE_FPS))
    kill_keywords    = [str(k).lower() for k in
                        ocr_cfg.get("kill_keywords", _DEFAULT_KILL_KEYWORDS)]

    # Build the set of sample timestamps
    sample_times = _build_sample_times(
        kill_timestamps, sample_window, frames_per_kill, clip_path, sample_fps
    )

    logger.info(
        f"[kill_feed_ocr] {clip_path.name}: backend={backend}  "
        f"sampling {len(sample_times)} frames  kill_ts={len(kill_timestamps)}"
    )

    # Open video and collect OCR detections
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return _write_result(meta_path, _skipped(f"could not open {clip_path.name}"))

    ocr_detections: list[dict] = []
    try:
        for t in sorted(sample_times):
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if not ok:
                continue

            h, w = frame.shape[:2]
            if w != _TARGET_W or h != _TARGET_H:
                frame = cv2.resize(frame, (_TARGET_W, _TARGET_H), interpolation=cv2.INTER_LINEAR)

            roi_bgr = frame[ry: ry + rh, rx: rx + rw]
            if roi_bgr.size == 0:
                continue

            processed = _preprocess(roi_bgr)
            texts = _run_ocr(processed, backend, conf_threshold)
            if not texts:
                continue

            keyword_found = _has_keyword(texts, kill_keywords)
            merged = " ".join(t["text"] for t in texts)
            ocr_detections.append({
                "timestamp":         round(t, 3),
                "texts":             texts,
                "merged_text":       merged,
                "kill_keyword_found": keyword_found,
            })
    finally:
        cap.release()

    # Match OCR detections back to kill_timestamps
    confirmed, unconfirmed = _match_to_kills(
        kill_timestamps, ocr_detections, sample_window
    )

    total = len(kill_timestamps)
    rate = round(len(confirmed) / total, 3) if total > 0 else 0.0

    result = {
        "ocr_detections":              ocr_detections,
        "confirmed_kill_timestamps":   confirmed,
        "unconfirmed_kill_timestamps": unconfirmed,
        "ocr_confirmation_rate":       rate,
        "method":                      backend,
        "passed":                      True,  # OCR is supplementary; never blocks
    }

    logger.info(
        f"[kill_feed_ocr] {clip_path.name}: confirmed={len(confirmed)}/{total}  "
        f"rate={rate:.0%}  detections={len(ocr_detections)}"
    )
    return _write_result(meta_path, result)


# ---------------------------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------------------------

def _build_sample_times(
    kill_timestamps: list[float],
    sample_window: float,
    frames_per_kill: int,
    clip_path: Path,
    fallback_fps: float,
) -> set[float]:
    """Build the set of timestamps to sample for OCR."""
    times: set[float] = set()

    if kill_timestamps:
        for ts in kill_timestamps:
            if frames_per_kill == 1:
                times.add(round(ts, 3))
            else:
                # Distribute frames evenly over [-sample_window/2, +sample_window/2]
                half = sample_window / 2.0
                step = sample_window / max(frames_per_kill - 1, 1)
                for i in range(frames_per_kill):
                    t = round(ts - half + i * step, 3)
                    if t >= 0:
                        times.add(t)
    else:
        # No kill timestamps — sample at fixed rate across full clip
        try:
            cap = cv2.VideoCapture(str(clip_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            duration = total_frames / fps
            cap.release()
        except Exception:
            duration = 30.0

        t = 0.0
        interval = 1.0 / fallback_fps
        while t < duration:
            times.add(round(t, 3))
            t += interval

    return times


# ---------------------------------------------------------------------------
# OCR matching
# ---------------------------------------------------------------------------

def _match_to_kills(
    kill_timestamps: list[float],
    ocr_detections: list[dict],
    window: float,
) -> tuple[list[float], list[float]]:
    """For each kill_timestamp, check whether any OCR detection falls within window."""
    confirmed: list[float] = []
    unconfirmed: list[float] = []
    ocr_times = [d["timestamp"] for d in ocr_detections]

    for ts in kill_timestamps:
        near = [t for t in ocr_times if abs(t - ts) <= window / 2]
        if near:
            confirmed.append(ts)
        else:
            unconfirmed.append(ts)

    return confirmed, unconfirmed


def _has_keyword(texts: list[dict], keywords: list[str]) -> bool:
    combined = " ".join(t["text"].lower() for t in texts)
    return any(kw in combined for kw in keywords)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _preprocess(roi_bgr: "np.ndarray") -> "np.ndarray":
    """Convert ROI to grayscale, upscale, and enhance contrast for OCR."""
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    # Upscale small ROIs so OCR has enough resolution
    h, w = gray.shape
    if h < _OCR_MIN_HEIGHT:
        scale = _OCR_MIN_HEIGHT / h
        gray = cv2.resize(gray, (int(w * scale), _OCR_MIN_HEIGHT),
                          interpolation=cv2.INTER_CUBIC)

    # CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


# ---------------------------------------------------------------------------
# OCR backends
# ---------------------------------------------------------------------------

def _pick_backend(preferred: str) -> str | None:
    """Return the first available OCR backend, preferring the requested one."""
    order = [preferred] + [b for b in ("easyocr", "tesseract") if b != preferred]
    for name in order:
        if name == "easyocr" and _easyocr_available():
            return "easyocr"
        if name == "tesseract" and _tesseract_available():
            return "tesseract"
    return None


def _easyocr_available() -> bool:
    try:
        import easyocr  # noqa: F401
        return True
    except ImportError:
        return False


def _tesseract_available() -> bool:
    try:
        import pytesseract  # noqa: F401
        return True
    except ImportError:
        return False


def _run_ocr(img: "np.ndarray", backend: str, conf_threshold: float) -> list[dict]:
    if backend == "easyocr":
        return _run_easyocr(img, conf_threshold)
    return _run_tesseract(img, conf_threshold)


def _run_easyocr(img: "np.ndarray", conf_threshold: float) -> list[dict]:
    global _easyocr_reader
    try:
        import easyocr
        if _easyocr_reader is None:
            logger.info("[kill_feed_ocr] Initialising EasyOCR reader (first run — may take ~10s)...")
            _easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)

        results = _easyocr_reader.readtext(img)
        texts = []
        for (_, text, conf) in results:
            text = text.strip()
            if text and float(conf) >= conf_threshold:
                texts.append({"text": text, "confidence": round(float(conf), 3)})
        return texts
    except Exception as exc:
        logger.debug(f"[kill_feed_ocr] EasyOCR error: {exc}")
        return []


def _run_tesseract(img: "np.ndarray", conf_threshold: float) -> list[dict]:
    try:
        import pytesseract
        data = pytesseract.image_to_data(
            img,
            output_type=pytesseract.Output.DICT,
            lang="eng",
            config="--psm 6",  # treat as uniform block of text
        )
        texts = []
        for i, word in enumerate(data["text"]):
            word = word.strip()
            raw_conf = data["conf"][i]
            if word and raw_conf >= 0:
                conf = raw_conf / 100.0
                if conf >= conf_threshold:
                    texts.append({"text": word, "confidence": round(conf, 3)})
        return texts
    except Exception as exc:
        logger.debug(f"[kill_feed_ocr] Tesseract error: {exc}")
        return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _skipped(reason: str) -> dict:
    return {
        "ocr_detections": [],
        "confirmed_kill_timestamps": [],
        "unconfirmed_kill_timestamps": [],
        "ocr_confirmation_rate": 0.0,
        "method": "skipped",
        "reason": reason,
        "passed": True,
    }


def _write_result(meta_path: Path, result: dict) -> dict:
    try:
        existing = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    except (json.JSONDecodeError, OSError):
        existing = {}
    existing["kill_feed_ocr"] = result
    meta_path.write_text(json.dumps(existing, indent=2))
    return result
