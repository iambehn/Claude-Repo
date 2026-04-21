"""
Weapon Detector — HUD icon recognition via OpenCV matchTemplate

Identifies the active weapon by matching the weapon-icon ROI against a
library of 64×64 PNG reference images stored in:
    assets/weapon_icons/{game}/{weapon_id}.png

Resolution normalization: every frame is resized to 1920×1080 before the
ROI crop, keeping config coordinates stable across stream resolutions.

Config block (config.yaml → weapon_detector):
    enabled: false
    confidence_threshold: 0.80
    frame_sample: "middle"      # "middle" | "kill_timestamps" | "all"
    icon_dir: "assets/weapon_icons"

Per-game weapon/hero tables + ROI live in the game pack:
    assets/games/<slug>/hud.yaml       →  rois.weapon_detector
    assets/games/<slug>/entities.yaml  →  entities with kind: hero|weapon

Building the icon library:
    1. Run a clip through the pipeline with kill_feed.enabled: true so
       kill_timestamps are written to meta.json.
    2. Use frame_sample: "kill_timestamps" to land on action frames.
    3. Manually crop the weapon icon area from a frame and save it as:
           assets/weapon_icons/{game}/{weapon_id}.png
    4. Add the entity to assets/games/<slug>/entities.yaml.
"""

from __future__ import annotations

import json
from pathlib import Path

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
        "OpenCV not installed — Weapon Detector will be skipped.\n"
        "  Fix: pip install opencv-python-headless"
    )

TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080
_DEFAULT_CONFIDENCE = 0.80
_DEFAULT_ICON_DIR = "assets/weapon_icons"


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def run_weapon_detector(clip_path: Path, game: str, config: dict) -> dict:
    """Detect the active weapon from the HUD weapon-icon ROI.

    Idempotent: skips if meta.json already contains a weapon_detection key.

    Returns:
        {
            'weapon_id':    str | None,
            'display_name': str | None,
            'confidence':   float,
            'method':       str,    # "template_match" | "no_match" | "disabled"
            'frame_time':   float,  # seconds into clip where the best frame was taken
        }
    """
    wd_cfg = config.get("weapon_detector", {})
    meta_path = clip_path.with_suffix(".meta.json")

    # Idempotency
    if meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text())
            if "weapon_detection" in existing:
                logger.debug(f"[weapon_detector] Already processed: {clip_path.name}")
                return existing["weapon_detection"]
        except (json.JSONDecodeError, OSError):
            pass

    if not _CV2_AVAILABLE:
        return _write_and_return(meta_path, _disabled("opencv not installed"))

    if not wd_cfg.get("enabled", False):
        return _write_and_return(meta_path, _disabled("weapon_detector disabled"))

    try:
        pack = game_pack.load(game)
    except (FileNotFoundError, game_pack.GamePackError) as exc:
        return _write_and_return(meta_path, _disabled(f"game pack load failed: {exc}"))

    roi = pack.hud.get("weapon_detector")
    if roi is None:
        return _write_and_return(meta_path, _disabled(f"no weapon_detector roi in pack '{game}'"))

    icon_dir = Path(wd_cfg.get("icon_dir", _DEFAULT_ICON_DIR)) / game
    # Back-compat with legacy shape: pass {id: display_name} to _load_templates.
    # Entities of any kind (hero or weapon) map to an icon lookup; weapon_detector
    # only cares that an icon PNG exists per id.
    weapon_names = pack.entities.display_name_map()
    match_mode = wd_cfg.get("match_mode", "color")   # "color" | "grayscale"
    templates = _load_templates(icon_dir, weapon_names, match_mode)

    if not templates:
        logger.debug(f"[weapon_detector] No icons in {icon_dir} — skipping.")
        return _write_and_return(meta_path, _disabled("no weapon icon assets"))
    threshold = float(wd_cfg.get("confidence_threshold", _DEFAULT_CONFIDENCE))
    frame_sample = wd_cfg.get("frame_sample", "middle")

    # Optionally align with kill-feed event timestamps
    kill_timestamps: list[float] = []
    if frame_sample == "kill_timestamps" and meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text())
            kf = existing.get("kill_feed", {})
            kill_timestamps = kf.get("kill_timestamps", []) + kf.get("headshot_timestamps", [])
        except (json.JSONDecodeError, OSError):
            pass

    result = _detect(clip_path, roi, templates, threshold, frame_sample, kill_timestamps, match_mode)
    _write_and_return(meta_path, result)

    if result["weapon_id"]:
        logger.info(
            f"[weapon_detector] {clip_path.name}: '{result['display_name']}' "
            f"(confidence={result['confidence']:.2f}, t={result['frame_time']}s)"
        )
    else:
        logger.debug(f"[weapon_detector] No weapon matched in {clip_path.name}.")

    return result


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _detect(
    clip_path: Path,
    roi: dict,
    templates: list[dict],
    threshold: float,
    frame_sample: str,
    kill_timestamps: list[float],
    match_mode: str = "color",
) -> dict:
    """Open the clip, extract frames, and return the best template match."""
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return _disabled(f"could not open {clip_path.name}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration = total_frames / fps

        if frame_sample == "kill_timestamps" and kill_timestamps:
            sample_times = sorted(set(kill_timestamps))
        elif frame_sample == "all":
            sample_times = list(range(0, max(1, int(duration)), 2))
        else:
            sample_times = [duration / 2]

        rx = roi.get("x", 0)
        ry = roi.get("y", 0)
        rw = roi.get("w", 150)
        rh = roi.get("h", 80)

        best_weapon: dict | None = None
        best_conf = 0.0
        best_time = 0.0

        for t in sample_times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if not ok:
                continue

            h, w = frame.shape[:2]
            if w != TARGET_WIDTH or h != TARGET_HEIGHT:
                frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)

            roi_bgr = frame[ry:ry + rh, rx:rx + rw]
            if roi_bgr.size == 0:
                continue

            # Grayscale mode: reduces sensitivity to background colour shifts
            # caused by semi-transparent or game-world-tinted HUD elements.
            search_frame = (
                cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
                if match_mode == "grayscale"
                else roi_bgr
            )

            for tmpl in templates:
                img = tmpl["image"]
                if img.shape[0] > search_frame.shape[0] or img.shape[1] > search_frame.shape[1]:
                    continue
                res = cv2.matchTemplate(search_frame, img, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                if max_val > best_conf:
                    best_conf = max_val
                    best_weapon = tmpl
                    best_time = t

        if best_weapon and best_conf >= threshold:
            return {
                "weapon_id":    best_weapon["weapon_id"],
                "display_name": best_weapon["display_name"],
                "confidence":   round(best_conf, 3),
                "method":       "template_match",
                "frame_time":   round(best_time, 2),
            }

        return {
            "weapon_id":    None,
            "display_name": None,
            "confidence":   round(best_conf, 3),
            "method":       "no_match",
            "frame_time":   round(best_time, 2),
        }
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_templates(icon_dir: Path, weapon_name_map: dict, match_mode: str = "color") -> list[dict]:
    """Load weapon PNG reference images from the icon directory.

    When match_mode is "grayscale", templates are loaded as single-channel
    images to match the grayscale ROI crop used during detection.
    """
    read_flag = cv2.IMREAD_GRAYSCALE if match_mode == "grayscale" else cv2.IMREAD_COLOR
    templates = []
    if not icon_dir.exists():
        return templates
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        for path in sorted(icon_dir.glob(ext)):
            img = cv2.imread(str(path), read_flag)
            if img is None:
                continue
            weapon_id = path.stem
            display_name = weapon_name_map.get(weapon_id, weapon_id.replace("_", " ").title())
            templates.append({"weapon_id": weapon_id, "display_name": display_name, "image": img})
            logger.debug(f"[weapon_detector] Loaded ({match_mode}): {path.name} → '{display_name}'")
    return templates


def _write_and_return(meta_path: Path, result: dict) -> dict:
    try:
        existing = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        existing["weapon_detection"] = result
        meta_path.write_text(json.dumps(existing, indent=2))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"[weapon_detector] Could not write meta: {e}")
    return result


def _disabled(reason: str) -> dict:
    return {
        "weapon_id":    None,
        "display_name": None,
        "confidence":   0.0,
        "method":       "disabled",
        "frame_time":   0.0,
        "reason":       reason,
    }
