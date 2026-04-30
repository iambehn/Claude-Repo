"""
pipeline/roi_matcher.py — Template-matching pipeline stage

Reads the `templates` list from hud.yaml via game_pack, runs
cv2.matchTemplate against the named ROI region for each template,
and writes `roi_matches` to meta.json.

Runs after weapon_detector, before transcription.

Config block (config.yaml → roi_matcher):
    enabled: false
    frame_sample: "middle"    # "middle" | "kill_timestamps" | "all"
    match_mode: "color"       # "color" | "grayscale"
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
        "OpenCV not installed — ROI Matcher will be skipped.\n"
        "  Fix: pip install opencv-python-headless"
    )

TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080
_DEFAULT_THRESHOLD = 0.80


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def run_roi_matcher(clip_path: Path, game: str, config: dict) -> dict:
    """Match hud.yaml templates against their ROI regions; write roi_matches to meta.json.

    Idempotent: skips if meta.json already contains a roi_matches key.

    Returns:
        {
            "matches": [{"id": str, "roi_name": str, "confidence": float, "frame_time": float}],
            "method":  "color" | "grayscale" | "skipped" | "disabled",
        }
    """
    rm_cfg = config.get("roi_matcher", {})
    meta_path = clip_path.with_suffix(".meta.json")

    # Idempotency check
    if meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text())
            if "roi_matches" in existing:
                logger.debug(f"[roi_matcher] Already processed: {clip_path.name}")
                return existing["roi_matches"]
        except (json.JSONDecodeError, OSError):
            pass

    if not _CV2_AVAILABLE:
        return _write_and_return(meta_path, _skipped("opencv not installed"))

    if not rm_cfg.get("enabled", False):
        return _write_and_return(meta_path, _skipped("roi_matcher disabled"))

    try:
        pack = game_pack.load(game)
    except (FileNotFoundError, game_pack.GamePackError) as exc:
        return _write_and_return(meta_path, _skipped(f"game pack load failed: {exc}"))

    templates = list(pack.hud.templates)
    if not templates:
        logger.debug(f"[roi_matcher] No templates in hud.yaml for '{game}' — skipping.")
        return _write_and_return(meta_path, _skipped("no templates in hud.yaml"))

    match_mode = rm_cfg.get("match_mode", "color")
    frame_sample = rm_cfg.get("frame_sample", "middle")

    kill_timestamps: list[float] = []
    if frame_sample == "kill_timestamps" and meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text())
            kf = existing.get("kill_feed", {})
            kill_timestamps = kf.get("kill_timestamps", []) + kf.get("headshot_timestamps", [])
        except (json.JSONDecodeError, OSError):
            pass

    # Load template images (keyed by template id)
    loaded = _load_template_images(templates, match_mode)
    if not loaded:
        return _write_and_return(meta_path, _skipped("no template images could be loaded"))

    neg_bank = None
    nb_threshold = 0.70
    if config.get("negative_bank", {}).get("enabled", False):
        from pipeline.negative_bank import NegativeBank
        nb_cfg = config["negative_bank"]
        neg_bank = NegativeBank(nb_cfg.get("bank_dir", "data/negative_bank"))
        nb_threshold = float(nb_cfg.get("check_threshold", 0.70))

    result = _match(
        clip_path, pack, loaded, frame_sample, kill_timestamps, match_mode,
        game=game, neg_bank=neg_bank, nb_threshold=nb_threshold,
    )
    _write_and_return(meta_path, result)

    n = len(result["matches"])
    if n:
        ids = ", ".join(m["id"] for m in result["matches"])
        logger.info(f"[roi_matcher] {clip_path.name}: {n} match(es): {ids}")
    else:
        logger.debug(f"[roi_matcher] No template matches in {clip_path.name}.")

    return result


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _match(
    clip_path: Path,
    pack: "game_pack.GamePack",
    loaded: list[dict],
    frame_sample: str,
    kill_timestamps: list[float],
    match_mode: str,
    game: str = "",
    neg_bank: "NegativeBank | None" = None,  # noqa: F821
    nb_threshold: float = 0.70,
) -> dict:
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return _skipped(f"could not open {clip_path.name}")

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

        # best_per_id: {template_id: {"id", "roi_name", "confidence", "frame_time"}}
        best_per_id: dict[str, dict] = {}

        for t in sample_times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if not ok:
                continue

            h, w = frame.shape[:2]
            if w != TARGET_WIDTH or h != TARGET_HEIGHT:
                frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)

            if neg_bank is not None:
                is_neg, scene_type, nb_score = neg_bank.check_frame(frame, game, nb_threshold)
                if is_neg:
                    logger.debug(
                        f"[roi_matcher] @{t:.1f}s suppressed by bank: {scene_type} ({nb_score:.2f})"
                    )
                    continue

            for tmpl in loaded:
                roi_name = tmpl["roi_name"]
                roi_px = pack.hud.get(roi_name)
                if roi_px is None:
                    continue

                rx, ry, rw, rh = roi_px["x"], roi_px["y"], roi_px["w"], roi_px["h"]
                roi_bgr = frame[ry: ry + rh, rx: rx + rw]
                if roi_bgr.size == 0:
                    continue

                search = (
                    cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
                    if match_mode == "grayscale"
                    else roi_bgr
                )

                img = tmpl["image"]
                if img.shape[0] > search.shape[0] or img.shape[1] > search.shape[1]:
                    continue

                res = cv2.matchTemplate(search, img, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)

                threshold = tmpl.get("match_threshold", _DEFAULT_THRESHOLD)
                if max_val < threshold:
                    continue

                tid = tmpl["id"]
                if tid not in best_per_id or max_val > best_per_id[tid]["confidence"]:
                    best_per_id[tid] = {
                        "id": tid,
                        "roi_name": roi_name,
                        "confidence": round(max_val, 3),
                        "frame_time": round(t, 2),
                    }

        matches = sorted(best_per_id.values(), key=lambda m: m["confidence"], reverse=True)
        return {"matches": matches, "method": match_mode}
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_template_images(templates: list[dict], match_mode: str) -> list[dict]:
    """Load PNG images for each template entry; skip entries with missing files."""
    read_flag = cv2.IMREAD_GRAYSCALE if match_mode == "grayscale" else cv2.IMREAD_COLOR
    loaded = []
    for tmpl in templates:
        img_path = Path(tmpl.get("image", ""))
        if not img_path.exists():
            logger.warning(f"[roi_matcher] Template image not found: {img_path} — skipping.")
            continue
        img = cv2.imread(str(img_path), read_flag)
        if img is None:
            logger.warning(f"[roi_matcher] Could not load image: {img_path} — skipping.")
            continue
        loaded.append({
            "id": tmpl["id"],
            "roi_name": tmpl["roi"],
            "image": img,
            "match_threshold": float(tmpl.get("match_threshold", _DEFAULT_THRESHOLD)),
        })
        logger.debug(f"[roi_matcher] Loaded ({match_mode}): {img_path.name} → '{tmpl['id']}'")
    return loaded


def _write_and_return(meta_path: Path, result: dict) -> dict:
    try:
        existing = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        existing["roi_matches"] = result
        meta_path.write_text(json.dumps(existing, indent=2))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"[roi_matcher] Could not write meta: {e}")
    return result


def _skipped(reason: str) -> dict:
    return {"matches": [], "method": "disabled", "reason": reason}
