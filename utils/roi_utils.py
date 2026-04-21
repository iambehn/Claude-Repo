"""
utils/roi_utils.py — Shared ROI snipping logic for CLI and Flask UI.

Used by:
  tools/snip_roi.py           — CLI extraction tool
  pipeline/review/app.py      — POST /api/save_roi_template endpoint
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from pipeline import game_pack
from utils.logger import get_logger

logger = get_logger(__name__)

TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080

try:
    import cv2
    import numpy as np
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


def extract_frame(clip_path: Path, timestamp: float) -> "np.ndarray":
    """Extract a single frame from clip_path at timestamp seconds, normalized to 1920×1080."""
    if not _CV2_AVAILABLE:
        raise RuntimeError("OpenCV not installed — run: pip install opencv-python-headless")

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open clip: {clip_path}")

    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read frame at t={timestamp:.2f}s from {clip_path.name}")
        h, w = frame.shape[:2]
        if w != TARGET_WIDTH or h != TARGET_HEIGHT:
            frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)
        return frame
    finally:
        cap.release()


def save_template(
    game: str,
    clip_path: Path,
    frame_time: float,
    crop: dict[str, int],
    template_id: str,
    roi_name: str,
    match_threshold: float,
    config: dict[str, Any],
) -> Path:
    """Crop frame, save PNG + provenance sidecar, append entry to hud.yaml templates list.

    Args:
        game:             game slug (e.g. "marvel_rivals")
        clip_path:        source clip Path (used for provenance only if already extracted)
        frame_time:       timestamp in seconds where the frame was taken
        crop:             {x, y, w, h} pixel coords at 1920×1080
        template_id:      snake_case identifier (used as PNG stem)
        roi_name:         name of the ROI in hud.yaml (e.g. "kill_feed")
        match_threshold:  confidence threshold for this template (0.0–1.0)
        config:           top-level config dict (for icon_dir and ui_version lookups)

    Returns:
        Path to the saved PNG.
    """
    if not _CV2_AVAILABLE:
        raise RuntimeError("OpenCV not installed — run: pip install opencv-python-headless")

    pack = game_pack.load(game)
    roi_pixel = pack.hud.get(roi_name)
    if roi_pixel is None:
        raise ValueError(f"ROI '{roi_name}' not found in hud.yaml for game '{game}'")

    frame = extract_frame(clip_path, frame_time)

    x, y, w, h = crop["x"], crop["y"], crop["w"], crop["h"]
    icon = frame[y: y + h, x: x + w]
    if icon.size == 0:
        raise ValueError(f"Crop region is empty: x={x}, y={y}, w={w}, h={h}")

    roi_dir_key = config.get("weapon_detector", {}).get("icon_dir", "assets/weapon_icons")
    out_dir = Path("assets/roi_library") / game
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{template_id}.png"

    # Overwrite if template_id already exists (idempotent update).
    cv2.imwrite(str(out_path), icon)
    logger.info(f"[roi_utils] Saved template: {out_path} ({icon.shape[1]}×{icon.shape[0]} px)")

    _write_provenance(
        out_dir=out_dir,
        template_id=template_id,
        game=game,
        roi_name=roi_name,
        ui_version=pack.info.ui_version,
        match_threshold=match_threshold,
        source_clip_stem=clip_path.stem,
        frame_time=frame_time,
    )

    _upsert_hud_template(game, template_id, roi_name, out_path, match_threshold)

    game_pack.clear_cache()
    return out_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_provenance(
    out_dir: Path,
    template_id: str,
    game: str,
    roi_name: str,
    ui_version: str,
    match_threshold: float,
    source_clip_stem: str,
    frame_time: float,
) -> None:
    meta = {
        "id": template_id,
        "game_id": game,
        "roi_name": roi_name,
        "ui_version": ui_version,
        "match_threshold": match_threshold,
        "source_clip_id": source_clip_stem,
        "frame_time_seconds": round(frame_time, 3),
        "added_at": datetime.now(timezone.utc).isoformat(),
    }
    sidecar = out_dir / f"{template_id}.meta.yaml"
    with sidecar.open("w") as fh:
        yaml.dump(meta, fh, default_flow_style=False, sort_keys=False)
    logger.debug(f"[roi_utils] Wrote provenance sidecar: {sidecar}")


def _upsert_hud_template(
    game: str,
    template_id: str,
    roi_name: str,
    png_path: Path,
    match_threshold: float,
) -> None:
    """Append or update a template entry in assets/games/{game}/hud.yaml."""
    hud_path = Path("assets/games") / game / "hud.yaml"
    with hud_path.open("r") as fh:
        data = yaml.safe_load(fh) or {}

    templates: list[dict] = data.get("templates") or []

    new_entry = {
        "id": template_id,
        "roi": roi_name,
        "image": str(png_path).replace("\\", "/"),
        "match_threshold": match_threshold,
    }

    # Replace existing entry with same id; otherwise append.
    replaced = False
    for i, tmpl in enumerate(templates):
        if tmpl.get("id") == template_id:
            templates[i] = new_entry
            replaced = True
            break
    if not replaced:
        templates.append(new_entry)

    data["templates"] = templates
    with hud_path.open("w") as fh:
        yaml.dump(data, fh, default_flow_style=False, sort_keys=False)
    action = "updated" if replaced else "appended"
    logger.info(f"[roi_utils] {action} template '{template_id}' in {hud_path}")
