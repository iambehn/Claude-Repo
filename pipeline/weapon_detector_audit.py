"""
Weapon Detector Audit — scan all clip directories and rank weapons that need
better reference icons.

Reads weapon_detection metadata from .meta.json sidecars across every pipeline
stage (inbox, quarantine, processing, accepted) and ranks weapons by detection
frequency and confidence. Exports ROI crops and candidate match crops for the
top_k most common weapons so they can be reviewed and promoted to reference icons.

Usage (run.py):
    python run.py --audit-weapon-detector marvel_rivals

Output:
    assets/games/{game}/reports/weapon_detector/{timestamp}.json
    assets/games/{game}/reports/weapon_detector/{timestamp}_crops/{weapon}_{clip}_roi.png
    assets/games/{game}/reports/weapon_detector/{timestamp}_crops/{weapon}_{clip}_candidate.png

Config block (config.yaml → weapon_detector.audit):
    top_k: 20           # rank this many weapons in the report
    min_confidence: 0.45
    export_crops: true
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline.game_pack import GAMES_ROOT
from pipeline.weapon_detector import TARGET_HEIGHT, TARGET_WIDTH
from utils.logger import get_logger

logger = get_logger(__name__)

_STAGE_KEYS = ("inbox", "quarantine", "processing", "accepted")
_VIDEO_EXTENSIONS = (".mp4", ".mov", ".m4v", ".webm", ".mkv")
_DEFAULT_TOP_K = 20
_DEFAULT_MIN_CONFIDENCE = 0.45

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


def audit_weapon_detector(game: str, config: dict) -> dict[str, Any]:
    wd_cfg = config.get("weapon_detector") or {}
    audit_cfg = wd_cfg.get("audit") or {}
    top_k = max(1, int(audit_cfg.get("top_k", _DEFAULT_TOP_K)))
    min_confidence = float(audit_cfg.get("min_confidence", _DEFAULT_MIN_CONFIDENCE))
    export_crops = bool(audit_cfg.get("export_crops", True))

    detector_cfg = _build_detector_cfg(game, config)
    report_dir = _report_dir(game)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    crop_dir = report_dir / f"{timestamp}_crops"
    if export_crops:
        crop_dir.mkdir(parents=True, exist_ok=True)

    scanned = 0
    audited = 0
    skipped_missing = 0
    skipped_no_detection = 0
    stage_counts: Counter[str] = Counter()
    method_counts: Counter[str] = Counter()
    candidate_items: list[dict[str, Any]] = []
    warnings: list[str] = []

    for stage, meta_path in _iter_sidecars(game, config):
        scanned += 1
        meta = _load_json(meta_path)
        weapon_detection = meta.get("weapon_detection")
        if not isinstance(weapon_detection, dict):
            skipped_no_detection += 1
            continue

        clip_path = _resolve_clip_path(meta_path, meta)
        if clip_path is None or not clip_path.exists():
            skipped_missing += 1
            continue

        audited += 1
        item = _summarize_detection_item(
            stage=stage,
            meta_path=meta_path,
            clip_path=clip_path,
            meta=meta,
            detector_cfg=detector_cfg,
        )
        stage_counts[stage] += 1
        method_counts[item["method"]] += 1
        if item["is_correction_candidate"] and float(item["candidate_confidence"]) >= min_confidence:
            candidate_items.append(item)

    candidate_counts = Counter(
        item["candidate_weapon_id"]
        for item in candidate_items
        if item.get("candidate_weapon_id")
    )
    for item in candidate_items:
        item["candidate_frequency"] = int(
            candidate_counts.get(item.get("candidate_weapon_id"), 0)
        )

    candidate_items.sort(
        key=lambda item: (
            int(item.get("candidate_frequency", 0)),
            float(item.get("candidate_confidence", 0.0)),
            len(item.get("frame_observations") or []),
        ),
        reverse=True,
    )

    ranked = candidate_items[:top_k]
    exported = 0
    if export_crops and not _CV2_AVAILABLE:
        warnings.append("opencv not installed; skipped crop export")
    elif export_crops:
        for index, item in enumerate(ranked, start=1):
            exported_paths = _export_candidate_assets(item, crop_dir, index)
            if exported_paths:
                item["exported_assets"] = exported_paths
                exported += 1

    target_summary = _recommended_targets(candidate_items)
    report = {
        "game": game,
        "generated_at": _now_iso(),
        "status": "ok",
        "scanned_meta_files": scanned,
        "audited_clips": audited,
        "skipped_missing": skipped_missing,
        "skipped_no_detection": skipped_no_detection,
        "top_k": top_k,
        "min_confidence": round(min_confidence, 3),
        "export_crops": export_crops,
        "exported_crop_count": exported,
        "stage_counts": dict(stage_counts),
        "method_counts": dict(method_counts),
        "recommended_targets": target_summary,
        "ranked_candidates": ranked,
        "warnings": warnings,
    }

    report_path = report_dir / f"{timestamp}.json"
    report_path.write_text(json.dumps(report, indent=2))
    logger.info(
        f"[weapon_detector_audit] {game}: scanned={scanned}, audited={audited}, "
        f"candidates={len(candidate_items)}, report={report_path}"
    )

    result = dict(report)
    result["report_path"] = str(report_path)
    result["crop_dir"] = str(crop_dir) if export_crops else None
    return result


def _build_detector_cfg(game: str, config: dict) -> dict[str, Any]:
    cfg = dict(config.get("weapon_detector") or {})
    try:
        from pipeline import game_pack as gp
        pack = gp.load(game)
        roi = pack.hud.get("weapon_detector")
        if roi:
            cfg["roi"] = roi
    except Exception:
        pass
    return cfg


def _report_dir(game: str) -> Path:
    path = GAMES_ROOT / game / "reports" / "weapon_detector"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _iter_sidecars(game: str, config: dict):
    for stage in _STAGE_KEYS:
        root = Path(config["paths"].get(stage, "")) / game
        if not root.exists():
            continue
        for meta_path in sorted(root.rglob("*.meta.json")):
            yield stage, meta_path


def _summarize_detection_item(
    *,
    stage: str,
    meta_path: Path,
    clip_path: Path,
    meta: dict[str, Any],
    detector_cfg: dict[str, Any],
) -> dict[str, Any]:
    weapon_detection = meta.get("weapon_detection") or {}
    top_candidates = list(weapon_detection.get("top_candidates") or [])
    frame_observations = list(weapon_detection.get("frame_observations") or [])
    top_candidate = top_candidates[0] if top_candidates else {}
    best_observation = _best_observation(
        frame_observations, str(top_candidate.get("weapon_id") or "")
    )
    roi = weapon_detection.get("roi") or detector_cfg.get("roi") or {}

    quarantine_reason = (
        (meta.get("quarantine") or {}).get("reason")
    ) or meta.get("quarantine_reason")
    if not quarantine_reason and stage == "quarantine":
        quarantine_reason = _quarantine_reason_from_path(meta_path)

    candidate_confidence = float(top_candidate.get("confidence", 0.0) or 0.0)
    return {
        "clip_id": meta.get("clip_id") or clip_path.stem,
        "clip_stem": clip_path.stem,
        "clip_path": str(clip_path),
        "meta_path": str(meta_path),
        "stage": stage,
        "quarantine_reason": quarantine_reason,
        "method": weapon_detection.get("method", "missing"),
        "matched_weapon_id": weapon_detection.get("weapon_id"),
        "matched_display_name": weapon_detection.get("display_name"),
        "confidence": float(weapon_detection.get("confidence", 0.0) or 0.0),
        "frame_time": float(weapon_detection.get("frame_time", 0.0) or 0.0),
        "roi": roi,
        "best_match_box": weapon_detection.get("best_match_box"),
        "best_match_variant": weapon_detection.get("best_match_variant"),
        "best_match_scale": weapon_detection.get("best_match_scale"),
        "top_candidate": top_candidate or None,
        "top_candidates": top_candidates,
        "frame_observations": frame_observations,
        "candidate_weapon_id": top_candidate.get("weapon_id"),
        "candidate_display_name": top_candidate.get("display_name"),
        "candidate_confidence": candidate_confidence,
        "candidate_timestamp": float(
            best_observation.get("timestamp", weapon_detection.get("frame_time", 0.0))
            or 0.0
        ),
        "candidate_match_box": (
            (best_observation.get("match_box") if best_observation else None)
            or top_candidate.get("match_box")
            or weapon_detection.get("best_match_box")
        ),
        "candidate_match_variant": (
            (best_observation.get("match_variant") if best_observation else None)
            or top_candidate.get("match_variant")
            or weapon_detection.get("best_match_variant")
        ),
        "candidate_match_scale": (
            (best_observation.get("match_scale") if best_observation else None)
            or top_candidate.get("match_scale")
            or weapon_detection.get("best_match_scale")
        ),
        "is_correction_candidate": (
            weapon_detection.get("method") != "template_match" and bool(top_candidate)
        ),
    }


def _best_observation(
    frame_observations: list[dict[str, Any]], weapon_id: str
) -> dict[str, Any]:
    candidates = [
        item for item in frame_observations if item.get("weapon_id") == weapon_id
    ]
    if not candidates:
        return {}
    return max(candidates, key=lambda item: float(item.get("confidence", 0.0) or 0.0))


def _recommended_targets(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for item in items:
        weapon_id = item.get("candidate_weapon_id")
        if not weapon_id:
            continue
        payload = grouped.setdefault(
            weapon_id,
            {
                "weapon_id": weapon_id,
                "display_name": item.get("candidate_display_name")
                    or weapon_id.replace("_", " ").title(),
                "count": 0,
                "confidence_total": 0.0,
                "max_confidence": 0.0,
                "example_clips": [],
            },
        )
        confidence = float(item.get("candidate_confidence", 0.0) or 0.0)
        payload["count"] += 1
        payload["confidence_total"] += confidence
        payload["max_confidence"] = max(float(payload["max_confidence"]), confidence)
        if len(payload["example_clips"]) < 3:
            payload["example_clips"].append(item.get("clip_stem"))

    ranked = sorted(
        grouped.values(),
        key=lambda item: (int(item["count"]), float(item["max_confidence"])),
        reverse=True,
    )
    for item in ranked:
        count = max(1, int(item["count"]))
        item["average_confidence"] = round(float(item["confidence_total"]) / count, 3)
        item["max_confidence"] = round(float(item["max_confidence"]), 3)
        del item["confidence_total"]
    return ranked


def _export_candidate_assets(
    item: dict[str, Any], crop_dir: Path, index: int
) -> dict[str, str]:
    frame = _read_frame(
        Path(item["clip_path"]),
        float(item.get("candidate_timestamp", 0.0) or 0.0),
    )
    if frame is None:
        return {}

    clip_slug = _safe_slug(item.get("clip_stem") or "clip")
    weapon_slug = _safe_slug(item.get("candidate_weapon_id") or "unknown")
    prefix = f"{index:02d}_{weapon_slug}_{clip_slug}"

    roi = item.get("roi") or {}
    roi_crop = _crop_box(frame, roi)
    if roi_crop is None:
        return {}

    exported: dict[str, str] = {}
    roi_path = crop_dir / f"{prefix}_roi.png"
    cv2.imwrite(str(roi_path), roi_crop)
    exported["roi_crop_path"] = str(roi_path)

    match_box = item.get("candidate_match_box")
    candidate_crop = _crop_box(frame, match_box) if match_box else None
    if candidate_crop is not None:
        candidate_path = crop_dir / f"{prefix}_candidate.png"
        cv2.imwrite(str(candidate_path), candidate_crop)
        exported["candidate_crop_path"] = str(candidate_path)

    return exported


def _read_frame(clip_path: Path, timestamp: float):
    if not _CV2_AVAILABLE:
        return None

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        logger.warning(f"[weapon_detector_audit] Could not open clip: {clip_path}")
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(timestamp)) * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        height, width = frame.shape[:2]
        if width != TARGET_WIDTH or height != TARGET_HEIGHT:
            frame = cv2.resize(
                frame, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR
            )
        return frame
    finally:
        cap.release()


def _crop_box(frame, box: dict[str, Any] | None):
    if box is None:
        return None
    try:
        x = max(0, int(box.get("x", 0)))
        y = max(0, int(box.get("y", 0)))
        w = max(1, int(box.get("w", 0)))
        h = max(1, int(box.get("h", 0)))
    except (TypeError, ValueError):
        return None

    crop = frame[y:y + h, x:x + w]
    if crop is None or crop.size == 0:
        return None
    return crop


def _resolve_clip_path(meta_path: Path, meta: dict[str, Any]) -> Path | None:
    raw = meta.get("clip_path")
    if raw:
        clip_path = Path(raw)
        if clip_path.exists():
            return clip_path

    for extension in _VIDEO_EXTENSIONS:
        candidate = meta_path.with_suffix(extension)
        if candidate.exists():
            return candidate
    return None


def _quarantine_reason_from_path(meta_path: Path) -> str | None:
    parts = meta_path.parts
    if "quarantine" not in parts:
        return None
    idx = parts.index("quarantine")
    if len(parts) > idx + 3:
        return parts[idx + 2]
    return None


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _safe_slug(value: str) -> str:
    slug = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in str(value).lower()
    )
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "unknown"


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
