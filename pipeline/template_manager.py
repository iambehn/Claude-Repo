"""
pipeline/template_manager.py — self-improving OpenCV template management.

Three commands, meant to be run in sequence:

  refresh      — scan training_images/ → crop screenshots to ROIs → register
                 icon templates in hud.yaml; safe to re-run any time
  calibrate    — sample clips from inbox/accepted/, run each registered template
                 against sampled frames, find score distribution valley, update
                 match_threshold in hud.yaml if it would improve by > 0.05
  audit        — report coverage gaps (missing images, unregistered templates,
                 stale files) and quality issues (low/high hit rates)

Typical workflow:
    python run.py --scrape-images marvel_rivals          # get raw images
    python run.py --refresh-templates marvel_rivals      # crop + register
    python run.py --game marvel_rivals                   # run pipeline
    python run.py --audit-templates marvel_rivals        # see what's missing
    python run.py --calibrate-templates marvel_rivals    # tune thresholds
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import yaml

from pipeline import game_pack

logger = logging.getLogger(__name__)

try:
    import cv2
    import numpy as np
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080
_MIN_SHARPNESS = 80.0       # Laplacian variance; below this = blurry, skip
_MIN_CALIBRATION_FRAMES = 20  # frames needed before calibrating a template
_CALIBRATION_SAMPLE_EVERY = 3  # seconds between sampled frames
_THRESHOLD_UPDATE_MIN_DELTA = 0.03  # only update if suggested differs by this much


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def refresh_templates(game_slug: str, config: dict | None = None) -> dict:
    """
    Scan training_images/ for icon templates and screenshots.
    - Icons (hero_icon, medal, ability): look for normalized PNGs in templates/ subdirs
    - Screenshots (kill_feed, hud_screenshot): crop to each ROI, check sharpness, save crop
    Update hud.yaml templates: list (add new, preserve verified, remove missing).
    """
    pack = game_pack.load(game_slug)
    pack_dir = Path("assets/games") / game_slug
    hud_path = pack_dir / "hud.yaml"

    existing_templates = _load_hud_templates(hud_path)
    existing_by_id = {t["id"]: t for t in existing_templates}

    candidates: list[dict] = []

    # --- Icon templates (already normalized by image_scraper) ---
    template_root = Path(f"assets/games/{game_slug}/templates")
    icon_map = {
        "heroes": ("hero_icon", "hero_portrait"),
        "medals": ("medal", "medal_popup"),
        "abilities": ("ability", "ability_ultimate"),
    }
    for subdir, (category, roi_name) in icon_map.items():
        icon_dir = template_root / subdir
        if not icon_dir.exists():
            continue
        for png in sorted(icon_dir.glob("*.png")):
            entity_id = png.stem.split(".")[0]  # strip .64 suffix if present
            tid = f"{category}.{entity_id}"
            candidates.append({
                "id": tid,
                "image": str(png),
                "roi": roi_name,
                "match_threshold": 0.88,
                "_source": "icon",
            })

    # --- Screenshot crops ---
    if _CV2_AVAILABLE:
        training_root = Path(f"assets/games/{game_slug}/training_images")
        crops_dir = template_root / "crops"
        for screen_category, roi_names in [
            ("kill_feed", ["kill_feed"]),
            ("hud_screenshot", list(pack.hud.rois.keys())),
        ]:
            screen_dir = training_root / screen_category
            if not screen_dir.exists():
                continue
            img_paths = sorted(screen_dir.glob("*.png")) + sorted(screen_dir.glob("*.jpg"))
            for img_path in img_paths:
                frame = cv2.imread(str(img_path))
                if frame is None:
                    continue
                h, w = frame.shape[:2]
                if w != TARGET_WIDTH or h != TARGET_HEIGHT:
                    frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
                for roi_name in roi_names:
                    roi_px = pack.hud.get(roi_name)
                    if roi_px is None:
                        continue
                    rx, ry, rw, rh = roi_px["x"], roi_px["y"], roi_px["w"], roi_px["h"]
                    crop = frame[ry: ry + rh, rx: rx + rw]
                    if crop.size == 0:
                        continue
                    sharpness = _laplacian_variance(crop)
                    if sharpness < _MIN_SHARPNESS:
                        logger.debug(
                            f"[template_mgr] Blurry crop skipped ({roi_name}/{img_path.name},"
                            f" sharpness={sharpness:.1f})"
                        )
                        continue
                    stem = img_path.stem
                    tid = f"{roi_name}.crop.{stem}"
                    out_dir = crops_dir / roi_name
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"{stem}.png"
                    cv2.imwrite(str(out_path), crop)
                    candidates.append({
                        "id": tid,
                        "image": str(out_path),
                        "roi": roi_name,
                        "match_threshold": 0.82,
                        "_source": "screenshot_crop",
                    })
    elif Path(f"assets/games/{game_slug}/training_images").exists():
        logger.warning(
            "[template_mgr] OpenCV not installed — screenshot cropping skipped. "
            "Only icon templates will be registered. "
            "Fix: pip install opencv-python-headless"
        )

    # --- Merge candidates into existing list ---
    added = updated = removed = skipped = 0
    final: list[dict] = []

    seen_ids: set[str] = set()
    for cand in candidates:
        tid = cand["id"]
        seen_ids.add(tid)
        entry = {k: v for k, v in cand.items() if not k.startswith("_")}
        if tid in existing_by_id:
            old = existing_by_id[tid]
            # Preserve user-tuned threshold and qa_status if already verified
            if old.get("qa_status") == "verified":
                entry["match_threshold"] = old.get("match_threshold", entry["match_threshold"])
            entry["qa_status"] = old.get("qa_status", "draft")
            if entry["image"] != old.get("image") or entry["match_threshold"] != old.get("match_threshold"):
                updated += 1
            else:
                skipped += 1
        else:
            entry["qa_status"] = "draft"
            added += 1
        final.append(entry)

    # Keep verified templates even if image is gone (warn), remove drafts with missing files
    for tid, old in existing_by_id.items():
        if tid in seen_ids:
            continue
        img = old.get("image", "")
        if Path(img).exists():
            final.append(old)
            skipped += 1
        elif old.get("qa_status") == "verified":
            final.append(old)
            logger.warning(
                f"[template_mgr] Verified template {tid!r} image file missing: {img}"
            )
        else:
            removed += 1
            logger.info(f"[template_mgr] Removed stale draft template: {tid!r}")

    _write_hud_templates(hud_path, final)
    logger.info(
        f"[template_mgr:{game_slug}] refresh: "
        f"+{added} added  ~{updated} updated  -{removed} removed  ={skipped} unchanged"
    )
    return {"added": added, "updated": updated, "removed": removed, "skipped": skipped,
            "total": len(final)}


def calibrate_thresholds(game_slug: str, config: dict | None = None) -> dict:
    """
    For each registered template, sample frames from processed clips and record
    the full confidence score distribution. Find the natural threshold (valley
    between noise floor and true-match cluster). Update hud.yaml if the suggested
    threshold differs from the current one by more than _THRESHOLD_UPDATE_MIN_DELTA.
    """
    if not _CV2_AVAILABLE:
        logger.error("[template_mgr] OpenCV required for calibration.")
        return {"error": "opencv not installed"}

    cfg = (config or {}).get("roi_matcher", {})
    match_mode = cfg.get("match_mode", "color")

    pack = game_pack.load(game_slug)
    pack_dir = Path("assets/games") / game_slug
    hud_path = pack_dir / "hud.yaml"
    templates = _load_hud_templates(hud_path)
    if not templates:
        return {"error": "no templates registered — run --refresh-templates first"}

    # Collect clips from inbox, accepted, quarantine
    clip_paths: list[Path] = []
    for bucket in ("inbox", "accepted", "quarantine"):
        bucket_dir = Path(config.get("paths", {}).get(bucket, bucket)) / game_slug
        if bucket_dir.exists():
            clip_paths.extend(sorted(bucket_dir.glob("*.mp4")))
    if not clip_paths:
        return {"error": f"no clips found for {game_slug}", "calibrated": 0}

    read_flag = cv2.IMREAD_GRAYSCALE if match_mode == "grayscale" else cv2.IMREAD_COLOR
    loaded = _cv_load_templates(templates, read_flag)
    if not loaded:
        return {"error": "no template images could be loaded from disk"}

    # scores_by_id: {tid: [float, ...]}
    scores_by_id: dict[str, list[float]] = {t["id"]: [] for t in loaded}

    for clip_path in clip_paths[:30]:  # cap at 30 clips to keep calibration fast
        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            continue
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            duration = total_frames / fps
            t = 0.0
            while t < duration:
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ok, frame = cap.read()
                if not ok:
                    break
                h, w = frame.shape[:2]
                if w != TARGET_WIDTH or h != TARGET_HEIGHT:
                    frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
                for tmpl in loaded:
                    roi_px = pack.hud.get(tmpl["roi_name"])
                    if roi_px is None:
                        continue
                    rx, ry, rw, rh = roi_px["x"], roi_px["y"], roi_px["w"], roi_px["h"]
                    roi_region = frame[ry: ry + rh, rx: rx + rw]
                    if roi_region.size == 0:
                        continue
                    search = (
                        cv2.cvtColor(roi_region, cv2.COLOR_BGR2GRAY)
                        if match_mode == "grayscale"
                        else roi_region
                    )
                    img = tmpl["image"]
                    if img.shape[0] > search.shape[0] or img.shape[1] > search.shape[1]:
                        continue
                    res = cv2.matchTemplate(search, img, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(res)
                    scores_by_id[tmpl["id"]].append(float(max_val))
                t += _CALIBRATION_SAMPLE_EVERY
        finally:
            cap.release()

    # Analyze distributions and build suggestions
    calibrated = unchanged = insufficient = 0
    suggestions: dict[str, float] = {}
    for tmpl in templates:
        tid = tmpl["id"]
        scores = scores_by_id.get(tid, [])
        if len(scores) < _MIN_CALIBRATION_FRAMES:
            insufficient += 1
            continue
        suggested = _suggest_threshold(scores)
        current = float(tmpl.get("match_threshold", 0.80))
        suggestions[tid] = suggested
        if abs(suggested - current) >= _THRESHOLD_UPDATE_MIN_DELTA:
            calibrated += 1
        else:
            unchanged += 1

    if calibrated:
        updated_templates = []
        for tmpl in templates:
            entry = dict(tmpl)
            tid = tmpl["id"]
            if tid in suggestions:
                delta = abs(suggestions[tid] - float(tmpl.get("match_threshold", 0.80)))
                if delta >= _THRESHOLD_UPDATE_MIN_DELTA:
                    old_t = entry.get("match_threshold", 0.80)
                    entry["match_threshold"] = round(suggestions[tid], 3)
                    logger.info(
                        f"[template_mgr] {tid}: threshold {old_t:.3f} → {entry['match_threshold']:.3f}"
                    )
            updated_templates.append(entry)
        _write_hud_templates(hud_path, updated_templates)

    logger.info(
        f"[template_mgr:{game_slug}] calibrate: "
        f"{calibrated} updated  {unchanged} unchanged  {insufficient} insufficient data"
    )
    return {
        "calibrated": calibrated,
        "unchanged": unchanged,
        "insufficient_data": insufficient,
        "suggestions": {k: round(v, 3) for k, v in suggestions.items()},
    }


def audit_templates(game_slug: str, config: dict | None = None) -> dict:
    """
    Report coverage gaps and quality issues for a game pack.
    Returns a structured dict with issues by category; also prints a summary.
    """
    pack = game_pack.load(game_slug)
    pack_dir = Path("assets/games") / game_slug
    hud_path = pack_dir / "hud.yaml"
    registered = {t["id"]: t for t in _load_hud_templates(hud_path)}
    training_root = Path(f"assets/games/{game_slug}/training_images")
    template_root = Path(f"assets/games/{game_slug}/templates")

    issues: dict[str, list[str]] = {
        "missing_image": [],      # entity has no downloaded image at all
        "unregistered": [],       # image exists but not in hud.yaml
        "stale_file": [],         # registered but image file deleted
        "low_hit": [],            # registered, zero matches in processed clips
        "high_hit": [],           # matching in > 80% of clips (too permissive)
    }

    # --- Coverage: heroes, medals, abilities ---
    def _check_entity(category: str, entity_id: str) -> None:
        tid = f"{category}.{entity_id}"
        subdir_map = {"hero_icon": "heroes", "medal": "medals", "ability": "abilities"}
        subdir = subdir_map.get(category, category)
        img_dir = training_root / category / entity_id
        tmpl_dir = template_root / subdir
        has_raw = img_dir.exists() and any(img_dir.iterdir())
        has_template = any(tmpl_dir.glob(f"{entity_id}*.png")) if tmpl_dir.exists() else False

        if not has_raw and not has_template:
            issues["missing_image"].append(tid)
        elif (has_raw or has_template) and tid not in registered:
            issues["unregistered"].append(tid)

    for entity in pack.entities.by_kind("hero"):
        _check_entity("hero_icon", entity.id)
    medals_path = pack_dir / "medals.yaml"
    if medals_path.exists():
        raw = yaml.safe_load(medals_path.read_text()) or {}
        for m in raw.get("medals", []):
            if m.get("medal_id"):
                _check_entity("medal", m["medal_id"])
    for ability in pack.abilities.abilities:
        _check_entity("ability", ability.id)

    # --- Stale files ---
    for tid, tmpl in registered.items():
        img = tmpl.get("image", "")
        if img and not Path(img).exists():
            issues["stale_file"].append(tid)

    # --- Hit rates from meta.json files ---
    clip_count, hit_counts = _collect_hit_rates(game_slug, config, set(registered.keys()))
    if clip_count > 0:
        for tid in registered:
            hits = hit_counts.get(tid, 0)
            rate = hits / clip_count
            if rate == 0.0 and tmpl.get("qa_status") != "draft":
                issues["low_hit"].append(f"{tid} (0/{clip_count} clips)")
            elif rate > 0.8:
                issues["high_hit"].append(f"{tid} ({hits}/{clip_count} clips, {rate:.0%})")

    # --- Print report ---
    total_issues = sum(len(v) for v in issues.values())
    print(f"\nTemplate audit — {game_slug}")
    print(f"  {len(registered)} registered  |  {clip_count} clips sampled  |  "
          f"{total_issues} issue(s)\n")

    labels = {
        "missing_image":   ("MISSING IMAGE  ", "run: --scrape-images {g} --scrape-categories {cat}"),
        "unregistered":    ("UNREGISTERED   ", "run: --refresh-templates {g}"),
        "stale_file":      ("STALE FILE     ", "image on disk deleted — re-scrape or remove from hud.yaml"),
        "low_hit":         ("LOW HIT RATE   ", "check threshold, template quality, or ROI coords"),
        "high_hit":        ("HIGH HIT RATE  ", "threshold too low or wrong ROI — run --calibrate-templates"),
    }
    for key, (label, hint) in labels.items():
        items = issues[key]
        if items:
            print(f"  {label} ({len(items)})")
            for item in items[:10]:
                print(f"    • {item}")
            if len(items) > 10:
                print(f"    … and {len(items) - 10} more")
            print(f"    → {hint.format(g=game_slug, cat=key.replace('missing_', ''))}\n")

    if total_issues == 0:
        print("  All templates look healthy.\n")

    return {"game": game_slug, "registered": len(registered),
            "clips_sampled": clip_count, "issues": issues}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _laplacian_variance(img: "np.ndarray") -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _suggest_threshold(scores: list[float]) -> float:
    """
    Given a list of matchTemplate confidence scores, suggest a threshold that
    sits just above the noise floor. Strategy:
    - Sort scores descending
    - Find the largest drop between consecutive sorted values (the "elbow")
    - Threshold = midpoint of the elbow, clamped to [0.70, 0.97]
    Falls back to mean + 2*std if no clear elbow is found.
    """
    arr = sorted(scores, reverse=True)
    if len(arr) < 4:
        return 0.85

    # Find largest gap in top half of scores
    mid = max(1, len(arr) // 2)
    top = arr[:mid]
    if len(top) < 2:
        return 0.85

    gaps = [(top[i] - top[i + 1], i) for i in range(len(top) - 1)]
    max_gap, elbow_i = max(gaps, key=lambda x: x[0])

    if max_gap > 0.05:  # clear separation
        threshold = (top[elbow_i] + top[elbow_i + 1]) / 2
    else:
        # No clear elbow: use mean + 2σ of all scores
        mean = sum(scores) / len(scores)
        var = sum((s - mean) ** 2 for s in scores) / len(scores)
        std = var ** 0.5
        threshold = mean + 2 * std

    return max(0.70, min(0.97, round(threshold, 3)))


def _collect_hit_rates(
    game_slug: str,
    config: dict | None,
    template_ids: set[str],
) -> tuple[int, dict[str, int]]:
    """Scan meta.json files for roi_matches data; return (clip_count, hits_per_template)."""
    cfg = config or {}
    hit_counts: dict[str, int] = {tid: 0 for tid in template_ids}
    clip_count = 0
    for bucket in ("inbox", "accepted", "rejected", "quarantine"):
        bucket_dir = Path(cfg.get("paths", {}).get(bucket, bucket)) / game_slug
        if not bucket_dir.exists():
            continue
        for meta_path in bucket_dir.glob("*.meta.json"):
            try:
                meta = json.loads(meta_path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            roi_data = meta.get("roi_matches", {})
            matches = roi_data.get("matches", [])
            if roi_data.get("method") in ("disabled", "skipped"):
                continue
            clip_count += 1
            for m in matches:
                tid = m.get("id", "")
                if tid in hit_counts:
                    hit_counts[tid] += 1
    return clip_count, hit_counts


def _cv_load_templates(templates: list[dict], read_flag: int) -> list[dict]:
    loaded = []
    for tmpl in templates:
        img_path = Path(tmpl.get("image", ""))
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path), read_flag)
        if img is None:
            continue
        loaded.append({
            "id": tmpl["id"],
            "roi_name": tmpl["roi"],
            "image": img,
            "match_threshold": float(tmpl.get("match_threshold", 0.80)),
        })
    return loaded


def _load_hud_templates(hud_path: Path) -> list[dict]:
    if not hud_path.exists():
        return []
    try:
        raw = yaml.safe_load(hud_path.read_text()) or {}
        return list(raw.get("templates") or [])
    except Exception:
        return []


def _write_hud_templates(hud_path: Path, templates: list[dict]) -> None:
    """Write the templates list back into hud.yaml, preserving all other keys."""
    raw = yaml.safe_load(hud_path.read_text()) or {}
    raw["templates"] = templates
    hud_path.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True))
