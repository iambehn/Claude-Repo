"""
pipeline/negative_bank.py — Hard-Negative Scene Bank for OpenCV False-Positive Suppression

Stores labeled 320×180 reference thumbnails for known false-positive scene types
(scoreboard, spectator_ui, loading_screen, replay_screen, team_select, end_screen).

Before running kill_feed or roi_matcher on a frame, detectors call check_frame();
if the frame matches a stored negative above the NCC threshold the frame is skipped.

Secondary use: export_for_training() copies labeled thumbnails into a directory
layout suitable for YOLO negative-class training.

CLI:
    python run.py --add-negative-sample GAME SCENE_TYPE IMAGE_PATH [--neg-notes TEXT]
    python run.py --audit-negative-bank [GAME]

Config block (config.yaml):
    negative_bank:
      enabled: false
      bank_dir: "data/negative_bank"
      check_threshold: 0.70
"""
from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_BANK_DIR = "data/negative_bank"
_THUMB_W, _THUMB_H = 320, 180
_DEFAULT_THRESHOLD = 0.70

SCENE_TYPES: tuple[str, ...] = (
    "scoreboard",
    "spectator_ui",
    "loading_screen",
    "replay_screen",
    "team_select",
    "end_screen",
)

try:
    import cv2
    import numpy as np
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class NegativeSample:
    sample_id: str       # SHA256[:12] of original file bytes
    game: str
    scene_type: str
    thumb_path: Path     # absolute path to 320×180 PNG thumbnail
    added_at: str        # ISO timestamp
    notes: str


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class NegativeBank:
    """Stores and queries hard-negative scene thumbnails for detector pre-screening."""

    def __init__(self, bank_dir: str = _DEFAULT_BANK_DIR) -> None:
        self._bank_dir = Path(bank_dir)
        self._bank_dir.mkdir(parents=True, exist_ok=True)
        # {game: [(scene_type, np.ndarray)]} — populated lazily on first check_frame() call
        self._cache: dict[str, list[tuple[str, Any]]] = {}
        self._loaded_games: set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_sample(
        self,
        game: str,
        scene_type: str,
        source_path: str | Path,
        notes: str = "",
    ) -> NegativeSample:
        """Add an image to the bank as a hard-negative sample.

        Validates the image, resizes to 320×180, saves as PNG thumbnail, and
        updates manifest.yaml. Returns silently if the SHA already exists.

        Raises:
            ValueError: If the file is not a valid image or scene_type is unknown.
            FileNotFoundError: If source_path does not exist.
            RuntimeError: If OpenCV is not installed.
        """
        if scene_type not in SCENE_TYPES:
            raise ValueError(
                f"Unknown scene_type '{scene_type}'. Valid: {', '.join(SCENE_TYPES)}"
            )

        src = Path(source_path)
        if not src.exists():
            raise FileNotFoundError(f"Image not found: {src}")

        raw = src.read_bytes()
        _validate_image_bytes(raw)

        sha = hashlib.sha256(raw).hexdigest()[:12]

        manifest = self._load_manifest(game)
        for entry in manifest.get("samples", []):
            if entry.get("sample_id") == sha:
                logger.debug(
                    f"[negative_bank] Duplicate SHA {sha} for {game}/{scene_type} — skipped."
                )
                return NegativeSample(
                    sample_id=sha,
                    game=entry["game"],
                    scene_type=entry["scene_type"],
                    thumb_path=self._bank_dir / entry["thumb_path"],
                    added_at=entry["added_at"],
                    notes=entry.get("notes", ""),
                )

        if not _CV2_AVAILABLE:
            raise RuntimeError("OpenCV not installed — required for thumbnail generation.")

        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"cv2 could not decode image: {src}")
        thumb = cv2.resize(img, (_THUMB_W, _THUMB_H), interpolation=cv2.INTER_AREA)

        scene_dir = self._bank_dir / game / scene_type
        scene_dir.mkdir(parents=True, exist_ok=True)
        thumb_rel = Path(game) / scene_type / f"{sha}.png"
        thumb_abs = self._bank_dir / thumb_rel
        cv2.imwrite(str(thumb_abs), thumb)

        added_at = datetime.now(timezone.utc).isoformat()
        entry = {
            "sample_id": sha,
            "game": game,
            "scene_type": scene_type,
            "thumb_path": str(thumb_rel),
            "added_at": added_at,
            "notes": notes,
        }
        manifest.setdefault("samples", []).append(entry)
        self._save_manifest(game, manifest)

        # Invalidate thumbnail cache for this game
        self._cache.pop(game, None)
        self._loaded_games.discard(game)

        sample = NegativeSample(
            sample_id=sha,
            game=game,
            scene_type=scene_type,
            thumb_path=thumb_abs,
            added_at=added_at,
            notes=notes,
        )
        logger.info(f"[negative_bank] Added {game}/{scene_type}/{sha}")
        return sample

    def check_frame(
        self,
        frame_bgr: "np.ndarray",
        game: str,
        threshold: float = _DEFAULT_THRESHOLD,
    ) -> tuple[bool, str | None, float]:
        """Check whether a frame matches any stored negative sample.

        Resizes frame to 320×180 and runs TM_CCOEFF_NORMED against each stored
        thumbnail.  When template size == source size matchTemplate returns a 1×1
        result — the normalized cross-correlation of the whole image.

        Returns:
            (is_negative, scene_type, best_score)
        """
        if not _CV2_AVAILABLE:
            return False, None, 0.0

        samples = self._get_cache(game)
        if not samples:
            return False, None, 0.0

        small = cv2.resize(frame_bgr, (_THUMB_W, _THUMB_H), interpolation=cv2.INTER_AREA)
        best_score = 0.0
        best_scene: str | None = None

        for scene_type, thumb in samples:
            res = cv2.matchTemplate(small, thumb, cv2.TM_CCOEFF_NORMED)
            score = float(res[0][0])
            if score > best_score:
                best_score = score
                best_scene = scene_type
            if best_score >= threshold:
                break

        return best_score >= threshold, best_scene, round(best_score, 4)

    def list_samples(self, game: str | None = None) -> list[NegativeSample]:
        """Return all samples, optionally filtered to a single game."""
        games = [game] if game else self._all_game_dirs()
        out: list[NegativeSample] = []
        for g in games:
            manifest = self._load_manifest(g)
            for entry in manifest.get("samples", []):
                out.append(NegativeSample(
                    sample_id=entry["sample_id"],
                    game=entry["game"],
                    scene_type=entry["scene_type"],
                    thumb_path=self._bank_dir / entry["thumb_path"],
                    added_at=entry["added_at"],
                    notes=entry.get("notes", ""),
                ))
        return out

    def audit(self, game: str | None = None) -> dict:
        """Return {game: {scene_type: count}} coverage dict."""
        games = [game] if game else self._all_game_dirs()
        result: dict[str, dict[str, int]] = {}
        for g in games:
            counts: dict[str, int] = {st: 0 for st in SCENE_TYPES}
            manifest = self._load_manifest(g)
            for entry in manifest.get("samples", []):
                st = entry.get("scene_type", "unknown")
                counts[st] = counts.get(st, 0) + 1
            result[g] = counts
        return result

    def export_for_training(self, game: str, output_dir: str | Path) -> int:
        """Copy thumbnails to output_dir/{scene_type}/ for YOLO negative-class training.

        Returns the number of files exported.
        """
        out = Path(output_dir)
        count = 0
        for sample in self.list_samples(game):
            if not sample.thumb_path.exists():
                continue
            dest_dir = out / sample.scene_type
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sample.thumb_path, dest_dir / sample.thumb_path.name)
            count += 1
        logger.info(f"[negative_bank] Exported {count} thumbnails for {game} → {out}")
        return count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_cache(self, game: str) -> list[tuple[str, Any]]:
        if game in self._loaded_games:
            return self._cache.get(game, [])

        if not _CV2_AVAILABLE:
            return []

        manifest = self._load_manifest(game)
        loaded: list[tuple[str, Any]] = []
        for entry in manifest.get("samples", []):
            thumb_path = self._bank_dir / entry["thumb_path"]
            if not thumb_path.exists():
                logger.warning(f"[negative_bank] Missing thumbnail: {thumb_path}")
                continue
            img = cv2.imread(str(thumb_path))
            if img is None:
                continue
            if img.shape[:2] != (_THUMB_H, _THUMB_W):
                img = cv2.resize(img, (_THUMB_W, _THUMB_H), interpolation=cv2.INTER_AREA)
            loaded.append((entry["scene_type"], img))

        self._cache[game] = loaded
        self._loaded_games.add(game)
        logger.debug(f"[negative_bank] Cached {len(loaded)} thumbnails for '{game}'")
        return loaded

    def _load_manifest(self, game: str) -> dict:
        path = self._bank_dir / game / "manifest.yaml"
        if not path.exists():
            return {"samples": []}
        try:
            with open(path, encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {"samples": []}
        except Exception as exc:
            logger.warning(f"[negative_bank] Could not read manifest for {game}: {exc}")
            return {"samples": []}

    def _save_manifest(self, game: str, manifest: dict) -> None:
        path = self._bank_dir / game / "manifest.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(manifest, fh, default_flow_style=False, sort_keys=False)

    def _all_game_dirs(self) -> list[str]:
        if not self._bank_dir.exists():
            return []
        return [
            d.name for d in sorted(self._bank_dir.iterdir())
            if d.is_dir() and (d / "manifest.yaml").exists()
        ]


# ---------------------------------------------------------------------------
# Image validation
# ---------------------------------------------------------------------------

def _validate_image_bytes(raw: bytes) -> None:
    """Raise ValueError if raw bytes are not a recognized image format."""
    if raw[:4] == b"\x89PNG":
        return
    if raw[:3] == b"\xff\xd8\xff":
        return
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return
    raise ValueError("File is not a recognized image (expected PNG, JPEG, or WebP).")


# ---------------------------------------------------------------------------
# Audit printer
# ---------------------------------------------------------------------------

def print_audit(coverage: dict) -> None:
    """Print a per-game coverage table from audit() output."""
    sep = "  " + "─" * 42

    for game, counts in sorted(coverage.items()):
        print()
        print("  ╔══════════════════════════════════════════╗")
        print(f"  ║  Negative Bank Audit  [{game}]")
        print("  ╚══════════════════════════════════════════╝")
        print()
        print(f"  {'Scene Type':<22} {'Samples':>7}")
        print(sep)
        gaps = []
        total = 0
        for scene_type in SCENE_TYPES:
            n = counts.get(scene_type, 0)
            total += n
            flag = "   ⚠ no samples" if n == 0 else ""
            print(f"  {scene_type:<22} {n:>7}{flag}")
            if n == 0:
                gaps.append(scene_type)
        print(sep)
        print(f"  {'Total':<22} {total:>7}")
        print()
        if gaps:
            print(f"  Coverage gaps: {', '.join(gaps)}")
            print(f"  Add:  python run.py --add-negative-sample {game} SCENE_TYPE IMAGE_PATH")
        else:
            print("  Full coverage across all scene types.")
        print()
