"""
pipeline/training_exporter.py — Window-Level Training Data Exporter

Serializes CandidateWindow objects from proxy_scanner.py into JSONL rows for
window-level ML training. Called during download_candidate_windows() before
each clip segment is fetched, so every candidate window—downloaded or not—gets
recorded with its proxy signals.

Output path: data/training_sets/windows/YYYY-MM-DD.jsonl
Schema:       ml/feature_schema.yaml (window_v1)

This module only writes data. It does not train models or run inference.
Train with: ml/train_fusion.py (deferred — needs 100–500 labeled windows first)
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_EXPORT_DIR = "data/training_sets/windows"
_SCHEMA_VERSION = "window_v1"


def _infer_platform(vod_url: str) -> str:
    if "twitch.tv" in vod_url:
        return "twitch"
    if "youtube.com" in vod_url or "youtu.be" in vod_url:
        return "youtube"
    return "unknown"


def _extract_vod_id(vod_url: str) -> str:
    """Best-effort VOD ID from URL (Twitch /videos/123 or YouTube ?v=abc)."""
    m = re.search(r"/videos/(\d+)", vod_url)
    if m:
        return m.group(1)
    m = re.search(r"[?&]v=([^&]+)", vod_url)
    if m:
        return m.group(1)
    return "unknown"


_UNLOADED = object()  # sentinel distinguishing "not tried" from None (no model found)

_DEFAULT_MODEL_PATH = Path("data/models/window_fusion/model.pkl")


def _load_window_model():
    """Lazily load the window fusion model. Returns None if not found or sklearn missing."""
    try:
        import joblib
    except ImportError:
        return None
    if not _DEFAULT_MODEL_PATH.exists():
        return None
    try:
        model = joblib.load(_DEFAULT_MODEL_PATH)
        logger.debug(f"[training_exporter] Loaded window fusion model → {_DEFAULT_MODEL_PATH}")
        return model
    except Exception as exc:
        logger.warning(f"[training_exporter] Could not load window fusion model: {exc}")
        return None


class TrainingExporter:
    """Appends one JSONL row per candidate window to the daily training file."""

    def __init__(self, export_dir: str = _DEFAULT_EXPORT_DIR) -> None:
        self._export_dir = Path(export_dir)
        self._export_dir.mkdir(parents=True, exist_ok=True)
        self._model = _UNLOADED  # loaded on first export_window() call

    def _get_model(self):
        if self._model is _UNLOADED:
            self._model = _load_window_model()
        return self._model

    def _daily_path(self) -> Path:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self._export_dir / f"{date_str}.jsonl"

    def export_window(
        self,
        window: "CandidateWindow",  # noqa: F821 — imported at call site
        game: str,
        source_meta: dict,
    ) -> dict:
        """Build a training row from a CandidateWindow and append it to JSONL.

        Args:
            window:      CandidateWindow from proxy_scanner.py
            game:        e.g. "marvel_rivals"
            source_meta: dict with keys platform, vod_url, vod_id

        Returns:
            The row dict (for testing / logging — not needed by callers).
        """
        detail = window.signal_detail

        def _max_field(source: str, field: str) -> float:
            vals = [s[field] for s in detail if s.get("source") == source and field in s]
            return max(vals) if vals else 0.0

        def _count(source: str) -> int:
            return sum(1 for s in detail if s.get("source") == source)

        chat_ts = [s["timestamp"] for s in detail
                   if s.get("source") == "chat_spike" and "timestamp" in s]
        audio_ts = [s["timestamp"] for s in detail
                    if s.get("source") == "audio_spike" and "timestamp" in s]
        lag = (min(audio_ts) - min(chat_ts)) if chat_ts and audio_ts else -1.0

        features = {
            "window_duration":            window.end - window.start,
            "proxy_score":                window.proxy_score,
            "signal_count":               float(window.signal_count),
            "unique_signal_sources":      float(len(set(window.signals))),
            "chat_spike_max_strength":    _max_field("chat_spike", "strength"),
            "chat_spike_max_confidence":  _max_field("chat_spike", "confidence"),
            "chat_spike_count":           float(_count("chat_spike")),
            "audio_spike_max_strength":   _max_field("audio_spike", "strength"),
            "audio_spike_max_confidence": _max_field("audio_spike", "confidence"),
            "audio_spike_count":          float(_count("audio_spike")),
            "viewer_clip_count":          float(_count("viewer_clips")),
            "viewer_clip_max_strength":   _max_field("viewer_clips", "strength"),
            "stream_marker_present":      any(
                s.get("source") in ("stream_marker", "youtube_chapter")
                for s in detail
            ),
            "chat_audio_lag_seconds":     lag,
        }

        vod_url = source_meta.get("vod_url", "")
        platform = source_meta.get("platform") or _infer_platform(vod_url)
        vod_id = source_meta.get("vod_id") or _extract_vod_id(vod_url)
        window_id = f"{game}_{vod_id}_t{int(window.start)}"

        # Run window fusion model inference if a trained model is available.
        fusion_score: float | None = None
        disagreement: float | None = None
        model = self._get_model()
        if model is not None:
            try:
                from ml.train_fusion import features_to_vector
                vec = features_to_vector(features)
                proba = model.predict_proba([vec])[0]
                pos_idx = list(model.classes_).index(1) if 1 in model.classes_ else 1
                fusion_score = round(float(proba[pos_idx]), 4)
                disagreement = round(abs(fusion_score - window.proxy_score), 4)
                # Attach to the window so scan_report and callers can read it.
                window.fusion_model_score = fusion_score
            except Exception as exc:
                logger.debug(f"[training_exporter] Inference failed for {window_id}: {exc}")

        row = {
            "schema_version":  _SCHEMA_VERSION,
            "feature_version": _SCHEMA_VERSION,
            "window_id":       window_id,
            "game":            game,
            "source": {
                "platform": platform,
                "vod_url":  vod_url,
                "vod_id":   vod_id,
            },
            "time_window": {
                "start_sec":    window.start,
                "end_sec":      window.end,
                "duration_sec": window.end - window.start,
            },
            "features": features,
            "scores": {
                "heuristic_score":    window.proxy_score,
                "fusion_model_score": fusion_score,
                "model_disagreement": disagreement,
            },
            "human_label": {
                "decision":            None,
                "reason":              None,
                "reviewer_confidence": None,
            },
        }

        out_path = self._daily_path()
        with open(out_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

        score_str = (
            f"heuristic={window.proxy_score:.2f}  model={fusion_score:.2f}"
            if fusion_score is not None
            else f"heuristic={window.proxy_score:.2f}  model=n/a"
        )
        logger.debug(f"[training_exporter] {window_id}  {score_str} → {out_path.name}")
        return row
