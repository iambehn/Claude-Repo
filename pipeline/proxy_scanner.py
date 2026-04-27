"""
pipeline/proxy_scanner.py — Proxy Signal Detection for VOD Mining

Instead of downloading and scanning a full VOD frame-by-frame, this module
uses cheap proxy signals to identify candidate time windows, then downloads
only those short segments for full pipeline analysis.

Funnel (cheapest to most expensive):
  1. Twitch chat velocity  → normalized spike detection on IRC chat log
  2. Twitch viewer clips   → timestamps where viewers already clipped
  3. Audio spike detection → FFmpeg RMS scan, find loud/active windows
  4. (future) stream markers, YT chapter timestamps

Usage from run.py:
    python run.py --scan-vod https://www.twitch.tv/videos/12345 marvel_rivals
    python run.py --scan-vod https://www.twitch.tv/videos/12345 marvel_rivals --chat-log chat.log

Each candidate window that passes the proxy score threshold is downloaded
as a short MP4 into inbox/{game}/ with a .meta.json sidecar, then the
normal pipeline processes it.
"""

from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import requests
import yt_dlp
import yaml

from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ProxySignal:
    source: str           # "viewer_clips" | "audio_spike" | "stream_marker"
    timestamp: float      # seconds from VOD start
    strength: float       # raw signal magnitude (0.0–1.0)
    confidence: float     # how reliable this source is (0.0–1.0)
    reason: str           # human-readable description


@dataclass
class CandidateWindow:
    start: float
    end: float
    proxy_score: float
    signals: list[str]    # which sources contributed
    signal_count: int
    signal_detail: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Signal weights (overridden by config proxy_scanner.signals.*.weight)
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS = {
    "chat_spike":     3.5,
    "viewer_clips":   5.0,
    "audio_spike":    2.0,
    "stream_marker":  5.0,
}

_DEFAULT_CONFIDENCES = {
    "viewer_clips":   0.90,
    "audio_spike":    0.60,
    "stream_marker":  0.95,
    "chat_spike":     0.70,
}


# ---------------------------------------------------------------------------
# Per-game config loading
# ---------------------------------------------------------------------------


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge override into base in-place (override wins on conflicts)."""
    for key, val in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val


def _load_proxy_config(game: str, global_config: dict) -> dict:
    """Load assets/games/{game}/proxy.yaml and deep-merge over global config defaults."""
    game_path = Path(f"assets/games/{game}/proxy.yaml")
    base = copy.deepcopy(global_config.get("proxy_scanner", {}))
    if game_path.exists():
        game_data = yaml.safe_load(game_path.read_text(encoding="utf-8")) or {}
        game_ps = game_data.get("proxy_scanner", {})
        _deep_merge(base, game_ps)
    return base


# ---------------------------------------------------------------------------
# Signal deduplication
# ---------------------------------------------------------------------------


def _dedupe_signals(signals: list[ProxySignal], min_gap: float = 3.0) -> list[ProxySignal]:
    """Drop same-source signals that arrive within min_gap seconds of each other.

    Prevents a burst of identical-source events (e.g., 10 consecutive chat
    buckets all flagged) from artificially inflating window scores.
    """
    signals = sorted(signals, key=lambda s: s.timestamp)
    deduped: list[ProxySignal] = []
    last_by_source: dict[str, ProxySignal] = {}
    for sig in signals:
        last = last_by_source.get(sig.source)
        if last and abs(sig.timestamp - last.timestamp) < min_gap:
            continue
        deduped.append(sig)
        last_by_source[sig.source] = sig
    return deduped


# ---------------------------------------------------------------------------
# Twitch helpers (reuses credentials from config)
# ---------------------------------------------------------------------------


def _get_twitch_token(config: dict) -> Optional[str]:
    client_id = os.environ.get("TWITCH_CLIENT_ID")
    client_secret = os.environ.get("TWITCH_CLIENT_SECRET")
    if not client_id or not client_secret:
        logger.warning("[proxy] TWITCH_CLIENT_ID or TWITCH_CLIENT_SECRET not set — skipping Twitch signals")
        return None
    try:
        resp = requests.post(
            "https://id.twitch.tv/oauth2/token",
            params={
                "client_id": client_id,
                "client_secret": client_secret,
                "grant_type": "client_credentials",
            },
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()["access_token"]
    except Exception as e:
        logger.warning(f"[proxy] Twitch token fetch failed: {e}")
        return None


def _extract_twitch_video_id(url: str) -> Optional[str]:
    """Extract numeric video ID from a Twitch VOD URL."""
    m = re.search(r"twitch\.tv/videos/(\d+)", url)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Signal: Twitch viewer clips
# ---------------------------------------------------------------------------


def _fetch_viewer_clips(
    video_id: str,
    config: dict,
    weight: float,
) -> list[ProxySignal]:
    """
    Fetch all clips created from this VOD via GET /helix/clips?video_id=.

    `vod_offset` on each clip is the exact second within the VOD where the
    clip starts. Clusters of clips near the same timestamp are strong signals.
    """
    token = _get_twitch_token(config)
    if not token:
        return []

    client_id = os.environ.get("TWITCH_CLIENT_ID", "")
    signals: list[ProxySignal] = []
    cursor = None

    try:
        while True:
            params: dict = {"video_id": video_id, "first": 100}
            if cursor:
                params["after"] = cursor

            resp = requests.get(
                "https://api.twitch.tv/helix/clips",
                params=params,
                headers={"Client-Id": client_id, "Authorization": f"Bearer {token}"},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            for clip in data.get("data", []):
                offset = clip.get("vod_offset")
                if offset is None:
                    continue
                view_count = int(clip.get("view_count", 1))
                # Normalise view count into strength (cap at 10k views)
                strength = min(1.0, view_count / 10_000)
                signals.append(ProxySignal(
                    source="viewer_clips",
                    timestamp=float(offset),
                    strength=max(0.1, strength),
                    confidence=_DEFAULT_CONFIDENCES["viewer_clips"],
                    reason=f"viewer clip '{clip.get('title', '')}' ({view_count} views)",
                ))

            pagination = data.get("pagination", {})
            cursor = pagination.get("cursor")
            if not cursor:
                break

    except Exception as e:
        logger.warning(f"[proxy] Twitch clips fetch failed: {e}")

    logger.info(f"[proxy] viewer_clips: {len(signals)} clips found for VOD {video_id}")
    return signals


# ---------------------------------------------------------------------------
# Signal: audio spike detection
# ---------------------------------------------------------------------------


def _detect_audio_spikes(
    vod_url: str,
    config: dict,
    weight: float,
    z_threshold: float = 3.5,
) -> list[ProxySignal]:
    """
    Download only the audio track of the VOD, run FFmpeg astats to extract
    per-second RMS levels, then flag windows above Z-score threshold.

    Returns ProxySignals at the centre of each spike window.
    """
    import numpy as np

    signals: list[ProxySignal] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = Path(tmpdir) / "audio.m4a"

        # Download audio-only (much faster than full video)
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": str(audio_path.with_suffix("")),
            "quiet": True,
            "no_warnings": True,
            "postprocessors": [],
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([vod_url])
        except Exception as e:
            logger.warning(f"[proxy] Audio download failed: {e}")
            return signals

        # Find the downloaded file (extension may vary)
        audio_files = list(Path(tmpdir).glob("audio.*"))
        if not audio_files:
            logger.warning("[proxy] No audio file found after download")
            return signals
        audio_file = audio_files[0]

        # Extract per-second RMS with FFmpeg astats
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-i", str(audio_file),
                    "-af", "astats=metadata=1:reset=1,"
                           "ametadata=print:key=lavfi.astats.Overall.RMS_level:file=-",
                    "-f", "null", "-",
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )
            rms_output = result.stderr + result.stdout
        except Exception as e:
            logger.warning(f"[proxy] FFmpeg astats failed: {e}")
            return signals

        # Parse: lines like "pts_time:14.2\nlavfi.astats.Overall.RMS_level=-24.3"
        timestamps: list[float] = []
        rms_values: list[float] = []
        current_time: Optional[float] = None

        for line in rms_output.splitlines():
            line = line.strip()
            if line.startswith("pts_time:"):
                try:
                    current_time = float(line.split(":")[1])
                except ValueError:
                    pass
            elif "RMS_level=" in line and current_time is not None:
                try:
                    val = float(line.split("=")[1])
                    if val > -120:  # -inf dB means silence
                        timestamps.append(current_time)
                        rms_values.append(val)
                        current_time = None
                except ValueError:
                    pass

        if len(rms_values) < 10:
            logger.warning("[proxy] Not enough RMS data points for spike detection")
            return signals

        arr = np.array(rms_values)
        mean, std = arr.mean(), arr.std()
        if std == 0:
            return signals

        z_scores = (arr - mean) / std

        # Group consecutive above-threshold frames into spike windows
        in_spike = False
        spike_start = 0.0
        spike_frames: list[float] = []

        for i, z in enumerate(z_scores):
            if z >= z_threshold:
                if not in_spike:
                    in_spike = True
                    spike_start = timestamps[i]
                    spike_frames = []
                spike_frames.append(z)
            else:
                if in_spike:
                    # Emit signal at the centre of the spike
                    centre = spike_start + (timestamps[i - 1] - spike_start) / 2
                    peak_z = max(spike_frames)
                    strength = min(1.0, (peak_z - z_threshold) / z_threshold)
                    signals.append(ProxySignal(
                        source="audio_spike",
                        timestamp=centre,
                        strength=max(0.1, strength),
                        confidence=_DEFAULT_CONFIDENCES["audio_spike"],
                        reason=f"audio spike z={peak_z:.2f} at {centre:.1f}s",
                    ))
                    in_spike = False

    logger.info(f"[proxy] audio_spikes: {len(signals)} spikes detected")
    return signals


# ---------------------------------------------------------------------------
# Window merging and scoring
# ---------------------------------------------------------------------------


def _merge_signals_into_windows(
    signals: list[ProxySignal],
    weights: dict[str, float],
    merge_gap: float,
    pre_pad: float,
    post_pad: float,
    min_score: float,
    max_windows: int,
) -> list[CandidateWindow]:
    """
    Cluster signals that fall within merge_gap seconds of each other into
    candidate windows. Score each window by weighted signal sum + agreement bonus.
    """
    if not signals:
        return []

    signals = sorted(signals, key=lambda s: s.timestamp)

    # Step 1: cluster into groups
    groups: list[list[ProxySignal]] = []
    current: list[ProxySignal] = [signals[0]]

    for sig in signals[1:]:
        if sig.timestamp - current[-1].timestamp <= merge_gap:
            current.append(sig)
        else:
            groups.append(current)
            current = [sig]
    groups.append(current)

    # Step 2: score each group
    windows: list[CandidateWindow] = []

    for group in groups:
        sources_seen: set[str] = set()
        raw_score = 0.0

        for sig in group:
            w = weights.get(sig.source, 1.0)
            raw_score += w * sig.strength * sig.confidence
            sources_seen.add(sig.source)

        # Agreement bonus: +10% per unique additional source beyond the first
        agreement_bonus = 1.0 + 0.10 * max(0, len(sources_seen) - 1)
        raw_score *= agreement_bonus

        # Normalise against the maximum possible score for a 3-source window
        max_possible = sum(weights.values()) * agreement_bonus
        proxy_score = min(1.0, raw_score / max(max_possible, 1.0))

        if proxy_score < min_score:
            continue

        start = max(0.0, group[0].timestamp - pre_pad)
        end = group[-1].timestamp + post_pad

        windows.append(CandidateWindow(
            start=round(start, 2),
            end=round(end, 2),
            proxy_score=round(proxy_score, 4),
            signals=sorted(sources_seen),
            signal_count=len(group),
            signal_detail=[
                {"source": s.source, "t": round(s.timestamp, 2), "reason": s.reason}
                for s in group
            ],
        ))

    windows.sort(key=lambda w: w.proxy_score, reverse=True)
    return windows[:max_windows]


# ---------------------------------------------------------------------------
# Candidate window download
# ---------------------------------------------------------------------------


def _download_window(
    vod_url: str,
    window: CandidateWindow,
    game: str,
    config: dict,
    output_dir: Path,
) -> Optional[dict]:
    """Download a single candidate window as a short MP4."""
    from datetime import datetime

    stem = f"proxy_{int(window.start):06d}_{int(window.end):06d}"
    out_path = output_dir / f"{stem}.mp4"

    if out_path.exists():
        logger.debug(f"[proxy] {stem} already downloaded — skipping")
        return None

    ydl_opts = {
        "format": "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
        "outtmpl": str(out_path),
        "quiet": True,
        "no_warnings": True,
        "download_ranges": yt_dlp.utils.download_range_func(
            None, [(window.start, window.end)]
        ),
        "force_keyframes_at_cuts": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([vod_url])
    except Exception as e:
        logger.error(f"[proxy] Download failed for window {stem}: {e}")
        return None

    if not out_path.exists():
        logger.warning(f"[proxy] yt-dlp finished but {out_path.name} not found")
        return None

    meta = {
        "clip_id": stem,
        "game": game,
        "clip_path": str(out_path),
        "meta_path": str(out_path.with_suffix(".meta.json")),
        "source": "proxy_scanner",
        "vod_url": vod_url,
        "proxy_score": window.proxy_score,
        "proxy_signals": window.signals,
        "proxy_window_start": window.start,
        "proxy_window_end": window.end,
        "downloaded_at": datetime.now().isoformat(timespec="seconds"),
    }
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    return meta


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def scan_vod(
    url: str,
    game: str,
    config: dict,
    chat_log: Optional[Path] = None,
) -> list[CandidateWindow]:
    """Run all enabled proxy signals against a VOD URL and return ranked windows.

    This is the cheap analysis pass — no video frames are read.
    Call download_candidate_windows() to fetch the winners.

    Args:
        url:      VOD URL (Twitch or YouTube).
        game:     Game key (must match assets/games/{game}/).
        config:   Global config dict (config.yaml).
        chat_log: Optional path to a pre-downloaded Twitch chat log.
                  When provided and chat_velocity.enabled is true in proxy.yaml,
                  chat velocity signals are included in the scan.
    """
    ps_cfg = _load_proxy_config(game, config)
    sig_cfg = ps_cfg.get("signals", {})
    cand_cfg = ps_cfg.get("candidate_selection", {})
    source_weights = ps_cfg.get("weights", {})

    weights: dict[str, float] = {}
    all_signals: list[ProxySignal] = []

    # Signal: chat velocity (cheapest — pure text processing)
    chat_cfg = sig_cfg.get("chat_velocity", {})
    if chat_cfg.get("enabled", False) and chat_log is not None:
        from pipeline.chat_scanner import scan_chat_log
        w = float(source_weights.get("chat_spike", _DEFAULT_WEIGHTS["chat_spike"]))
        weights["chat_spike"] = w
        try:
            chat_signals = scan_chat_log(chat_log, chat_cfg)
            all_signals.extend(chat_signals)
            logger.info(f"[proxy] chat_spike: {len(chat_signals)} velocity signal(s) from {chat_log.name}")
        except Exception as e:
            logger.warning(f"[proxy] Chat log scan failed: {e}")
    elif chat_cfg.get("enabled", False) and chat_log is None:
        logger.debug("[proxy] chat_velocity enabled but no --chat-log provided — skipping")

    # Signal: Twitch viewer clips
    vc_cfg = sig_cfg.get("viewer_clips", {})
    if vc_cfg.get("enabled", True):
        w = float(source_weights.get("viewer_clips", _DEFAULT_WEIGHTS["viewer_clips"]))
        weights["viewer_clips"] = w
        video_id = _extract_twitch_video_id(url)
        if video_id:
            all_signals.extend(_fetch_viewer_clips(video_id, config, w))
        else:
            logger.debug("[proxy] URL is not a Twitch VOD — skipping viewer_clips signal")

    # Signal: audio spikes
    as_cfg = sig_cfg.get("audio_spikes", {})
    if as_cfg.get("enabled", True):
        w = float(source_weights.get("audio_spike", _DEFAULT_WEIGHTS["audio_spike"]))
        weights["audio_spike"] = w
        z_thresh = float(as_cfg.get("z_score_threshold", 3.5))
        all_signals.extend(_detect_audio_spikes(url, config, w, z_thresh))

    if not all_signals:
        logger.warning("[proxy] No proxy signals found for this VOD")
        return []

    # Deduplicate same-source signals within 3s before merging
    all_signals = _dedupe_signals(all_signals)
    logger.info(f"[proxy] Total signals after dedup: {len(all_signals)} from {len(weights)} source(s)")

    windows = _merge_signals_into_windows(
        signals=all_signals,
        weights=weights,
        merge_gap=float(cand_cfg.get("merge_gap_seconds", 30)),
        pre_pad=float(cand_cfg.get("window_pre_seconds", 10)),
        post_pad=float(cand_cfg.get("window_post_seconds", 25)),
        min_score=float(cand_cfg.get("min_proxy_score", 0.30)),
        max_windows=int(cand_cfg.get("max_windows", 20)),
    )

    logger.info(f"[proxy] {len(windows)} candidate window(s) after merge + filter")
    for w in windows[:5]:
        logger.info(
            f"  [{w.proxy_score:.2f}] {w.start:.0f}s–{w.end:.0f}s "
            f"signals={w.signals} n={w.signal_count}"
        )

    return windows


def download_candidate_windows(
    vod_url: str,
    windows: list[CandidateWindow],
    game: str,
    config: dict,
) -> list[dict]:
    """Download each candidate window into inbox/{game}/ as a short MP4.

    Returns a list of meta dicts (same shape as run_ingestion output) so the
    existing pipeline loop in run.py can process them without modification.
    """
    inbox = Path(config.get("paths", {}).get("inbox", "inbox")) / game
    inbox.mkdir(parents=True, exist_ok=True)

    clips: list[dict] = []
    for i, window in enumerate(windows):
        logger.info(
            f"[proxy] Downloading window {i+1}/{len(windows)}: "
            f"{window.start:.0f}s–{window.end:.0f}s (score={window.proxy_score:.2f})"
        )
        meta = _download_window(vod_url, window, game, config, inbox)
        if meta:
            clips.append(meta)

    logger.info(f"[proxy] {len(clips)}/{len(windows)} windows downloaded to {inbox}")
    return clips


def save_scan_report(windows: list[CandidateWindow], vod_url: str, game: str) -> Path:
    """Write a JSON scan report to logs/proxy_scan_{game}_{timestamp}.json."""
    from datetime import datetime, timezone
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    report_path = logs_dir / f"proxy_scan_{game}_{ts}.json"
    report = {
        "vod_url": vod_url,
        "game": game,
        "scanned_at": datetime.now(timezone.utc).isoformat(),
        "windows_found": len(windows),
        "windows": [w.to_dict() for w in windows],
    }
    report_path.write_text(json.dumps(report, indent=2))
    logger.info(f"[proxy] Scan report saved → {report_path}")
    return report_path
