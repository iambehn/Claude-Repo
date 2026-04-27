"""
pipeline/youtube_scanner.py — YouTube VOD proxy signal detection

Extracts candidate timestamps from YouTube VODs without downloading video:
  1. Video chapters  — via yt-dlp metadata (free, no API key required)
  2. Timestamped comments — via YouTube Data API v3 (requires YOUTUBE_API_KEY)

Both sources produce ProxySignal objects that feed into the proxy_scanner's
window merger alongside Twitch signals.

Required env var (comments only):
    YOUTUBE_API_KEY — read-only key from Google Cloud Console
                      (same key used by the Game Scout dashboard)

Chapters are always attempted regardless of API key — yt-dlp extracts them
from the video's description timestamps for free.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from typing import Any, Optional

import requests

from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Local ProxySignal — structurally identical to proxy_scanner.ProxySignal;
# defined here to avoid a circular import (proxy_scanner imports us lazily).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProxySignal:
    source: str
    timestamp: float
    strength: float
    confidence: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

_YT_ID_PATTERNS = [
    re.compile(r"youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})"),
    re.compile(r"youtu\.be/([a-zA-Z0-9_-]{11})"),
    re.compile(r"youtube\.com/live/([a-zA-Z0-9_-]{11})"),
    re.compile(r"youtube\.com/v/([a-zA-Z0-9_-]{11})"),
]

# Matches MM:SS and HH:MM:SS — e.g. "1:04", "12:45", "1:04:22"
_TIMESTAMP_RE = re.compile(
    r"\b(?:(?P<hours>\d{1,2}):)?(?P<minutes>\d{1,2}):(?P<seconds>\d{2})\b"
)


def _extract_video_id(url: str) -> Optional[str]:
    for pattern in _YT_ID_PATTERNS:
        m = pattern.search(url)
        if m:
            return m.group(1)
    return None


def _parse_timestamps(text: str) -> list[float]:
    """Extract every HH:MM:SS / MM:SS timestamp from a string as seconds."""
    results: list[float] = []
    for m in _TIMESTAMP_RE.finditer(text):
        hours = int(m.group("hours") or 0)
        minutes = int(m.group("minutes"))
        seconds = int(m.group("seconds"))
        total = hours * 3600 + minutes * 60 + seconds
        if total > 0:
            results.append(float(total))
    return results


# ---------------------------------------------------------------------------
# Signal: YouTube chapters (via yt-dlp, no API key)
# ---------------------------------------------------------------------------


def _fetch_chapters(url: str, confidence: float = 0.50) -> list[ProxySignal]:
    """Use yt-dlp --skip-download -j to extract chapter start times.

    YouTube auto-generates chapters from description timestamps, so this
    works even without the creator explicitly using the chapters feature.
    """
    try:
        result = subprocess.run(
            ["yt-dlp", "--skip-download", "-j", "--no-warnings", "--quiet", url],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.debug(f"[youtube_scanner] yt-dlp metadata fetch returned {result.returncode}")
            return []

        data = json.loads(result.stdout)
        chapters: list[dict] = data.get("chapters") or []

        signals: list[ProxySignal] = []
        for ch in chapters:
            t = float(ch.get("start_time", 0))
            title = str(ch.get("title", "")).strip()
            if t <= 0:
                continue
            signals.append(ProxySignal(
                source="youtube_chapter",
                timestamp=t,
                strength=0.50,
                confidence=confidence,
                reason=f"chapter: {title!r}",
            ))

        logger.info(f"[youtube_scanner] chapters: {len(signals)} found")
        return signals

    except FileNotFoundError:
        logger.warning("[youtube_scanner] yt-dlp not found — cannot fetch chapters")
        return []
    except json.JSONDecodeError as e:
        logger.warning(f"[youtube_scanner] yt-dlp output parse error: {e}")
        return []
    except Exception as e:
        logger.warning(f"[youtube_scanner] Chapter fetch failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Signal: timestamped YouTube comments (YouTube Data API v3)
# ---------------------------------------------------------------------------


def _fetch_timestamped_comments(
    url: str,
    max_comments: int = 100,
    confidence: float = 0.65,
) -> list[ProxySignal]:
    """Fetch top comments via YouTube Data API v3 and extract timestamp mentions.

    Comments like "12:45 insane clip" or "skip to 1:04:22" become ProxySignals.
    Strength scales with the comment's like count (more liked = stronger signal).

    Requires YOUTUBE_API_KEY environment variable.
    """
    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        logger.debug(
            "[youtube_scanner] YOUTUBE_API_KEY not set — skipping comment signals. "
            "Set YOUTUBE_API_KEY in .env to enable."
        )
        return []

    video_id = _extract_video_id(url)
    if not video_id:
        logger.debug(f"[youtube_scanner] Could not extract video ID from {url!r}")
        return []

    signals: list[ProxySignal] = []
    try:
        resp = requests.get(
            "https://www.googleapis.com/youtube/v3/commentThreads",
            params={
                "part": "snippet",
                "videoId": video_id,
                "order": "relevance",
                "maxResults": min(max_comments, 100),
                "key": api_key,
            },
            timeout=15,
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])

        for item in items:
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            text = snippet.get("textDisplay", "")
            like_count = int(snippet.get("likeCount", 0))

            for t in _parse_timestamps(text):
                # Liked comments are stronger signals — 0 likes → 0.30, 1000 likes → 1.0
                strength = round(min(1.0, 0.30 + like_count / 1000), 4)
                preview = text[:70].replace("\n", " ")
                signals.append(ProxySignal(
                    source="youtube_comment",
                    timestamp=t,
                    strength=strength,
                    confidence=confidence,
                    reason=f"{like_count} likes: {preview!r}",
                ))

        logger.info(
            f"[youtube_scanner] comments: {len(signals)} timestamp mention(s) "
            f"across {len(items)} top comments"
        )
    except requests.HTTPError as e:
        logger.warning(f"[youtube_scanner] Comment API error: {e}")
    except Exception as e:
        logger.warning(f"[youtube_scanner] Comment fetch failed: {e}")

    return signals


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def scan_youtube_vod(url: str, yt_cfg: dict, global_config: dict) -> list[ProxySignal]:
    """Run all enabled YouTube proxy signals and return combined ProxySignals.

    Called from proxy_scanner.scan_vod() when a YouTube URL is detected.
    """
    signals: list[ProxySignal] = []

    ch_cfg = yt_cfg.get("chapters", {})
    if ch_cfg.get("enabled", True):
        confidence = float(ch_cfg.get("confidence", 0.50))
        signals.extend(_fetch_chapters(url, confidence=confidence))

    cm_cfg = yt_cfg.get("comments", {})
    if cm_cfg.get("enabled", True):
        max_comments = int(cm_cfg.get("max_comments", 100))
        confidence = float(cm_cfg.get("confidence", 0.65))
        signals.extend(_fetch_timestamped_comments(url, max_comments=max_comments, confidence=confidence))

    return signals
