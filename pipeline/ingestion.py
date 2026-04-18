"""
Stage 1 — Ingestion

Uses the Twitch Helix API to fetch the top clips for a game, then downloads
each clip via yt-dlp. Each downloaded clip is probed with FFprobe to extract
duration, resolution, fps, and audio presence. Clips are assigned a quality
tag (high / medium / low) or quarantined if they fall outside the configured
thresholds.

Good clips land in inbox/{game}/ with a sidecar .meta.json file.
Bad clips are moved to quarantine/{game}/ and excluded from the returned list.

Required environment variables (in .env):
    TWITCH_CLIENT_ID      — from dev.twitch.tv
    TWITCH_CLIENT_SECRET  — from dev.twitch.tv

Sidecar .meta.json fields:
    clip_id, game, clip_path, meta_path, quality_tag,
    duration_seconds, resolution_width, resolution_height,
    fps, has_audio, downloaded_at
"""

import json
import os
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
import yt_dlp

from utils.file_utils import move_to_quarantine
from utils.logger import get_logger

logger = get_logger(__name__)

_TWITCH_TOKEN_URL = "https://id.twitch.tv/oauth2/token"
_TWITCH_GAMES_URL = "https://api.twitch.tv/helix/games"
_TWITCH_CLIPS_URL = "https://api.twitch.tv/helix/clips"


# ---------------------------------------------------------------------------
# Twitch API helpers
# ---------------------------------------------------------------------------

def _get_twitch_token(client_id: str, client_secret: str) -> str:
    """Obtain a Twitch app access token via client credentials flow."""
    resp = requests.post(
        _TWITCH_TOKEN_URL,
        params={
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "client_credentials",
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def _get_game_id(display_name: str, client_id: str, token: str) -> str | None:
    """Look up the Twitch numeric game ID by display name."""
    resp = requests.get(
        _TWITCH_GAMES_URL,
        params={"name": display_name},
        headers={"Client-Id": client_id, "Authorization": f"Bearer {token}"},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json().get("data", [])
    if not data:
        logger.error(f"Game '{display_name}' not found on Twitch.")
        return None
    game_id = data[0]["id"]
    logger.debug(f"Resolved '{display_name}' → Twitch game_id={game_id}")
    return game_id


def _fetch_clip_metadata(
    game_id: str, client_id: str, token: str, count: int, max_age_hours: float | None = None
) -> list[dict]:
    """Return metadata for up to `count` clips for a game, sorted by view count.

    Each dict contains the fields needed by the rank_looker and download stages:
    url, broadcaster_id, broadcaster_name, title, view_count, duration, created_at.
    """
    params: dict = {"game_id": game_id, "first": min(count, 100)}
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours) if max_age_hours else None
    if cutoff:
        # Filter server-side so Twitch returns only recent clips, not all-time top clips
        params["started_after"] = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")

    resp = requests.get(
        _TWITCH_CLIPS_URL,
        params=params,
        headers={"Client-Id": client_id, "Authorization": f"Bearer {token}"},
        timeout=15,
    )
    resp.raise_for_status()
    clips = resp.json().get("data", [])

    metas = []
    skipped_stale = 0

    for c in clips:
        if not c.get("url"):
            continue
        if cutoff and c.get("started_at"):
            clip_time = datetime.fromisoformat(c["started_at"].replace("Z", "+00:00"))
            if clip_time < cutoff:
                skipped_stale += 1
                continue
        metas.append({
            "url":                c["url"],
            "broadcaster_id":     c.get("broadcaster_id", ""),
            "broadcaster_name":   c.get("broadcaster_name", ""),
            "title":              c.get("title", ""),
            "view_count":         c.get("view_count", 0),
            "duration":           c.get("duration", 0),
            "created_at":         c.get("created_at", ""),
        })

    if skipped_stale:
        logger.debug(f"Freshness filter: skipped {skipped_stale} clip(s) older than {max_age_hours}h.")
    logger.debug(f"Fetched metadata for {len(metas)} clip(s) from Twitch API.")
    return metas


def _get_follower_count(broadcaster_id: str, client_id: str, token: str) -> int:
    """Return total follower count for a broadcaster. Returns 0 on any error."""
    try:
        resp = requests.get(
            "https://api.twitch.tv/helix/channels/followers",
            params={"broadcaster_id": broadcaster_id},
            headers={"Client-Id": client_id, "Authorization": f"Bearer {token}"},
            timeout=10,
        )
        resp.raise_for_status()
        return int(resp.json().get("total", 0))
    except Exception as e:
        logger.debug(f"Follower count lookup failed for broadcaster {broadcaster_id}: {e}")
        return 0


# ---------------------------------------------------------------------------
# FFprobe helper
# ---------------------------------------------------------------------------

def _probe_clip(clip_path: Path) -> dict | None:
    """Run FFprobe on a clip and return parsed media info."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_streams",
                "-show_format",
                str(clip_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        data = json.loads(result.stdout)
    except (subprocess.SubprocessError, json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"FFprobe failed for {clip_path.name}: {e}")
        return None

    streams = data.get("streams", [])
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    if not video_stream:
        logger.warning(f"No video stream found in {clip_path.name}")
        return None

    fmt = data.get("format", {})
    duration = float(fmt.get("duration") or video_stream.get("duration", 0))

    fps_raw = video_stream.get("r_frame_rate", "30/1")
    try:
        num, den = fps_raw.split("/")
        fps = round(int(num) / int(den), 3)
    except (ValueError, ZeroDivisionError):
        fps = 30.0

    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    return {
        "duration_seconds": round(duration, 3),
        "resolution_width": int(video_stream.get("width", 0)),
        "resolution_height": int(video_stream.get("height", 0)),
        "fps": fps,
        "has_audio": has_audio,
    }


# ---------------------------------------------------------------------------
# Quality classification
# ---------------------------------------------------------------------------

def _assign_quality_tag(probe: dict, thresholds: dict) -> str | None:
    """Assign a quality tag or return None to quarantine the clip."""
    h = probe["resolution_height"]
    dur = probe["duration_seconds"]

    if dur < thresholds["min_duration_seconds"]:
        return None
    if dur > thresholds["max_duration_seconds"]:
        return None
    if h < thresholds["min_resolution_height"]:
        return None

    if h >= 1080:
        return "high"
    if h >= 720:
        return "medium"
    return "low"


def _write_meta(clip_path: Path, game: str, probe: dict, quality_tag: str) -> Path:
    """Write the sidecar .meta.json for a clip and return its path."""
    meta = {
        "clip_id": clip_path.stem,
        "game": game,
        "clip_path": str(clip_path),
        "meta_path": str(clip_path.with_suffix(".meta.json")),
        "quality_tag": quality_tag,
        "duration_seconds": probe["duration_seconds"],
        "resolution_width": probe["resolution_width"],
        "resolution_height": probe["resolution_height"],
        "fps": probe["fps"],
        "has_audio": probe["has_audio"],
        "downloaded_at": datetime.now().isoformat(timespec="seconds"),
    }
    meta_path = clip_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta_path


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def run_ingestion(game: str, config: dict) -> list[dict]:
    """Download and classify clips for a single game.

    Fetches top clip URLs from the Twitch Helix API, downloads each via
    yt-dlp, probes with FFprobe, assigns quality tags, quarantines bad clips,
    and writes a sidecar .meta.json per good clip.

    Args:
        game: Game key matching a key in config['games'] (e.g. 'deadlock').
        config: Full parsed config.yaml dict.

    Returns:
        List of clip manifest dicts for all successfully ingested clips.
    """
    game_cfg = config["games"][game]
    display_name = game_cfg["display_name"]
    output_dir = Path(config["paths"]["inbox"]) / game
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = config["ingestion"]["quality_thresholds"]
    max_clips = config["ingestion"]["max_clips_per_run"]

    # --- Twitch API: resolve game ID and fetch clip URLs ---
    client_id = os.environ.get("TWITCH_CLIENT_ID", "")
    client_secret = os.environ.get("TWITCH_CLIENT_SECRET", "")

    if not client_id or not client_secret:
        logger.error(
            "TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET must be set in .env. "
            "Create a free app at dev.twitch.tv to get these."
        )
        return []

    try:
        token = _get_twitch_token(client_id, client_secret)
        game_id = _get_game_id(display_name, client_id, token)
        if not game_id:
            return []
        max_age_hours = config["ingestion"].get("max_clip_age_hours")
        clip_metas = _fetch_clip_metadata(game_id, client_id, token, max_clips, max_age_hours)
    except requests.RequestException as e:
        logger.error(f"[{game}] Twitch API error: {e}")
        return []

    if not clip_metas:
        logger.warning(f"[{game}] No clips returned from Twitch API.")
        return []

    # --- Rank Looker: filter clips by skill signals before downloading ---
    from pipeline.rank_looker import check_clip as rank_check
    rl_enabled = config.get("rank_looker", {}).get("enabled", False)

    if rl_enabled:
        follower_cache: dict[str, int] = {}
        passing_metas = []
        skipped_rank = 0
        for meta in clip_metas:
            bid = meta["broadcaster_id"]
            if bid not in follower_cache:
                follower_cache[bid] = _get_follower_count(bid, client_id, token)
            meta["broadcaster_follower_count"] = follower_cache[bid]
            meta["game"] = game
            result = rank_check(meta, config)
            if result["passed"]:
                passing_metas.append(meta)
            else:
                skipped_rank += 1
                logger.info(
                    f"[rank_looker] Skipped '{meta['broadcaster_name']}' "
                    f"({meta['title'][:60]}): {result['reason']}"
                )
        if skipped_rank:
            logger.info(
                f"[{game}] Rank Looker: {skipped_rank} clip(s) skipped, "
                f"{len(passing_metas)} clip(s) passing."
            )
        clip_metas = passing_metas

    if not clip_metas:
        logger.info(f"[{game}] No clips passed rank filter.")
        return []

    clip_urls = [m["url"] for m in clip_metas]
    logger.info(f"[{game}] Downloading {len(clip_urls)} clip(s) for '{display_name}'...")

    # --- yt-dlp: download each clip URL ---
    ydl_opts = {
        "outtmpl": str(output_dir / "%(title)s_%(id)s.%(ext)s"),
        "format": "best[ext=mp4]/best",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "ignoreerrors": True,
    }

    downloaded_paths: list[Path] = []

    def _on_progress(d: dict) -> None:
        if d.get("status") == "finished":
            path = d.get("filepath") or d.get("filename")
            if path:
                downloaded_paths.append(Path(path))

    ydl_opts["progress_hooks"] = [_on_progress]

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download(clip_urls)
    except Exception as e:
        logger.error(f"[{game}] yt-dlp error: {e}")
        if not downloaded_paths:
            return []

    logger.info(f"[{game}] Downloaded {len(downloaded_paths)} file(s). Classifying...")

    # --- Classify each downloaded clip ---
    manifests: list[dict] = []

    for clip_path in downloaded_paths:
        if not clip_path.exists():
            logger.warning(f"File missing after download: {clip_path}")
            continue

        meta_path = clip_path.with_suffix(".meta.json")
        if meta_path.exists():
            logger.debug(f"Skipping already-processed clip: {clip_path.name}")
            manifests.append(json.loads(meta_path.read_text()))
            continue

        probe = _probe_clip(clip_path)
        if probe is None:
            logger.warning(f"[{game}] Quarantining (probe failed): {clip_path.name}")
            move_to_quarantine(clip_path, game, config)
            continue

        quality_tag = _assign_quality_tag(probe, thresholds)
        if quality_tag is None:
            reason = (
                f"duration={probe['duration_seconds']}s, "
                f"height={probe['resolution_height']}px"
            )
            logger.warning(
                f"[{game}] Quarantining (thresholds): {clip_path.name} ({reason})"
            )
            move_to_quarantine(clip_path, game, config)
            continue

        _write_meta(clip_path, game, probe, quality_tag)
        manifest = json.loads(clip_path.with_suffix(".meta.json").read_text())
        manifests.append(manifest)
        logger.info(
            f"[{game}] Ingested [{quality_tag}] {clip_path.name} "
            f"({probe['duration_seconds']}s, {probe['resolution_height']}p)"
        )

    logger.info(f"[{game}] Ingestion complete: {len(manifests)} clip(s) ready.")
    return manifests
