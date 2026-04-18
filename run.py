"""
Gaming Clip Farming Bot — Pipeline Entry Point

Usage:
    python run.py --game arc_raiders
    python run.py --game marvel_rivals
    python run.py --game deadlock
    python run.py --game all          # run for all configured games
    python run.py --distribute        # upload all approved clips, log analytics, backup

The pipeline runs each stage in sequence for the given game:
  1. Ingestion          — download clips from Twitch
  2. Transcription      — Whisper speech-to-text
  3. Feature Extraction — build metadata JSON per clip
  4. Decision Engine    — select template per clip
  5. Processing         — FFmpeg render
  6. AI Scoring         — Claude API virality score
  7. [Manual Review launched separately: python -m pipeline.review.app]
  8. Distribution       — python run.py --distribute
     a. Upload to enabled social media platforms
     b. Log row to Google Sheets (analytics)
     c. Back up clip + meta to Google Drive
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

from utils.analytics import log_clip
from utils.backup import backup_clip
from utils.file_utils import ensure_dirs
from utils.logger import get_logger
from utils.metadata_injector import inject_metadata

from pipeline.ingestion import run_ingestion
from pipeline.audio_detector import run_audio_detector
from pipeline.kill_feed import run_kill_feed_parser
from pipeline.weapon_detector import run_weapon_detector
from pipeline.title_engine import generate_title
from pipeline.transcription import run_transcription
from pipeline.feature_extraction import run_feature_extraction
from pipeline.decision_engine import select_template
from pipeline.processing import run_processing
from pipeline.scoring import run_scoring
from pipeline.distribution import run_distribution, poll_tiktok_pending, list_reddit_flairs
from pipeline.montage import run_montage

logger = get_logger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_pipeline_for_game(game: str, config: dict) -> None:
    if game not in config["games"]:
        logger.error(f"Unknown game '{game}'. Valid options: {list(config['games'].keys())}")
        sys.exit(1)

    logger.info(f"Starting pipeline for: {config['games'][game]['display_name']}")

    clips = run_ingestion(game, config)
    if not clips:
        logger.info("No clips ingested. Exiting.")
        return

    for clip in clips:
        if clip.get("review_status"):
            logger.debug(f"Skipping already-reviewed clip: {clip.get('clip_id')}")
            continue

        clip_path = clip["clip_path"]
        logger.info(f"Processing clip: {clip_path}")

        # Audio Detector: runs first — cheapest signal, narrows down action timestamps
        # for the kill-feed OpenCV pass that follows.
        if config.get("audio_detector", {}).get("enabled", False):
            run_audio_detector(Path(clip_path), game, config)

        # Kill-Feed Parser: analyse kill events in the ROI before expensive stages.
        # When audio_detector is also enabled, it reads audio spike_timestamps from
        # meta.json and restricts frame sampling to those windows automatically.
        if config.get("kill_feed", {}).get("enabled", False):
            kf_result = run_kill_feed_parser(Path(clip_path), game, config)
            if not kf_result["passed"]:
                logger.debug(
                    f"[kill_feed] Low sweat score ({kf_result['sweat_score']}) "
                    f"for {Path(clip_path).name} — continuing with reduced priority."
                )

        # Weapon Detector: identify active weapon from HUD icon ROI.
        # Uses kill_timestamps from kill_feed meta when frame_sample: "kill_timestamps".
        if config.get("weapon_detector", {}).get("enabled", False):
            wd_result = run_weapon_detector(Path(clip_path), game, config)
            if (
                config["weapon_detector"].get("require_detection", False)
                and not wd_result.get("weapon_id")
            ):
                logger.info(
                    f"[weapon_detector] No weapon detected in {Path(clip_path).name} "
                    f"— quarantining (require_detection=true)."
                )
                from utils.file_utils import move_to_quarantine
                move_to_quarantine(Path(clip_path), game, config)
                continue

        transcript = run_transcription(clip_path, config)
        if transcript is None:
            logger.info(f"Skipping clip (language filter): {clip_path}")
            continue
        metadata = run_feature_extraction(clip_path, transcript, config)
        template = select_template(metadata, config)
        processed_path = run_processing(clip_path, template, metadata, config)
        score = run_scoring(processed_path, metadata, config)

        # Title Engine: generate upload title after scoring so sweat_score,
        # kill_feed, and weapon_detection are all in meta.json.
        if config.get("title_engine", {}).get("enabled", False):
            title_result = generate_title(Path(clip_path), game, config)
            logger.info(f"[title_engine] Proposed title: '{title_result.get('title')}'")
            # Embed title and hashtags into the processed MP4 file tags.
            inject_metadata(Path(processed_path), Path(clip_path), config)

        logger.info(
            f"Clip ready for review — score: {score.get('highlight_score', 'n/a')} "
            f"| template: {template.get('template_id', 'n/a')}"
        )

    logger.info(f"Pipeline complete for {game}. Launch review UI: python -m pipeline.review.app")


def run_distribution_for_all(config: dict, dry_run: bool = False) -> None:
    """Distribute all approved clips that have not yet been posted.

    Scans accepted/{game}/ for .mp4 files, loads their metadata from
    inbox/{game}/, then runs distribution → analytics → backup in sequence.
    Fully idempotent: already-distributed platforms are skipped.

    When dry_run=True, logs what would be distributed without uploading anything.
    """
    accepted_root = Path(config["paths"]["accepted"])
    inbox_root = Path(config["paths"]["inbox"])
    total = 0
    distributed = 0

    for game in config["games"]:
        game_dir = accepted_root / game
        if not game_dir.exists():
            continue

        for clip_file in sorted(game_dir.glob("*.mp4")):
            total += 1
            meta_path = _find_meta_for_clip(clip_file, inbox_root / game)
            if meta_path is None:
                logger.warning(f"No meta.json found for {clip_file.name} — skipping distribution.")
                continue

            metadata = json.loads(meta_path.read_text())

            # Only distribute accepted clips
            if metadata.get("review_status") != "accepted":
                continue

            if dry_run:
                enabled_platforms = [
                    p for p, cfg in config.get("distribution", {}).get("platforms", {}).items()
                    if cfg.get("enabled")
                ]
                score = metadata.get("scoring", {}).get("highlight_score", "n/a")
                logger.info(
                    f"[DRY RUN] {clip_file.name} | score={score} "
                    f"| platforms={enabled_platforms or ['none enabled']}"
                )
                distributed += 1
                continue

            logger.info(f"Distributing: {clip_file.name}")
            dist_results = run_distribution(str(clip_file), metadata, config)

            # Reload metadata after distribution (it may have been updated)
            metadata = json.loads(meta_path.read_text())

            log_clip(metadata, dist_results, config)
            backup_clip(str(clip_file), metadata, config)
            distributed += 1

    if dry_run:
        logger.info(f"[DRY RUN] {distributed}/{total} clip(s) would be distributed.")
    else:
        logger.info(f"Distribution complete: {distributed}/{total} clip(s) processed.")
    if distributed == 0 and total == 0:
        logger.info("No clips in accepted/ yet. Run the pipeline then approve clips in the review UI.")


def _find_meta_for_clip(clip_file: Path, inbox_game_dir: Path) -> Path | None:
    """Locate the inbox .meta.json that matches an accepted clip.

    The accepted clip filename format is: {game}_{date}_{clip_id}.mp4
    We look for a .meta.json whose 'clip_id' field matches the clip_id
    portion of the filename, or fall back to a full-stem name match.
    """
    # Extract clip_id: everything after {game}_{date}_
    parts = clip_file.stem.split("_", 2)
    clip_id_guess = parts[2] if len(parts) == 3 else clip_file.stem

    # Fast path: look for <clip_id>.meta.json directly
    candidate = inbox_game_dir / f"{clip_id_guess}.meta.json"
    if candidate.exists():
        return candidate

    # Slow path: scan all meta files and match by clip_id field
    for meta_file in inbox_game_dir.glob("*.meta.json"):
        try:
            meta = json.loads(meta_file.read_text())
            if meta.get("clip_id") == clip_id_guess:
                return meta_file
        except (json.JSONDecodeError, OSError):
            continue

    return None


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Gaming Clip Farming Bot")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--game",
        help="Game key to process (e.g. arc_raiders) or 'all' for every configured game.",
    )
    group.add_argument(
        "--distribute",
        action="store_true",
        help="Upload all approved clips from accepted/ to social media, log analytics, and back up to Drive.",
    )
    group.add_argument(
        "--watch",
        action="store_true",
        help="Continuously run the pipeline for all games on a loop (interval set by pipeline.watch_interval_seconds in config).",
    )
    group.add_argument(
        "--poll-tiktok",
        action="store_true",
        dest="poll_tiktok",
        help="Check TikTok processing status for uploaded clips that don't have a URL yet.",
    )
    group.add_argument(
        "--list-reddit-flairs",
        action="store_true",
        dest="list_reddit_flairs",
        help="Print available link flairs for each configured subreddit, then exit.",
    )
    group.add_argument(
        "--montage",
        metavar="GAME",
        help="Assemble a montage from accepted clips for GAME (or 'all'). Requires montage.enabled: true in config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --distribute: show what would be uploaded without actually posting anything.",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dirs(config)

    if args.montage:
        games_to_process = list(config["games"].keys()) if args.montage == "all" else [args.montage]
        for game in games_to_process:
            if game not in config["games"]:
                logger.error(f"Unknown game '{game}'. Valid options: {list(config['games'].keys())}")
                sys.exit(1)
            run_montage(game, config)
    elif args.distribute:
        run_distribution_for_all(config, dry_run=args.dry_run)
    elif args.poll_tiktok:
        poll_tiktok_pending(config)
    elif args.list_reddit_flairs:
        list_reddit_flairs(config)
    elif args.watch:
        interval = config.get("pipeline", {}).get("watch_interval_seconds", 300)
        logger.info(f"Watch mode active — running all games every {interval}s. Ctrl+C to stop.")
        while True:
            for game in config["games"]:
                run_pipeline_for_game(game, config)
            logger.info(f"Watch mode: next run in {interval}s...")
            time.sleep(interval)
    elif args.game == "all":
        for game in config["games"]:
            run_pipeline_for_game(game, config)
    else:
        run_pipeline_for_game(args.game, config)


if __name__ == "__main__":
    main()
