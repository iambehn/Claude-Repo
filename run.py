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

from pipeline import game_pack
from pipeline.ingestion import run_ingestion
from pipeline.audio_detector import run_audio_detector
from pipeline.clip_judge import run_clip_judge, evaluate
from pipeline.kill_feed import run_kill_feed_parser
from pipeline.roi_matcher import run_roi_matcher
from pipeline.weapon_detector import run_weapon_detector
from pipeline.title_engine import generate_title
from pipeline.transcription import run_transcription
from pipeline.feature_extraction import run_feature_extraction
from pipeline.decision_engine import select_template
from pipeline.processing import run_processing
from pipeline.scoring import run_scoring
from pipeline.distribution import run_distribution, poll_tiktok_pending, list_reddit_flairs
from pipeline.montage import run_montage
from pipeline.proxy_scanner import scan_vod, download_candidate_windows, save_scan_report
from pipeline.hook_enforcer import run_hook_enforcer

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
                move_to_quarantine(Path(clip_path), game, config, reason="missing_context")
                continue

        # ROI Matcher: template-match HUD elements against saved reference PNGs.
        # Runs after weapon_detector so kill_timestamps are available in meta.json.
        if config.get("roi_matcher", {}).get("enabled", False):
            run_roi_matcher(Path(clip_path), game, config)

        # Hook Enforcer: verify first 1.5s has an anchor event; propose hard_trim if not.
        if config.get("hook_enforcer", {}).get("enabled", False):
            run_hook_enforcer(Path(clip_path), game, config)

        transcript = run_transcription(clip_path, config)
        if transcript is None:
            logger.info(f"Skipping clip (language filter): {clip_path}")
            continue
        metadata = run_feature_extraction(clip_path, transcript, config)

        # Composite Clip Judge — decides accept | reject | quarantine before any
        # expensive downstream stage (FFmpeg render + Claude scoring). Short-circuits
        # on a non-accept decision to save API cost.
        worthiness = run_clip_judge(clip_path, game, config)
        if worthiness.get("decision") != "accept":
            logger.info(
                f"[clip_judge] Skipping downstream stages for {Path(clip_path).name} "
                f"(decision={worthiness.get('decision')})."
            )
            continue

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


def run_enrich_quarantine(game_arg: str, config: dict) -> None:
    """Re-evaluate quarantined clips after new ROI templates have been added.

    Clears worthiness + roi_matches, re-runs roi_matcher and clip_judge, then
    promotes clips whose decision flips to "accept" back to inbox/{game}/.
    """
    import shutil

    quarantine_root = Path("quarantine")
    inbox_root = Path(config["paths"]["inbox"])

    games = list(config["games"].keys()) if game_arg == "all" else [game_arg]

    for game in games:
        if game not in config["games"]:
            logger.warning(f"[enrich] Unknown game '{game}' — skipping.")
            continue

        try:
            pack = game_pack.load(game)
        except (FileNotFoundError, game_pack.GamePackError) as exc:
            logger.error(f"[enrich] Could not load game pack for '{game}': {exc}")
            continue

        game_quarantine = quarantine_root / game
        if not game_quarantine.exists():
            logger.info(f"[enrich] {game}: no quarantine directory — nothing to enrich.")
            continue

        evaluated = promoted = bucket_changed = 0

        for reason_dir in sorted(game_quarantine.iterdir()):
            if not reason_dir.is_dir():
                continue
            for clip_file in sorted(reason_dir.glob("*.mp4")):
                meta_path = clip_file.with_suffix(".meta.json")
                if not meta_path.exists():
                    continue

                evaluated += 1
                try:
                    meta = json.loads(meta_path.read_text())
                except (json.JSONDecodeError, OSError) as exc:
                    logger.warning(f"[enrich] Could not read meta for {clip_file.name}: {exc}")
                    continue

                old_reason = meta.get("worthiness", {}).get("quarantine_reason")

                # Clear so stages re-run from scratch
                meta.pop("worthiness", None)
                meta.pop("roi_matches", None)
                try:
                    meta_path.write_text(json.dumps(meta, indent=2))
                except OSError as exc:
                    logger.warning(f"[enrich] Could not clear meta for {clip_file.name}: {exc}")
                    continue

                if config.get("roi_matcher", {}).get("enabled", False):
                    run_roi_matcher(clip_file, game, config)

                try:
                    worthiness = evaluate(meta_path, pack, config)
                except Exception as exc:
                    logger.warning(f"[enrich] evaluate() failed for {clip_file.name}: {exc}")
                    continue

                decision = worthiness.get("decision")
                new_reason = worthiness.get("quarantine_reason")

                if decision == "accept":
                    inbox_game = inbox_root / game
                    inbox_game.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(clip_file), str(inbox_game / clip_file.name))
                    if meta_path.exists():
                        shutil.move(str(meta_path), str(inbox_game / meta_path.name))
                    promoted += 1
                    logger.info(f"[enrich] Promoted to inbox: {clip_file.name}")
                elif new_reason and new_reason != old_reason:
                    new_dir = game_quarantine / new_reason
                    new_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(clip_file), str(new_dir / clip_file.name))
                    if meta_path.exists():
                        shutil.move(str(meta_path), str(new_dir / meta_path.name))
                    bucket_changed += 1
                    logger.debug(
                        f"[enrich] {clip_file.name}: bucket {old_reason!r} → {new_reason!r}"
                    )

        logger.info(
            f"[enrich] {game}: {evaluated} evaluated, {promoted} promoted to inbox, "
            f"{bucket_changed} changed bucket, "
            f"{evaluated - promoted - bucket_changed} unchanged."
        )


def _fmt_hms(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _print_scan_preview(windows: list, game: str) -> None:
    print()
    print(f"  Proxy scan results — {game}  ({len(windows)} candidate window(s))")
    print(f"  {'#':<4} {'Score':<7} {'Start':<9} {'End':<9} {'Signals':<36} {'n':>3}")
    print("  " + "-" * 72)
    for i, w in enumerate(windows, 1):
        signals_str = ",".join(w.signals)
        print(
            f"  {i:<4} [{w.proxy_score:.2f}]  "
            f"{_fmt_hms(w.start):<9} {_fmt_hms(w.end):<9} "
            f"{signals_str:<36} {w.signal_count:>3}"
        )
    print()
    print(f"  Run without --dry-run to download these windows into inbox/{game}/")
    print()


def run_scan_vod(
    vod_url: str,
    game: str,
    config: dict,
    chat_log: "Optional[Path]" = None,
    dry_run: bool = False,
) -> None:
    """Scan a VOD for candidate clip windows via proxy signals, then download them."""
    if game not in config["games"]:
        logger.error(f"Unknown game '{game}'. Valid: {list(config['games'].keys())}")
        sys.exit(1)

    logger.info(
        f"[proxy] Scanning {vod_url} for game '{game}'"
        + (f" with chat log {chat_log}" if chat_log else "")
        + (" [dry-run]" if dry_run else "")
    )
    windows = scan_vod(vod_url, game, config, chat_log=chat_log)
    if not windows:
        logger.info("[proxy] No candidate windows found.")
        return

    if dry_run:
        _print_scan_preview(windows, game)
        return

    save_scan_report(windows, vod_url, game)
    clips = download_candidate_windows(vod_url, windows, game, config)
    logger.info(
        f"[proxy] {len(clips)} clip(s) staged in inbox/{game}/. "
        f"Run: python run.py --game {game}"
    )


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
    group.add_argument(
        "--validate-game-pack",
        metavar="SLUG",
        dest="validate_game_pack",
        help="Validate assets/games/SLUG/ and exit 0 on pass, 1 on fail.",
    )
    group.add_argument(
        "--init-game",
        metavar="SLUG",
        dest="init_game",
        help="Scaffold a new assets/games/SLUG/ directory with placeholder YAML.",
    )
    group.add_argument(
        "--enrich-quarantine",
        metavar="GAME",
        dest="enrich_quarantine",
        help="Re-evaluate quarantined clips for GAME (or 'all') after adding new ROI templates.",
    )
    group.add_argument(
        "--scan-vod",
        nargs=2,
        metavar=("URL", "GAME"),
        dest="scan_vod",
        help="Scan a full VOD for candidate clip windows. "
             "Args: URL GAME (e.g. https://twitch.tv/videos/123 marvel_rivals). "
             "Downloads candidate windows into inbox/GAME/ for normal pipeline processing.",
    )
    group.add_argument(
        "--wiki-enrich",
        nargs=2,
        metavar=("GAME", "URL"),
        dest="wiki_enrich",
        help="Fetch a Fandom wiki page and write a draft entities.yaml for GAME. "
             "Args: GAME URL (e.g. marvel_rivals https://marvelrivals.fandom.com/wiki/Heroes).",
    )
    group.add_argument(
        "--audit-weapon-detector",
        metavar="GAME",
        dest="audit_weapon_detector",
        help="Scan all clip directories for GAME and rank weapons that need better reference icons.",
    )
    group.add_argument(
        "--export-training-data",
        metavar="GAME",
        dest="export_training_data",
        help="Backfill training records for all reviewed clips in GAME (or 'all') from inbox/. "
             "Useful for exporting data from clips reviewed before training logging was added.",
    )
    parser.add_argument(
        "--chat-log",
        metavar="PATH",
        dest="chat_log",
        help="Path to a Twitch chat log file (.txt/.log) for chat velocity signal with --scan-vod.",
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

    # --- Game-pack-only commands: handle before startup validation ---
    if args.validate_game_pack:
        slug = args.validate_game_pack
        result = game_pack.validate(slug)
        if result.ok:
            logger.info(f"[game_pack:{slug}] OK")
            for warning in result.warnings:
                logger.warning(f"[game_pack:{slug}] warning: {warning}")
            sys.exit(0)
        for err in result.errors:
            logger.error(f"[game_pack:{slug}] {err}")
        sys.exit(1)

    if args.init_game:
        slug = args.init_game
        try:
            game_pack.init_scaffold(slug)
        except game_pack.GamePackError as exc:
            logger.error(f"[init_game:{slug}] {exc}")
            sys.exit(1)
        logger.info(
            f"Scaffolded assets/games/{slug}/. Next: calibrate HUD ROIs and fill entities.yaml."
        )
        sys.exit(0)

    if args.enrich_quarantine:
        run_enrich_quarantine(args.enrich_quarantine, config)
        sys.exit(0)

    if args.scan_vod:
        vod_url, game = args.scan_vod
        chat_log = Path(args.chat_log) if args.chat_log else None
        run_scan_vod(vod_url, game, config, chat_log=chat_log, dry_run=args.dry_run)
        sys.exit(0)

    if args.wiki_enrich:
        game, wiki_url = args.wiki_enrich
        from pipeline.wiki_enrichment import enrich_game_from_wiki
        result = enrich_game_from_wiki(game, wiki_url, config)
        logger.info(
            f"[wiki_enrich] {game}: {result['status']} — "
            f"{result['entities_found']} entities, {result['icons_downloaded']} icons "
            f"→ {result.get('draft_dir') or 'no output'}"
        )
        for warning in result.get("warnings") or []:
            logger.warning(f"[wiki_enrich] {warning}")
        sys.exit(0 if result["status"] in ("ok", "partial") else 1)

    if args.audit_weapon_detector:
        from pipeline.weapon_detector_audit import audit_weapon_detector
        result = audit_weapon_detector(args.audit_weapon_detector, config)
        logger.info(
            f"[weapon_audit] {args.audit_weapon_detector}: "
            f"{result['audited_clips']} clips audited, "
            f"{len(result['recommended_targets'])} targets recommended "
            f"→ {result.get('report_path') or 'no report'}"
        )
        sys.exit(0)

    if args.export_training_data:
        from utils.training_logger import log_review_decision
        game_arg = args.export_training_data
        games = list(config["games"].keys()) if game_arg == "all" else [game_arg]
        inbox_root = Path(config["paths"]["inbox"])
        total = exported = 0
        for game in games:
            game_dir = inbox_root / game
            if not game_dir.exists():
                continue
            for meta_path in sorted(game_dir.glob("*.meta.json")):
                try:
                    meta = json.loads(meta_path.read_text())
                except (json.JSONDecodeError, OSError):
                    continue
                if not meta.get("review_status"):
                    continue
                total += 1
                clip_id = meta.get("clip_id") or meta_path.stem
                if log_review_decision(meta, clip_id, game, config):
                    exported += 1
        logger.info(
            f"[export_training] Exported {exported}/{total} reviewed clips to "
            f"{config.get('training', {}).get('output_dir', 'data/training_sets')}/clip_judge/"
        )
        sys.exit(0)

    # --- Pipeline-running commands: validate every game pack first ---
    if args.montage or args.game or args.watch:
        try:
            game_pack.ensure_game_packs_valid(config)
        except game_pack.GamePackError as exc:
            logger.error(f"Game-pack validation failed:\n{exc}")
            sys.exit(1)

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
