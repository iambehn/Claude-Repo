"""
Pipeline orchestrator — deterministic clip state machine.

Wraps existing stage functions with per-clip exception isolation, retry
tracking, and resumable state in each clip's meta.json "orchestrator" block.

Usage:  python run.py --orchestrate [GAME]

Lifecycle (stages in order):
    detect → judge → process → review_ready   (accepted path)
                  ↘ quarantined               (terminal — moved to quarantine/)
                  ↘ rejected                  (terminal — stays in inbox/)
    failed                                    (terminal — exhausted max_attempts)

On restart, each clip resumes from its last recorded stage because all detector
stages are already idempotent (they check their meta.json key before running).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from utils.logger import get_logger

logger = get_logger(__name__)

TERMINAL_STAGES = frozenset({"quarantined", "rejected", "review_ready", "failed"})

_DEFAULT_MAX_ATTEMPTS = 3
_DEFAULT_ZOMBIE_TIMEOUT = 10  # minutes


def run_orchestrator(game: str, config: dict) -> dict[str, int]:
    """Run orchestrated pipeline for one game. Returns summary counts."""
    inbox = Path(config["paths"]["inbox"]) / game
    if not inbox.exists():
        logger.info(f"[orchestrator] No inbox directory for {game}: {inbox}")
        return {"processed": 0, "quarantined": 0, "failed": 0, "skipped": 0}

    orch_cfg = config.get("orchestrator", {})
    max_attempts = orch_cfg.get("max_attempts", _DEFAULT_MAX_ATTEMPTS)
    zombie_timeout = orch_cfg.get("zombie_timeout_minutes", _DEFAULT_ZOMBIE_TIMEOUT)

    summary = {"processed": 0, "quarantined": 0, "failed": 0, "skipped": 0}

    meta_files = sorted(inbox.glob("*.meta.json"))
    if not meta_files:
        logger.info(f"[orchestrator] inbox/{game}/ is empty — nothing to process")
        return summary

    logger.info(f"[orchestrator] {game}: checking {len(meta_files)} clips")

    for meta_path in meta_files:
        try:
            meta = _load_meta(meta_path)
        except Exception as exc:
            logger.warning(f"[orchestrator] Could not load {meta_path.name}: {exc}")
            summary["skipped"] += 1
            continue

        # Clips the review UI already handled — skip silently
        if meta.get("review_status"):
            summary["skipped"] += 1
            continue

        # Infer initial stage from existing meta state for clips processed
        # by the old --game pipeline (no orchestrator block yet).
        orch = _ensure_orch_block(meta, meta_path, max_attempts)

        # Terminal stages are permanently done — orchestrator never re-enters them
        if orch["stage"] in TERMINAL_STAGES:
            summary["skipped"] += 1
            continue

        # Reset clips stuck in "running" (zombie detection)
        _zombie_check(orch, meta, meta_path, zombie_timeout)
        if orch["status"] == "running":
            summary["skipped"] += 1
            continue

        clip_path = meta.get("clip_path", "")
        if not clip_path or not Path(clip_path).exists():
            logger.warning(
                f"[orchestrator] Clip file not found: {clip_path!r} ({meta_path.name}) — skipping"
            )
            summary["skipped"] += 1
            continue

        logger.info(
            f"[orchestrator] {meta_path.stem}  stage={orch['stage']}  "
            f"attempt={orch['attempts'] + 1}/{orch['max_attempts']}"
        )

        _mark_running(orch, meta, meta_path)

        try:
            outcome = _process_clip(clip_path, meta_path, meta, orch, game, config)
        except Exception as exc:
            logger.error(f"[orchestrator] {meta_path.stem} failed: {exc}", exc_info=True)
            _handle_failure(orch, meta, meta_path, clip_path, game, config, exc)
            summary["failed"] += 1
            continue

        if outcome == "quarantined":
            summary["quarantined"] += 1
        elif outcome in ("advanced", "review_ready", "rejected"):
            summary["processed"] += 1
        else:
            summary["skipped"] += 1

    logger.info(
        f"[orchestrator] {game} complete — "
        f"processed={summary['processed']} quarantined={summary['quarantined']} "
        f"failed={summary['failed']} skipped={summary['skipped']}"
    )
    return summary


# ---------------------------------------------------------------------------
# Stage dispatch
# ---------------------------------------------------------------------------

def _process_clip(
    clip_path: str,
    meta_path: Path,
    meta: dict,
    orch: dict,
    game: str,
    config: dict,
) -> str:
    """Execute the next pending stage for one clip. Returns an outcome string."""
    stage = orch["stage"]

    if stage == "detect":
        try:
            _run_detect_stage(clip_path, game, config)
        except _QuarantineEarlyExit:
            # Clip already moved to quarantine/ inside the stage; just record it.
            _advance(orch, meta, meta_path, "quarantined")
            return "quarantined"
        except _LanguageFilterExit:
            _advance(orch, meta, meta_path, "rejected")
            return "rejected"
        _advance(orch, meta, meta_path, "judge")
        return "advanced"

    if stage == "judge":
        decision = _run_judge_stage(clip_path, game, config)
        if decision == "accept":
            _advance(orch, meta, meta_path, "process")
            return "advanced"
        if decision == "quarantine":
            _advance(orch, meta, meta_path, "quarantined")
            return "quarantined"
        # decision == "reject"
        _advance(orch, meta, meta_path, "rejected")
        return "rejected"

    if stage == "process":
        _run_process_stage(clip_path, meta_path, game, config)
        _advance(orch, meta, meta_path, "review_ready")
        return "review_ready"

    logger.warning(f"[orchestrator] Unknown stage '{stage}' for {meta_path.stem} — skipping")
    return "skipped"


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------

def _run_detect_stage(clip_path: str, game: str, config: dict) -> None:
    """Run all enabled detector stages in pipeline order. Each is idempotent."""
    from pipeline.audio_detector import run_audio_detector
    from pipeline.kill_feed import run_kill_feed_parser
    from pipeline.weapon_detector import run_weapon_detector
    from pipeline.roi_matcher import run_roi_matcher
    from pipeline.hook_enforcer import run_hook_enforcer
    from pipeline.transcription import run_transcription
    from pipeline.feature_extraction import run_feature_extraction

    path = Path(clip_path)

    if config.get("audio_detector", {}).get("enabled"):
        run_audio_detector(path, game, config)

    if config.get("kill_feed", {}).get("enabled"):
        run_kill_feed_parser(path, game, config)

    if config.get("weapon_detector", {}).get("enabled"):
        wd = run_weapon_detector(path, game, config)
        if config["weapon_detector"].get("require_detection") and not wd.get("weapon_id"):
            from utils.file_utils import move_to_quarantine
            move_to_quarantine(path, game, config, reason="missing_context")
            raise _QuarantineEarlyExit("require_detection=true but no weapon found")

    if config.get("roi_matcher", {}).get("enabled"):
        run_roi_matcher(path, game, config)

    if config.get("hook_enforcer", {}).get("enabled"):
        run_hook_enforcer(path, game, config)

    transcript = run_transcription(clip_path, config)
    if transcript is None:
        raise _LanguageFilterExit("language filter rejected clip")

    run_feature_extraction(clip_path, transcript, config)


def _run_judge_stage(clip_path: str, game: str, config: dict) -> str:
    """Run clip_judge and move quarantined clips. Returns 'accept', 'quarantine', or 'reject'."""
    from pipeline.clip_judge import run_clip_judge
    from utils.file_utils import move_to_quarantine

    worthiness = run_clip_judge(clip_path, game, config)
    decision = worthiness.get("decision", "quarantine")

    if decision == "quarantine":
        reason = worthiness.get("quarantine_reason") or "low_confidence"
        try:
            move_to_quarantine(Path(clip_path), game, config, reason=reason)
        except (OSError, FileNotFoundError) as exc:
            logger.warning(f"[orchestrator] Could not move clip to quarantine: {exc}")

    return decision


def _run_process_stage(clip_path: str, meta_path: Path, game: str, config: dict) -> None:
    """Run the post-judge processing chain for an accepted clip."""
    from pipeline.decision_engine import select_template
    from pipeline.processing import run_processing
    from pipeline.scoring import run_scoring
    from pipeline.title_engine import generate_title
    from utils.metadata_injector import inject_metadata

    # Re-load meta so all detector outputs written since stage init are included
    meta = _load_meta(meta_path)

    # Idempotent: if scoring already exists this clip was processed by --game
    if meta.get("scoring"):
        logger.debug(f"[orchestrator] process stage already done for {meta_path.stem} — skipping render")
        return

    template = select_template(meta, config)
    processed_path = run_processing(clip_path, template, meta, config)
    run_scoring(processed_path, meta, config)

    if config.get("title_engine", {}).get("enabled"):
        title_result = generate_title(Path(clip_path), game, config)
        logger.info(f"[orchestrator] Proposed title: {title_result.get('title')!r}")
        inject_metadata(Path(processed_path), Path(clip_path), config)


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def _ensure_orch_block(meta: dict, meta_path: Path, max_attempts: int) -> dict:
    """Return the orchestrator block, creating it if absent.

    For clips already processed by --game, the initial stage is inferred from
    existing meta.json state so the orchestrator doesn't re-run completed work.
    """
    if "orchestrator" not in meta:
        stage = _infer_stage(meta)
        meta["orchestrator"] = {
            "stage": stage,
            "status": "done" if stage in TERMINAL_STAGES else "pending",
            "attempts": 0,
            "max_attempts": max_attempts,
            "last_error": None,
            "last_run_at": None,
        }
        _save_meta(meta_path, meta)
    return meta["orchestrator"]


def _infer_stage(meta: dict) -> str:
    """Infer the correct starting stage from an existing (pre-orchestrator) meta dict."""
    worthiness = meta.get("worthiness") or {}
    decision = worthiness.get("decision")
    if meta.get("scoring"):
        return "review_ready"
    if decision == "quarantine":
        return "quarantined"
    if decision == "reject":
        return "rejected"
    return "detect"


def _zombie_check(orch: dict, meta: dict, meta_path: Path, timeout_minutes: int) -> None:
    """Reset clips stuck in 'running' status longer than timeout_minutes."""
    if orch.get("status") != "running":
        return
    last_run = orch.get("last_run_at")
    if not last_run:
        orch["status"] = "pending"
        _save_meta(meta_path, meta)
        return
    try:
        age = (datetime.now() - datetime.fromisoformat(last_run)).total_seconds()
        if age > timeout_minutes * 60:
            logger.warning(
                f"[orchestrator] Zombie reset: {meta_path.stem} "
                f"(stuck running {age / 60:.1f}m)"
            )
            orch["status"] = "pending"
            _save_meta(meta_path, meta)
    except (ValueError, TypeError):
        orch["status"] = "pending"
        _save_meta(meta_path, meta)


def _handle_failure(
    orch: dict,
    meta: dict,
    meta_path: Path,
    clip_path: str,
    game: str,
    config: dict,
    error: Exception,
) -> None:
    """Increment attempt counter; quarantine clip when max_attempts is exhausted."""
    orch["attempts"] = orch.get("attempts", 0) + 1
    orch["last_error"] = str(error)[:500]
    orch["last_run_at"] = datetime.now().isoformat(timespec="seconds")

    if orch["attempts"] >= orch.get("max_attempts", _DEFAULT_MAX_ATTEMPTS):
        orch["stage"] = "failed"
        orch["status"] = "failed"
        logger.warning(
            f"[orchestrator] {meta_path.stem} exhausted {orch['attempts']} attempts "
            f"— quarantining. Last error: {orch['last_error']}"
        )
        try:
            from utils.file_utils import move_to_quarantine
            move_to_quarantine(Path(clip_path), game, config, reason="default")
        except (OSError, FileNotFoundError) as exc:
            logger.warning(f"[orchestrator] Could not quarantine failed clip: {exc}")
    else:
        orch["status"] = "pending"  # will be retried on next run

    _save_meta(meta_path, meta)


def _mark_running(orch: dict, meta: dict, meta_path: Path) -> None:
    orch["status"] = "running"
    orch["last_run_at"] = datetime.now().isoformat(timespec="seconds")
    _save_meta(meta_path, meta)


def _advance(orch: dict, meta: dict, meta_path: Path, next_stage: str) -> None:
    orch["stage"] = next_stage
    orch["status"] = "done" if next_stage in TERMINAL_STAGES else "pending"
    orch["last_run_at"] = datetime.now().isoformat(timespec="seconds")
    _save_meta(meta_path, meta)
    logger.info(f"[orchestrator] {meta_path.stem} → {next_stage}")


def _load_meta(meta_path: Path) -> dict:
    return json.loads(meta_path.read_text())


def _save_meta(meta_path: Path, meta: dict) -> None:
    meta_path.write_text(json.dumps(meta, indent=2))


# ---------------------------------------------------------------------------
# Sentinel exceptions (used inside _run_detect_stage only)
# ---------------------------------------------------------------------------

class _QuarantineEarlyExit(Exception):
    """Raised when a detect sub-stage already moved the clip to quarantine/."""


class _LanguageFilterExit(Exception):
    """Raised when transcription rejects the clip via the language filter."""
