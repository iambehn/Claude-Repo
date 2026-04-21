"""
File system helpers shared across pipeline stages.

Handles directory creation, clip movement between pipeline folders,
and deriving the game key from a clip's path.
"""

import shutil
from pathlib import Path

from utils.logger import get_logger

logger = get_logger(__name__)

GAMES = ["arc_raiders", "marvel_rivals", "deadlock"]


def ensure_dirs(config: dict) -> None:
    """Create all pipeline folders and per-game subfolders if they don't exist.

    Should be called once at startup before any stage runs.

    Args:
        config: Full parsed config.yaml dict.
    """
    stage_folders = [
        config["paths"]["inbox"],
        config["paths"]["quarantine"],
        config["paths"]["processing"],
        config["paths"]["accepted"],
        config["paths"]["rejected"],
    ]
    for folder in stage_folders:
        for game in config["games"]:
            Path(folder, game).mkdir(parents=True, exist_ok=True)

    Path(config["paths"]["assets"], "music").mkdir(parents=True, exist_ok=True)
    Path(config["paths"]["logs"]).mkdir(parents=True, exist_ok=True)

    logger.debug("Pipeline directories verified.")


def get_game_from_path(clip_path: str | Path) -> str | None:
    """Derive the game key from a clip's file path.

    Looks for a known game folder name in the path components.

    Args:
        clip_path: Path to the clip file.

    Returns:
        Game key string (e.g. 'arc_raiders') or None if not found.
    """
    parts = Path(clip_path).parts
    for part in parts:
        if part in GAMES:
            return part
    return None


QUARANTINE_REASONS = (
    "default",
    "missing_context",
    "hook_not_resolved",
    "low_confidence",
    "ui_drift",
    "needs_roi_template",
)


def move_to_quarantine(
    clip_path: str | Path,
    game: str,
    config: dict,
    reason: str = "default",
) -> Path:
    """Move a clip to a bucketed quarantine folder.

    Clips are routed to quarantine/{game}/{reason}/ so enrichment work can be
    targeted. Unknown reasons fall through to "default".
    """
    if reason not in QUARANTINE_REASONS:
        logger.warning(f"Unknown quarantine reason '{reason}', using 'default'")
        reason = "default"
    src = Path(clip_path)
    dest_dir = Path(config["paths"]["quarantine"], game, reason)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    shutil.move(str(src), str(dest))
    logger.info(f"Quarantined ({reason}): {src.name} → {dest}")
    return dest


def move_clip(clip_path: str | Path, destination_stage: str, game: str, config: dict) -> Path:
    """Move a clip between pipeline stage folders.

    Args:
        clip_path: Current path to the clip.
        destination_stage: Config path key for the target folder
                           (e.g. 'processing', 'accepted', 'rejected').
        game: Game key (e.g. 'arc_raiders').
        config: Full parsed config.yaml dict.

    Returns:
        New path of the clip in the destination folder.
    """
    src = Path(clip_path)
    dest_dir = Path(config["paths"][destination_stage], game)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    shutil.move(str(src), str(dest))
    logger.info(f"Moved {src.name}: → {destination_stage}/{game}/")
    return dest
