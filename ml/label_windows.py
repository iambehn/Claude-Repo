"""
ml/label_windows.py — Auto-labeler for window fusion model training data.

Cross-references proxy scanner window records in data/training_sets/windows/*.jsonl
with clip review outcomes in inbox/accepted/rejected/ to assign human_label.decision.

Matching strategy: window records store time_window.start_sec; downloaded clips are
named proxy_{int(start):06d}_{int(end):06d}.mp4 with a sidecar meta.json that has
proxy_window_start. The labeler finds the meta.json and reads review_status / bucket.

Label assignment:
  accepted/ bucket OR review_status == "accepted"  → "accept"
  rejected/ bucket OR review_status == "rejected"  → "reject"
  quarantine/ or not yet reviewed                  → skip (label stays null)
  window never downloaded (no clip file anywhere)  → "skip" if --label-undownloaded

Usage:
  python run.py --label-windows                   # label all games
  python run.py --label-windows marvel_rivals     # label one game
  python run.py --label-windows --label-undownloaded  # also label undownloaded windows
"""

from __future__ import annotations

import json
from pathlib import Path

from utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_TRAINING_DIR = "data/training_sets/windows"
_DECIDED_BUCKETS = {
    "accepted": "accept",
    "rejected": "reject",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def label_windows(
    game_filter: str | None = None,
    config: dict | None = None,
    label_undownloaded: bool = False,
) -> dict:
    """
    Scan JSONL window records and backfill human_label.decision from clip outcomes.

    Args:
        game_filter:        None / "all" → all games; game slug → one game.
        config:             Full config dict (reads paths.* buckets).
        label_undownloaded: If True, windows with no matching clip anywhere are
                            labeled "skip" with reason "never_downloaded". These
                            are weak negatives — the heuristic filtered them out.

    Returns:
        {labeled: int, already_labeled: int, not_found: int,
         skipped_unreviewed: int, total: int}
    """
    cfg = config or {}
    training_dir = Path(
        cfg.get("training", {}).get("output_dir", "data/training_sets")
    ) / "windows"

    if not training_dir.exists():
        logger.info("[label_windows] No window training dir found — nothing to label.")
        return {"labeled": 0, "already_labeled": 0, "not_found": 0,
                "skipped_unreviewed": 0, "total": 0}

    game = None if (game_filter is None or game_filter == "all") else game_filter

    # Build a lookup: (game, start_sec_int) → decision
    clip_outcomes = _build_clip_outcome_index(cfg, game)
    logger.info(
        f"[label_windows] Found {len(clip_outcomes)} reviewed clip(s) "
        f"{'for ' + game if game else 'across all games'}"
    )

    labeled = already_labeled = not_found = skipped_unreviewed = 0

    for jsonl_path in sorted(training_dir.glob("*.jsonl")):
        rows, changed = _process_file(
            jsonl_path, game, clip_outcomes, label_undownloaded
        )
        if changed:
            jsonl_path.write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n",
                encoding="utf-8",
            )

        for r in rows:
            decision = r.get("human_label", {}).get("decision")
            reason = r.get("human_label", {}).get("reason", "")
            if decision is not None and reason == "_just_labeled":
                labeled += 1
            elif decision is not None:
                already_labeled += 1
            elif reason == "_not_found":
                not_found += 1
            else:
                skipped_unreviewed += 1

    # Strip internal tracking markers after counting
    _strip_markers(training_dir)

    total = labeled + already_labeled + not_found + skipped_unreviewed
    logger.info(
        f"[label_windows] Done — "
        f"{labeled} newly labeled  {already_labeled} already labeled  "
        f"{not_found} not found  {skipped_unreviewed} unreviewed  "
        f"({total} total)"
    )
    return {
        "labeled": labeled,
        "already_labeled": already_labeled,
        "not_found": not_found,
        "skipped_unreviewed": skipped_unreviewed,
        "total": total,
    }


# ---------------------------------------------------------------------------
# Outcome index
# ---------------------------------------------------------------------------

def _build_clip_outcome_index(
    config: dict,
    game_filter: str | None,
) -> dict[tuple[str, int], str]:
    """
    Returns {(game, start_sec_int): decision} by scanning all clip buckets.

    Decision priority: accepted > rejected > quarantine (quarantine = skip).
    Clips whose review is still pending (inbox, no review_status) are excluded.
    """
    paths = config.get("paths", {})
    outcomes: dict[tuple[str, int], str] = {}

    # Definitive buckets first (accepted / rejected)
    for bucket, decision in _DECIDED_BUCKETS.items():
        bucket_root = Path(paths.get(bucket, bucket))
        if not bucket_root.exists():
            continue
        for game_dir in bucket_root.iterdir():
            if not game_dir.is_dir():
                continue
            g = game_dir.name
            if game_filter and g != game_filter:
                continue
            for meta_path in game_dir.glob("proxy_*.meta.json"):
                start = _start_from_stem(meta_path.stem)
                if start is not None:
                    key = (g, start)
                    # accepted wins over rejected if clip somehow appears in both
                    if key not in outcomes or decision == "accept":
                        outcomes[key] = decision

    # inbox / processing: check review_status in meta.json
    for bucket in ("inbox", "processing"):
        bucket_root = Path(paths.get(bucket, bucket))
        if not bucket_root.exists():
            continue
        for game_dir in bucket_root.iterdir():
            if not game_dir.is_dir():
                continue
            g = game_dir.name
            if game_filter and g != game_filter:
                continue
            for meta_path in game_dir.glob("proxy_*.meta.json"):
                start = _start_from_stem(meta_path.stem)
                if start is None:
                    continue
                try:
                    meta = json.loads(meta_path.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                rs = meta.get("review_status", "")
                decision = None
                if rs in ("accepted",):
                    decision = "accept"
                elif rs in ("rejected",):
                    decision = "reject"
                if decision:
                    key = (g, start)
                    if key not in outcomes or decision == "accept":
                        outcomes[key] = decision

    return outcomes


def _start_from_stem(stem: str) -> int | None:
    """Extract start_sec from proxy_{start:06d}_{end:06d} stem."""
    parts = stem.split("_")
    # stem looks like: proxy_000120_000155  (possibly with a suffix)
    if len(parts) >= 3 and parts[0] == "proxy":
        try:
            return int(parts[1])
        except ValueError:
            pass
    return None


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def _process_file(
    path: Path,
    game_filter: str | None,
    clip_outcomes: dict[tuple[str, int], str],
    label_undownloaded: bool,
) -> tuple[list[dict], bool]:
    """
    Load JSONL, update human_label.decision where possible.
    Returns (rows, changed).
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        logger.warning(f"[label_windows] Could not read {path.name}: {exc}")
        return [], False

    rows: list[dict] = []
    changed = False

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue

        game = row.get("game", "")
        if game_filter and game != game_filter:
            rows.append(row)
            continue

        hl = row.setdefault("human_label", {})
        if hl.get("decision") is not None:
            rows.append(row)
            continue

        start_sec = row.get("time_window", {}).get("start_sec")
        if start_sec is None:
            rows.append(row)
            continue

        start_int = int(start_sec)
        key = (game, start_int)
        decision = clip_outcomes.get(key)

        if decision is not None:
            hl["decision"] = decision
            hl["reason"] = "_just_labeled"  # stripped after counting
            hl["reviewer_confidence"] = 1.0
            changed = True
        elif label_undownloaded:
            # Weak negative: window was scored but never downloaded
            hl["decision"] = "skip"
            hl["reason"] = "never_downloaded"
            hl["reviewer_confidence"] = 0.5
            changed = True
        else:
            hl["reason"] = "_not_found"  # temp marker for counting, stripped after

        rows.append(row)

    return rows, changed


def _strip_markers(training_dir: Path) -> None:
    """Remove internal _just_labeled / _not_found markers written during counting."""
    for jsonl_path in sorted(training_dir.glob("*.jsonl")):
        try:
            lines = jsonl_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        cleaned = []
        dirty = False
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                cleaned.append(line)
                continue
            hl = row.get("human_label", {})
            reason = hl.get("reason", "")
            if reason in ("_just_labeled", "_not_found"):
                hl.pop("reason", None)
                dirty = True
            cleaned.append(json.dumps(row))
        if dirty:
            jsonl_path.write_text("\n".join(cleaned) + "\n", encoding="utf-8")
