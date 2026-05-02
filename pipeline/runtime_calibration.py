"""
pipeline/runtime_calibration.py — Runtime Review Calibration V1

Reads reviewed .meta.json sidecars, compares current runtime scores/actions
against human review outcomes, and emits a diagnostics report for tuning.

Report-only: does not mutate sidecars, config, or exports.

Sidecar eligibility:
  - has `worthiness` block  (schema_version == runtime_analysis_v1 equivalent)
  - `review_status` in {"accepted", "rejected"}

Field mapping (plan concept → meta.json field):
  highlight_score         = worthiness.final_score
  recommended_action      = derived from worthiness.decision
                              "accept"     → "highlight_candidate"
                              "quarantine" → "inspect"
                              "reject"     → "skip"
  runtime_review_status   = review_status ("accepted" → approved, "rejected" → rejected)
  medal_seen              = atomic event "medal_awarded" OR roi_match roi_name "medal_popup"
  ability_seen            = atomic event "ultimate_used" OR roi_match roi_name "ability_ultimate"
  pov_character_identified = weapon_detection.weapon_id is not None / non-empty

Run:
    python run.py --calibrate-runtime-review inbox/marvel_rivals
    python run.py --calibrate-runtime-review . --calibrate-game marvel_rivals
    python run.py --calibrate-runtime-review . --calibrate-output report.json
    python run.py --calibrate-runtime-review . --calibrate-debug-dir data/calibration/
"""
from __future__ import annotations

import csv
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from pipeline import game_pack
from utils.logger import get_logger

logger = get_logger(__name__)

_MIN_REVIEWED_DEFAULT = 5

_DECISION_TO_ACTION: dict[str, str] = {
    "accept":     "highlight_candidate",
    "quarantine": "inspect",
    "reject":     "skip",
}
_ACTION_LABELS = ["highlight_candidate", "inspect", "skip"]

_KNOWN_GAMES = ("marvel_rivals", "deadlock", "arc_raiders")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CalibrationRecord:
    sidecar_path: str
    game: str
    clip_id: str
    review_label: str           # "approved" | "rejected"
    highlight_score: float      # worthiness.final_score (0.0–1.0)
    recommended_action: str     # "highlight_candidate" | "inspect" | "skip"
    context_confidence: float
    hook_confidence: float
    postability_score: float
    medal_seen: bool
    ability_seen: bool
    pov_character_identified: bool
    kill_detected: bool
    kill_count: int
    headshot_count: int
    audio_spike_count: int
    total_excitement: float


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calibrate_runtime_review(
    sidecar_root: str | Path,
    config: dict,
    game_filter: str | None = None,
    min_reviewed: int = _MIN_REVIEWED_DEFAULT,
    include_unreviewed: bool = False,
) -> dict:
    """Scan sidecars, compute diagnostics, return structured calibration result.

    Report-only: does not mutate any sidecar or config file.

    Returns a dict with keys:
        ok, status, sidecar_root, game_filter, scanned_sidecar_count,
        reviewed_sidecar_count, approved_count, rejected_count,
        skipped_sidecar_count, current_scoring, diagnostics,
        recommendations, warnings
    """
    root = Path(sidecar_root)
    if not root.exists():
        return {
            "ok": False,
            "status": "error",
            "error": f"sidecar_root does not exist: {root}",
            "sidecar_root": str(root),
            "game_filter": game_filter,
            "scanned_sidecar_count": 0,
            "reviewed_sidecar_count": 0,
            "approved_count": 0,
            "rejected_count": 0,
            "skipped_sidecar_count": 0,
            "current_scoring": {},
            "diagnostics": {},
            "recommendations": {},
            "warnings": [],
        }

    records, skipped, warnings, unreviewed_count = _scan_sidecars(
        root, game_filter, include_unreviewed
    )

    approved = [r for r in records if r.review_label == "approved"]
    rejected = [r for r in records if r.review_label == "rejected"]
    reviewed_count = len(approved) + len(rejected)
    scanned_count = reviewed_count + skipped + unreviewed_count

    current_scoring = _load_current_scoring(config, game_filter)

    if reviewed_count < min_reviewed:
        return {
            "ok": True,
            "status": "insufficient_review_data",
            "sidecar_root": str(root),
            "game_filter": game_filter,
            "scanned_sidecar_count": scanned_count,
            "reviewed_sidecar_count": reviewed_count,
            "approved_count": len(approved),
            "rejected_count": len(rejected),
            "skipped_sidecar_count": skipped,
            "current_scoring": current_scoring,
            "diagnostics": {},
            "recommendations": {
                "threshold_observations": [],
                "weight_observations": [],
                "candidate_threshold_ranges": {},
                "candidate_weight_adjustments": {},
                "data_quality_notes": [
                    f"Only {reviewed_count} reviewed sidecar(s) found; "
                    f"need at least {min_reviewed} for tuning output. "
                    "Review more clips via the review UI, then re-run."
                ],
            },
            "warnings": warnings,
        }

    diagnostics = _compute_diagnostics(approved, rejected, current_scoring)
    recommendations = _generate_recommendations(
        diagnostics, current_scoring, approved, rejected
    )

    return {
        "ok": True,
        "status": "success",
        "sidecar_root": str(root),
        "game_filter": game_filter,
        "scanned_sidecar_count": scanned_count,
        "reviewed_sidecar_count": reviewed_count,
        "approved_count": len(approved),
        "rejected_count": len(rejected),
        "skipped_sidecar_count": skipped,
        "current_scoring": current_scoring,
        "diagnostics": diagnostics,
        "recommendations": recommendations,
        "warnings": warnings,
    }


def write_calibration_artifacts(
    result: dict,
    output_path: str | Path | None = None,
    debug_dir: str | Path | None = None,
) -> list[str]:
    """Write calibration output files. Returns list of paths written."""
    written: list[str] = []

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        # Strip internal _records_raw before writing top-level report
        report = _strip_internal_keys(result)
        out.write_text(json.dumps(report, indent=2))
        written.append(str(out))
        logger.info(f"[calibration] Report written to {out}")

    if debug_dir:
        ddir = Path(debug_dir)
        ddir.mkdir(parents=True, exist_ok=True)

        report_path = ddir / "runtime_calibration_report.json"
        report_path.write_text(json.dumps(_strip_internal_keys(result), indent=2))
        written.append(str(report_path))

        diag = result.get("diagnostics") or {}

        records_raw = diag.get("_records_raw") or []
        if records_raw:
            clips_csv = ddir / "reviewed_clips.csv"
            _write_csv(clips_csv, records_raw)
            written.append(str(clips_csv))

        action_outcomes = diag.get("action_bucket_outcomes") or {}
        if action_outcomes:
            rows = [
                {
                    "action": action,
                    "approved": counts.get("approved", 0),
                    "rejected": counts.get("rejected", 0),
                    "total": counts.get("total", 0),
                }
                for action, counts in action_outcomes.items()
            ]
            buckets_csv = ddir / "score_buckets.csv"
            _write_csv(buckets_csv, rows)
            written.append(str(buckets_csv))

        event_incidence = diag.get("event_incidence") or {}
        if event_incidence:
            rows = [{"event_type": evt, **rates} for evt, rates in event_incidence.items()]
            events_csv = ddir / "event_diagnostics.csv"
            _write_csv(events_csv, rows)
            written.append(str(events_csv))

        warnings = result.get("warnings") or []
        if warnings:
            warn_path = ddir / "warnings.json"
            warn_path.write_text(json.dumps(warnings, indent=2))
            written.append(str(warn_path))

        logger.info(f"[calibration] Debug artifacts written to {ddir}/")

    return written


# ---------------------------------------------------------------------------
# Sidecar scanning
# ---------------------------------------------------------------------------

def _scan_sidecars(
    root: Path,
    game_filter: str | None,
    include_unreviewed: bool,
) -> tuple[list[CalibrationRecord], int, list[str], int]:
    """Scan root recursively for .meta.json files.

    Returns: (records, skipped_count, warnings, unreviewed_count)
    """
    records: list[CalibrationRecord] = []
    warnings: list[str] = []
    skipped = 0
    unreviewed_count = 0

    for meta_path in sorted(root.rglob("*.meta.json")):
        try:
            meta = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            warnings.append(f"Malformed JSON in {meta_path.name}: {exc}")
            skipped += 1
            continue

        if "worthiness" not in meta:
            warnings.append(
                f"Skipping {meta_path.name}: no 'worthiness' block (clip not yet judged)"
            )
            skipped += 1
            continue

        game = meta.get("game") or _infer_game_from_path(meta_path)
        if game_filter and game != game_filter:
            skipped += 1
            continue

        review_status = (meta.get("review_status") or "").lower()
        if review_status not in ("accepted", "rejected"):
            if include_unreviewed:
                unreviewed_count += 1
            else:
                skipped += 1
            continue

        review_label = "approved" if review_status == "accepted" else "rejected"

        try:
            rec = _extract_record(meta, meta_path, game, review_label)
            records.append(rec)
        except Exception as exc:
            warnings.append(
                f"Could not extract calibration fields from {meta_path.name}: {exc}"
            )
            skipped += 1

    return records, skipped, warnings, unreviewed_count


def _infer_game_from_path(path: Path) -> str:
    for part in path.parts:
        if part in _KNOWN_GAMES:
            return part
    return "unknown"


def _extract_record(
    meta: dict, path: Path, game: str, review_label: str
) -> CalibrationRecord:
    worthiness = meta.get("worthiness") or {}
    kf = meta.get("kill_feed") or {}
    ad = meta.get("audio_detector") or meta.get("audio_events") or {}
    ae = meta.get("atomic_events") or {}
    roi = meta.get("roi_matches") or {}
    wd = meta.get("weapon_detection") or {}

    decision = worthiness.get("decision", "")
    recommended_action = _DECISION_TO_ACTION.get(decision, "inspect")
    highlight_score = float(worthiness.get("final_score") or 0.0)

    ae_events = ae.get("events") or []
    ae_types = {e.get("event_type", "") for e in ae_events}
    roi_names = {m.get("roi_name", "") for m in (roi.get("matches") or [])}

    medal_seen   = "medal_awarded" in ae_types or "medal_popup" in roi_names
    ability_seen = "ultimate_used" in ae_types or "ability_ultimate" in roi_names

    weapon_id = (wd.get("weapon_id") or "").strip()
    pov_char = bool(weapon_id)

    kill_count     = int(kf.get("kill_count") or 0)
    headshot_count = int(kf.get("headshot_count") or 0)
    kill_detected  = kill_count > 0 or headshot_count > 0

    spikes = ad.get("spike_timestamps") or []
    audio_spike_count = len(spikes) if isinstance(spikes, list) else 0

    total_excitement = float(ae.get("total_excitement") or 0.0)

    return CalibrationRecord(
        sidecar_path=str(path),
        game=game,
        clip_id=meta.get("clip_id") or path.stem,
        review_label=review_label,
        highlight_score=highlight_score,
        recommended_action=recommended_action,
        context_confidence=float(worthiness.get("context_confidence") or 0.0),
        hook_confidence=float(worthiness.get("hook_confidence") or 0.0),
        postability_score=float(worthiness.get("postability_score") or 0.0),
        medal_seen=medal_seen,
        ability_seen=ability_seen,
        pov_character_identified=pov_char,
        kill_detected=kill_detected,
        kill_count=kill_count,
        headshot_count=headshot_count,
        audio_spike_count=audio_spike_count,
        total_excitement=total_excitement,
    )


# ---------------------------------------------------------------------------
# Scoring config loader
# ---------------------------------------------------------------------------

def _load_current_scoring(config: dict, game_filter: str | None) -> dict:
    """Load current scoring thresholds from weights.yaml for the given game."""
    game = game_filter
    if not game:
        games = list((config.get("games") or {}).keys())
        game = games[0] if games else None

    if game:
        try:
            pack = game_pack.load(game)
            w = pack.weights
            thresholds = w.get("thresholds") or {}
            composite = w.get("composite") or {}
            return {
                "highlight_candidate_threshold": thresholds.get("accept", 0.65),
                "inspect_min_threshold":         thresholds.get("reject", 0.35),
                "min_context_threshold":         thresholds.get("min_context", 0.40),
                "hook_gate_threshold":           thresholds.get("hook_gate", 0.40),
                "reliable_confidence_threshold": thresholds.get("reliable_confidence", 0.60),
                "context_weight":                composite.get("context_weight", 0.30),
                "hook_weight":                   composite.get("hook_weight", 0.25),
                "postability_weight":            composite.get("postability_weight", 0.45),
                "game": game,
            }
        except Exception as exc:
            logger.debug(f"[calibration] Could not load weights.yaml for {game}: {exc}")

    return {
        "highlight_candidate_threshold": 0.65,
        "inspect_min_threshold":         0.35,
        "min_context_threshold":         0.40,
        "hook_gate_threshold":           0.40,
        "reliable_confidence_threshold": 0.60,
        "context_weight":    0.30,
        "hook_weight":       0.25,
        "postability_weight": 0.45,
        "game": game or "default",
    }


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def _compute_diagnostics(
    approved: list[CalibrationRecord],
    rejected: list[CalibrationRecord],
    current_scoring: dict,
) -> dict:
    all_records = approved + rejected
    hi = current_scoring["highlight_candidate_threshold"]
    lo = current_scoring["inspect_min_threshold"]

    score_dist = {
        "approved": _score_stats([r.highlight_score for r in approved]) if approved else {},
        "rejected": _score_stats([r.highlight_score for r in rejected]) if rejected else {},
        "all":      _score_stats([r.highlight_score for r in all_records]),
    }

    bucket_outcomes: dict[str, dict[str, int]] = {
        a: {"approved": 0, "rejected": 0, "total": 0} for a in _ACTION_LABELS
    }
    for r in all_records:
        action = r.recommended_action
        if action not in bucket_outcomes:
            bucket_outcomes[action] = {"approved": 0, "rejected": 0, "total": 0}
        bucket_outcomes[action][r.review_label] += 1
        bucket_outcomes[action]["total"] += 1

    event_types = [
        "medal_seen",
        "ability_seen",
        "pov_character_identified",
        "kill_detected",
    ]
    event_incidence: dict[str, dict] = {}
    for evt in event_types:
        approved_with = sum(1 for r in approved if getattr(r, evt))
        rejected_with = sum(1 for r in rejected if getattr(r, evt))
        event_incidence[evt] = {
            "approved_rate":  round(approved_with / len(approved), 3) if approved else 0.0,
            "rejected_rate":  round(rejected_with / len(rejected), 3) if rejected else 0.0,
            "approved_count": approved_with,
            "rejected_count": rejected_with,
        }

    approved_below_hi = [r.clip_id for r in approved if r.highlight_score < hi]
    rejected_above_hi = [r.clip_id for r in rejected if r.highlight_score >= hi]
    approved_below_lo = [r.clip_id for r in approved if r.highlight_score < lo]

    fn_rate = round(len(approved_below_hi) / len(approved), 3) if approved else 0.0
    fp_rate = round(len(rejected_above_hi) / len(rejected), 3) if rejected else 0.0

    threshold_analysis = {
        "highlight_candidate_threshold":      hi,
        "inspect_min_threshold":              lo,
        "approved_below_highlight_threshold": approved_below_hi,
        "approved_below_highlight_count":     len(approved_below_hi),
        "rejected_above_highlight_threshold": rejected_above_hi,
        "rejected_above_highlight_count":     len(rejected_above_hi),
        "approved_below_inspect_threshold":   approved_below_lo,
        "approved_below_inspect_count":       len(approved_below_lo),
        "false_negative_rate":                fn_rate,
        "false_positive_rate":                fp_rate,
    }

    weight_analysis = _compute_weight_analysis(approved, rejected)

    return {
        "score_distribution":    score_dist,
        "action_bucket_outcomes": bucket_outcomes,
        "event_incidence":        event_incidence,
        "threshold_analysis":     threshold_analysis,
        "weight_analysis":        weight_analysis,
        "_records_raw":           [asdict(r) for r in all_records],
    }


def _score_stats(scores: list[float]) -> dict:
    if not scores:
        return {}
    n = len(scores)
    sorted_s = sorted(scores)
    return {
        "count":  n,
        "mean":   round(statistics.mean(scores), 4),
        "median": round(statistics.median(scores), 4),
        "min":    round(sorted_s[0], 4),
        "max":    round(sorted_s[-1], 4),
        "p25":    round(sorted_s[max(0, int(n * 0.25) - 1)], 4),
        "p75":    round(sorted_s[min(n - 1, int(n * 0.75))], 4),
        "stdev":  round(statistics.stdev(scores), 4) if n >= 2 else 0.0,
    }


def _compute_weight_analysis(
    approved: list[CalibrationRecord],
    rejected: list[CalibrationRecord],
) -> dict:
    all_records = approved + rejected

    per_event_approval_rate: dict[str, dict] = {}
    for evt in [
        "medal_seen", "ability_seen", "pov_character_identified", "kill_detected"
    ]:
        with_evt    = [r for r in all_records if getattr(r, evt)]
        without_evt = [r for r in all_records if not getattr(r, evt)]
        apr_with = (
            sum(1 for r in with_evt if r.review_label == "approved") / len(with_evt)
            if with_evt else None
        )
        apr_without = (
            sum(1 for r in without_evt if r.review_label == "approved") / len(without_evt)
            if without_evt else None
        )
        per_event_approval_rate[evt] = {
            "approval_rate_when_present": round(apr_with, 3) if apr_with is not None else None,
            "approval_rate_when_absent":  round(apr_without, 3) if apr_without is not None else None,
            "n_present": len(with_evt),
            "n_absent":  len(without_evt),
        }

    score_contributions: dict[str, dict] = {}
    for comp in ["context_confidence", "hook_confidence", "postability_score"]:
        app_vals = [getattr(r, comp) for r in approved]
        rej_vals = [getattr(r, comp) for r in rejected]
        score_contributions[comp] = {
            "approved_mean": round(statistics.mean(app_vals), 4) if app_vals else None,
            "rejected_mean": round(statistics.mean(rej_vals), 4) if rej_vals else None,
        }

    exc_app = [r.total_excitement for r in approved]
    exc_rej = [r.total_excitement for r in rejected]
    excitement_analysis = {
        "approved_mean_excitement": round(statistics.mean(exc_app), 2) if exc_app else 0.0,
        "rejected_mean_excitement": round(statistics.mean(exc_rej), 2) if exc_rej else 0.0,
    }

    return {
        "per_event_approval_rate": per_event_approval_rate,
        "score_contributions":     score_contributions,
        "excitement_analysis":     excitement_analysis,
    }


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

def _generate_recommendations(
    diagnostics: dict,
    current_scoring: dict,
    approved: list[CalibrationRecord],
    rejected: list[CalibrationRecord],
) -> dict:
    threshold_obs: list[str] = []
    weight_obs:    list[str] = []
    threshold_ranges: dict   = {}
    weight_adj:    dict      = {}
    data_quality:  list[str] = []

    hi = current_scoring["highlight_candidate_threshold"]
    lo = current_scoring["inspect_min_threshold"]
    ta = diagnostics.get("threshold_analysis") or {}
    ei = diagnostics.get("event_incidence") or {}
    wa = diagnostics.get("weight_analysis") or {}
    sd = diagnostics.get("score_distribution") or {}
    bo = diagnostics.get("action_bucket_outcomes") or {}

    fn_rate = ta.get("false_negative_rate", 0.0)
    fp_rate = ta.get("false_positive_rate", 0.0)
    n_approved = len(approved)
    n_rejected = len(rejected)
    total = n_approved + n_rejected
    overall_rate = n_approved / total if total else 0.0

    # --- Threshold observations ---
    if fn_rate > 0.20:
        threshold_obs.append(
            f"highlight_candidate threshold ({hi:.2f}) may be too high: "
            f"{ta.get('approved_below_highlight_count', 0)} approved clips "
            f"({fn_rate:.0%}) fall below it. "
            f"Consider lowering toward {max(lo + 0.05, hi - 0.10):.2f}."
        )
    elif fn_rate <= 0.05 and n_approved >= 3:
        threshold_obs.append(
            f"highlight_candidate threshold ({hi:.2f}) appears well-calibrated: "
            f"only {fn_rate:.0%} of approved clips fall below it."
        )

    if fp_rate > 0.15:
        threshold_obs.append(
            f"highlight_candidate threshold ({hi:.2f}) may be too low: "
            f"{ta.get('rejected_above_highlight_count', 0)} rejected clips "
            f"({fp_rate:.0%}) score above it. "
            f"Consider raising toward {min(1.0, hi + 0.05):.2f}."
        )

    if ta.get("approved_below_inspect_count", 0) > 0:
        threshold_obs.append(
            f"{ta['approved_below_inspect_count']} approved clip(s) score below the "
            f"inspect threshold ({lo:.2f}) and would be hard-skipped. "
            "Consider lowering the inspect threshold."
        )

    if not threshold_obs:
        threshold_obs.append(
            "No strong threshold misalignment detected with current data."
        )

    # --- Candidate threshold ranges ---
    app_p25 = (sd.get("approved") or {}).get("p25")
    rej_p75 = (sd.get("rejected") or {}).get("p75")
    if app_p25 is not None and rej_p75 is not None:
        if rej_p75 < app_p25:
            suggested_hi = round((app_p25 + rej_p75) / 2.0, 2)
            threshold_ranges["highlight_candidate"] = {
                "current": hi,
                "suggested_trial": suggested_hi,
                "rationale": (
                    f"midpoint of approved p25 ({app_p25:.2f}) "
                    f"and rejected p75 ({rej_p75:.2f})"
                ),
            }
        else:
            threshold_ranges["highlight_candidate"] = {
                "current": hi,
                "note": (
                    "approved/rejected score distributions overlap — "
                    "more reviewed data needed for a reliable suggestion"
                ),
            }

    # --- Weight observations ---
    for evt in ["medal_seen", "ability_seen", "pov_character_identified"]:
        evd = ei.get(evt) or {}
        apr_present = evd.get("approval_rate_when_present")
        apr_absent  = evd.get("approval_rate_when_absent")
        if apr_present is not None and apr_absent is not None:
            lift = apr_present - overall_rate
            if lift > 0.20 and evd.get("n_present", 0) >= 3:
                weight_obs.append(
                    f"{evt}: approval rate when present={apr_present:.0%} vs "
                    f"absent={apr_absent:.0%} (lift={lift:+.0%}) — "
                    "signal appears underweighted relative to review outcomes."
                )
                weight_adj[evt] = "increase"
            elif lift < -0.10 and evd.get("n_present", 0) >= 3:
                weight_obs.append(
                    f"{evt}: approval rate when present={apr_present:.0%} — "
                    "this signal does not reliably indicate clip quality."
                )
                weight_adj[evt] = "hold"
            else:
                weight_adj[evt] = "hold"

    exc = (wa.get("excitement_analysis") or {})
    exc_app = exc.get("approved_mean_excitement", 0.0)
    exc_rej = exc.get("rejected_mean_excitement", 0.0)
    if exc_rej > 0 and exc_app > exc_rej * 2:
        weight_obs.append(
            f"Atomic excitement scores are notably higher for approved clips "
            f"(avg {exc_app:.1f} vs rejected {exc_rej:.1f}). "
            "The atomic_events excitement bonus in clip_judge is providing useful signal."
        )
    elif exc_app > 0 and exc_app <= exc_rej:
        weight_obs.append(
            "Atomic excitement scores are not separating approved from rejected clips — "
            "review event weighting in atomic_events.py."
        )

    if not weight_obs:
        weight_obs.append(
            "No strong weight misalignment detected with current data."
        )

    # --- Data quality notes ---
    if total < 10:
        data_quality.append(
            f"Only {total} reviewed clips available. "
            "Recommendations are directional only — collect ≥20 for reliable tuning."
        )
    if total >= 5:
        imbalance = n_approved / total
        if imbalance < 0.2 or imbalance > 0.8:
            data_quality.append(
                f"Class imbalance: {n_approved} approved vs {n_rejected} rejected. "
                "Calibration metrics may be skewed; aim for a more balanced review set."
            )

    inspect_total = (bo.get("inspect") or {}).get("total", 0)
    if inspect_total < 3:
        data_quality.append(
            "Fewer than 3 clips in the 'inspect' (quarantine) band — "
            "inspect threshold analysis is unreliable with this sample."
        )

    return {
        "threshold_observations":       threshold_obs,
        "weight_observations":          weight_obs,
        "candidate_threshold_ranges":   threshold_ranges,
        "candidate_weight_adjustments": weight_adj,
        "data_quality_notes":           data_quality,
    }


# ---------------------------------------------------------------------------
# Print helper
# ---------------------------------------------------------------------------

def print_calibration_report(result: dict) -> None:
    """Print a human-readable calibration report to stdout."""
    status = result.get("status", "unknown")
    game   = result.get("game_filter") or "all games"
    root   = result.get("sidecar_root", "")

    print()
    print("  ╔══════════════════════════════════════════════════════════╗")
    print(f"  ║  Runtime Calibration Report  [{game}]")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print()
    print(f"  Root:   {root}")
    print(f"  Status: {status}")
    print()

    if status in ("error", "insufficient_review_data"):
        recs = result.get("recommendations") or {}
        for note in recs.get("data_quality_notes") or []:
            print(f"  ⚠  {note}")
        _print_warnings(result.get("warnings") or [])
        print()
        return

    print(f"  {'Scanned:':<26} {result.get('scanned_sidecar_count', 0)}")
    print(f"  {'Reviewed:':<26} {result.get('reviewed_sidecar_count', 0)}")
    print(f"  {'  Approved:':<26} {result.get('approved_count', 0)}")
    print(f"  {'  Rejected:':<26} {result.get('rejected_count', 0)}")
    print(f"  {'Skipped / unlabeled:':<26} {result.get('skipped_sidecar_count', 0)}")
    print()

    diag = result.get("diagnostics") or {}

    # Score distributions
    sd = diag.get("score_distribution") or {}
    if sd:
        print("  Score distribution  (highlight_score 0.0–1.0):")
        print(f"  {'':4}{'Group':<12}{'Mean':>7}{'Median':>8}{'Min':>7}{'Max':>7}{'P25':>7}{'P75':>7}")
        print("  " + "─" * 56)
        for group in ("approved", "rejected", "all"):
            s = sd.get(group) or {}
            if s:
                print(
                    f"  {'':4}{group:<12}"
                    f"{s.get('mean', 0):>7.3f}{s.get('median', 0):>8.3f}"
                    f"{s.get('min', 0):>7.3f}{s.get('max', 0):>7.3f}"
                    f"{s.get('p25', 0):>7.3f}{s.get('p75', 0):>7.3f}"
                )
        print()

    # Action bucket outcomes
    bo = diag.get("action_bucket_outcomes") or {}
    if bo:
        print("  Action bucket outcomes:")
        print(f"  {'Action':<24}{'Approved':>10}{'Rejected':>10}{'Total':>7}")
        print("  " + "─" * 52)
        for action in _ACTION_LABELS:
            counts = bo.get(action) or {}
            if counts.get("total", 0) > 0:
                print(
                    f"  {action:<24}"
                    f"{counts.get('approved', 0):>10}"
                    f"{counts.get('rejected', 0):>10}"
                    f"{counts.get('total', 0):>7}"
                )
        print()

    # Event incidence
    ei = diag.get("event_incidence") or {}
    if ei:
        print("  Event signal incidence by review outcome:")
        print(f"  {'Signal':<32}{'Approved':>10}{'Rejected':>10}")
        print("  " + "─" * 54)
        for evt, rates in ei.items():
            print(
                f"  {evt:<32}"
                f"{rates.get('approved_rate', 0):>9.0%}"
                f"{rates.get('rejected_rate', 0):>10.0%}"
            )
        print()

    # Threshold analysis
    ta = diag.get("threshold_analysis") or {}
    if ta:
        hi = ta.get("highlight_candidate_threshold", 0)
        lo = ta.get("inspect_min_threshold", 0)
        print(
            f"  Threshold analysis  "
            f"(highlight_candidate={hi:.2f}  inspect_min={lo:.2f}):"
        )
        print(
            f"    False negative rate: {ta.get('false_negative_rate', 0):.1%}  "
            f"({ta.get('approved_below_highlight_count', 0)} approved clips "
            f"below highlight threshold)"
        )
        print(
            f"    False positive rate: {ta.get('false_positive_rate', 0):.1%}  "
            f"({ta.get('rejected_above_highlight_count', 0)} rejected clips "
            f"above highlight threshold)"
        )
        print()

    recs = result.get("recommendations") or {}

    threshold_obs = recs.get("threshold_observations") or []
    if threshold_obs:
        print("  Threshold observations:")
        for obs in threshold_obs:
            print(f"    • {obs}")
        print()

    ranges = recs.get("candidate_threshold_ranges") or {}
    if ranges:
        print("  Candidate threshold ranges:")
        for name, info in ranges.items():
            if "suggested_trial" in info:
                print(
                    f"    {name}: current={info['current']:.2f}  "
                    f"trial={info['suggested_trial']:.2f}  "
                    f"({info.get('rationale', '')})"
                )
            else:
                print(f"    {name}: {info.get('note', '')}")
        print()

    weight_obs = recs.get("weight_observations") or []
    if weight_obs:
        print("  Weight observations:")
        for obs in weight_obs:
            print(f"    • {obs}")
        print()

    adj = recs.get("candidate_weight_adjustments") or {}
    if adj:
        print("  Candidate weight adjustments:")
        for signal, direction in adj.items():
            print(f"    {signal:<35} → {direction}")
        print()

    data_quality = recs.get("data_quality_notes") or []
    if data_quality:
        print("  Data quality notes:")
        for note in data_quality:
            print(f"    ⚠  {note}")
        print()

    _print_warnings(result.get("warnings") or [])


def _print_warnings(warnings: list[str]) -> None:
    if not warnings:
        return
    print(f"  Warnings ({len(warnings)}):")
    for w in warnings[:5]:
        print(f"    - {w}")
    if len(warnings) > 5:
        print(f"    ... and {len(warnings) - 5} more (use --calibrate-debug-dir to save all)")
    print()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _strip_internal_keys(result: dict) -> dict:
    """Return a copy of the result with internal _* keys removed from diagnostics."""
    import copy
    out = copy.deepcopy(result)
    diag = out.get("diagnostics") or {}
    for k in list(diag.keys()):
        if k.startswith("_"):
            del diag[k]
    return out
