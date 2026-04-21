"""
Game Pack Loader — per-game configuration as drop-in directories.

Each game is described by a directory of YAML files under
`assets/games/<slug>/` with this layout:

    game.yaml       (required) — id, display_name, detectors, resolution profiles
    entities.yaml   (required) — heroes and weapons with id / display_name / role / aliases
    hud.yaml        (required) — rois (pct-based), anchors, colors, pixel spike threshold
    moments.yaml    (optional) — narrative "moments" with score multipliers (pilot: MR)
    weights.yaml    (optional) — composite score weights and thresholds (pilot: MR)
    abilities.yaml  (optional) — character abilities with worthiness_boost (pilot: MR)

Public API:
    load(slug)                 -> GamePack
    validate(slug)             -> ValidationResult
    init_scaffold(slug)        -> None     (creates a minimal, self-validating stub)
    clear_cache()              -> None

Pixel coordinates are derived from pct-based values at the reference resolution
of 1920x1080. Downstream consumers that still speak in pixels (kill_feed.py,
weapon_detector.py) call GamePack.hud.get(name) to receive {x, y, w, h} ints.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

from utils.logger import get_logger

logger = get_logger(__name__)

# Reference resolution for pct -> pixel conversion. All detectors internally
# resize frames to this before reading ROIs.
REF_WIDTH = 1920
REF_HEIGHT = 1080

# Baseline weights applied to any game that omits weights.yaml.
BASELINE_WEIGHTS: dict[str, Any] = {
    "composite": {
        "context_weight": 0.30,
        "hook_weight": 0.25,
        "postability_weight": 0.45,
    },
    "thresholds": {
        "accept": 0.65,
        "reject": 0.35,
        "min_context": 0.40,
        "hook_gate": 0.40,
        "reliable_confidence": 0.60,
        "hook_window_seconds": 1.5,
    },
    "context_inputs": {
        "weapon_confidence": 0.40,
        "kill_detection_saturation": 0.35,
        "audio_saturation": 0.25,
    },
    "postability_inputs": {
        "sweat_score": 0.25,
        "audio_energy": 0.15,
        "motion_level": 0.15,
        "keyword_count": 0.10,
        "weapon_confidence": 0.15,
        "audio_spike_count": 0.20,
    },
}

GAMES_ROOT = Path("assets/games")


class GamePackError(Exception):
    """Raised when a game-pack directory is malformed or missing required fields."""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GameInfo:
    game_id: str
    display_name: str
    genre: str
    camera_mode: str
    ui_version: str
    detectors: dict[str, bool]
    resolution_profiles: tuple[str, ...]


@dataclass(frozen=True)
class Entity:
    id: str
    kind: str  # "hero" | "weapon"
    display_name: str
    role: str
    aliases: tuple[str, ...]
    primary_weapon: str | None = None
    asset_icon: str | None = None


@dataclass(frozen=True)
class EntityTable:
    entries: tuple[Entity, ...]

    def by_kind(self, kind: str) -> tuple[Entity, ...]:
        return tuple(e for e in self.entries if e.kind == kind)

    def display_name_map(self, kind: str | None = None) -> dict[str, str]:
        """Return {id: display_name} for entities, optionally filtered by kind.

        Back-compat shim for weapon_detector.py, which used to read
        `weapon_detector.games.<slug>.weapons` as {id: display_name}.
        """
        source = self.entries if kind is None else self.by_kind(kind)
        return {e.id: e.display_name for e in source}

    def resolve_alias(self, text: str) -> Entity | None:
        """Find an entity whose display_name or an alias appears in `text`.

        Case-insensitive substring match. Used for transcript matching in
        Phase 2's title/moments logic.
        """
        if not text:
            return None
        t = text.lower()
        for entity in self.entries:
            if entity.display_name.lower() in t:
                return entity
            for alias in entity.aliases:
                if alias.lower() in t:
                    return entity
        return None


@dataclass(frozen=True)
class Roi:
    name: str
    anchor: str
    x_pct: float
    y_pct: float
    w_pct: float
    h_pct: float
    kill_colors: tuple[dict[str, tuple[int, int, int]], ...] = field(default_factory=tuple)
    headshot_colors: tuple[dict[str, tuple[int, int, int]], ...] = field(default_factory=tuple)
    pixel_spike_threshold: int | None = None


@dataclass(frozen=True)
class HudTable:
    anchors: dict[str, dict[str, float]]
    rois: dict[str, Roi]
    templates: tuple[dict[str, Any], ...]

    def get(self, name: str) -> dict[str, int] | None:
        """Return pixel ROI at 1920x1080 for the named roi.

        Returns {x, y, w, h} or None when the roi is not defined.
        """
        roi = self.rois.get(name)
        if roi is None:
            return None
        return {
            "x": int(round(roi.x_pct * REF_WIDTH)),
            "y": int(round(roi.y_pct * REF_HEIGHT)),
            "w": int(round(roi.w_pct * REF_WIDTH)),
            "h": int(round(roi.h_pct * REF_HEIGHT)),
        }

    def colors(self, name: str, kind: str) -> list[dict[str, list[int]]]:
        """Return HSV color bands for an roi's detection list.

        `kind` is "kill" or "headshot". Returns [] if the roi doesn't define them.
        """
        roi = self.rois.get(name)
        if roi is None:
            return []
        source = roi.kill_colors if kind == "kill" else roi.headshot_colors
        return [{"lower": list(c["lower"]), "upper": list(c["upper"])} for c in source]

    def pixel_spike_threshold(self, name: str) -> int | None:
        roi = self.rois.get(name)
        return roi.pixel_spike_threshold if roi else None


@dataclass(frozen=True)
class Moment:
    id: str
    category: str
    description: str
    evidence: dict[str, Any]
    score_weight: float


@dataclass(frozen=True)
class MomentTable:
    moments: tuple[Moment, ...]

    def ids(self) -> tuple[str, ...]:
        return tuple(m.id for m in self.moments)


@dataclass(frozen=True)
class Ability:
    id: str
    character_id: str
    display_name: str
    aliases: tuple[str, ...]
    clazz: str  # "ultimate" | "signature" | "utility"
    worthiness_boost: float


@dataclass(frozen=True)
class AbilityTable:
    abilities: tuple[Ability, ...]

    def for_character(self, character_id: str) -> tuple[Ability, ...]:
        return tuple(a for a in self.abilities if a.character_id == character_id)


@dataclass(frozen=True)
class Weights:
    composite: dict[str, float]
    thresholds: dict[str, float]
    context_inputs: dict[str, float]
    postability_inputs: dict[str, float]


@dataclass(frozen=True)
class GamePack:
    slug: str
    info: GameInfo
    entities: EntityTable
    hud: HudTable
    moments: MomentTable
    abilities: AbilityTable
    weights: Weights


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

_cache: dict[str, tuple[float, GamePack]] = {}


def clear_cache() -> None:
    _cache.clear()


def _dir_mtime(pack_dir: Path) -> float:
    latest = pack_dir.stat().st_mtime
    for child in pack_dir.iterdir():
        if child.is_file():
            latest = max(latest, child.stat().st_mtime)
    return latest


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------


def _read_yaml(path: Path) -> Any:
    try:
        with path.open("r") as fh:
            return yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise GamePackError(f"Invalid YAML in {path}: {exc}") from exc
    except OSError as exc:
        raise GamePackError(f"Could not read {path}: {exc}") from exc


def _require(data: dict, key: str, where: str) -> Any:
    if key not in data:
        raise GamePackError(f"Missing required field '{key}' in {where}")
    return data[key]


def _pack_dir(slug: str) -> Path:
    return GAMES_ROOT / slug


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def _parse_game(path: Path) -> GameInfo:
    data = _read_yaml(path) or {}
    if not isinstance(data, dict):
        raise GamePackError(f"{path}: expected a mapping at top-level")
    return GameInfo(
        game_id=_require(data, "game_id", str(path)),
        display_name=_require(data, "display_name", str(path)),
        genre=data.get("genre", "unknown"),
        camera_mode=data.get("camera_mode", "unknown"),
        ui_version=data.get("ui_version", "unknown"),
        detectors={k: bool(v) for k, v in (data.get("detectors") or {}).items()},
        resolution_profiles=tuple(data.get("resolution_profiles") or []),
    )


def _parse_entities(path: Path) -> EntityTable:
    data = _read_yaml(path) or {}
    raw = data.get("entities") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        raise GamePackError(f"{path}: 'entities' must be a list")

    seen: set[str] = set()
    entries: list[Entity] = []
    for idx, row in enumerate(raw):
        if not isinstance(row, dict):
            raise GamePackError(f"{path}: entities[{idx}] must be a mapping")
        eid = _require(row, "id", f"{path} entities[{idx}]")
        if eid in seen:
            raise GamePackError(f"{path}: duplicate entity id '{eid}'")
        seen.add(eid)
        kind = _require(row, "kind", f"{path} entities[{idx}]")
        if kind not in ("hero", "weapon"):
            raise GamePackError(f"{path}: entities[{idx}].kind must be 'hero' or 'weapon'")
        entries.append(
            Entity(
                id=eid,
                kind=kind,
                display_name=_require(row, "display_name", f"{path} entities[{idx}]"),
                role=row.get("role", ""),
                aliases=tuple(row.get("aliases") or ()),
                primary_weapon=row.get("primary_weapon"),
                asset_icon=row.get("asset_icon"),
            )
        )
    return EntityTable(entries=tuple(entries))


def _parse_hud(path: Path) -> HudTable:
    data = _read_yaml(path) or {}
    if not isinstance(data, dict):
        raise GamePackError(f"{path}: expected a mapping at top-level")

    anchors_raw = data.get("anchors") or {}
    rois_raw = data.get("rois") or {}
    templates_raw = data.get("templates") or []

    rois: dict[str, Roi] = {}
    for name, row in rois_raw.items():
        if not isinstance(row, dict):
            raise GamePackError(f"{path}: rois.{name} must be a mapping")
        for field_name in ("x_pct", "y_pct", "w_pct", "h_pct"):
            if field_name not in row:
                raise GamePackError(f"{path}: rois.{name}.{field_name} missing")
            val = row[field_name]
            if not isinstance(val, (int, float)) or not (0.0 <= float(val) <= 1.0):
                raise GamePackError(
                    f"{path}: rois.{name}.{field_name}={val} must be in [0, 1]"
                )
        x_pct, y_pct = float(row["x_pct"]), float(row["y_pct"])
        w_pct, h_pct = float(row["w_pct"]), float(row["h_pct"])
        if x_pct + w_pct > 1.0001 or y_pct + h_pct > 1.0001:
            raise GamePackError(
                f"{path}: rois.{name} extends beyond screen "
                f"(x+w={x_pct+w_pct:.3f}, y+h={y_pct+h_pct:.3f})"
            )
        rois[name] = Roi(
            name=name,
            anchor=row.get("anchor", ""),
            x_pct=x_pct,
            y_pct=y_pct,
            w_pct=w_pct,
            h_pct=h_pct,
            kill_colors=_to_color_tuple(row.get("kill_colors")),
            headshot_colors=_to_color_tuple(row.get("headshot_colors")),
            pixel_spike_threshold=row.get("pixel_spike_threshold"),
        )

    return HudTable(
        anchors={k: dict(v) for k, v in anchors_raw.items()},
        rois=rois,
        templates=tuple(templates_raw),
    )


def _to_color_tuple(raw: Any) -> tuple[dict[str, tuple[int, int, int]], ...]:
    if not raw:
        return ()
    out: list[dict[str, tuple[int, int, int]]] = []
    for band in raw:
        lower = tuple(band.get("lower", []))
        upper = tuple(band.get("upper", []))
        if len(lower) != 3 or len(upper) != 3:
            raise GamePackError(
                f"kill/headshot color band requires 3-element lower/upper (HSV)"
            )
        out.append({"lower": lower, "upper": upper})
    return tuple(out)


def _parse_moments(path: Path) -> MomentTable:
    if not path.exists():
        return MomentTable(moments=())
    data = _read_yaml(path) or {}
    raw = data.get("moments") if isinstance(data, dict) else None
    if raw is None:
        return MomentTable(moments=())
    if not isinstance(raw, list):
        raise GamePackError(f"{path}: 'moments' must be a list")
    seen: set[str] = set()
    moments: list[Moment] = []
    for idx, row in enumerate(raw):
        if not isinstance(row, dict):
            raise GamePackError(f"{path}: moments[{idx}] must be a mapping")
        mid = _require(row, "id", f"{path} moments[{idx}]")
        if mid in seen:
            raise GamePackError(f"{path}: duplicate moment id '{mid}'")
        seen.add(mid)
        moments.append(
            Moment(
                id=mid,
                category=row.get("category", ""),
                description=row.get("description", ""),
                evidence=dict(row.get("evidence") or {}),
                score_weight=float(row.get("score_weight", 1.0)),
            )
        )
    return MomentTable(moments=tuple(moments))


def _parse_abilities(path: Path) -> AbilityTable:
    if not path.exists():
        return AbilityTable(abilities=())
    data = _read_yaml(path) or {}
    raw = data.get("abilities") if isinstance(data, dict) else None
    if raw is None:
        return AbilityTable(abilities=())
    if not isinstance(raw, list):
        raise GamePackError(f"{path}: 'abilities' must be a list")
    seen: set[str] = set()
    abilities: list[Ability] = []
    for idx, row in enumerate(raw):
        if not isinstance(row, dict):
            raise GamePackError(f"{path}: abilities[{idx}] must be a mapping")
        aid = _require(row, "id", f"{path} abilities[{idx}]")
        if aid in seen:
            raise GamePackError(f"{path}: duplicate ability id '{aid}'")
        seen.add(aid)
        abilities.append(
            Ability(
                id=aid,
                character_id=_require(row, "character_id", f"{path} abilities[{idx}]"),
                display_name=_require(row, "display_name", f"{path} abilities[{idx}]"),
                aliases=tuple(row.get("aliases") or ()),
                clazz=row.get("class", "utility"),
                worthiness_boost=float(row.get("worthiness_boost", 0.0)),
            )
        )
    return AbilityTable(abilities=tuple(abilities))


def _parse_weights(path: Path) -> Weights:
    if not path.exists():
        src = BASELINE_WEIGHTS
    else:
        data = _read_yaml(path) or {}
        if not isinstance(data, dict):
            raise GamePackError(f"{path}: expected a mapping at top-level")
        # Merge over baseline so omitted sections still work.
        src = _merge_weights(BASELINE_WEIGHTS, data)
    composite = src["composite"]
    cw = composite["context_weight"] + composite["hook_weight"] + composite["postability_weight"]
    if abs(cw - 1.0) > 0.01:
        raise GamePackError(
            f"{path}: composite weights must sum to 1.0 (got {cw:.3f})"
        )
    return Weights(
        composite=dict(composite),
        thresholds=dict(src["thresholds"]),
        context_inputs=dict(src["context_inputs"]),
        postability_inputs=dict(src["postability_inputs"]),
    )


def _merge_weights(base: dict, override: dict) -> dict:
    out = {k: dict(v) for k, v in base.items()}
    for section, values in override.items():
        if section not in out or not isinstance(values, dict):
            out[section] = values
            continue
        out[section].update(values)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load(slug: str) -> GamePack:
    """Load a game pack by slug, using an mtime cache keyed by directory.

    Raises FileNotFoundError if the slug directory is missing and GamePackError
    if any file is malformed.
    """
    pack_dir = _pack_dir(slug)
    if not pack_dir.is_dir():
        raise FileNotFoundError(
            f"Game pack '{slug}' not found at {pack_dir}. "
            f"Create it with: python run.py --init-game {slug}"
        )

    mtime = _dir_mtime(pack_dir)
    cached = _cache.get(slug)
    if cached and cached[0] >= mtime:
        return cached[1]

    info = _parse_game(pack_dir / "game.yaml")
    entities = _parse_entities(pack_dir / "entities.yaml")
    hud = _parse_hud(pack_dir / "hud.yaml")
    moments = _parse_moments(pack_dir / "moments.yaml")
    abilities = _parse_abilities(pack_dir / "abilities.yaml")
    weights = _parse_weights(pack_dir / "weights.yaml")

    pack = GamePack(
        slug=slug,
        info=info,
        entities=entities,
        hud=hud,
        moments=moments,
        abilities=abilities,
        weights=weights,
    )
    _cache[slug] = (mtime, pack)
    return pack


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate(slug: str) -> ValidationResult:
    """Validate a game pack. Missing asset PNGs emit warnings, not errors."""
    result = ValidationResult(ok=True)
    try:
        pack = load(slug)
    except (GamePackError, FileNotFoundError) as exc:
        return ValidationResult(ok=False, errors=[str(exc)])

    # Warn on missing asset icons rather than fail — libraries are still being populated.
    for entity in pack.entities.entries:
        if entity.asset_icon:
            if not Path(entity.asset_icon).exists():
                result.warnings.append(
                    f"entity '{entity.id}': asset_icon {entity.asset_icon} does not exist"
                )

    for tmpl in pack.hud.templates:
        img = tmpl.get("image") if isinstance(tmpl, dict) else None
        if img and not Path(img).exists():
            result.warnings.append(f"hud template image missing: {img}")

    # abilities.character_id must resolve to an entity of kind=hero
    hero_ids = {e.id for e in pack.entities.by_kind("hero")}
    for ability in pack.abilities.abilities:
        if ability.character_id not in hero_ids:
            result.errors.append(
                f"ability '{ability.id}': character_id '{ability.character_id}' "
                f"is not a hero entity"
            )
            result.ok = False

    return result


def init_scaffold(slug: str) -> None:
    """Create a minimal, self-validating game-pack directory for `slug`.

    Produces game.yaml, entities.yaml, hud.yaml with TODO placeholders. The
    scaffold passes validate() so CI and iteration can proceed while HUD
    calibration and entity lists are filled in.
    """
    pack_dir = _pack_dir(slug)
    if pack_dir.exists():
        raise GamePackError(f"Directory already exists: {pack_dir}")
    pack_dir.mkdir(parents=True, exist_ok=False)

    (pack_dir / "game.yaml").write_text(
        f"""game_id: {slug}
display_name: "{slug.replace('_', ' ').title()}"
genre: "unknown"           # TODO: replace
camera_mode: "first_person" # TODO: replace
ui_version: "unknown"      # TODO: replace

detectors:
  kill_feed: false         # TODO: flip to true when hud.yaml roi is calibrated
  weapon_detector: false   # TODO: flip when weapon_detector roi is calibrated
  audio_detector: false

resolution_profiles:
  - "1920x1080"
"""
    )

    (pack_dir / "entities.yaml").write_text(
        """entities:
  # TODO: replace this placeholder with real heroes/weapons
  - id: placeholder
    kind: weapon
    display_name: "Placeholder"
    role: "unknown"
    aliases: []
"""
    )

    # Dummy 0.01% roi so pct validator passes but nothing real is detected.
    (pack_dir / "hud.yaml").write_text(
        """anchors:
  top_right: {x_pct: 1.0, y_pct: 0.0}

rois:
  # TODO: calibrate kill_feed roi on a frame screenshot
  kill_feed:
    anchor: top_right
    x_pct: 0.0
    y_pct: 0.0
    w_pct: 0.01
    h_pct: 0.01
    pixel_spike_threshold: 50
    kill_colors: []
    headshot_colors: []

  # TODO: calibrate weapon_detector roi on a frame screenshot
  weapon_detector:
    anchor: top_right
    x_pct: 0.0
    y_pct: 0.0
    w_pct: 0.01
    h_pct: 0.01

templates: []
"""
    )

    # Clear any stale cache entry so the next load re-reads from disk.
    _cache.pop(slug, None)

    # Self-check — the scaffold must pass validate() or init_scaffold raises.
    result = validate(slug)
    if not result.ok:
        raise GamePackError(
            f"init_scaffold produced an invalid pack: {result.errors}"
        )


def available_slugs(games_root: Path | None = None) -> list[str]:
    root = games_root or GAMES_ROOT
    if not root.exists():
        return []
    return sorted(d.name for d in root.iterdir() if d.is_dir())


def ensure_game_packs_valid(config: dict) -> None:
    """Validate every game pack referenced by config['games']. Raises on failure.

    Called at startup so the pipeline fails loud and early rather than in
    mid-stage on a missing or broken pack.
    """
    missing: list[str] = []
    errors: list[tuple[str, list[str]]] = []
    for slug in config.get("games", {}):
        if not _pack_dir(slug).exists():
            missing.append(slug)
            continue
        result = validate(slug)
        if not result.ok:
            errors.append((slug, result.errors))

    if missing or errors:
        msg_parts = []
        if missing:
            msg_parts.append(
                f"Missing game pack(s): {', '.join(missing)}. "
                f"Create with: python run.py --init-game <slug>"
            )
        for slug, errs in errors:
            msg_parts.append(f"[{slug}] {'; '.join(errs)}")
        raise GamePackError("\n".join(msg_parts))
