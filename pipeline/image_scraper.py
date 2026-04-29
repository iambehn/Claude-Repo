"""
pipeline/image_scraper.py — generic image scraper for OpenCV training assets.

Builds search targets from a game pack (heroes, medals, abilities), queries
Google Custom Search API or DuckDuckGo images, downloads results, validates
them, normalizes icon categories to a fixed template size, and writes an
asset manifest.

Usage:
    python run.py --scrape-images marvel_rivals
    python run.py --scrape-images marvel_rivals --scrape-categories hero_icon medal
    python run.py --scrape-images marvel_rivals --scrape-max 3
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import struct
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

import requests
import yaml

from pipeline import game_pack

logger = logging.getLogger(__name__)

_DEFAULT_NORMALIZE_SIZE = 64
_REQUEST_TIMEOUT = 15
_MIN_IMAGE_BYTES = 1024
_MIN_IMAGE_PIXELS = 16

# Only these categories get normalized to fixed-size template PNGs
_ICON_CATEGORIES = {"hero_icon", "medal", "ability"}

_DDG_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}
_IMG_HEADERS = {
    "User-Agent": _DDG_HEADERS["User-Agent"],
    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
}


@dataclass
class SearchTarget:
    game_id: str
    category: str          # hero_icon | kill_feed | medal | ability | hud_screenshot
    entity_id: str | None  # hero/medal/ability slug; None for game-level searches
    display_name: str
    query_override: str | None = None


@dataclass
class ScrapeResult:
    target: SearchTarget
    source_url: str
    local_path: Path
    file_hash: str
    fetch_timestamp: float
    template_path: Path | None = None
    qa_status: str = "draft"
    error: str | None = None


def scrape_images_for_game(
    game_slug: str,
    categories: list[str] | None = None,
    max_per_entity: int = 5,
    config: dict | None = None,
) -> list[ScrapeResult]:
    """
    Main entry point. Loads the game pack, builds search targets, searches,
    downloads, normalizes icon categories, and writes the asset manifest.
    """
    cfg = (config or {}).get("image_scraper", {})
    pack = game_pack.load(game_slug)
    targets = _build_targets(pack, categories)
    if not targets:
        logger.warning(f"[image_scraper:{game_slug}] No search targets built — check game pack")
        return []

    manifest_path = Path(
        cfg.get("manifest_path", "assets/games/{game}/image_scraper_manifest.yaml")
        .replace("{game}", game_slug)
    )
    existing = _load_manifest(manifest_path)
    seen_urls: set[str] = {e.get("source_url", "") for e in existing}
    seen_hashes: set[str] = {e.get("file_hash", "") for e in existing}

    delay = float(cfg.get("request_delay_seconds", 0.8))
    normalize_size = int(cfg.get("normalize_size", _DEFAULT_NORMALIZE_SIZE))
    template_dir_tpl = cfg.get("template_dir", "assets/games/{game}/templates")
    output_dir_tpl = cfg.get("output_dir", "assets/games/{game}/training_images")

    results: list[ScrapeResult] = []
    game_display = pack.info.display_name

    for target in targets:
        query = target.query_override or _build_query(game_display, target)
        logger.info(
            f"[image_scraper] [{target.category}] "
            f"{target.entity_id or 'game'}: {query!r}"
        )

        urls = _search_images(query, max_per_entity, cfg)
        fetched = 0
        for url in urls:
            if fetched >= max_per_entity:
                break
            if url in seen_urls:
                logger.debug(f"[image_scraper] skip already-fetched: {url}")
                continue

            out_dir = _resolve_output_dir(output_dir_tpl, game_slug, target)
            out_dir.mkdir(parents=True, exist_ok=True)

            existing_count = (
                len(list(out_dir.glob("*.png"))) + len(list(out_dir.glob("*.jpg")))
            )
            slug_part = target.entity_id or "screenshot"
            dest = out_dir / f"{slug_part}_{existing_count + 1:03d}.png"

            data = _download_image(url, dest)
            if data is None:
                results.append(ScrapeResult(
                    target=target,
                    source_url=url,
                    local_path=dest,
                    file_hash="",
                    fetch_timestamp=round(time.time(), 3),
                    error="download_failed",
                ))
                time.sleep(delay)
                continue

            file_hash = _compute_hash(data)
            if file_hash in seen_hashes:
                logger.debug(f"[image_scraper] skip duplicate hash: {url}")
                dest.unlink(missing_ok=True)
                time.sleep(delay)
                continue

            seen_urls.add(url)
            seen_hashes.add(file_hash)

            template_path: Path | None = None
            if target.category in _ICON_CATEGORIES and target.entity_id:
                t_dir = _resolve_template_dir(template_dir_tpl, game_slug, target)
                t_dir.mkdir(parents=True, exist_ok=True)
                t_path = t_dir / f"{target.entity_id}.{normalize_size}.png"
                template_path = _normalize_icon(dest, t_path, normalize_size)

            results.append(ScrapeResult(
                target=target,
                source_url=url,
                local_path=dest,
                file_hash=file_hash,
                fetch_timestamp=round(time.time(), 3),
                template_path=template_path,
                qa_status="draft",
            ))
            fetched += 1
            time.sleep(delay)

    _write_manifest(manifest_path, game_slug, existing, results)
    return results


# ---------------------------------------------------------------------------
# Target building
# ---------------------------------------------------------------------------

def _build_targets(pack: game_pack.GamePack, categories: list[str] | None) -> list[SearchTarget]:
    cats = (
        set(categories)
        if categories
        else {"hero_icon", "kill_feed", "medal", "ability", "hud_screenshot"}
    )
    targets: list[SearchTarget] = []

    if "hero_icon" in cats:
        for entity in pack.entities.by_kind("hero"):
            targets.append(SearchTarget(
                game_id=pack.info.game_id,
                category="hero_icon",
                entity_id=entity.id,
                display_name=entity.display_name,
            ))

    if "ability" in cats:
        for ability in pack.abilities.abilities:
            targets.append(SearchTarget(
                game_id=pack.info.game_id,
                category="ability",
                entity_id=ability.id,
                display_name=ability.display_name,
            ))

    if "medal" in cats:
        medals_path = Path("assets/games") / pack.info.game_id / "medals.yaml"
        if medals_path.exists():
            raw = yaml.safe_load(medals_path.read_text()) or {}
            for m in raw.get("medals", []):
                mid = m.get("medal_id")
                if mid:
                    targets.append(SearchTarget(
                        game_id=pack.info.game_id,
                        category="medal",
                        entity_id=mid,
                        display_name=m.get("display_name", mid),
                    ))

    if "kill_feed" in cats:
        targets.append(SearchTarget(
            game_id=pack.info.game_id,
            category="kill_feed",
            entity_id=None,
            display_name=pack.info.display_name,
        ))

    if "hud_screenshot" in cats:
        targets.append(SearchTarget(
            game_id=pack.info.game_id,
            category="hud_screenshot",
            entity_id=None,
            display_name=pack.info.display_name,
        ))

    return targets


# ---------------------------------------------------------------------------
# Query construction
# ---------------------------------------------------------------------------

def _build_query(game_display: str, target: SearchTarget) -> str:
    g = f'"{game_display}"'
    name = target.display_name
    cat = target.category
    if cat == "hero_icon":
        return f"{g} {name} hero icon transparent"
    if cat == "ability":
        return f"{g} {name} ability icon"
    if cat == "medal":
        return f"{g} {name} medal notification screenshot"
    if cat == "kill_feed":
        return f"{g} kill feed screenshot gameplay 1080p"
    if cat == "hud_screenshot":
        return f"{g} gameplay screenshot HUD 1080p"
    return f"{g} {name} screenshot"


# ---------------------------------------------------------------------------
# Search backends
# ---------------------------------------------------------------------------

def _search_images(query: str, n: int, cfg: dict) -> list[str]:
    backend = cfg.get("search_backend", "auto")
    use_google = backend == "google_cse" or (
        backend == "auto"
        and os.environ.get("GOOGLE_CSE_API_KEY")
        and os.environ.get("GOOGLE_CSE_ID")
    )
    if use_google:
        urls = _search_google_cse(query, n)
        if urls:
            return urls
        logger.debug("[image_scraper] Google CSE empty, falling back to DuckDuckGo")
    return _search_duckduckgo(query, n)


def _search_google_cse(query: str, n: int) -> list[str]:
    api_key = os.environ.get("GOOGLE_CSE_API_KEY", "")
    cx = os.environ.get("GOOGLE_CSE_ID", "")
    if not api_key or not cx:
        return []
    try:
        resp = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={
                "key": api_key,
                "cx": cx,
                "q": query,
                "searchType": "image",
                "num": min(n, 10),
            },
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return [item["link"] for item in resp.json().get("items", []) if "link" in item]
    except Exception as exc:
        logger.warning(f"[image_scraper] Google CSE error: {exc}")
        return []


def _search_duckduckgo(query: str, n: int) -> list[str]:
    """
    DDG image search via their undocumented JSON endpoint.
    Step 1: fetch vqd token from the search homepage response.
    Step 2: fetch image results from the i.js endpoint.
    """
    try:
        encoded = urllib.parse.quote_plus(query)
        r1 = requests.get(
            f"https://duckduckgo.com/?q={encoded}&ia=images",
            headers=_DDG_HEADERS,
            timeout=_REQUEST_TIMEOUT,
        )
        # vqd appears as vqd='...' or "vqd":"..." depending on DDG version
        m = re.search(r'vqd=(["\'])([^"\']+)\1', r1.text)
        if not m:
            m = re.search(r'"vqd"\s*:\s*"([^"]+)"', r1.text)
            vqd = m.group(1) if m else None
        else:
            vqd = m.group(2)

        if not vqd:
            logger.debug("[image_scraper] DDG: could not extract vqd token")
            return []

        time.sleep(0.5)

        r2 = requests.get(
            "https://duckduckgo.com/i.js",
            params={"l": "us-en", "o": "json", "q": query, "vqd": vqd, "f": ",,,,,", "p": "1"},
            headers=_DDG_HEADERS,
            timeout=_REQUEST_TIMEOUT,
        )
        return [item["image"] for item in r2.json().get("results", []) if "image" in item][:n]
    except Exception as exc:
        logger.warning(f"[image_scraper] DuckDuckGo error: {exc}")
        return []


# ---------------------------------------------------------------------------
# Download and validation
# ---------------------------------------------------------------------------

_IMAGE_MAGIC: list[tuple[bytes, str]] = [
    (b"\x89PNG", "png"),
    (b"\xff\xd8\xff", "jpg"),
    (b"RIFF", "webp"),
    (b"GIF8", "gif"),
]


def _detect_format(data: bytes) -> str | None:
    for magic, fmt in _IMAGE_MAGIC:
        if data[: len(magic)] == magic:
            if fmt == "webp":
                return "webp" if len(data) >= 12 and data[8:12] == b"WEBP" else None
            return fmt
    return None


def _download_image(url: str, dest: Path) -> bytes | None:
    data: bytes | None = None
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=_IMG_HEADERS, timeout=_REQUEST_TIMEOUT, stream=True)
            resp.raise_for_status()
            data = resp.content
            break
        except Exception as exc:
            if attempt == 2:
                logger.debug(f"[image_scraper] download failed ({url}): {exc}")
                return None
            time.sleep(1.5**attempt)

    if data is None or len(data) < _MIN_IMAGE_BYTES:
        return None

    fmt = _detect_format(data)
    if fmt is None:
        return None

    if fmt == "webp":
        data = _convert_webp_to_png(data)
        if data is None:
            return None
        fmt = "png"

    if not _check_min_dimensions(data, fmt):
        return None

    dest.write_bytes(data)
    return data


def _convert_webp_to_png(data: bytes) -> bytes | None:
    try:
        import io
        from PIL import Image
        buf = io.BytesIO()
        Image.open(io.BytesIO(data)).save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


def _check_min_dimensions(data: bytes, fmt: str) -> bool:
    try:
        if fmt == "png" and len(data) >= 24:
            w = struct.unpack(">I", data[16:20])[0]
            h = struct.unpack(">I", data[20:24])[0]
            return w >= _MIN_IMAGE_PIXELS and h >= _MIN_IMAGE_PIXELS
        if fmt == "jpg":
            i = 2
            while i < len(data) - 4:
                if data[i] != 0xFF:
                    break
                marker = data[i + 1]
                if marker in (0xC0, 0xC2):
                    h = struct.unpack(">H", data[i + 5 : i + 7])[0]
                    w = struct.unpack(">H", data[i + 7 : i + 9])[0]
                    return w >= _MIN_IMAGE_PIXELS and h >= _MIN_IMAGE_PIXELS
                seg_len = struct.unpack(">H", data[i + 2 : i + 4])[0]
                i += 2 + seg_len
    except Exception:
        pass
    return True  # can't determine dimensions — accept optimistically


def _compute_hash(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_icon(src: Path, dest: Path, size: int) -> Path | None:
    try:
        import io
        from PIL import Image
        img = Image.open(src).convert("RGBA")
        img = img.resize((size, size), Image.LANCZOS)
        img.save(dest, format="PNG")
        return dest
    except Exception as exc:
        logger.debug(f"[image_scraper] normalize failed ({src.name}): {exc}")
        return None


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _resolve_output_dir(tpl: str, game_slug: str, target: SearchTarget) -> Path:
    base = Path(tpl.replace("{game}", game_slug))
    if target.category in ("kill_feed", "hud_screenshot"):
        return base / target.category
    return base / target.category / (target.entity_id or "unknown")


def _resolve_template_dir(tpl: str, game_slug: str, target: SearchTarget) -> Path:
    base = Path(tpl.replace("{game}", game_slug))
    return base / {"hero_icon": "heroes", "medal": "medals", "ability": "abilities"}.get(
        target.category, target.category
    )


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        return (yaml.safe_load(path.read_text()) or {}).get("entries", [])
    except Exception:
        return []


def _write_manifest(
    path: Path,
    game_slug: str,
    existing: list[dict],
    new_results: list[ScrapeResult],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_entries = [
        {
            "category": r.target.category,
            "entity_id": r.target.entity_id,
            "source_url": r.source_url,
            "local_path": str(r.local_path),
            "file_hash": r.file_hash,
            "fetch_timestamp": r.fetch_timestamp,
            "template_path": str(r.template_path) if r.template_path else None,
            "qa_status": r.qa_status,
        }
        for r in new_results
        if not r.error
    ]
    path.write_text(
        yaml.safe_dump(
            {"game_id": game_slug, "last_run": round(time.time(), 3), "entries": existing + new_entries},
            sort_keys=False,
            allow_unicode=True,
        )
    )
