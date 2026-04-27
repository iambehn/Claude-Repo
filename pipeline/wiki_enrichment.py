"""
Wiki Enrichment — fetch and parse a game wiki page into a draft game-pack.

Fetches a Fandom wiki page, extracts entity names and portrait icons from
tables and card grids, and writes a draft YAML file that can be reviewed and
merged into the game pack's entities.yaml.

Usage (run.py):
    python run.py --wiki-enrich marvel_rivals https://marvelrivals.fandom.com/wiki/Heroes

Output:
    assets/games/{game}/drafts/wiki/{timestamp}/entities.draft.yaml
    assets/games/{game}/drafts/wiki/{timestamp}/assets_manifest.json
    assets/games/{game}/drafts/wiki/{timestamp}/icons/{entity_id}.png

Only Fandom wiki URLs are supported. Other domains will return status=failed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
import yaml

from pipeline.game_pack import GAMES_ROOT, init_scaffold, GamePackError
from utils.logger import get_logger

logger = get_logger(__name__)

_IMAGE_SIGNATURES = {
    ".png": b"\x89PNG\r\n\x1a\n",
    ".jpg": b"\xff\xd8\xff",
    ".gif": b"GIF8",
    ".webp": b"RIFF",
}
_CONTENT_TYPE_EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
}


@dataclass
class ParsedEntity:
    entity_id: str
    display_name: str
    role: str | None = None
    aliases: list[str] = field(default_factory=list)
    source_url: str | None = None
    source_icon_url: str | None = None
    local_icon_path: str | None = None
    scrape_confidence: float = 0.0
    scrape_status: str = "pending"


@dataclass
class _Node:
    tag: str
    attrs: dict[str, str] = field(default_factory=dict)
    children: list["_Node"] = field(default_factory=list)
    text_parts: list[str] = field(default_factory=list)
    parent: "_Node | None" = None


class _TreeParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = _Node("document")
        self.stack = [self.root]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = _Node(
            tag.lower(),
            {key.lower(): value or "" for key, value in attrs},
            parent=self.stack[-1],
        )
        self.stack[-1].children.append(node)
        if tag.lower() not in {
            "area", "base", "br", "col", "embed", "hr", "img", "input",
            "link", "meta", "param", "source", "track", "wbr",
        }:
            self.stack.append(node)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                break

    def handle_data(self, data: str) -> None:
        if data:
            self.stack[-1].text_parts.append(data)


def enrich_game_from_wiki(
    game: str,
    wiki_url: str,
    config: dict,
    *,
    html: str | None = None,
    timestamp: str | None = None,
    image_fetcher: Any | None = None,
) -> dict[str, Any]:
    """Fetch and parse a supported wiki page into a draft game-pack enrichment."""
    parser_name = detect_parser(wiki_url)
    if parser_name != "fandom":
        return {
            "game": game,
            "source_url": wiki_url,
            "parser": parser_name or "unsupported",
            "status": "failed",
            "entities_found": 0,
            "icons_downloaded": 0,
            "warnings": [
                f"Unsupported wiki URL domain: {urlparse(wiki_url).netloc or wiki_url}"
            ],
            "draft_dir": None,
        }

    fetched_at = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    warnings: list[str] = []
    if html is None:
        try:
            html = fetch_html(wiki_url)
        except requests.RequestException as e:
            return {
                "game": game,
                "source_url": wiki_url,
                "parser": parser_name,
                "status": "failed",
                "entities_found": 0,
                "icons_downloaded": 0,
                "warnings": [f"Failed to fetch wiki URL: {e}"],
                "draft_dir": None,
            }

    entities, parse_warnings = parse_fandom_entities(html, wiki_url)
    warnings.extend(parse_warnings)

    pack_dir = GAMES_ROOT / game
    if not pack_dir.exists():
        try:
            init_scaffold(game)
        except GamePackError:
            pass

    draft_dir = pack_dir / "drafts" / "wiki" / fetched_at
    icons_dir = draft_dir / "icons"
    icons_dir.mkdir(parents=True, exist_ok=True)

    icons_downloaded = _download_entity_icons(
        entities, icons_dir, wiki_url, warnings, image_fetcher=image_fetcher
    )
    status = "ok" if entities and not warnings else "partial" if entities else "failed"
    _write_draft_files(
        draft_dir=draft_dir,
        game=game,
        source_url=wiki_url,
        parser_name=parser_name,
        fetched_at=fetched_at,
        status=status,
        entities=entities,
        icons_downloaded=icons_downloaded,
        warnings=warnings,
    )

    return {
        "game": game,
        "source_url": wiki_url,
        "parser": parser_name,
        "status": status,
        "entities_found": len(entities),
        "icons_downloaded": icons_downloaded,
        "warnings": warnings,
        "draft_dir": _path_for_response(draft_dir),
    }


def detect_parser(wiki_url: str) -> str | None:
    netloc = urlparse(wiki_url).netloc.lower()
    if "fandom.com" in netloc:
        return "fandom"
    return None


def fetch_html(url: str) -> str:
    response = requests.get(
        url,
        timeout=20,
        headers={"User-Agent": "Claude-Repo-GamePackEnricher/1.0"},
    )
    response.raise_for_status()
    return response.text


def parse_fandom_entities(
    html: str, source_url: str
) -> tuple[list[ParsedEntity], list[str]]:
    parser = _TreeParser()
    parser.feed(html)
    warnings: list[str] = []
    candidates: list[ParsedEntity] = []
    candidates.extend(_parse_table_entities(parser.root, source_url))
    candidates.extend(_parse_card_entities(parser.root, source_url))

    unique = _dedupe_entities(candidates)
    if not unique:
        warnings.append("No playable entities found in Fandom-like tables or cards.")
    return unique, warnings


def slugify_entity_id(raw: str) -> str:
    text = raw.strip().lower()
    text = re.sub(r"['']", "", text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown_entity"


def _parse_table_entities(root: _Node, source_url: str) -> list[ParsedEntity]:
    entities: list[ParsedEntity] = []
    for table in _find_all(root, lambda node: node.tag == "table"):
        table_class = _class_text(table)
        if table_class and not any(
            token in table_class
            for token in ("article-table", "wikitable", "fandom-table", "sortable")
        ):
            continue

        rows = [node for node in _find_all(table, lambda node: node.tag == "tr")]
        if len(rows) < 2:
            continue

        headers = [
            _clean_text(_text_content(cell)).lower()
            for cell in _direct_children(rows[0], {"th", "td"})
        ]
        for row in rows[1:]:
            cells = _direct_children(row, {"td", "th"})
            if not cells:
                continue
            name_index = _first_header_index(
                headers, ("name", "character", "hero", "operator", "legend")
            )
            role_index = _first_header_index(headers, ("role", "class", "type"))
            name_cell = cells[name_index] if 0 <= name_index < len(cells) else cells[0]
            role_cell = cells[role_index] if 0 <= role_index < len(cells) else None
            entity = _entity_from_node(
                name_cell, source_url, role_cell=role_cell, confidence=0.86
            )
            if entity:
                if not entity.source_icon_url:
                    row_img = _first_descendant(row, lambda item: item.tag == "img")
                    entity.source_icon_url = (
                        _image_url(row_img, source_url) if row_img else None
                    )
                entities.append(entity)
    return entities


def _parse_card_entities(root: _Node, source_url: str) -> list[ParsedEntity]:
    entities: list[ParsedEntity] = []
    for node in _find_all(root, _looks_like_card):
        link = _first_descendant(
            node, lambda item: item.tag == "a" and item.attrs.get("href")
        )
        img = _first_descendant(node, lambda item: item.tag == "img")
        text_source = link or node
        name = _clean_text(_text_content(text_source))
        if not name and img:
            name = _clean_text(img.attrs.get("alt", ""))
        if not _valid_entity_name(name):
            continue
        entities.append(
            ParsedEntity(
                entity_id=slugify_entity_id(name),
                display_name=name,
                aliases=[],
                source_url=urljoin(source_url, link.attrs.get("href", ""))
                if link else source_url,
                source_icon_url=_image_url(img, source_url) if img else None,
                scrape_confidence=0.68,
                scrape_status="pending",
            )
        )
    return entities


def _entity_from_node(
    name_cell: _Node,
    source_url: str,
    *,
    role_cell: _Node | None,
    confidence: float,
) -> ParsedEntity | None:
    link = _first_descendant(
        name_cell, lambda item: item.tag == "a" and item.attrs.get("href")
    )
    img = _first_descendant(name_cell, lambda item: item.tag == "img")
    name = _clean_text(_text_content(link or name_cell))
    if not name and img:
        name = _clean_text(img.attrs.get("alt", ""))
    if not _valid_entity_name(name):
        return None

    return ParsedEntity(
        entity_id=slugify_entity_id(name),
        display_name=name,
        role=_clean_text(_text_content(role_cell)) if role_cell else None,
        aliases=[],
        source_url=urljoin(source_url, link.attrs.get("href", "")) if link else source_url,
        source_icon_url=_image_url(img, source_url) if img else None,
        scrape_confidence=confidence,
        scrape_status="pending",
    )


def _download_entity_icons(
    entities: list[ParsedEntity],
    icons_dir: Path,
    source_url: str,
    warnings: list[str],
    *,
    image_fetcher: Any | None,
) -> int:
    downloaded = 0
    for entity in entities:
        if not entity.source_icon_url:
            entity.scrape_status = "no_icon"
            warnings.append(f"No icon URL found for {entity.display_name}.")
            continue

        icon_url = urljoin(source_url, entity.source_icon_url)
        try:
            image_bytes, content_type = _fetch_image(icon_url, image_fetcher=image_fetcher)
            ext = _image_extension(icon_url, content_type, image_bytes)
            if not ext:
                raise ValueError("downloaded content is not a supported image")
        except Exception as e:
            entity.scrape_status = "icon_download_failed"
            warnings.append(f"Failed to download icon for {entity.display_name}: {e}")
            continue

        icon_path = icons_dir / f"{entity.entity_id}{ext}"
        icon_path.write_bytes(image_bytes)
        entity.local_icon_path = _path_for_response(icon_path)
        entity.scrape_status = "ok"
        downloaded += 1
    return downloaded


def _fetch_image(icon_url: str, *, image_fetcher: Any | None) -> tuple[bytes, str]:
    if image_fetcher:
        result = image_fetcher(icon_url)
        if isinstance(result, tuple):
            return result[0], result[1] if len(result) > 1 else ""
        return result, ""

    response = requests.get(
        icon_url,
        timeout=20,
        headers={"User-Agent": "Claude-Repo-GamePackEnricher/1.0"},
    )
    response.raise_for_status()
    return response.content, response.headers.get("Content-Type", "")


def _write_draft_files(
    *,
    draft_dir: Path,
    game: str,
    source_url: str,
    parser_name: str,
    fetched_at: str,
    status: str,
    entities: list[ParsedEntity],
    icons_downloaded: int,
    warnings: list[str],
) -> None:
    draft_dir.mkdir(parents=True, exist_ok=True)
    entities_payload = {
        "primary_kind": "heroes",
        "heroes": {
            entity.entity_id: {
                "display_name": entity.display_name,
                "role": entity.role,
                "aliases": entity.aliases,
                "source_url": entity.source_url,
                "source_icon_url": entity.source_icon_url,
                "local_icon_path": entity.local_icon_path,
                "scrape_confidence": entity.scrape_confidence,
                "scrape_status": entity.scrape_status,
            }
            for entity in entities
        },
        "wiki_enrichment": {
            "game": game,
            "source_url": source_url,
            "parser": parser_name,
            "fetched_at": fetched_at,
            "status": status,
            "warnings": warnings,
        },
    }
    manifest = {
        "game": game,
        "source_url": source_url,
        "parser": parser_name,
        "fetched_at": fetched_at,
        "status": status,
        "entities_found": len(entities),
        "icons_downloaded": icons_downloaded,
        "warnings": warnings,
        "entities": [
            {
                "entity_id": entity.entity_id,
                "display_name": entity.display_name,
                "role": entity.role,
                "source_url": entity.source_url,
                "source_icon_url": entity.source_icon_url,
                "local_icon_path": entity.local_icon_path,
                "scrape_confidence": entity.scrape_confidence,
                "scrape_status": entity.scrape_status,
            }
            for entity in entities
        ],
    }

    (draft_dir / "entities.draft.yaml").write_text(yaml.safe_dump(entities_payload, sort_keys=False))
    (draft_dir / "assets_manifest.json").write_text(json.dumps(manifest, indent=2))


def _dedupe_entities(candidates: list[ParsedEntity]) -> list[ParsedEntity]:
    by_id: dict[str, ParsedEntity] = {}
    counts: dict[str, int] = {}
    for candidate in candidates:
        base_id = slugify_entity_id(candidate.display_name)
        entity_id = base_id
        if entity_id in by_id:
            existing = by_id[entity_id]
            if existing.display_name.lower() == candidate.display_name.lower():
                if not existing.source_icon_url and candidate.source_icon_url:
                    existing.source_icon_url = candidate.source_icon_url
                if not existing.role and candidate.role:
                    existing.role = candidate.role
                existing.scrape_confidence = max(
                    existing.scrape_confidence, candidate.scrape_confidence
                )
                continue
            counts[base_id] = counts.get(base_id, 1) + 1
            entity_id = f"{base_id}_{counts[base_id]}"
        candidate.entity_id = entity_id
        by_id[entity_id] = candidate
    return list(by_id.values())


def _image_extension(icon_url: str, content_type: str, image_bytes: bytes) -> str | None:
    content_type = content_type.split(";")[0].strip().lower()
    if content_type in _CONTENT_TYPE_EXTENSIONS:
        return _CONTENT_TYPE_EXTENSIONS[content_type]

    path_ext = Path(urlparse(icon_url).path).suffix.lower()
    if path_ext in _IMAGE_SIGNATURES:
        return path_ext

    for ext, signature in _IMAGE_SIGNATURES.items():
        if image_bytes.startswith(signature):
            return ext
    return None


def _find_all(node: _Node, predicate: Any) -> list[_Node]:
    found = []
    if predicate(node):
        found.append(node)
    for child in node.children:
        found.extend(_find_all(child, predicate))
    return found


def _first_descendant(node: _Node, predicate: Any) -> _Node | None:
    for item in _find_all(node, predicate):
        if item is not node:
            return item
    return None


def _direct_children(node: _Node, tags: set[str]) -> list[_Node]:
    return [child for child in node.children if child.tag in tags]


def _text_content(node: _Node | None) -> str:
    if node is None:
        return ""
    parts = list(node.text_parts)
    for child in node.children:
        if child.tag == "img":
            continue
        parts.append(_text_content(child))
    return " ".join(part for part in parts if part)


def _clean_text(text: str) -> str:
    text = re.sub(r"\[[^\]]+\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _class_text(node: _Node) -> str:
    return node.attrs.get("class", "").lower()


def _first_header_index(headers: list[str], needles: tuple[str, ...]) -> int:
    for index, header in enumerate(headers):
        if any(needle in header for needle in needles):
            return index
    return -1


def _looks_like_card(node: _Node) -> bool:
    class_text = _class_text(node)
    return node.tag in {"div", "li"} and any(
        token in class_text
        for token in ("character", "hero", "card", "gallerybox", "portal")
    )


def _image_url(img: _Node | None, source_url: str) -> str | None:
    if img is None:
        return None
    raw = (
        img.attrs.get("data-src")
        or img.attrs.get("data-image-key")
        or img.attrs.get("src")
        or ""
    ).strip()
    if not raw or raw.startswith("data:"):
        return None
    return urljoin(source_url, raw)


def _valid_entity_name(name: str) -> bool:
    if not name or len(name) > 80:
        return False
    lowered = name.lower()
    return lowered not in {
        "name", "character", "hero", "role", "class", "edit", "unknown"
    }


def _path_for_response(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)
