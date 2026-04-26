"""
tools/scrape_hero_icons.py — Marvel Rivals hero icon scraper

Downloads hero portrait icons from marvelrivals.wiki.gg into a staging
directory, then reports a manifest of what was successfully fetched.

Usage:
    python tools/scrape_hero_icons.py
    python tools/scrape_hero_icons.py --hero iron_man           # single hero
    python tools/scrape_hero_icons.py --dry-run                 # show URLs only
    python tools/scrape_hero_icons.py --promote                 # copy staged → active

Directory layout produced:
    assets/hero_icons/marvel_rivals/
        _sources/          <- raw downloads (staging)
            iron_man.png
            ...
        _manifest.json     <- source URL, resolution, status per hero
        iron_man.png       <- promoted active asset (after --promote)
        ...

After running, check _manifest.json to see what succeeded and what needs
manual sourcing. Only --promote moves images into the active slot.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
import random
from pathlib import Path
from typing import Optional

import requests
from PIL import Image
import io

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
STAGING_DIR = REPO_ROOT / "assets/hero_icons/marvel_rivals/_sources"
ACTIVE_DIR = REPO_ROOT / "assets/hero_icons/marvel_rivals"
MANIFEST_PATH = ACTIVE_DIR / "_manifest.json"

# ---------------------------------------------------------------------------
# Hero → wiki page name mapping
# wiki.gg uses Title_Case with underscores; some names have hyphens
# ---------------------------------------------------------------------------

HERO_WIKI_NAMES: dict[str, str] = {
    # Vanguards
    "angela":           "Angela",
    "bruce_banner":     "Bruce_Banner",
    "captain_america":  "Captain_America",
    "deadpool":         "Deadpool",
    "doctor_strange":   "Doctor_Strange",
    "emma_frost":       "Emma_Frost",
    "groot":            "Groot",
    "magneto":          "Magneto",
    "peni_parker":      "Peni_Parker",
    "rogue":            "Rogue",
    "the_thing":        "The_Thing",
    "thor":             "Thor",
    "venom":            "Venom",
    # Duelists
    "black_cat":        "Black_Cat",
    "black_panther":    "Black_Panther",
    "black_widow":      "Black_Widow",
    "blade":            "Blade",
    "daredevil":        "Daredevil",
    "elsa_bloodstone":  "Elsa_Bloodstone",
    "hawkeye":          "Hawkeye",
    "hela":             "Hela",
    "human_torch":      "Human_Torch",
    "iron_fist":        "Iron_Fist",
    "iron_man":         "Iron_Man",
    "magik":            "Magik",
    "mister_fantastic": "Mister_Fantastic",
    "moon_knight":      "Moon_Knight",
    "namor":            "Namor",
    "phoenix":          "Phoenix",
    "psylocke":         "Psylocke",
    "scarlet_witch":    "Scarlet_Witch",
    "spider_man":       "Spider-Man",
    "squirrel_girl":    "Squirrel_Girl",
    "star_lord":        "Star-Lord",
    "storm":            "Storm",
    "the_punisher":     "The_Punisher",
    "winter_soldier":   "Winter_Soldier",
    "wolverine":        "Wolverine",
    # Strategists
    "adam_warlock":         "Adam_Warlock",
    "cloak_and_dagger":     "Cloak_%26_Dagger",
    "gambit":               "Gambit",
    "invisible_woman":      "Invisible_Woman",
    "jeff_the_land_shark":  "Jeff_the_Land_Shark",
    "loki":                 "Loki",
    "luna_snow":            "Luna_Snow",
    "mantis":               "Mantis",
    "rocket_raccoon":       "Rocket_Raccoon",
    "ultron":               "Ultron",
    "white_fox":            "White_Fox",
}

WIKI_BASE = "https://marvelrivals.wiki.gg/wiki"

# Realistic browser headers — required to pass Cloudflare on wiki.gg
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://marvelrivals.wiki.gg/",
    "DNT": "1",
}

# ---------------------------------------------------------------------------
# Scraping logic
# ---------------------------------------------------------------------------


def _get_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def _wiki_url(hero_id: str) -> str:
    wiki_name = HERO_WIKI_NAMES.get(hero_id, hero_id.replace("_", " ").title())
    return f"{WIKI_BASE}/{wiki_name}"


def _find_portrait_url(session: requests.Session, hero_id: str) -> Optional[str]:
    """Fetch the wiki hero page and extract the main portrait image URL."""
    from bs4 import BeautifulSoup

    url = _wiki_url(hero_id)
    try:
        resp = session.get(url, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [FAIL] {hero_id}: page fetch failed — {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Strategy 1: look for the infobox/aside portrait (wiki.gg uses <aside> infoboxes)
    for selector in [
        "aside.portable-infobox figure img",
        ".pi-image img",
        ".infobox img",
        "figure.pi-item img",
        ".character-image img",
    ]:
        el = soup.select_one(selector)
        if el and el.get("src"):
            src = el["src"]
            # Remove thumbnail sizing parameters to get full image
            # wiki.gg CDN URLs look like: /images/thumb/a/ab/File.png/250px-File.png
            # Strip the thumbnail suffix to get the original
            if "/thumb/" in src:
                # e.g. .../thumb/a/ab/Iron_Man_portrait.png/250px-Iron_Man_portrait.png
                # → .../a/ab/Iron_Man_portrait.png
                parts = src.split("/thumb/")
                if len(parts) == 2:
                    rest = parts[1]  # a/ab/Iron_Man_portrait.png/250px-Iron_Man_portrait.png
                    # drop the last path segment (the resized filename)
                    original = "/".join(rest.split("/")[:-1])
                    src = parts[0] + "/" + original
            if src.startswith("//"):
                src = "https:" + src
            return src

    # Strategy 2: first image in main content area
    content = soup.select_one("#mw-content-text, .mw-parser-output")
    if content:
        img = content.find("img")
        if img and img.get("src"):
            src = img["src"]
            if src.startswith("//"):
                src = "https:" + src
            return src

    print(f"  [WARN] {hero_id}: page loaded but no portrait image found")
    return None


def _download_image(
    session: requests.Session,
    url: str,
    dest: Path,
    hero_id: str,
) -> Optional[dict]:
    """Download image to dest, return metadata dict or None on failure."""
    try:
        resp = session.get(url, timeout=15)
        resp.raise_for_status()

        img = Image.open(io.BytesIO(resp.content))
        img.save(dest, format="PNG")

        return {
            "hero_id": hero_id,
            "source_url": url,
            "wiki_page": _wiki_url(hero_id),
            "resolution": f"{img.width}x{img.height}",
            "status": "ok",
            "staged_path": str(dest.relative_to(REPO_ROOT)),
        }
    except Exception as e:
        print(f"  [FAIL] {hero_id}: image download failed — {e}")
        return None


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------


def scrape_heroes(
    hero_ids: list[str],
    dry_run: bool = False,
    delay_range: tuple[float, float] = (1.5, 3.5),
) -> dict[str, dict]:
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    ACTIVE_DIR.mkdir(parents=True, exist_ok=True)

    session = _get_session()
    manifest: dict[str, dict] = _load_manifest()
    results = {"ok": [], "failed": [], "skipped": []}

    for hero_id in hero_ids:
        dest = STAGING_DIR / f"{hero_id}.png"

        if dest.exists():
            print(f"  [SKIP] {hero_id}: already staged")
            results["skipped"].append(hero_id)
            continue

        print(f"  {hero_id} → {_wiki_url(hero_id)}")

        if dry_run:
            results["ok"].append(hero_id)
            continue

        img_url = _find_portrait_url(session, hero_id)
        if not img_url:
            manifest[hero_id] = {"hero_id": hero_id, "status": "failed", "source_url": None}
            results["failed"].append(hero_id)
        else:
            print(f"         image: {img_url}")
            entry = _download_image(session, img_url, dest, hero_id)
            if entry:
                manifest[hero_id] = entry
                results["ok"].append(hero_id)
                print(f"         saved: {dest.relative_to(REPO_ROOT)}")
            else:
                manifest[hero_id] = {"hero_id": hero_id, "status": "failed", "source_url": img_url}
                results["failed"].append(hero_id)

        _save_manifest(manifest)

        # Polite delay — avoid hammering the server
        delay = random.uniform(*delay_range)
        time.sleep(delay)

    return results


def promote_staged(hero_ids: list[str]) -> None:
    """Copy staged images to the active directory."""
    for hero_id in hero_ids:
        src = STAGING_DIR / f"{hero_id}.png"
        dst = ACTIVE_DIR / f"{hero_id}.png"
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  promoted: {hero_id}.png → {dst.relative_to(REPO_ROOT)}")
        else:
            print(f"  [MISS]  {hero_id}: no staged file to promote")


def print_summary(manifest: dict, results: dict) -> None:
    total = len(HERO_WIKI_NAMES)
    ok = len(results["ok"])
    skipped = len(results["skipped"])
    failed = len(results["failed"])

    print()
    print("=" * 56)
    print(f"  Hero Icons — Scrape Summary")
    print("=" * 56)
    print(f"  Total heroes : {total}")
    print(f"  Downloaded   : {ok}")
    print(f"  Skipped      : {skipped}  (already staged)")
    print(f"  Failed       : {failed}")
    print()

    if results["ok"]:
        print("  Successfully staged:")
        for h in results["ok"]:
            entry = manifest.get(h, {})
            res = entry.get("resolution", "?")
            print(f"    {h:<30} {res}")

    if results["failed"]:
        print()
        print("  Failed — needs manual sourcing:")
        for h in results["failed"]:
            print(f"    {h}")
        print()
        print("  For failed heroes, find a clean portrait PNG and save to:")
        print(f"    {STAGING_DIR.relative_to(REPO_ROOT)}/<hero_id>.png")
        print("  Then run:  python tools/scrape_hero_icons.py --promote")

    print()
    print(f"  Manifest saved: {MANIFEST_PATH.relative_to(REPO_ROOT)}")
    print()


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def _save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape Marvel Rivals hero icons from wiki.gg")
    parser.add_argument(
        "--hero",
        metavar="HERO_ID",
        help="Scrape a single hero by entity ID (e.g. iron_man)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print target URLs without downloading",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Copy all staged images to the active directory",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Only attempt heroes marked as failed in the manifest",
    )
    args = parser.parse_args()

    if args.promote:
        manifest = _load_manifest()
        staged = [h for h in manifest if manifest[h].get("status") == "ok"]
        promote_staged(staged)
        return

    if args.hero:
        if args.hero not in HERO_WIKI_NAMES:
            print(f"Unknown hero ID: {args.hero}")
            print(f"Valid IDs: {', '.join(sorted(HERO_WIKI_NAMES))}")
            return
        hero_ids = [args.hero]
    elif args.retry_failed:
        manifest = _load_manifest()
        hero_ids = [h for h, v in manifest.items() if v.get("status") == "failed"]
        if not hero_ids:
            print("No failed heroes in manifest.")
            return
        print(f"Retrying {len(hero_ids)} failed heroes...")
        # Remove failed entries so they aren't skipped
        for h in hero_ids:
            src = STAGING_DIR / f"{h}.png"
            src.unlink(missing_ok=True)
    else:
        hero_ids = list(HERO_WIKI_NAMES.keys())

    print(f"Scraping {len(hero_ids)} hero(es) from wiki.gg...")
    print(f"Staging dir: {STAGING_DIR.relative_to(REPO_ROOT)}")
    print()

    results = scrape_heroes(hero_ids, dry_run=args.dry_run)
    manifest = _load_manifest()
    print_summary(manifest, results)


if __name__ == "__main__":
    main()
