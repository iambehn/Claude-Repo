"""
tools/extract_weapon_icon.py — Extract and save a weapon icon from a gameplay clip

Reads the weapon-icon ROI defined in config.yaml for the given game, crops it
from a frame of the clip (normalized to 1920×1080), and saves it as a PNG
reference image ready for use by the Weapon Detector.

Usage:
    python tools/extract_weapon_icon.py \\
        --clip inbox/deadlock/myclip.mp4 \\
        --game deadlock \\
        --weapon-id sniper_rifle \\
        --display-name "Sniper Rifle"

    # Use a specific timestamp (seconds) instead of the middle frame:
    python tools/extract_weapon_icon.py \\
        --clip inbox/deadlock/myclip.mp4 \\
        --game deadlock \\
        --weapon-id sniper_rifle \\
        --display-name "Sniper Rifle" \\
        --time 12.5

    # Save a full debug frame with the ROI rectangle drawn on it:
    python tools/extract_weapon_icon.py ... --debug

After running:
    1. Check assets/weapon_icons/{game}/{weapon_id}.png looks correct.
    2. Add the entity to assets/games/{game}/entities.yaml:
           - id: sniper_rifle
             kind: weapon
             display_name: "Sniper Rifle"
             role: "sniper"
             aliases: ["sniper"]
    3. Set weapon_detector.enabled: true and title_engine.enabled: true in config.yaml.
    4. Run the pipeline — the title engine will now inject the detected weapon name.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import game_pack  # noqa: E402

TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a weapon HUD icon from a gameplay clip for use as a Weapon Detector reference image."
    )
    parser.add_argument("--clip", required=True, help="Path to the source gameplay clip (.mp4)")
    parser.add_argument("--game", required=True, help="Game key matching config.yaml (e.g. deadlock)")
    parser.add_argument("--weapon-id", required=True, dest="weapon_id",
                        help="Identifier for this weapon (snake_case, e.g. sniper_rifle). Used as the PNG filename.")
    parser.add_argument("--display-name", dest="display_name", default=None,
                        help="Human-readable name for config.yaml (e.g. 'Sniper Rifle'). Defaults to weapon-id title-cased.")
    parser.add_argument("--time", type=float, default=None,
                        help="Timestamp in seconds to sample. Defaults to mid-clip.")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config.yaml (default: config.yaml)")
    parser.add_argument("--debug", action="store_true",
                        help="Also save a full frame with the ROI drawn on it for visual inspection.")
    args = parser.parse_args()

    try:
        import cv2
    except ImportError:
        print("ERROR: OpenCV not installed. Run: pip install opencv-python-headless", file=sys.stderr)
        sys.exit(1)

    import numpy as np  # noqa: F401 — available when cv2 is

    config_path = ROOT / args.config
    if not config_path.exists():
        print(f"ERROR: config not found at {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    wd_cfg = config.get("weapon_detector", {})
    try:
        pack = game_pack.load(args.game)
    except (FileNotFoundError, game_pack.GamePackError) as exc:
        print(f"ERROR: could not load game pack '{args.game}': {exc}", file=sys.stderr)
        sys.exit(1)

    roi = pack.hud.get("weapon_detector")
    if roi is None:
        print(
            f"ERROR: No 'weapon_detector' roi in assets/games/{args.game}/hud.yaml",
            file=sys.stderr,
        )
        sys.exit(1)
    rx, ry, rw, rh = roi["x"], roi["y"], roi["w"], roi["h"]

    clip_path = Path(args.clip)
    if not clip_path.exists():
        print(f"ERROR: Clip not found: {clip_path}", file=sys.stderr)
        sys.exit(1)

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        print(f"ERROR: Could not open clip: {clip_path}", file=sys.stderr)
        sys.exit(1)

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration = total_frames / fps

        sample_time = args.time if args.time is not None else duration / 2
        if sample_time > duration:
            print(f"WARNING: --time {sample_time}s exceeds clip duration {duration:.1f}s — using mid-clip.")
            sample_time = duration / 2

        cap.set(cv2.CAP_PROP_POS_MSEC, sample_time * 1000)
        ok, frame = cap.read()
        if not ok:
            print("ERROR: Could not read frame from clip.", file=sys.stderr)
            sys.exit(1)

        # Normalize to 1920×1080 (same as detector)
        h, w = frame.shape[:2]
        if w != TARGET_WIDTH or h != TARGET_HEIGHT:
            print(f"  Resizing frame from {w}×{h} → {TARGET_WIDTH}×{TARGET_HEIGHT}")
            frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)

        icon = frame[ry:ry + rh, rx:rx + rw]
        if icon.size == 0:
            print(f"ERROR: ROI crop is empty — check roi coordinates (x={rx}, y={ry}, w={rw}, h={rh})", file=sys.stderr)
            sys.exit(1)

        out_dir = ROOT / wd_cfg.get("icon_dir", "assets/weapon_icons") / args.game
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{args.weapon_id}.png"
        cv2.imwrite(str(out_path), icon)
        print(f"Saved icon: {out_path.relative_to(ROOT)}  ({icon.shape[1]}×{icon.shape[0]} px)")

        if args.debug:
            debug_frame = frame.copy()
            cv2.rectangle(debug_frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
            cv2.putText(debug_frame, f"ROI: {args.weapon_id}", (rx, max(ry - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            debug_path = out_dir / f"{args.weapon_id}_debug.png"
            cv2.imwrite(str(debug_path), debug_frame)
            print(f"Debug frame: {debug_path.relative_to(ROOT)}")

        display_name = args.display_name or args.weapon_id.replace("_", " ").title()
        print()
        print(f"Next step — add to assets/games/{args.game}/entities.yaml:")
        print("  entities:")
        print(f"    - id: {args.weapon_id}")
        print("      kind: weapon")
        print(f"      display_name: \"{display_name}\"")
        print("      role: \"unknown\"   # TODO: replace with a real role")
        print("      aliases: []")
        print()
        print("Then enable detection:")
        print("  weapon_detector:")
        print("    enabled: true")
        print("  title_engine:")
        print("    enabled: true")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
