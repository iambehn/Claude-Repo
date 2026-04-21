"""
tools/preview_roi.py — Preview all configured ROIs on a frame from a gameplay clip

Saves a single annotated PNG showing every ROI rectangle (kill_feed, weapon_detector)
that is configured for the given game. Use this to verify coordinates are correct
before running the full pipeline, or after a HUD patch changes icon positions.

Usage:
    python tools/preview_roi.py --clip inbox/deadlock/myclip.mp4 --game deadlock

    # Sample a specific timestamp instead of mid-clip:
    python tools/preview_roi.py --clip inbox/deadlock/myclip.mp4 --game deadlock --time 8.0

Output:
    assets/roi_preview_{game}.png   — open this file to inspect ROI placement
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import game_pack  # noqa: E402

TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080

# (label, BGR colour)
_ROI_STYLES = [
    ("kill_feed",       (0,  200,  0)),   # green
    ("weapon_detector", (0,  180, 255)),  # orange
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save an annotated preview frame showing all configured ROIs for a game."
    )
    parser.add_argument("--clip", required=True, help="Path to a gameplay clip (.mp4)")
    parser.add_argument("--game", required=True, help="Game key matching config.yaml")
    parser.add_argument("--time", type=float, default=None,
                        help="Timestamp in seconds to sample (default: mid-clip)")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    try:
        import cv2
    except ImportError:
        print("ERROR: OpenCV not installed. Run: pip install opencv-python-headless", file=sys.stderr)
        sys.exit(1)

    try:
        pack = game_pack.load(args.game)
    except (FileNotFoundError, game_pack.GamePackError) as exc:
        print(f"ERROR: could not load game pack '{args.game}': {exc}", file=sys.stderr)
        sys.exit(1)

    clip_path = Path(args.clip)
    if not clip_path.exists():
        print(f"ERROR: clip not found: {clip_path}", file=sys.stderr)
        sys.exit(1)

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        print(f"ERROR: could not open clip: {clip_path}", file=sys.stderr)
        sys.exit(1)

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration = total_frames / fps

        sample_time = args.time if args.time is not None else duration / 2
        cap.set(cv2.CAP_PROP_POS_MSEC, sample_time * 1000)
        ok, frame = cap.read()
        if not ok:
            print("ERROR: could not read frame.", file=sys.stderr)
            sys.exit(1)

        h, w = frame.shape[:2]
        if w != TARGET_WIDTH or h != TARGET_HEIGHT:
            frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)

        rois_drawn = 0

        # Kill-feed ROI
        kf_roi = pack.hud.get("kill_feed")
        if kf_roi:
            rx, ry, rw, rh = kf_roi["x"], kf_roi["y"], kf_roi["w"], kf_roi["h"]
            color = _ROI_STYLES[0][1]
            cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), color, 2)
            cv2.putText(frame, "kill_feed ROI", (rx, max(ry - 6, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            rois_drawn += 1

        # Weapon-detector ROI
        wd_roi = pack.hud.get("weapon_detector")
        if wd_roi:
            rx, ry, rw, rh = wd_roi["x"], wd_roi["y"], wd_roi["w"], wd_roi["h"]
            color = _ROI_STYLES[1][1]
            cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), color, 2)
            cv2.putText(frame, "weapon ROI", (rx, max(ry - 6, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            rois_drawn += 1

        out_path = ROOT / "assets" / f"roi_preview_{args.game}.png"
        cv2.imwrite(str(out_path), frame)
        print(f"Saved preview: {out_path.relative_to(ROOT)}  ({rois_drawn} ROI(s) drawn)")
        print(f"  Sampled at t={sample_time:.1f}s of {duration:.1f}s total")
        if rois_drawn == 0:
            print(f"  WARNING: No ROIs found in assets/games/{args.game}/hud.yaml")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
