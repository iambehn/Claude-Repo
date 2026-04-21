"""
tools/snip_roi.py — Snip a HUD ROI template from a gameplay clip

Extracts a crop from a normalized 1920×1080 frame and registers it as a
template in assets/games/{game}/hud.yaml for use by roi_matcher.

Usage:
    # Non-interactive (headless): provide --crop
    python tools/snip_roi.py \\
        --clip inbox/marvel_rivals/myclip.mp4 \\
        --game marvel_rivals \\
        --template-id headshot_medal \\
        --roi-name kill_feed \\
        --time 12.5 \\
        --crop "1620,10,300,380" \\
        --match-threshold 0.84

    # Interactive (requires display): omit --crop to draw rectangle with mouse
    python tools/snip_roi.py \\
        --clip inbox/marvel_rivals/myclip.mp4 \\
        --game marvel_rivals \\
        --template-id headshot_medal \\
        --roi-name kill_feed \\
        --time 12.5

    # Save annotated debug frame alongside the template
    python tools/snip_roi.py ... --debug

After running:
    1. Check assets/roi_library/{game}/{template_id}.png looks correct.
    2. Enable roi_matcher in config.yaml:
           roi_matcher:
             enabled: true
    3. Optionally re-evaluate quarantined clips:
           python run.py --enrich-quarantine {game}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from utils.roi_utils import extract_frame, save_template, TARGET_WIDTH, TARGET_HEIGHT  # noqa: E402
from pipeline import game_pack  # noqa: E402

_DISPLAY_ENV_VARS = ("DISPLAY", "WAYLAND_DISPLAY")


def _has_display() -> bool:
    import os
    return any(os.environ.get(v) for v in _DISPLAY_ENV_VARS)


def _interactive_crop(frame: "import numpy; numpy.ndarray", window_title: str) -> dict[str, int] | None:
    """Open an OpenCV window; user drags a rectangle. Returns {x,y,w,h} or None on cancel."""
    import cv2
    import numpy as np

    clone = frame.copy()
    drawing = False
    start = (0, 0)
    rect = [None]   # mutable container

    # Scale down to fit screen if needed
    disp_scale = 1.0
    if frame.shape[1] > 1600:
        disp_scale = 1600 / frame.shape[1]

    def _scale(pt: tuple[int, int]) -> tuple[int, int]:
        return (int(pt[0] / disp_scale), int(pt[1] / disp_scale))

    disp_frame = frame if disp_scale == 1.0 else cv2.resize(
        frame, (int(frame.shape[1] * disp_scale), int(frame.shape[0] * disp_scale))
    )
    canvas = disp_frame.copy()

    def on_mouse(event, x, y, flags, param):
        nonlocal drawing, start, canvas
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start = (x, y)
            canvas = disp_frame.copy()
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            canvas = disp_frame.copy()
            cv2.rectangle(canvas, start, (x, y), (0, 255, 0), 2)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            # Normalize rect so w/h are always positive
            x1, y1 = min(start[0], x), min(start[1], y)
            x2, y2 = max(start[0], x), max(start[1], y)
            rect[0] = {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}
            canvas = disp_frame.copy()
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 200, 255), 2)

    cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_title, on_mouse)

    print("Draw a rectangle with the mouse, then press Enter to confirm or Esc to cancel.")

    while True:
        cv2.imshow(window_title, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key == 13 and rect[0] is not None:   # Enter
            break
        if key == 27:   # Esc
            rect[0] = None
            break
        if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
            rect[0] = None
            break

    cv2.destroyWindow(window_title)

    if rect[0] is None:
        return None

    # Scale coords back to 1920×1080
    r = rect[0]
    return {
        "x": int(r["x"] / disp_scale),
        "y": int(r["y"] / disp_scale),
        "w": int(r["w"] / disp_scale),
        "h": int(r["h"] / disp_scale),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and register a HUD template from a gameplay clip."
    )
    parser.add_argument("--clip", required=True, help="Path to the source gameplay clip (.mp4)")
    parser.add_argument("--game", required=True, help="Game slug (e.g. marvel_rivals)")
    parser.add_argument("--template-id", required=True, dest="template_id",
                        help="snake_case identifier used as the PNG filename")
    parser.add_argument("--roi-name", required=True, dest="roi_name",
                        help="ROI name in hud.yaml (e.g. kill_feed, weapon_detector)")
    parser.add_argument("--time", type=float, default=None,
                        help="Timestamp in seconds to sample (default: mid-clip)")
    parser.add_argument("--crop", default=None,
                        help="Crop region as 'x,y,w,h' at 1920×1080. Omit for interactive mode.")
    parser.add_argument("--match-threshold", type=float, default=0.84, dest="match_threshold",
                        help="Confidence threshold for this template (default: 0.84)")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config.yaml (default: config.yaml)")
    parser.add_argument("--debug", action="store_true",
                        help="Save annotated full-frame debug image alongside the template.")
    args = parser.parse_args()

    try:
        import cv2
        import numpy as np
    except ImportError:
        print("ERROR: OpenCV not installed. Run: pip install opencv-python-headless", file=sys.stderr)
        sys.exit(1)

    config_path = ROOT / args.config
    if not config_path.exists():
        print(f"ERROR: config not found at {config_path}", file=sys.stderr)
        sys.exit(1)
    with open(config_path) as fh:
        config = yaml.safe_load(fh)

    clip_path = (ROOT / args.clip).resolve() if not Path(args.clip).is_absolute() else Path(args.clip)
    if not clip_path.exists():
        print(f"ERROR: clip not found: {clip_path}", file=sys.stderr)
        sys.exit(1)

    # Validate game pack exists and ROI is defined
    try:
        pack = game_pack.load(args.game)
    except (FileNotFoundError, game_pack.GamePackError) as exc:
        print(f"ERROR: could not load game pack '{args.game}': {exc}", file=sys.stderr)
        sys.exit(1)

    if pack.hud.get(args.roi_name) is None:
        print(f"ERROR: ROI '{args.roi_name}' not found in assets/games/{args.game}/hud.yaml", file=sys.stderr)
        sys.exit(1)

    # Determine sample timestamp
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        print(f"ERROR: Could not open clip: {clip_path}", file=sys.stderr)
        sys.exit(1)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = total_frames / fps
    cap.release()

    sample_time = args.time if args.time is not None else duration / 2
    if sample_time > duration:
        print(f"WARNING: --time {sample_time}s exceeds duration {duration:.1f}s — using mid-clip.")
        sample_time = duration / 2

    print(f"Extracting frame at t={sample_time:.2f}s ({duration:.1f}s clip)…")
    try:
        frame = extract_frame(clip_path, sample_time)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    # Resolve crop
    if args.crop:
        parts = [p.strip() for p in args.crop.split(",")]
        if len(parts) != 4:
            print("ERROR: --crop must be 'x,y,w,h'", file=sys.stderr)
            sys.exit(1)
        try:
            crop = {"x": int(parts[0]), "y": int(parts[1]), "w": int(parts[2]), "h": int(parts[3])}
        except ValueError:
            print("ERROR: --crop values must be integers", file=sys.stderr)
            sys.exit(1)
    elif _has_display():
        crop = _interactive_crop(frame, f"snip_roi: {args.template_id} — draw rectangle, Enter to save")
        if crop is None:
            print("Cancelled.")
            sys.exit(0)
        print(f"  Crop captured: x={crop['x']}, y={crop['y']}, w={crop['w']}, h={crop['h']}")
        print(f"  Tip: re-run with --crop \"{crop['x']},{crop['y']},{crop['w']},{crop['h']}\" for headless use.")
    else:
        # Save a full frame so the user can inspect it and supply --crop
        out_dir = ROOT / "assets" / "roi_library" / args.game
        out_dir.mkdir(parents=True, exist_ok=True)
        frame_path = out_dir / f"{args.template_id}_frame.png"
        cv2.imwrite(str(frame_path), frame)
        print(f"Headless mode: no display detected.")
        print(f"Full frame saved to: {frame_path.relative_to(ROOT)}")
        print(f"  Open it, note the pixel coordinates, then re-run with:")
        print(f"      --crop \"x,y,w,h\"")
        sys.exit(0)

    # Save the template
    try:
        out_path = save_template(
            game=args.game,
            clip_path=clip_path,
            frame_time=sample_time,
            crop=crop,
            template_id=args.template_id,
            roi_name=args.roi_name,
            match_threshold=args.match_threshold,
            config=config,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Saved template: {out_path.relative_to(ROOT)}  ({crop['w']}×{crop['h']} px)")
    print(f"Updated:        assets/games/{args.game}/hud.yaml  (templates list)")

    if args.debug:
        debug_frame = frame.copy()
        roi_px = pack.hud.get(args.roi_name)
        if roi_px:
            rx, ry, rw, rh = roi_px["x"], roi_px["y"], roi_px["w"], roi_px["h"]
            cv2.rectangle(debug_frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
            cv2.putText(debug_frame, f"ROI: {args.roi_name}", (rx, max(ry - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cx, cy, cw, ch = crop["x"], crop["y"], crop["w"], crop["h"]
        cv2.rectangle(debug_frame, (cx, cy), (cx + cw, cy + ch), (0, 180, 255), 2)
        cv2.putText(debug_frame, args.template_id, (cx, max(cy - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2)
        debug_path = out_path.parent / f"{args.template_id}_debug.png"
        cv2.imwrite(str(debug_path), debug_frame)
        print(f"Debug frame:    {debug_path.relative_to(ROOT)}")

    print()
    print("Next steps:")
    print(f"  1. Inspect the saved PNG to confirm it looks correct.")
    print(f"  2. Enable roi_matcher in config.yaml:")
    print(f"         roi_matcher:")
    print(f"           enabled: true")
    print(f"  3. Re-evaluate quarantined clips (optional):")
    print(f"         python run.py --enrich-quarantine {args.game}")


if __name__ == "__main__":
    main()
