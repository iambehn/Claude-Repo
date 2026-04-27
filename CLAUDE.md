# ClipBot — Session Context

This file is read automatically by Claude Code at the start of every session.
It replaces the need to re-explain the project each time.

---

## What This Project Is

A gaming clip farming bot that automatically detects highlight moments in
gameplay videos, scores them, and routes them through a review queue before
distribution to YouTube Shorts, TikTok, and Reddit.

**Games:** Marvel Rivals, Deadlock, Arc Raiders
**Stack:** OpenCV + YOLO (detection), FFmpeg (processing), Whisper (transcription),
Flask (review UI), Claude API (AI scoring), yt-dlp (ingestion)

---

## Pipeline Order

**VOD Mining (optional pre-ingestion):**
```
VOD URL → Proxy Scanner (chat + viewer clips + audio spikes)
        → Candidate Windows → yt-dlp segment download → inbox/{game}/
```

**Clip Processing:**
```
Ingestion → Transcription → Feature Extraction → Kill Feed → Weapon Detector
→ Audio Detector → ROI Matcher → Clip Judge → Processing → AI Scoring
→ Manual Review → Distribution
```

Each stage writes its output to a `.meta.json` sidecar file next to the clip.
The clip judge reads all detector outputs and makes the final accept / quarantine / reject decision.

**Run commands:**
```bash
python run.py --game marvel_rivals   # process one game
python run.py --game all             # process all games
python run.py --watch                # continuous loop
python run.py --enrich-quarantine marvel_rivals  # re-evaluate quarantined clips

# VOD mining — finds highlight windows before downloading full video
python run.py --scan-vod https://twitch.tv/videos/12345 marvel_rivals
python run.py --scan-vod https://twitch.tv/videos/12345 marvel_rivals --chat-log chat.log
```

---

## Folder Structure

```
assets/games/{game}/     # per-game config: game.yaml, hud.yaml, weights.yaml, etc.
assets/gold_set/         # labeled clip snapshots for evaluation
inbox/{game}/            # clips waiting to be processed
quarantine/{game}/       # clips that need more context or a better hook
accepted/{game}/         # clips approved for distribution
rejected/{game}/         # clips that scored too low
pipeline/                # all pipeline stage modules
utils/                   # shared helpers
tools/                   # CLI utilities (snip_roi, etc.)
```

---

## Per-Game Config

Every game has a directory at `assets/games/{game}/` containing:
- `game.yaml` — display name, detector flags, resolution profiles
- `hud.yaml` — ROI coordinates (pct-based at 1920×1080), templates
- `weights.yaml` — scoring weights and decision thresholds
- `entities.yaml` — heroes and weapons
- `moments.yaml` — named gameplay moments with score multipliers
- `abilities.yaml` — character abilities with worthiness boosts
- `proxy.yaml` — proxy scanner signal weights and thresholds (overrides global config.yaml)

Load with: `from pipeline import game_pack; pack = game_pack.load("marvel_rivals")`

---

## Scoring Philosophy

The clip judge (`pipeline/clip_judge.py`) combines three sub-scores:

- `context_confidence` — how much detector evidence exists (kill feed, audio, weapon, ROI)
- `hook_confidence` — does something interesting happen in the first 1.5 seconds?
- `postability_score` — how likely is this clip to perform if posted?

**Key principle:** OpenCV and YOLO are signal extractors, not judges.
They produce timestamped events with confidence scores.
The clip judge decides what those events mean.

**Decision thresholds** (in `weights.yaml`):
- `accept >= 0.65` and `context >= 0.40` and `hook >= 0.40`
- `reject < 0.35` and `context >= 0.60`
- everything else → quarantine with a reason

---

## What Has Been Built

- Full ingestion → distribution pipeline (`run.py`)
- Clip judge with three-component scoring (`pipeline/clip_judge.py`)
- Per-game pack system (`pipeline/game_pack.py`)
- Kill feed detector (`pipeline/kill_feed.py`)
- Audio detector (`pipeline/audio_detector.py`)
- Weapon detector (`pipeline/weapon_detector.py`)
- ROI matcher (`pipeline/roi_matcher.py`)
- Quarantine enrichment loop (`run.py --enrich-quarantine`)
- Flask review UI with canvas ROI editor (`pipeline/review/app.py`)
- Gold set evaluation harness (`pipeline/evaluate.py`)
- Proxy scanner for full VOD mining (`pipeline/proxy_scanner.py`, `pipeline/chat_scanner.py`)
  — chat velocity, viewer clip clusters, audio spikes; per-game `proxy.yaml` config
  — `run.py --scan-vod URL GAME [--chat-log FILE]`

---

## Design Principles (Read Before Adding Features)

1. **Lean over complete.** One working detector beats five half-built ones.
2. **Meta.json is the shared state.** Every stage reads and writes this file.
   Nothing communicates through function arguments across stage boundaries.
3. **Per-game config, not hardcoded logic.** Thresholds, weights, and ROI
   coordinates live in `assets/games/{game}/`. Change behavior by editing YAML,
   not code.
4. **The review UI is the feedback loop.** Approved and rejected clips become
   labeled data. Don't skip the review step.
5. **Don't add a feature until you feel the pain of not having it.**
   Traceability, evaluation harnesses, and multi-engine fusion are legitimate
   future goals — but only after the core scoring loop is reliable on real clips.

---

## Current Focus

Building reliable per-game scoring logic using OpenCV as the detection
foundation. Priority order:

1. Get kill feed detection producing trustworthy timestamps on real footage
2. Validate audio spike detection is not firing on music/notifications
3. Tune `weights.yaml` thresholds using real accepted/rejected clip data
4. Expand the gold set (`assets/gold_set/`) as real clips are reviewed

---

## What To Avoid

- Adding more detectors before existing ones are measured
- Architectural abstractions before the core loop works end-to-end
- Changing weights without a before/after evaluation run
- Committing to `main` without testing on at least one real clip

---

## Key Files To Read First

When starting a new task, read these before writing any code:

| File | Why |
|---|---|
| `pipeline/clip_judge.py` | The scoring logic — understand this before touching weights |
| `pipeline/game_pack.py` | How per-game config loads — read before adding new YAML fields |
| `assets/games/marvel_rivals/weights.yaml` | Current thresholds for the primary test game |
| `config.yaml` | Global pipeline config — detector enable flags live here |
| `pipeline/evaluate.py` | Run this before and after any weight change |
