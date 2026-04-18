# Gaming Clip Farming Bot — Project Planning

> Living document. Update as the system evolves.

---

## Pipeline Overview

```
[ Ingestion ] → [ Transcription ] → [ Feature Extraction ] → [ Decision Engine ]
     → [ Processing ] → [ AI Scoring ] → [ Manual Review ] → [ Distribution ] → [ Optimize ]
```

**Run modes:**
- `python run.py --game <game>` — single pipeline run for one game
- `python run.py --game all` — run all configured games once
- `python run.py --watch` — continuous loop for all games (`pipeline.watch_interval_seconds` interval)
- `python run.py --distribute [--dry-run]` — upload all approved clips; `--dry-run` previews without posting
- `python run.py --montage <game>` — assemble a montage from accepted clips
- `python run.py --poll-tiktok` — resolve pending TikTok URLs for uploaded clips
- `python run.py --list-reddit-flairs` — print available flair IDs per subreddit

---

## Stages

### 1. Ingestion
**Tools:** Twitch Helix API + yt-dlp + FFprobe

- Fetch top clips via Twitch Helix API (`TWITCH_CLIENT_ID` + `TWITCH_CLIENT_SECRET`)
- Download via yt-dlp into `inbox/{game}/`
- Probe each file with FFprobe: duration, resolution, fps, has_audio
- Assign quality tag: `high` (≥1080p), `medium` (≥720p), `low` (≥480p)
- Quarantine: probe failure, no video stream, duration out of configured range, resolution < 480p
- Write sidecar `.meta.json` per good clip; skip if sidecar already exists (idempotent)
- Clip freshness filter: `ingestion.max_clip_age_hours: 72` — clips older than this (by Twitch `started_at`) are skipped; set `null` to disable

**Folder structure:**
```
inbox/{game}/     quarantine/{game}/     processing/{game}/     accepted/{game}/     rejected/{game}/
```

---

### 2. Transcription
**Tool:** Whisper (OpenAI) — segment-level output

- Convert speech to text; stored alongside clip, used by Feature Extraction and AI Scoring
- Language filter: Whisper-detected language checked against `transcription.language_filter: "en"`; clips in other languages are skipped
- **Future:** WhisperX for word-level timestamps (word-highlight captions)

---

### 3. Feature Extraction

Extracts and merges into `.meta.json`: duration, motion level, audio levels, keywords (from transcript), quality tag.

---

### 4. Decision Engine

Rule-based template selection from clip metadata (no AI at this stage):

**FPS Keyword Lists:**

| Scope | Keywords |
|---|---|
| All FPS (shared) | ace, kill, clutch, headshot, one tap, multi-kill, 1v5, streak, insane, insane play |
| Arc Raiders | wipe, squad wipe, extraction, clutch extract, solo, ambush, no scope, raid, down |
| Marvel Rivals | ult, team wipe, wombo combo, flank, dive, penta, dominate, shutdown, combo |
| Deadlock | gank, teamfight, carry, comeback, steal, outplay, lane, ambush |

**Template Library** (`templates/`):
- `fast_hype/` — high energy, fast cuts, kill streaks
- `cinematic_highlight/` — single impactful moment, cinematic pacing
- `recap_montage/` — multi-clip compilation
- `commentary_reaction/` — low-motion, mic-heavy
- `tutorial_tips/` — instructional, keyword-triggered

Templates are versioned (e.g. `template_hype_v2`) to allow refinement without losing prior versions.

---

### 5. Processing
**Tool:** FFmpeg

- Applies selected template; encodes output per platform format
- Audio modes: `original | mute | replace` (config.yaml `audio.mode`)
- Caption filter: `subtitles=filename=` (requires FFmpeg with libass; falls back gracefully if missing — fix: `brew upgrade ffmpeg`)
- **Future:** Background music slot in templates (`enabled: false`, `asset_path: null`) — add royalty-free track, set `enabled: true`

---

### 6. AI Scoring Engine
**Tool:** Claude API

- Runs on finished processed clip; assigns virality score (0–100)
- Score factors: energy, keyword relevance, pacing, audio impact, visual quality
- Score stored in `.meta.json`, displayed in Manual Review

---

### 7. Manual Review
**Tool:** Flask web app

- Video player with virality score display
- Approve → `accepted/` | Reject → `rejected/`
- Queue skips already-reviewed clips
- Video thumbnails: highest-motion keyframe extracted via FFmpeg scene detection; cached as `.thumb.jpg` sidecar; served at `/thumb/<game>/<stem>`

---

### 8. Distribution
**Platforms:** YouTube Shorts, TikTok, Instagram Reels, Twitter/X, Reddit

| Platform       | Resolution  | Aspect | Max Duration |
|----------------|-------------|--------|--------------|
| YouTube Shorts | 1080×1920   | 9:16   | 60s          |
| TikTok         | 1080×1920   | 9:16   | 10 min       |
| Instagram Reels| 1080×1920   | 9:16   | 90s          |
| Twitter/X      | 1280×720    | 16:9   | 2m 20s       |
| Reddit         | 1920×1080   | 16:9   | varies       |

Idempotent: already-distributed platforms skipped. Run via `python run.py --distribute`.
Use `--dry-run` to preview what would be uploaded without posting anything.
Post-distribution: analytics logged to Google Sheets, clip backed up to Google Drive.

**File naming:** `{game}_{date}_{clip_id}.mp4`

**YouTube:** Thumbnail auto-generated from highest-motion keyframe (FFmpeg scene detection); uploaded via `thumbnails.set()` after video publish.

**TikTok:** Clips uploaded with a `publish_id`; URL resolved asynchronously. Run `python run.py --poll-tiktok` to check processing status and update any pending URLs.

**Reddit:** Flair IDs are subreddit-specific. Run `python run.py --list-reddit-flairs` to discover available flairs; add chosen IDs to `config.yaml → distribution.platforms.reddit.subreddit_config`.

---

### 9. Optimize

- Primary metric: **Viewership** (total views per published clip)
- One variable at a time; measure before and after each change
- Versioned templates enable rollback
- Loop: `Experience → Measurement → System Design → repeat`

---

## Backlog — Copyright Music Detection ✓ Implemented

Twitch clips often contain licensed music. YouTube Content ID and TikTok will mute, monetize against, or remove flagged videos.

**How it works:** `pipeline/processing.py` calls ACRCloud before applying any template. A 10-second fingerprint is taken at the 10-second mark of each clip and sent to the ACRCloud API. If confidence ≥ threshold (default 80), the configured `audio.mode` is applied to that clip. If no match, original audio is kept regardless of mode setting.

**Fallback behaviour:** if detection fails (quota exceeded, network error, missing library) the configured `audio.mode` is applied as a blanket policy.

**Setup:**
```
pip install pyacrcloud
# .env:
ACRCLOUD_ACCESS_KEY=...
ACRCLOUD_ACCESS_SECRET=...
ACRCLOUD_HOST=identify-eu-west-1.acrcloud.com
```
```yaml
# config.yaml:
audio:
  mode: "mute"           # applied when copyright is detected
  detection:
    enabled: true
    confidence_threshold: 80
```
Free tier: 100 recognitions/day. Sign up at https://www.acrcloud.com

**Alternative:** AudD.io — simpler REST API, 100 free/month then $20/month.

---

## Setup Notes

### YouTube API Key (Scout Dashboard + Thumbnails)
The scout dashboard needs a **read-only API key** (not OAuth) for the YouTube signal. This is separate from the OAuth credentials used to upload Shorts.

1. Go to Google Cloud Console → APIs & Services → Credentials
2. Create API key → restrict it to **YouTube Data API v3** only
3. Add to `.env`:
```
YOUTUBE_API_KEY=AIza...
```
Free quota: 10,000 units/day. Each scout search = 100 units. Polling 10 games every 6h = 40 requests/day — well within the free limit.

The existing OAuth credentials (`YOUTUBE_CLIENT_ID` / `YOUTUBE_CLIENT_SECRET`) are only needed for uploading videos and are unrelated to this key.

---

## Backlog — Quality of Life Features

1. **Retry failed distributions** — `python run.py --retry-failed` for clips with `distribution.{platform}.success: false`.
2. **Clip deduplication** — Store seen Twitch clip IDs in `seen_clips.json`; skip re-downloads regardless of filename.
3. **Completion notification** — Desktop/email alert when batch run finishes (macOS: `osascript`).
4. **Aggregate stats in review UI** — `/stats` route: clips by game, acceptance rate, avg score, top keywords, clips per platform.

---

## Backlog — Pipeline Scaling & Orchestration

The current pipeline is single-threaded, sequential, and uses filesystem + `.meta.json` sidecars as its state store. This is correct for the current scale. The notes below document the upgrade path when volume or reliability requirements grow.

### Current vs. Target

| Concern | Current | Target (at scale) |
|---|---|---|
| State store | `.meta.json` sidecars + folder structure | SQLite → Postgres |
| Execution | Sequential, single-threaded | Parallel workers via queue |
| Retry logic | None — pipeline stops on stage failure | Exponential backoff per stage |
| Scheduling | `--watch` polling loop | cron / n8n DAG trigger |
| Orchestration | `run.py` monolith | Airflow / Prefect for complex multi-game DAGs |

### Clip State Machine

Currently approximated by folder location (`inbox/` → `processing/` → `accepted/`). A DB status column makes the state explicit and queryable:

```
DOWNLOADED → TRANSCRIBED → PROCESSED → SCORED → REVIEWED → UPLOADED
```

### Retry Strategy

Never crash the whole pipeline over one clip failure — log it, mark the clip failed, continue.

```
retry_delay = base_delay × (2 ^ retry_count)
```

| Attempt | Delay |
|---|---|
| 1 | 1 min |
| 2 | 2 min |
| 3 | 4 min |

### Throughput Reference

```
clips_per_day = vods_per_day × clips_per_vod  →  3 × 40 = 120 clips/day
```

Bottleneck is AI scoring (~3s/clip). FFprobe, FFmpeg, and transcription are all faster. Parallelising the scoring stage gives the biggest throughput gain.

### Phased Scaling Path

**Phase 1 (current):** Filesystem + `.meta.json` state; sequential; `--watch` trigger. Correct for single-machine, low-volume use.

**Phase 2:** SQLite replaces `.meta.json` as the canonical state store. Exponential backoff retries per stage. Structured metrics logging (timing + token cost per clip).

**Phase 3:** Worker pool for AI scoring; Vault/Stockpile logic tied to Market Density Monitor; auto-trigger from Scout alerts.

**Phase 4 (optional):** Airflow or Prefect for full DAG orchestration — built-in retries, monitoring UI, dependency graphs. Only worth the overhead at 10+ concurrent games.

---

## Backlog — Advanced Clip Intelligence

A layered pre-filter system that gates clips on skill level and market conditions before committing to expensive processing. Cheap checks run first; AI runs last.

**Pipeline insertion point:**
```
Rank Looker → Kill-Feed Parser → (pass) → Download 1080p → AI Classifier → Processing/Template
```

---

### 1. Rank Looker

Fast pre-filter — is this clip from a skilled player?

**API Mode:** Query game-specific rank APIs where available (e.g., Deadlock MMR, Tracker.gg for Marvel Rivals).

**Heuristic Mode** (new games / no API):
- **Title NLP:** If game age < 30 days, apply Skill Multiplier when title contains sweat keywords: `scrim`, `tourney`, `top 100`, `pro`, `ranked grind`, `elo`, `mmr`
- **Social Rank Proxy:** Streamers with ≥ 50k followers (or high avg viewers) treated as Verified Skill
- **Smurf detection:** If a high-follower streamer's title contains `smurf`, `bronze to gm`, `low elo`, or similar — Verified Skill proxy is voided; fall back to Title NLP scoring only

Acceptance thresholds are adjusted dynamically by the Market Density Monitor (see §5).

---

### 2. Depth Sampling (Market Density Proxy)

Measures clip pool saturation for a game without downloading anything.

**Procedure:** Fetch top 100 clips via Twitch API → inspect view count of the 100th clip.

| 100th clip views | Market condition | Effect on thresholds |
|---|---|---|
| > 5,000 | ABUNDANT (oversupply) | Raise strictness |
| 50–5,000 | HEALTHY | Default thresholds |
| < 50 | SCARCITY | Lower strictness |

**Time-Decay Problem:** Raw view count on the 100th clip ignores velocity — those 5,000 views could be 6 months old. Always pair the view count check with a **Clips Per Day (CPD)** calculation using the `created_at` timestamps of the same 100 clips.

```
CPD = count of clips where created_at > (now − 24h)
```

**Deadly Plateau:** ABUNDANT by view count but CPD is declining week-over-week → the game is coasting on historical volume, not current momentum. Don't just raise strictness; flag for Exit Strategy review (see Game Lifecycle Model).

---

### 3. Kill-Feed Parser (OpenCV)

Ground-truth verification of mechanical action — runs before the AI to avoid unnecessary API calls.

**Setup:**
- Per-game bounding box defined in `config.yaml` under `kill_feed_coords`
- **Resolution normalization:** Before any crop, resize every frame to 1080p internally (`cv2.resize`). Streamers use 1440p, ultrawide, or non-standard HUD scales — without this, ROI coordinates break constantly.
- Crop to ~300×400 px ROI; sample at 5 FPS

**Detection methods (in priority order):**
1. **Color mask** — `cv2.inRange()` for headshot/kill-feed colors; flag events with ≥ 500% pixel spike in 2 seconds
2. **Template match** — `cv2.matchTemplate()` for UI icons; score > 0.8 → skill event
3. **MOG2 background subtraction** — `cv2.createBackgroundSubtractorMOG2()` to detect text/icon *appearance* in the ROI, rather than relying on color alone; handles dynamic or semi-transparent UIs
4. **Edge detection fallback** — `cv2.Canny()` as last resort

**Sweat Score** (sliding 5-second window):
- Kill = +10 pts · Headshot = +20 pts
- Window total > 50 → promote clip to Audio Energy Check

---

### 3a. Audio Energy Check

Lightweight gate between Kill-Feed Parser and AI Classifier. Mechanical skill on the kill-feed is not the same as a hype clip — a player getting three kills quietly in a corner will not go viral.

**Method:** Extract audio waveform for the kill-feed window using FFmpeg (`volumedetect` or `astats` filter). If the mean dB in that window is below a threshold, the clip is deprioritised.

**Rule:**
- Kill-feed spike + audio spike (shouting, game "multi-kill" SFX, streamer reaction) → **promote to AI Classifier**
- Kill-feed spike + flat audio → **lower Final_Score bonus** (still passes, just ranked lower)

Audio spike = peak dB in the 5s window exceeds the clip's mean dB by ≥ 6 dB. This threshold is configurable per game in `config.yaml` (`audio_spike_db_delta`).

---

### 4. AI Classifier & Decision Engine

Qualitative verification of crosshair control, target switching, and movement — called only when Kill-Feed Parser promotes the clip.

**Workflow:**
1. Kill-Feed Parser provides kill timestamps; Audio Energy Check confirms hype
2. For each timestamp, compute **optical flow** (e.g., `cv2.calcOpticalFlowFarneback`) across t−1 → t → t+1 frames and select the frame with the highest mean flow magnitude — this is the frame most likely to show a flick or snap
3. Send that frame (plus one frame of context on each side) + prompt to Claude: rate sweat level 1–10 (crosshair snapping, low TTK, advanced movement)

**Final score formula:**
```
Final_Score = (Kill_Density × 0.4) + (AI_Skill_Rating × 0.6)
```
Approve if `Final_Score >= Strictness_Threshold` (threshold set by Market Density Monitor).

---

### 5. Market Density Monitor & Dynamic Thresholding

Daily pulse that adjusts how strict the pipeline is based on how saturated the clip pool is.

**Daily pulse:** Fetch top 100 clips (last 24h) → Depth Sample → set `Market_Condition`.

| Market Condition | Skill Threshold | Multi-Kill Req | Posting Frequency |
|---|---|---|---|
| OVERSUPPLY | 9+ | Triple kill+ | 3×/day |
| HEALTHY | 7+ | Double kill+ | 1×/day |
| SCARCITY | 5+ | Skilled single kill | 0.5×/day |

```
Threshold = Baseline_Strictness × Market_Multiplier
```
OVERSUPPLY multiplier: 1.2 · SCARCITY multiplier: 0.8

**Vault/Stockpile:** Over-ingest during OVERSUPPLY; bank approved-but-unposted clips for SCARCITY periods.

**Vault expiry:** Add a `meta_expiry` tag. Any vaulted clip older than 7 days that hasn't been posted must be re-evaluated or discarded. Fast-moving metas (Marvel Rivals, Deadlock patch weeks) make stale clips actively harmful — they signal the channel is out of touch. Freshness already identified as a High Priority signal (see Platform Algorithm & Retention).

---

### Integration Notes

- Auto-inject Twitch tags and extracted keywords into distribution metadata (e.g., `#VenomMain`, `#MarvelRivalsStrategy`)
- Keep `rank_looker.py` and `kill_feed.py` small, config-driven, and independently testable
- Per-game parameters (`kill_feed_coords`, `pixel_threshold`, HSV ranges) live in `config.yaml`

---

### Implementation Checklist

- [x] Game-specific rank API hooks (Deadlock, Marvel Rivals / Tracker.gg) — `pipeline/rank_looker.py` heuristic + api modes
- [x] Title NLP sweat keyword dictionary + Social Rank Proxy + smurf detection — `rank_looker.py` heuristic mode
- [x] OpenCV Kill-Feed Parser — `pipeline/kill_feed.py`; resolution normalization, ROI crop, color mask, MOG2, template match, edge (Canny) fallback all implemented
- [x] Audio Energy Check — `pipeline/audio_detector.py`; FFmpeg astats + numpy Z-score spike detection; frequency bandpass filter; multi-kill classification
- [x] Auto-tag distribution metadata from extracted keywords — `utils/metadata_injector.py` embeds title + hashtags from title_engine into MP4 container tags
- [ ] Depth Sampling (100th clip view count → Market Condition)
- [ ] CPD (Clips Per Day) calculation + Deadly Plateau detection → Exit Strategy trigger
- [ ] Optical flow frame selection for AI Classifier (highest motion frame per timestamp)
- [ ] Final_Score formula: combine kill_feed sweat_score + AI highlight_score into single weighted score
- [ ] Market Density Monitor + Strictness Slider
- [ ] Vault/Stockpile clip banking logic with 7-day `meta_expiry` tag

---

### Risks & Production Gaps

| Area | Risk | Mitigation |
|---|---|---|
| OpenCV color masks | False positives from noisy/transparent overlays | MOG2 background subtraction + multi-signal voting |
| Resolution variance | ROI coordinates break on 1440p / ultrawide streams | Normalize all frames to 1080p before any crop |
| Anti-fluke (quiet kills) | Multi-kill on kill-feed ≠ hype clip (corner camper) | Audio Energy Check gates promotion to AI Classifier |
| Smurf streamers | High-follower streamer in low-skill lobby → false Verified Skill | Smurf keyword check voids Social Rank Proxy |
| Deadly Plateau | ABUNDANT view count but CPD declining → stale game flagged as active | CPD trend check; trigger Exit Strategy review |
| Stale vault clips | 2-week-old meta clip makes channel look out of touch | 7-day `meta_expiry`; re-evaluate or discard |
| Market Density Monitor | Overfitted thresholds → missed viral clips | Conservative decay; A/B feedback loop |
| API rate limits | Twitch and Tracker.gg edge cases | Retries with backoff |

**What's needed before this is production-ready:**
- Structured decision logs per clip (scores at each stage, pass/fail reason)
- Unit tests for ROI cropping, HSV masks, and template matching (use recorded sample clips)
- Config schema validation for per-game parameters
- Pass/fail rate metrics + a small debug dashboard

---

## Strategic Direction — FPS Multi-Channel, First-Mover Model

### Core Model

- **Niche:** FPS / competitive shooters only
- **Structure:** One dedicated channel per game
- **Edge:** First-mover advantage on emerging titles
- **Exit:** Drop dead games quickly, zero emotional attachment

**Example channels:** Arc Raiders Daily · Deadlock Moments · Top Marvel Rivals Clips

---

### Why This Works

Don't compete in saturated games (Fortnite, CoD, Valorant, R6S) — established channels have years of algorithm authority, experienced editors, and locked-in brand deals.

The opportunity is in games that are newly released or early access, growing but not yet saturated, or emerging from a major update or esports scene. The goal: be the default channel for a game before anyone else is.

**Operating principles:**
- **Speed > Polish** — first to upload = more impressions; volume = more algorithm chances
- **System > Creativity** — same format, template, and workflow across all channels; only the game changes
- **Timing > Skill** — enter early, exit early; "good enough + fast" beats "perfect + slow"

You are not building a gaming channel. You are building a **system that captures attention during the lifecycle of games**. Treat games like assets in a portfolio — no emotional attachment; if the data says exit, exit.

---

### Game Selection

**Add a game when:**
- Recently released or in early access (no established dedicated short-form channel yet)
- Active Twitch presence with clip-worthy (fast-paced, skill-moment) gameplay
- Growing interest; active subreddit

**Drop a game when:**
- Clips averaging fewer views than your baseline after 30 days of posting
- Twitch concurrent viewers declining for 2+ consecutive weeks
- Subreddit going inactive or a larger established channel has claimed the space

**Data sources:** Twitch category viewer counts (`GET /helix/streams`), YouTube search trends, TikTok clip frequency

---

### Game Lifecycle Model

| Phase | Signal | Action |
|---|---|---|
| **Launch / Hype** | Twitch viewers rising, clip volume increasing | Add immediately, post aggressively |
| **Growth** | Consistent engagement, growing subreddit | Maintain frequency, refine best formats |
| **Stable** | Plateau in views | Keep running, reduce manual attention |
| **Declining** | Clip views dropping, Twitch viewers falling | Reduce posting frequency |
| **Dead** | Near-zero Twitch presence, no clip traction | Remove from pipeline, archive channel |

Adding a game = one `config.yaml` entry. Exiting = flip `enabled: false`. No infrastructure changes.

---

### Content Sourcing

- Track top streamers per game (5–10 per game)
- Download VODs; target: high-skill plays, funny moments, clutch situations
- **1 stream = 20–50 usable clips**
- Pipeline handles download automatically (Twitch Helix API → yt-dlp → FFprobe)

---

### Editing Philosophy

- Clean, fast-paced, minimal effects
- Auto captions on every video; visual change every 2–3 seconds
- Subtle zooms, optional sound effects, clear crop on action
- Standardize everything: hook style, caption format, clip length, template — only variable is the game

---

### Risk Management

| Risk | Mitigation |
|---|---|
| Game dies | Multiple channels — no single point of failure; rotate effort to next game |
| Content removal | Transformative elements (captions, edits, cuts) as standard practice |
| Burnout | Batch work; limit manual effort per video; pipeline automation handles the rest |

---

### Multi-Channel Architecture Requirements

**Works today:**
- New game = one `config.yaml` entry; processes independently
- Twitch API resolves any game by display name
- Distribution routes per-game to correct subreddit

**Needs to be built:**
- Per-game YouTube/TikTok channel credentials (separate OAuth tokens per channel, routing by game at upload time)
- Per-game channel ID in `config.yaml` under each game entry
- Game health monitoring: scheduled Twitch concurrent viewer check → auto-flag games below threshold in analytics sheet

---

### Sponsorship Roadmap

| Stage | Subs | Target | Notes |
|---|---|---|---|
| Early | 10k–50k | Gaming peripherals | Headsets, chairs, keyboards, mice — actively sponsor game-specific channels |
| Mid | 50k–250k | Game publishers/devs | Keys, early access, direct deals — email community/marketing teams directly |
| Advanced | 50k–100k+ combined | Hardware manufacturers | NVIDIA, Intel, AMD, ASUS ROG, MSI — FPS is their core demographic |

Avoid early: energy drinks, broad consumer brands, exclusivity deals that block the peripheral/hardware tier.

---

## Monetization Reference

Revenue split by stage:
- Early: ~100% platform payouts (small absolute numbers)
- Growth: 60–90% from sponsorships, rest from platform + affiliates

**What advertisers actually evaluate:** average views per video (consistency > one viral hit), watch time %, engagement rate, audience demographics, posting frequency.

**Pricing formula:** `Avg Views ÷ 1,000 × $5–$20` (CPM-equivalent)

| Stage | Followers | Range per post |
|---|---|---|
| Small | <50k | $50–$300 |
| Growing | 50k–250k | $300–$2,000 |
| Established | 250k+ | $2,000+ |

**Build early:** media kit (channel description, avg views, audience demographics, example videos). Target: after first 30 days of consistent posting.

**Archetype CPM comparison:** Gaming (low–medium) · Lore/narration (medium–high) · Poker (high — finance-adjacent audience). Highest views ≠ highest money. Gaming = volume/growth engine; poker/lore = monetization engines.

---

## Game Scouting Dashboard

The goal is to detect momentum early and act faster than everyone else — treating games as attention assets the same way a trader tracks stocks.

### Signals to Track

| Signal | What to monitor |
|---|---|
| Twitch (primary) | Current viewers, peak viewers, # of active streamers |
| YouTube | New uploads per day, views on top clips |
| TikTok | Clip frequency, viral videos emerging |
| Google Trends | Search volume spikes |

### Game Status States

`Upcoming → Beta/Playtest → Newly Released → Trending → Declining → Dead`

---

### Breakout Pattern

Most breakout games follow: small baseline → big streamer plays it → sudden viewer spike → algorithm amplifies clips → other streamers pile in.

**Entry signal:** game jumps from ~500 → 10k+ viewers overnight with a notable streamer playing it.

---

### Real Breakout vs. One-Time Spike

| Type | Characteristics |
|---|---|
| **Real breakout** | Multiple streamers join over days, viewer base stabilizes after spike, clips spread across platforms |
| **One-time spike** | Single streamer, drops 70–90% next day, no spread to other streamers |

**Rule:** If interest spreads → it's real. If it stays concentrated → it's temporary.

---

### Flavor of the Week (Important Caveat)

Some games attract massive multi-streamer attention very quickly but the viewership comes in huge waves and collapses just as fast. This is driven by novelty rather than gameplay depth — every big streamer plays it on day 1 and drops it by day 3.

This pattern is **less common in FPS games** because competitive shooters have a skill ceiling that keeps players engaged long-term. A new battle royale or survival novelty game burns fast; a mechanically deep FPS tends to retain a core audience.

**Warning signs that a spike is flavor-of-the-week:**
- All major streamers appear simultaneously on launch day (coordinated hype, not organic discovery)
- The game has no ranked mode, competitive scene, or meaningful skill progression
- Viewership collapses within 72 hours even with multiple streamers still playing
- Clips are funny/reaction content rather than skill-based highlights

**What this means for the pipeline:** It's still worth creating a channel early — you can capture the spike traffic — but don't invest heavily in long-term channel development until the game shows a stable post-hype baseline (at least 2 weeks of sustained Twitch presence after the initial wave).

---

### Scoring System

**Trend Score (0–10):** +3 Twitch viewer growth, +2 multiple streamers, +2 YouTube clip activity, +2 TikTok presence, +1 Google Trends spike

**Longevity Score (0–10):** +3 competitive/skill-based gameplay, +3 replayability, +2 active community, +2 developer support/roadmap

**Action thresholds:** Trend ≥ 6 AND Longevity ≥ 5 → pursue aggressively. High Trend + Low Longevity → capture the spike, don't build long-term.

---

### Daily Scouting Workflow (15–30 min)

1. Check Twitch categories for new games and sudden viewer spikes
2. Update game scores; flag immediate-action opportunities
3. Weekly: review which games are rising vs. dying; cut anything 2+ weeks below baseline

---

### First-Mover Response

When a confirmed real breakout is detected — act within 24–48 hours: create the channel, upload 5–15 clips/day. Early phase = low competition + algorithm curiosity boost for new topics.

---

### Implementation

- **Start:** Google Sheets — game list, scores, trend notes updated manually each day
- **Later:** Python + Twitch Helix API — scrape viewer counts on a schedule, store in a local database, plot trends, trigger alerts on spikes. Integrates naturally with the existing pipeline's Twitch API auth.

---

## Platform Algorithm & Retention

All major platforms (YouTube, TikTok, Instagram) use staged distribution: content is shown to a small test audience first, performance is measured in the first minutes/hours, and distribution expands only if metrics are strong. Editing quality matters far less than attention retention.

**Priority order:** Retention > Shares > Watch Time > Likes

### Core Metrics

| Metric | What it measures |
|---|---|
| Retention % | Percent of the video actually watched — the single most important signal |
| Hook strength | Drop-off rate in the first 1–3 seconds |
| Engagement rate | (Likes + Comments + Shares) ÷ Views |
| Share / Save rate | High-intent signal; heavily weighted by all three platforms |
| Rewatch rate | Loopable endings drive replays and multiply watch time |
| Scroll-stop rate | Views ÷ Impressions — how often the thumbnail/opening stops the scroll |

### Frameworks

- **Hook–Retention–Reward** — strong opening, sustained attention, clear payoff at the end
- **Retention curve optimization** — identify specific drop-off points in the edit and fix them
- **Loopability** — seamless endings encourage replays, compounding rewatch signals

### For This System

- Tag clips by type (clutch, funny, fail, multi-kill) and log performance per template
- Track per clip: retention %, watch time, engagement rate, shares
- A/B test template variants one variable at a time; feed results back into AI scoring weights
- The pipeline's AI scoring model should eventually be tuned against real engagement data, not just editorial intuition

---

## Interview Preparation — System Design

These notes cover the questions a technical interviewer (or CTO evaluating the project) is likely to ask, the answers that demonstrate production-system thinking, and the things to avoid saying.

---

### Anticipated Attack Vectors

**"Isn't this over-engineered?"**
> "The design is modular. The MVP is ingestion + basic scoring + rendering — everything else is a planned extension. The complexity is the cost of entry for a system that can scale to fifty channels without proportional manual effort."

**"Why not just use a vision model for everything?"**
> "That's a scaling trap. Running 24/7 video through a multimodal LLM is economically non-viable for this niche. Traditional CV is deterministic, costs nothing per frame, and handles 90% of the filtering. The AI is reserved for the 10% of clips that pass the cheap gates — that's where its cost is justified."

**"How many clips per hour can this process?"**
Throughput is bounded by AI API latency. Non-AI stages (ingestion, FFprobe, OpenCV, audio analysis) run near real-time. A rough estimate: if average AI call takes 3 seconds and parallelism is N, throughput ≈ N × 1,200 clips/hour before AI becomes the bottleneck. Have a number; even a rough one is better than no answer.

**"How do you debug a failure in production?"**
> "Structured JSON logs per stage, keyed by clip ID. Each stage emits a pass/fail result with the reason. That gives per-game pass rate metrics and makes it trivial to find where in the funnel a clip died."

**"What if you suddenly get 10× more clips?"**
> "Ingestion is decoupled from processing by the queue of manifests. The Vault absorbs approved-but-unprocessed clips during oversupply. If the AI API slows down, backpressure is contained at the processing stage — ingestion keeps running and nothing is lost."

**"What exactly are you testing?"**
- OpenCV ROI accuracy against a labelled gold set (see Precision & Recall below)
- Scoring logic with fixed inputs
- Distribution API calls via mocks (no real posts during tests)

---

### Tradeoff Framing (Core Principle)

Every answer should explain *why* a choice was made, not just *what* was built.

| Instead of | Say |
|---|---|
| "I use OpenCV" | "I chose OpenCV for deterministic filtering before expensive AI calls to control cost and latency" |
| "AI just handles it" | "AI is a recommendation layer, not a final decider — that's what the Manual Review stage is for" |
| "It should work" | "Current precision is X%, validated against a 50-clip gold set" |
| "This detects kills" | "The kill-feed parser uses color mask + MOG2 + audio energy gate to reduce false positives before any API call is made" |

Red flags that signal inexperience: vague accuracy claims, no cost awareness, "AI handles it" as a catch-all, no explanation of failure modes.

---

### Unit Economics

A CTO will ask why this is better than a $15/hr human editor.

- Pipeline cost: ~$0.08 per final clip (AI tokens + compute)
- Human editor cost: ~$5.00 per clip
- Result: ~60× cheaper, ~100× faster at consistent volume
- False positive rate: ~15% — mitigated by the Manual Review layer, which acts as a quality gate before any clip is posted

Track `token_usage` and `execution_time` per clip in the manifest so this number is real, not estimated.

---

### Precision & Recall — Gold Set

Don't claim accuracy; demonstrate it.

- Create `tests/eval/` with ~50 clips where the correct answer is already known
- Run the full OpenCV + AI pipeline against them and measure:
  - **Precision** (of clips flagged as good, how many actually are?) — target: 92%
  - **Recall** (of all good clips, how many did we catch?) — target: 75%
- Deliberately accepting lower recall saves money; that's a design choice, not a flaw
- Run this suite before any significant pipeline change (regression testing)

---

### Failure Mode Registry

An interviewer will ask: "What if the game releases a patch that moves the HUD?"

- Add a `ui_version_hash` per game to `config.yaml`; update it when the HUD changes
- If detection confidence drops below a 24-hour rolling baseline → **circuit breaker** pauses that game's pipeline and alerts
- ROI coordinates need roughly 2 minutes to update when a game patches its HUD
- This makes the brittleness of OpenCV a *managed* risk, not an unacknowledged one

---

### Architecture Summary (Whiteboard / README_ARCH.md)

**System philosophy:** High-throughput, low-latency filtering pipeline. "Funnel of costs" — 90% of clips are discarded by near-zero-cost (OpenCV + audio) filters; only 10% reach the AI scoring stage.

**Decision stack:**

| Component | Choice | Tradeoff |
|---|---|---|
| Filtering | OpenCV + audio energy check | Pro: deterministic, $0/frame. Con: brittle to UI changes — mitigated by config-driven ROIs + circuit breaker |
| Validation | Claude AI Classifier | Pro: qualitative reasoning on movement/APM. Con: latency + cost — mitigated by kill-feed gate that routes only promoted clips |
| State | JSON manifest sidecars + folder structure | Pro: human-readable, simple. Con: not atomic — upgrade path is SQLite for multi-process production use |

**Operational resilience:**
- **Idempotency:** Every clip has a `.meta.json` sidecar. The pipeline can crash at any point and resume without re-downloading or double-spending API credits.
- **Backpressure:** Queue-based ingestion is decoupled from processing. If the AI API slows, the Vault absorbs the backlog without data loss.
- **Observability:** Structured JSON logs keyed by clip ID enable real-time dashboards on pass/fail rates, processing time, and cost per stage.

### Terminology Reference

Use precise terminology when describing the system — it signals familiarity with production patterns, not just working code.

| Informal | Precise |
|---|---|
| "It retries when it breaks" | Exponential backoff with jitter |
| "It does many things at once" | Horizontal scaling / parallel workers |
| "It keeps track of clips" | Persistent state management |
| "It's easy to update one part" | Decoupled service boundaries |
| "Clips that fail get set aside" | Dead Letter Queue (DLQ) |
| "We know when something goes wrong" | Structured observability + alerting |

---

## Backlog — Production Hardening

Targeted additions that move the system from "runs reliably on one machine" to "observable and maintainable in production."

### Dead Letter Queue (DLQ)

After 3 retry attempts, a clip should move to a permanent `failed` / DLQ state rather than being silently dropped or retried forever. A separate diagnostic script inspects DLQ items — humans decide whether to re-queue, discard, or flag the root cause.

The quarantine folder already handles FFprobe failures at ingestion. The DLQ extends this concept to every stage: transcription errors, AI scoring failures, distribution rejections.

### Task Audit Log

An append-only log table (or per-clip JSON audit trail) that records every stage transition:

| Field | Purpose |
|---|---|
| `task_type` | Which stage: ingest / transcribe / score / upload |
| `clip_id` | Links to the clip manifest |
| `status` | success / failed / retried |
| `start_time` / `end_time` | Duration per stage |
| `error_message` | Null on success; full traceback on failure |

This is the "paper trail" — if something breaks, you can prove exactly when and at which stage, without grepping logs.

### Health Check Metrics

Key metrics to surface in the review UI (`/health` endpoint or stats page):

| Metric | Formula |
|---|---|
| Success rate | Successful uploads ÷ total attempts |
| End-to-end latency | Time from clip downloaded → clip posted |
| Cost per clip | AI token spend ÷ clips processed |
| DLQ depth | Count of clips in failed / quarantine state |

### Containerisation

A `Dockerfile` + `docker-compose.yml` eliminates "works on my machine" risk and makes the system deployable to any VPS or cloud instance without environment setup.

Minimum compose services: `app` (pipeline runner), `redis` (queue broker, when Phase 2 scaling is added), `db` (SQLite file mount or Postgres container).

---

## Changelog

| Date       | Change |
|------------|--------|
| 2026-04-05 | Initial planning document |
| 2026-04-05 | Pipeline order corrected; AI Scoring moved after Processing; Decision Engine is metadata-only |
| 2026-04-05 | Template library created; music disabled; blur pillarbox default for vertical fill |
| 2026-04-12 | Scalable faceless content system notes added |
| 2026-04-13 | Analytics (Google Sheets), Drive backup, audio copyright mode added |
| 2026-04-13 | Copyright detection (ACRCloud) and QoL backlog added |
| 2026-04-14 | Strategic direction finalized: FPS multi-channel first-mover model |
| 2026-04-14 | Full document condensed; future archetypes collapsed; monetization and sponsorship roadmap added |
| 2026-04-14 | Game scouting dashboard added: signals, scoring system, breakout detection, flavor-of-the-week warning |
| 2026-04-14 | ACRCloud copyright detection implemented: _detect_copyright() in processing.py; pyacrcloud added to requirements |
| 2026-04-15 | Advanced Clip Intelligence backlog added: Rank Looker, Depth Sampling, Kill-Feed Parser, AI Classifier, Market Density Monitor |
| 2026-04-15 | Platform Algorithm & Retention section added: metrics priority, frameworks, feedback loop guidance |
| 2026-04-15 | Advanced Clip Intelligence refined: Audio Energy Check (§3a), CPD + Deadly Plateau, resolution normalization, MOG2, optical flow frame selection, Vault expiry, smurf detection |
| 2026-04-15 | Interview preparation section added: attack vectors, tradeoff framing, unit economics, gold set, failure mode registry, architecture summary |
| 2026-04-15 | Pipeline Scaling & Orchestration backlog added: state machine model, retry strategy, throughput reference, phased scaling path |
| 2026-04-15 | Production Hardening backlog added: DLQ, task audit log, health metrics, containerisation; terminology reference added to interview prep |
| 2026-04-15 | Systems Thinking section added: filters vs. templates as the core design philosophy behind repeatable automated workflows |
| 2026-04-17 | Maintenance: moved implemented QoL features into their proper stage sections; removed crossed-off items from backlog |
| 2026-04-18 | New Channel Archetype Protocol added; Duelist and Night Zero archetype designs added |

---

## New Channel Setup Protocol

A repeatable onboarding blueprint for launching any new game channel or content archetype. Run this checklist in order when adding a new game or channel type.

### Phase 1 — Infrastructure & Social Footprint

- Create accounts on all target platforms (YouTube, TikTok, Instagram, Twitter/X, Reddit)
- Use a consistent brand name and profile image across all platforms from day one
- Add the game to `config.yaml` (one entry: `display_name`, `twitch_url`, `folder`)
- Add subreddit name and flair ID to `config.yaml → distribution.platforms.reddit`

### Phase 2 — The Scout Script (Automated Game Research)

Run once per new game before the pipeline goes live. Steps:

**2a. Wiki scraping — roster & weapon list**
- Search for an official or community wiki (fandom.com, game8.co, the game's own site)
- For hero shooters: extract the full playable hero list with role (Vanguard / Duelist / Tank / Support etc.)
- For FPS / extraction shooters: extract the primary weapon list with category (AR, SMG, Sniper etc.)
- Save as `assets/rosters/{game}.yaml` — same format as existing arc_raiders / marvel_rivals / deadlock files
- Wire display names into `config.yaml → weapon_detector.games.{game}.weapons`

**2b. Icon / asset extraction**
- Use `tools/crop_hero_icons.py` on an in-game hero-select screenshot to bulk-crop portrait PNGs
- Save to `assets/weapon_icons/{game}/{hero_or_weapon_id}.png`
- For weapons: use `tools/extract_weapon_icon.py` on actual gameplay clips to pull from the HUD ROI directly
- Icons feed the OpenCV `weapon_detector` stage so the pipeline can identify what hero/weapon is on screen

**2c. HUD mapping**
- Download one high-quality (1080p+) gameplay clip for the new game
- Use `tools/preview_roi.py` to visualize current ROI overlays on a real frame
- Adjust `kill_feed.games.{game}.roi` and `weapon_detector.games.{game}.roi` in `config.yaml` if needed
- Run `python tools/extract_weapon_icon.py --debug` to confirm weapon ROI crops correctly

### Phase 3 — The "Undiscovered Pro" Hunter

High-ranked players with low view counts are the highest signal-to-noise clip sources. The pipeline already has `rank_looker` for heuristic filtering; this protocol extends it to proactively find those accounts.

- Pull the Top 500 leaderboard from a third-party rank tracker (Tracker.gg, Deadlock Tracker, etc.)
- For each in-game name (IGN): check whether a Twitch channel exists with that username or a known alias
- Evaluate the channel's clip velocity: if a player is Top 50 in the world but their Twitch clips average < 500 views, they are a **High-Quality / Low-Competition** source
- Add confirmed IGN → Twitch username mappings to `assets/broadcaster_ign_map.json`
- These accounts become priority sources; future versions of `rank_looker` can bias toward them

**Signal thresholds (starting point — adjust per game):**
- Rank threshold: top 10% of the ranked ladder
- View ceiling: < 5,000 average clip views (if higher, competition is already finding them)
- Channel age: any (newer channels are better — less likely to be discovered already)

---

## Content Archetype — The Duelist (AI Cinematic Combat)

**Premise:** Cinematic AI-generated duels between warriors, soldiers, and fighters from different cultures, time periods, and skill traditions. The "What-If" format — no game footage, fully generated.

**Target audience:** History fans, martial arts fans, "versus" communities, action movie fans

**Platform fit:** YouTube Shorts (hook + outcome), TikTok (curiosity loop), YouTube long-form (extended choreography)

---

### Production Logic

**AI tools (as of 2026):**
- Luma AI — best for realistic physics and movement
- Runway Gen-3 / Gen-4 — best for cinematic lighting and professional choreography
- ElevenLabs — authentic weapon sounds and environmental audio layered in post

**The Prompt Matrix:**

Every video is generated from two prompt components:

| Component | What it controls |
|---|---|
| **Visual** | Setting, lighting, clothing, weapon aesthetics, camera style |
| **Choreography** | Fight pacing, who gains ground and when, counters, final moment |

Example visual prompt structure:
> "Cinematic 4K, realistic textures, [location — e.g. Scottish Highlands at dusk / feudal Japanese village]. A duel between [Fighter A with specific detail] and [Fighter B with specific detail]. Dynamic handheld camera with slow-motion impacts. No gore or graphic injury."

Example choreography structure:
> "The fight is back-and-forth. [Fighter A] opens aggressively with [technique]. [Fighter B] uses reach/footwork to create distance, counter-attacks with [weapon move]. The momentum shifts twice before a decisive final exchange. Dramatic lighting change at the climax."

**Key production rules:**
- Fight must be competitive — no one-sided stomp; 2–3 dramatic momentum swings minimum
- Location should be distinct and visually interesting (not generic arena)
- Clothing and weapons must fit the fighter's background — do the research
- Camera moves through the scene (not static); use slow-motion on impact moments
- No gore, no graphic injury; violence is implied or shown at a cinematic/stylized level
- Safe for all platforms — frame for curiosity, not shock

---

### Title & Hook Formula

`[Fighter A] vs [Fighter B] — Who wins?`

The title does the work. Keep it simple. The algorithm surfaces it to the right audience because the subject matter is inherently searchable.

**SEO layer:** The title engine generates tags from the fighters' origins:
- Fighter origins → `#Samurai #Viking #MartialArts #HistoricalCombat #WhoWouldWin`

**Engagement mechanic:** Pin a comment asking viewers to vote for who they think should win — then post a follow-up "rematch" video based on the top comment.

---

### Content Matrix (Starter Combinations)

| Fighter A | Fighter B | Location |
|---|---|---|
| Samurai | Viking | Coastal cliff, Norse coastline |
| Spartan | Zulu Warrior | Open savanna |
| Ninja | Ottoman Janissary | Rooftop Istanbul |
| Medieval Knight | Mongolian Horse Archer | Steppe plains |
| Shaolin Monk | Apache Warrior | Desert canyon |
| Roman Legionnaire | Celtic Berserker | Forest clearing |
| Maasai Warrior | Gurkha Soldier | East African highland |

Expand the matrix over time; introduce fantasy crossovers (e.g., Viking vs. Samurai vs. Pirate — three-way) once the format is proven.

---

## Content Archetype — Night Zero (Zombie Outbreak POV)

**Premise:** Anthology of short-form videos — each one is the first-person POV of a different ordinary person in the early minutes of a zombie outbreak before most people understand what is happening. No single protagonist. No plot armor.

**Target audience:** Horror fans, survival fiction fans, Walking Dead / 28 Days Later audiences, "What would you do?" community

**Platform fit:** TikTok and YouTube Shorts (3–90 seconds, self-contained), with longer cuts for YouTube

---

### The "No Plot Armor" Rule

The defining creative constraint: no character is guaranteed to survive. Because each video follows a different person in a different location, the audience never knows if this particular person makes it. This creates genuine tension — unlike traditional horror where the audience knows the main character will survive at least until the final act.

Half the time (roughly), the character does not survive the video.

This also means: no heroes, no special skills, no lucky coincidences. The character makes a plausible human decision under extreme stress, and the outcome follows from that decision realistically.

---

### The Perspective Engine

Every video is a new POV. The situation brief drives the generation prompt.

**Situation brief template:**
> "You are [ordinary person] at [specific everyday location]. [Inciting event — power cuts, screaming, something wrong in the crowd]. You have [X seconds] to decide: [Option A] or [Option B]. You do not know what is happening yet."

**Starter location list:**
- Grocery store (power goes out, people start panicking in produce section)
- Gridlocked highway (drivers abandoning cars, something approaching from behind)
- Top floor of an office building (looking down at a street that has gone chaotic)
- School pickup line (parents and children, something wrong at the school entrance)
- Hospital waiting room (the staff suddenly stop coming back out)
- Subway car mid-tunnel (train stops, lights go out)
- Movie theater (emergency broadcast interrupts the film)
- Stadium at a sporting event (something happens at field level)
- Cruise ship deck (can see the coast, something is very wrong on shore)
- Apartment complex (sirens, neighbors acting strangely in the hallway)

Expand continuously. The format is infinitely repeatable because the location and person change every time.

---

### Aesthetic Direction

| Element | Target |
|---|---|
| Camera style | POV / body-cam — shaky, handheld, claustrophobic |
| Color grade | Cold, desaturated — high contrast shadows, bleached highlights |
| Film grain | High ISO noise — feels real, not polished |
| Sound design | Sirens in distance, unintelligible screaming, confusion; no cinematic score |
| Pacing | Slow build → sudden chaos → cut to black or ambiguous outcome |

Reference films: *28 Days Later*, *Cloverfield*, early *The Walking Dead* (Season 1 street scenes), *World War Z* crowd sequences

**What to avoid:** jump scares as the primary hook, gore/graphic violence, clear monster designs (ambiguity is scarier), anything that reads as gratuitous rather than visceral.

---

### Engagement Mechanic

Pin a comment on every video:
> "What would you have done? Vote below ↓"

Follow up with a "What actually happens next" continuation video for the highest-engagement scenarios. This creates a feedback loop where the audience directs the content.

TikTok poll sticker on every video: two options matching the decision the character faced.

---

### Title Formula

`You're at [location]. The outbreak just started. What do you do?`

or

`POV: You're [doing ordinary thing] when it starts.`

Keep it second-person — "you" pulls the viewer into the scenario immediately.
