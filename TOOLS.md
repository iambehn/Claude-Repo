# Tools Reference

Every tool relevant to the gaming clip farming bot pipeline — whether in use, under consideration, or researched and ruled out.

**Status key:** `IN USE` | `CONSIDERING` | `RESEARCH ONLY` | `NOT USING`

---

## Table of Contents
- [Ingestion](#ingestion)
- [Transcription](#transcription)
- [Feature Extraction](#feature-extraction)
- [Processing](#processing)
- [Audio](#audio)
- [AI Scoring](#ai-scoring)
- [Review UI](#review-ui)
- [Distribution](#distribution)
- [Analytics / Optimize](#analytics--optimize)
  - [Optuna](#optuna)
  - [MLflow](#mlflow)
  - [VidIQ](#vidiq)
  - [TubeBuddy](#tubebuddy)
- [Competitive Reference](#competitive-reference)

---

## Ingestion

### yt-dlp
**Stage:** Ingestion  
**Status:** IN USE  
**Cost:** Open Source  
**What it does:** Downloads video and audio from 1000+ sites (YouTube, Twitch VODs, Twitter/X, Reddit, etc.) via CLI. Successor to youtube-dl with active maintenance and faster development.  
**Pipeline fit:** Primary download engine. Called by the Ingestion stage to pull source clips by URL or batch list. Outputs to `raw/` directory.  
**Notes:** https://github.com/yt-dlp/yt-dlp — supports cookies, rate limiting, format selection, SponsorBlock, and metadata extraction. JSON metadata output (`--write-info-json`) feeds directly into our source manifest.

---

### gallery-dl
**Stage:** Ingestion  
**Status:** CONSIDERING  
**Cost:** Open Source  
**What it does:** Downloads image and media galleries from 200+ sites (Reddit, Imgur, Twitter/X, Pixiv, etc.).  
**Pipeline fit:** Supplementary ingestion for image-heavy sources or Reddit posts that link to clips not handled by yt-dlp.  
**Notes:** https://github.com/mikf/gallery-dl — JSON metadata output; supports cookies and rate limiting.

---

### streamlink
**Stage:** Ingestion  
**Status:** CONSIDERING  
**Cost:** Open Source  
**What it does:** Extracts stream URLs from live streaming services (Twitch, YouTube Live, etc.) and pipes them to a local video player or file.  
**Pipeline fit:** Live stream capture — record Twitch VODs or live streams directly to `raw/` for later processing.  
**Notes:** https://github.com/streamlink/streamlink — pairs with FFmpeg for recording (`streamlink <url> best -o output.mp4`).

---

### Medal.tv
**Stage:** Ingestion  
**Status:** RESEARCH ONLY  
**Cost:** Free (platform); no public automation API  
**What it does:** Gaming clip sharing platform with automatic highlight detection for PC games via a client overlay.  
**Pipeline fit:** Potential clip source — users share short gaming moments here.  
**Notes:** No public CLI or download API. Web scraping would violate ToS. Clips discovered here would need to be ingested manually or via yt-dlp if they surface on YouTube/Twitter.  
**Reason not using:** No automatable ingestion path; platform-only consumption.

---

## Transcription

### Whisper
**Stage:** Transcription  
**Status:** IN USE  
**Cost:** Open Source  
**What it does:** OpenAI's speech recognition model. Transcribes audio to text with timestamps at the segment level. Runs locally.  
**Pipeline fit:** Transcription stage reads audio track from ingested clip and outputs SRT/JSON with segment timestamps used for caption burn-in and AI scoring context.  
**Notes:** https://github.com/openai/whisper — models range from `tiny` (fast, less accurate) to `large-v3` (slow, most accurate). `medium` is a practical default for gaming audio.

---

### faster-whisper
**Stage:** Transcription  
**Status:** CONSIDERING  
**Cost:** Open Source  
**What it does:** Drop-in reimplementation of Whisper using CTranslate2. 4x faster inference than the original at the same accuracy, with lower memory usage.  
**Pipeline fit:** Direct replacement for Whisper in the Transcription stage — same output format, significantly faster on CPU and GPU.  
**Notes:** https://github.com/SYSTRAN/faster-whisper — strong candidate to replace Whisper once pipeline is stable.

---

### WhisperX
**Stage:** Transcription  
**Status:** CONSIDERING  
**Cost:** Open Source  
**What it does:** Extends Whisper with forced alignment to produce word-level timestamps (not just segment-level).  
**Pipeline fit:** Required if we implement word-highlight caption style (each word lights up as spoken). Segment-level timestamps from vanilla Whisper are too coarse for this effect.  
**Notes:** https://github.com/m-bain/whisperX — depends on `pyannote.audio` for alignment; requires a Hugging Face token for the alignment model. Segment-level Whisper is the chosen approach for the initial build; WhisperX is the planned upgrade path when word-highlight captions are implemented.

---

### stable-ts
**Stage:** Transcription  
**Status:** CONSIDERING  
**Cost:** Open Source  
**What it does:** Modifies Whisper's internal processing to produce more accurate word and segment timestamps, especially around silences and music.  
**Pipeline fit:** Drop-in improvement over vanilla Whisper timestamps without changing the rest of the transcription pipeline.  
**Notes:** https://github.com/jianfch/stable-ts — compatible with faster-whisper; can be combined with WhisperX.

---

## Feature Extraction

### FFprobe
**Stage:** Feature Extraction  
**Status:** IN USE  
**Cost:** Open Source (ships with FFmpeg)  
**What it does:** CLI tool that reads media file metadata: codec, resolution, frame rate, duration, audio streams, bitrate, and more.  
**Pipeline fit:** Called after ingestion to populate `source_metadata` fields in the clip manifest (resolution, fps, duration, has_audio). Zero additional install cost — included with FFmpeg.  
**Notes:** JSON output via `ffprobe -v quiet -print_format json -show_streams -show_format`.

---

### PySceneDetect
**Stage:** Feature Extraction  
**Status:** CONSIDERING  
**Cost:** Open Source  
**What it does:** Detects scene changes and cuts in video using content-aware (histogram diff) or threshold-based algorithms.  
**Pipeline fit:** Generates `motion_level` and `scene_change_count` fields in the feature manifest. High scene-change density correlates with action highlights.  
**Notes:** https://github.com/Breakthrough/PySceneDetect — Python API and CLI; outputs CSV or JSON scene list. Can export scene images for thumbnail candidates.

---

### OpenCV
**Stage:** Feature Extraction  
**Status:** NOT USING  
**Cost:** Open Source  
**What it does:** Computer vision library for frame-level analysis: optical flow, object detection, motion estimation, color analysis.  
**Pipeline fit:** Could generate motion vectors and per-frame energy scores.  
**Reason not using:** PySceneDetect covers the motion detection need with far less complexity and no ML model overhead. OpenCV would be warranted only if we add object detection (e.g., HUD element recognition) — deferred until needed.

---

## Processing

### FFmpeg
**Stage:** Processing (primary), Ingestion (remux), Distribution (final encode)  
**Status:** IN USE  
**Cost:** Open Source  
**What it does:** Industry-standard CLI tool for video/audio encoding, decoding, filtering, muxing, and streaming. Stateless — one command per job.  
**Pipeline fit:** Core processing engine for all clip operations: lossless trim (`-c copy`), vertical fill (blur background + foreground composite), caption burn-in (subtitles filter), zoom punch (`zoompan` filter), loudness normalization, format conversion, and final social media encode.  
**Notes:** https://ffmpeg.org — comprehensive filter graph system. `libx264`/`libx265` for H.264/H.265 output. `-crf` controls quality/size tradeoff.

---

### Shutter Encoder
**Stage:** Processing (batch encode)  
**Status:** NOT USING  
**Cost:** Free (open source, Java)  
**What it does:** GUI-based batch video converter built on top of FFmpeg and HandBrake. Provides a visual preset library for encoding jobs.  
**Pipeline fit:** Useful for manual batch re-encoding passes or testing encode presets before codifying them as FFmpeg commands. Not part of the automated pipeline — operator-facing tool.  
**Notes:** https://www.shutterencoder.com — Windows/Mac/Linux. No CLI automation; interaction is through the GUI.  
**Reason not using:** FFmpeg is the permanent processing engine. Shutter Encoder is a GUI wrapper with no CLI automation path — it adds no capability to a scripted pipeline.

---

### HandBrake
**Stage:** Processing (final compression)  
**Status:** CONSIDERING  
**Cost:** Open Source  
**What it does:** Video transcoder with a well-tuned preset library, particularly for H.264/H.265 output targeting specific device profiles and file sizes.  
**Pipeline fit:** Optional final compression pass after FFmpeg assembly — HandBrake's `--preset` system can produce slightly smaller files at equivalent quality for upload. Also has a CLI (`HandBrakeCLI`) for automation.  
**Notes:** https://handbrake.fr — `HandBrakeCLI` is the automation path. Evaluate whether FFmpeg CRF output is already sufficient before adding this step.

---

### LosslessCut
**Stage:** Manual Review (operator tool)  
**Status:** CONSIDERING  
**Cost:** Free (open source)  
**What it does:** GUI tool for fast, lossless trimming of video files by cutting at keyframes without re-encoding.  
**Pipeline fit:** Operator use during the Manual Review stage — quickly trim a clip to precise in/out points before approving it for processing. Output feeds back into the pipeline as a pre-trimmed source.  
**Notes:** https://github.com/mifi/lossless-cut — Electron app, Windows/Mac/Linux. Not part of automated pipeline.

---

### mkvtoolnix
**Stage:** Ingestion (pre-processing)  
**Status:** CONSIDERING  
**Cost:** Open Source  
**What it does:** Tools for creating, modifying, and inspecting MKV files. `mkvmerge` can remux MKV → MP4 without re-encoding.  
**Pipeline fit:** Some sources (Twitch VOD downloads) arrive as `.mkv`. FFmpeg can handle MKV but remuxing to MP4 first avoids edge cases with container metadata in subsequent FFmpeg filter chains.  
**Notes:** https://mkvtoolnix.download — `mkvmerge -o output.mp4 input.mkv` for lossless remux.

---

### MoviePy
**Stage:** Processing  
**Status:** NOT USING  
**Cost:** Open Source  
**What it does:** Python library for programmatic video editing: clip trimming, concatenation, compositing, text overlays, audio manipulation, and frame-level effects. Wraps FFmpeg under the hood.  
**Pipeline fit:** Could replace direct FFmpeg subprocess calls with a higher-level Python API.  
**Reason not using:** Re-encodes at every operation by default — lossless trim operations that FFmpeg handles via `-c copy` in milliseconds become full re-encodes in MoviePy. Memory-heavy for long source clips (loads frames into RAM). `TextClip` requires ImageMagick as a separate fragile dependency. Raw FFmpeg subprocess calls give more control over codec, CRF, and filter graph settings for the operations this pipeline needs.

---

### Remotion
**Stage:** Processing  
**Status:** NOT USING  
**Cost:** Open Source (free for personal use; commercial license for companies)  
**What it does:** React/TypeScript framework for rendering videos programmatically — define video compositions as React components, render to MP4 via headless Chrome.  
**Pipeline fit:** Could generate title cards, animated overlays, and template-driven sequences.  
**Reason not using:** JavaScript/React-based — wrong language for this Python pipeline. Headless Chrome rendering is heavy and slow compared to FFmpeg filter chains for the same output.

---

### DaVinci Resolve
**Stage:** Processing (color grading, NLE timeline assembly)  
**Status:** RESEARCH ONLY  
**Cost:** Free tier (limited scripting); Studio ~$295 one-time  
**What it does:** Professional NLE with a color grading node graph, Fusion compositing engine, and a Python scripting API for render queue automation.  
**Pipeline fit:** Would enable template-driven LUT-based color grading and complex multi-clip timeline assembly beyond FFmpeg's filter graph capabilities.  
**Reason not using:** Free tier does not support external Python scripting (Studio license required). Python API has significant gaps: cannot programmatically build color nodes, no direct API for zoom/transitions/effects (requires pre-made DRX/Fusion macro files), and silent failures (returns `False`/`None` without error messages). FFmpeg is sufficient for social media clip quality at this scale. Revisit if professional color grading becomes a requirement.

---

## Audio

### Auphonic
**Stage:** Processing (audio normalization)  
**Status:** CONSIDERING  
**Cost:** Free tier (2 hrs/mo); ~$11/mo for 9 hrs  
**What it does:** Automated audio post-production service: loudness normalization (LUFS targeting), noise reduction, leveling, and encoding via API.  
**Pipeline fit:** Optional audio processing step after clip trim — normalize loudness to platform specs (YouTube: -14 LUFS, TikTok: -14 LUFS) before final encode.  
**Notes:** https://auphonic.com — REST API available; outputs normalized audio file. Evaluate whether FFmpeg's `loudnorm` filter is sufficient before adding this dependency.

---

### RX by iZotope
**Stage:** Audio repair  
**Status:** RESEARCH ONLY  
**Cost:** Subscription (~$99–$499/yr depending on tier)  
**What it does:** Industry-standard audio repair suite: de-noise, de-click, de-reverb, dialogue isolation, and spectral repair.  
**Pipeline fit:** Would improve audio quality for clips with background noise, game audio bleed, or mic issues.  
**Reason not using:** No automation API — manual-only workflow via standalone app or DAW plugin. Subscription cost not justified until audio quality is a proven bottleneck.

---

### Epidemic Sound
**Stage:** Distribution (background music)  
**Status:** CONSIDERING  
**Cost:** Subscription (~$15/mo personal, ~$49/mo commercial)  
**What it does:** Royalty-free music library with streaming platform clearance — tracks won't trigger Content ID claims on YouTube, TikTok, or Instagram.  
**Pipeline fit:** Background music layer for clips that need energy boost. Track selected by template `audio.background_music_id` field.  
**Notes:** https://www.epidemicsound.com — API available for licensed subscribers; track search by mood/genre/BPM.

---

### Artlist
**Stage:** Distribution (background music)  
**Status:** CONSIDERING  
**Cost:** Subscription (~$199/yr)  
**What it does:** Royalty-free music and SFX library with lifetime licensing — one annual subscription covers all content created that year permanently.  
**Pipeline fit:** Alternative to Epidemic Sound. Lifetime license model is better value if clip volume is high.  
**Notes:** https://artlist.io — no public API; track selection would be manual or via a curated ID list in templates.

---

## AI Scoring

### Claude API
**Stage:** AI Scoring  
**Status:** IN USE  
**Cost:** Token-based (~$3–15 per million tokens depending on model)  
**What it does:** Anthropic's Claude models via API. Multimodal (text + vision) — can analyze transcripts, metadata, and video frames to score clip quality and generate titles/descriptions.  
**Pipeline fit:** AI Scoring stage sends clip transcript + metadata to Claude with a scoring prompt. Returns structured JSON with `highlight_score`, `clip_type`, suggested title, and caption text.  
**Notes:** Claude 3.5 Sonnet or Claude 3 Haiku for cost efficiency. Vision input enables frame-level analysis without a separate model.

---

### GPT-4o
**Stage:** AI Scoring  
**Status:** CONSIDERING  
**Cost:** Token-based (~$5–15 per million tokens)  
**What it does:** OpenAI's multimodal model. Same use case as Claude API — transcript + metadata scoring with optional vision input.  
**Pipeline fit:** Drop-in alternative to Claude API in the AI Scoring stage. Both produce structured JSON output.  
**Notes:** Evaluate on accuracy vs. cost for the specific scoring prompt. Claude and GPT-4o can be A/B tested against the same ground truth labels.

---

### LangChain
**Stage:** AI Scoring / Orchestration  
**Status:** NOT USING  
**Cost:** Open Source  
**What it does:** Framework for building LLM-powered applications with chains, agents, memory, and tool use.  
**Pipeline fit:** Could orchestrate multi-step AI scoring workflows.  
**Reason not using:** Overhead not justified at current scale. Direct API calls to Claude or GPT-4o with a structured prompt are simpler, more debuggable, and faster to iterate on than a LangChain abstraction layer.

---

## Review UI

### Flask
**Stage:** Manual Review  
**Status:** IN USE  
**Cost:** Open Source  
**What it does:** Lightweight Python web framework for building server-side web applications and REST APIs.  
**Pipeline fit:** Backend for the Manual Review UI — serves clip data, accepts approve/reject/edit actions, updates the clip manifest.  
**Notes:** Minimal footprint; good for a simple review dashboard. Pair with HTMX or a lightweight JS frontend to avoid a full SPA.

---

### FastAPI
**Stage:** Manual Review / Pipeline API  
**Status:** NOT USING  
**Cost:** Open Source  
**What it does:** Modern Python web framework with automatic OpenAPI docs, async support, and Pydantic data validation.  
**Pipeline fit:** Alternative to Flask for the Review UI backend. Better choice if the pipeline exposes a REST API for external integrations (n8n webhooks, mobile review app).  
**Notes:** https://fastapi.tiangolo.com — async support is useful if the review UI needs to stream video or handle concurrent requests.  
**Reason not using:** Flask selected for the Manual Review UI. FastAPI's async benefits and auto-docs aren't needed at this scale.

---

### Streamlit
**Stage:** Review UI  
**Status:** NOT USING  
**Cost:** Open Source  
**What it does:** Python framework for building data dashboards and ML demos with minimal frontend code.  
**Pipeline fit:** Could display clip metrics and scoring outputs as a dashboard.  
**Reason not using:** No native video player component with editing controls (in/out trim markers, approve/reject buttons per clip). Streamlit is designed for data visualization, not interactive video review workflows.

---

## Distribution

### n8n
**Stage:** Distribution / Workflow Automation  
**Status:** NOT USING  
**Cost:** Open Source (self-hosted free); Cloud ~$24/mo  
**What it does:** Self-hosted workflow automation platform (similar to Zapier/Make but open source). Connects apps and APIs via a visual node editor.  
**Pipeline fit:** Watch folder → detect approved clips → trigger upload to TikTok/YouTube Shorts/Instagram Reels → log result. Keeps distribution logic out of the Python codebase.  
**Notes:** https://n8n.io — 400+ integrations. Docker-deployable. Webhook nodes can receive events from the pipeline.  
**Reason not using:** Direct platform API calls chosen for distribution. n8n overhead (separate service, visual editor, Docker) isn't justified at current scale.

---

### Buffer
**Stage:** Distribution (social scheduling)  
**Status:** CONSIDERING  
**Cost:** Free tier (3 channels); ~$6/mo per channel  
**What it does:** Social media scheduling tool with a REST API for programmatic post creation and queue management.  
**Pipeline fit:** Schedule approved clips for optimal posting times rather than posting immediately. API call at the end of the distribution stage enqueues the clip with caption and hashtags.  
**Notes:** https://buffer.com — supports YouTube, TikTok, Instagram, Twitter/X, LinkedIn. Evaluate against native platform APIs for upload reliability.

---

## Analytics / Optimize

### Optuna
**Stage:** Optimize (Calibration Layer)
**Status:** CONSIDERING
**Cost:** Open Source
**What it does:** Bayesian hyperparameter optimization framework. Defines an objective function and searches for the parameter combination that maximizes it using efficient sampling strategies (TPE, CMA-ES). Replaces grid search and manual tuning.
**Pipeline fit:** Once 50–100 clips have been labeled (accepted/rejected) via the review UI, Optuna can search over `BASELINE_WEIGHTS` (`kill_detection_saturation`, `weapon_confidence`, `audio_saturation`, `roi_match_count`) to find the weight combination that best predicts human review decisions. Replaces manual YAML weight editing with evidence-based calibration. Objective function uses the labeled clips in `accepted/` and `rejected/` as ground truth.
**Notes:** `pip install optuna`. Minimal study pattern:
```python
import optuna

def objective(trial):
    weights = {
        "kill_detection_saturation": trial.suggest_float("kill_detection_saturation", 0.1, 0.6),
        "weapon_confidence":         trial.suggest_float("weapon_confidence", 0.1, 0.5),
        "audio_saturation":          trial.suggest_float("audio_saturation", 0.05, 0.4),
        "roi_match_count":           trial.suggest_float("roi_match_count", 0.0, 0.3),
    }
    return evaluate_accuracy_on_labeled_clips(weights)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
print(study.best_params)
```
https://optuna.readthedocs.io — integrates with MLflow via `MLflowCallback`. Prerequisite: ~50 labeled clips from the review UI.

---

### MLflow
**Stage:** Optimize (Experiment Tracking)
**Status:** CONSIDERING
**Cost:** Open Source (self-hosted free)
**What it does:** Experiment tracking platform — log parameters, metrics, and artifacts per run. Compare runs side-by-side in a web UI. Tracks which weight configuration produced which accuracy result so tuning history is reproducible.
**Pipeline fit:** Each Optuna optimization run logs: weight combination tried, resulting accuracy on the labeled clip set, number of clips evaluated. The MLflow UI makes it trivial to compare runs, identify which weight combinations produce the best human-agreement rate, and roll back to a previous working config. Also useful for tracking AI scoring prompt iterations — log the prompt version and its impact on clip acceptance rate.
**Notes:** `pip install mlflow`. Start tracking server: `mlflow ui --port 5001` (defaults to SQLite backend, no cloud required). Log from Python:
```python
import mlflow

with mlflow.start_run():
    mlflow.log_params(weights)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_artifact("assets/games/marvel_rivals/weights.yaml")
```
https://mlflow.org — self-hosted with SQLite backend works fine at this scale. Pair with Optuna's `MLflowCallback` for automatic run logging.

---

### VidIQ
**Stage:** Optimize  
**Status:** CONSIDERING  
**Cost:** Free tier; Pro ~$7.50/mo  
**What it does:** YouTube analytics and SEO tool — keyword research, competitor analysis, tag suggestions, and performance tracking per video.  
**Pipeline fit:** Optimize stage — pull per-clip performance data (views, CTR, watch time) via API to feed back into scoring model and template parameters.  
**Notes:** https://vidiq.com — API access on paid tiers. Complements YouTube Analytics API data.

---

### TubeBuddy
**Stage:** Optimize  
**Status:** CONSIDERING  
**Cost:** Free tier; Pro ~$4.99/mo  
**What it does:** YouTube-focused browser extension and API for keyword research, A/B thumbnail testing, bulk processing, and analytics.  
**Pipeline fit:** Alternative or complement to VidIQ for keyword/tag optimization in the Optimize stage.  
**Notes:** https://www.tubebuddy.com — API available; evaluate alongside VidIQ to avoid paying for both.

---

## Competitive Reference

### Opus Clip
**Stage:** Reference  
**Status:** RESEARCH ONLY  
**Cost:** Free tier (limited); Pro ~$15–29/mo  
**What it does:** AI-powered tool that automatically identifies and extracts highlights from long-form video into short clips optimized for TikTok/Reels/Shorts. Adds captions, reframes to vertical, and scores each clip.  
**Pipeline fit:** Market leader in the exact problem space this bot is solving. Study for: clip selection scoring logic, caption style (word-highlight), vertical reframe quality, and UI/UX for review and approval.  
**Notes:** https://www.opus.pro — not a tool we integrate with; a product to analyze and learn from. Their "Clipping AI" scores clips on engagement potential — worth reverse-engineering the prompt strategy.  
**Reason not using:** External SaaS — we are building an equivalent system locally for control, cost, and customization.

---

### Descript
**Stage:** Reference  
**Status:** RESEARCH ONLY  
**Cost:** Free tier; Creator ~$24/mo  
**What it does:** Transcript-based video editor — edit video by editing the text transcript. Includes overdub (AI voice cloning), filler word removal, and caption generation.  
**Pipeline fit:** Study for: caption editing UX, transcript-synchronized clip trimming interface, and the overall paradigm of text-first video editing.  
**Notes:** https://www.descript.com — not integrated; reference for Manual Review UI design decisions.  
**Reason not using:** External SaaS; transcript-based editing UX is worth borrowing conceptually for our review interface.

---

### Captions App
**Stage:** Reference  
**Status:** RESEARCH ONLY  
**Cost:** Free tier; Pro ~$7/mo  
**What it does:** Mobile-first AI captions app with animated word-highlight style captions, auto-reframe, B-roll insertion, and clip trimming.  
**Pipeline fit:** Primary reference for the word-highlight caption style we want to implement — each word animates in sync with speech.  
**Notes:** https://www.captions.ai — mobile app (iOS/Android). Study caption animation timing, font choices, background pill/shadow styling, and line-break logic.  
**Reason not using:** Mobile-only; no API. Reference only for caption visual design.
