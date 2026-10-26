# Local Music Analyzer

Analyze music files locally, or capture macOS system audio and inspect it in a local web app. No Spotify credentials or cloud audio service is required.

**Related files:** [Live dependencies](requirements-audio-live.txt), [analysis dependencies](requirements-analysis.txt), [tests](tests/), [web interface](web/audio-live/index.html).

## Features

- Live waveform, spectrum, levels, onset candidates and tentative tempo.
- Save the captured signal as WAV and replay it with custom controls.
- Local analysis of rhythm, harmony, instrument/genre predictions, energy, sections and mastering metrics, depending on installed models.
- Demucs stems and a vocal melody contour with confidence and silence gaps.
- JSON and Markdown reports; individual stages report completed, skipped or failed status.

The interface is currently in Polish. Live capture requires **macOS 14.2 or newer**. The development runtime uses **Python 3.12**. File analysis also requires `ffmpeg` and `ffprobe` on PATH.

## Live system audio

From a clone of this repository:

```bash
python3.12 -m venv .tools/audio-live
.tools/audio-live/bin/python -m pip install -r requirements-audio-live.txt
bash scripts/run-audio-live.sh
```

Open <http://127.0.0.1:8766>. Allow system audio recording for the application hosting Python if macOS asks. The app uses [catap](https://github.com/sbetko/catap) / Core Audio taps, not the microphone. Click Start, then Stop and Save to keep a recording. Capture never starts automatically.

The launcher accepts `--port`, for example `bash scripts/run-audio-live.sh --port 8767`. The server always binds to `127.0.0.1` and checks request origins. The buffer is limited to 10 minutes and 128 MiB. Live updates contain at most 30 seconds of signal history.

Recordings and analysis jobs stay under `.tools/audio-live-data/`. This directory is ignored by Git. Stop capture before replaying a saved recording to avoid capturing that playback too.

## File analysis and deeper analysis in the app

The analysis runtime is separate from the capture runtime:

```bash
bash scripts/install-analysis-deps.sh
bash scripts/download-models.sh
bash scripts/analyze-song.sh --full /absolute/path/to/song.wav --skip-whisper
```

The installer creates `.venv/`, installs the analysis packages, tries Essentia with a DSP-only fallback, and prints import checks. Some optional packages or model backends may be unavailable on a particular platform; the report exposes failed/skipped stages instead of claiming complete results. Models are downloaded separately and are not part of this repository. First use of other analysis libraries can also download weights. Audio inference remains local.

With that runtime installed, the web app's Analyze button uses the same full pipeline with transcription disabled. Optional CLI transcription remains available by omitting `--skip-whisper`; its default language is `pl`, configurable with `--language auto` in full mode.

By default, an input such as `/music/song.wav` writes `/music/song.analysis/`. An input in an `audio/` directory retains the existing sibling `analysis/` layout. Use `--output-dir /path/to/report` with `--full` to choose a destination. Model paths default to this repository's `models/essentia/`, regardless of the current directory; `--models-dir` overrides that.

The main full-analysis outputs are `analysis.json` and `song-map.md`, plus stems and other stage outputs when available. `--include-frames` adds a NumPy frame-data file. See `bash scripts/analyze-song.sh --full --help` for individual `--skip-*` switches.

## Limits

Tempo, key, chord, structure and instrument labels are estimates. The pitch contour follows one dominant line, preferably an isolated vocal; it is not a polyphonic score or a complete per-instrument transcription. Browser plots use bounded data and at most 6,000 displayed melody points.

The repository contains no music, captured recordings, downloaded model weights, credentials or source-project history. Models and dependencies have their own licenses. Keep generated data private unless you intend to share it.

## Tests

```bash
python3.12 -m venv .tools/audio-live
.tools/audio-live/bin/python -m pip install -r requirements-dev.txt
.tools/audio-live/bin/python -m unittest discover -s tests
```

Tests use synthetic PCM and fake capture/model objects; they do not record the current system output or download model weights. Hardware capture and full model inference require separate checks on the target machine.
