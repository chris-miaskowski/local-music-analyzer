#!/usr/bin/env python3
"""
Analyze a song: vocal separation, beat detection, lyrics transcription.
Produces a platform-agnostic song-map.md.

Usage:
    python scripts/analyze-song.py /path/to/song.mp3 --skip-whisper
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path


def get_duration(audio_path: str) -> float:
    """Get audio duration in seconds using ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", audio_path],
        capture_output=True, text=True
    )
    return float(result.stdout.strip())


def run_demucs(audio_path: str, output_dir: str) -> str:
    """Separate vocals using demucs. Returns path to vocals.wav."""
    print("\n=== Step 1/3: Vocal separation (demucs) ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run(
            [sys.executable, "-m", "demucs", "--two-stems", "vocals",
             "-o", tmpdir, audio_path],
            check=True
        )
        # demucs outputs to tmpdir/htdemucs/<stem_name>/vocals.wav
        stem_name = Path(audio_path).stem
        src = os.path.join(tmpdir, "htdemucs", stem_name, "vocals.wav")
        dst = os.path.join(output_dir, "vocals.wav")
        os.rename(src, dst)
    print(f"  -> {dst}")
    return dst


def run_beat_this(audio_path: str, output_dir: str) -> dict:
    """Detect beats using beat_this. Returns beat data dict."""
    print("\n=== Step 2/3: Beat detection (beat_this) ===")
    from beat_this.inference import File2Beats

    file2beats = File2Beats(checkpoint_path="final0", device="cpu", dbn=False)
    beats, downbeats = file2beats(audio_path)

    beats_list = [float(b) for b in beats]
    downbeats_list = [float(d) for d in downbeats]

    if len(beats_list) >= 2:
        intervals = [beats_list[i+1] - beats_list[i] for i in range(min(len(beats_list)-1, 50))]
        avg_interval = sum(intervals) / len(intervals)
        bpm = round(60.0 / avg_interval) if avg_interval > 0 else 0
    else:
        bpm = 0

    beat_data = {
        "bpm": bpm,
        "beats": beats_list,
        "downbeats": downbeats_list,
    }

    out_path = os.path.join(output_dir, "beats.json")
    with open(out_path, "w") as f:
        json.dump(beat_data, f, indent=2)

    print(f"  -> {out_path}")
    print(f"  BPM: {bpm}, Beats: {len(beats_list)}, Downbeats: {len(downbeats_list)}")
    return beat_data


def run_whisper(vocals_path: str, output_dir: str) -> list:
    """Transcribe vocals with word-level timestamps. Returns word list."""
    print("\n=== Step 3/3: Lyrics transcription (whisper) ===")
    import whisper

    model = whisper.load_model("large-v2")
    result = model.transcribe(
        vocals_path,
        language="pl",
        word_timestamps=True,
        condition_on_previous_text=True,
    )

    words = []
    for segment in result.get("segments", []):
        for w in segment.get("words", []):
            words.append({
                "start": round(w["start"], 3),
                "end": round(w["end"], 3),
                "word": w["word"].strip(),
            })

    out_path = os.path.join(output_dir, "lyrics.tsv")
    with open(out_path, "w") as f:
        f.write("start\tend\tword\n")
        for w in words:
            f.write(f"{w['start']:.2f}\t{w['end']:.2f}\t{w['word']}\n")

    print(f"  -> {out_path}")
    print(f"  Words transcribed: {len(words)}")
    return words


def format_time(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m)}:{s:05.2f}"


def generate_song_map(
    track_name: str,
    duration: float,
    beat_data: dict,
    words: list,
    output_dir: str,
) -> str:
    """Generate the platform-agnostic song-map.md."""
    print("\n=== Generating song-map.md ===")

    bpm = beat_data["bpm"]
    beats = beat_data["beats"]
    downbeats = set(round(d, 3) for d in beat_data["downbeats"])

    dur_m, dur_s = divmod(int(duration), 60)
    date = datetime.now().strftime("%Y-%m-%d")

    lines = []
    lines.append(f"# {track_name} -- Song Map\n")
    lines.append(f"BPM: {bpm} | Duration: {dur_m}:{dur_s:02d} | Generated: {date}\n")

    # Beat grid
    lines.append("## Beat Grid\n")
    lines.append("| Beat | Time | Bar | Downbeat |")
    lines.append("|------|------|-----|----------|")

    bar = 0
    beat_in_bar = 0
    beats_per_bar = 4  # default assumption
    for i, t in enumerate(beats):
        t_rounded = round(t, 3)
        is_downbeat = t_rounded in downbeats or any(abs(t_rounded - d) < 0.05 for d in beat_data["downbeats"])
        if is_downbeat:
            bar += 1
            beat_in_bar = 1
        else:
            beat_in_bar += 1
        db_mark = "yes" if is_downbeat else ""
        lines.append(f"| {i+1} | {t_rounded:.3f} | {bar} | {db_mark} |")

    # Lyrics reference
    lines.append("\n## Timestamped Lyrics\n")
    lines.append(f"See `lyrics.tsv` ({len(words)} words)")
    lines.append("")

    # Sections template
    lines.append("\n## Sections (manual -- fill in after reviewing beats + lyrics)\n")
    lines.append("| Section | Start | End | Lyric cue | Notes |")
    lines.append("|---------|-------|-----|-----------|-------|")
    lines.append("| | | | | |")
    lines.append("")

    content = "\n".join(lines)
    out_path = os.path.join(output_dir, "song-map.md")
    with open(out_path, "w") as f:
        f.write(content)

    print(f"  -> {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Analyze a song for content production")
    parser.add_argument("audio", help="Path to audio file (mp3/wav)")
    parser.add_argument("--skip-demucs", action="store_true", help="Skip vocal separation")
    parser.add_argument("--skip-beats", action="store_true", help="Skip beat detection")
    parser.add_argument("--skip-whisper", action="store_true", help="Skip transcription")
    args = parser.parse_args()

    audio_path = os.path.abspath(args.audio)
    if not os.path.isfile(audio_path):
        print(f"Error: {audio_path} not found")
        sys.exit(1)

    p = Path(audio_path)
    output_dir = str(p.parent.parent / "analysis" if p.parent.name == "audio"
                     else p.parent / f"{p.stem}.analysis")
    os.makedirs(output_dir, exist_ok=True)

    track_name = (p.parent.parent.name if p.parent.name == "audio" else p.stem).replace("-", " ").title()
    print(f"Analyzing: {track_name}")
    print(f"Input: {audio_path}")
    print(f"Output: {output_dir}")

    duration = get_duration(audio_path)
    print(f"Duration: {format_time(duration)}")

    # Step 1: Vocal separation
    vocals_path = os.path.join(output_dir, "vocals.wav")
    if not args.skip_demucs:
        vocals_path = run_demucs(audio_path, output_dir)
    elif not os.path.isfile(vocals_path):
        print("Warning: --skip-demucs but no vocals.wav found, using original audio for whisper")
        vocals_path = audio_path

    # Step 2: Beat detection
    beat_data = None
    beats_json = os.path.join(output_dir, "beats.json")
    if not args.skip_beats:
        beat_data = run_beat_this(audio_path, output_dir)
    elif os.path.isfile(beats_json):
        with open(beats_json) as f:
            beat_data = json.load(f)
    else:
        beat_data = {"bpm": 0, "beats": [], "downbeats": []}

    # Step 3: Lyrics transcription
    words = []
    lyrics_tsv = os.path.join(output_dir, "lyrics.tsv")
    if not args.skip_whisper:
        words = run_whisper(vocals_path, output_dir)
    elif os.path.isfile(lyrics_tsv):
        words = []
        with open(lyrics_tsv) as f:
            next(f)  # skip header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 3:
                    words.append({"start": float(parts[0]), "end": float(parts[1]), "word": parts[2]})

    # Generate song map
    generate_song_map(track_name, duration, beat_data, words, output_dir)

    print("\n=== Done! ===")
    print(f"Review: {os.path.join(output_dir, 'song-map.md')}")


if __name__ == "__main__":
    main()
