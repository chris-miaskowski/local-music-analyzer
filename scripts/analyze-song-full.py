#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ANALYZER_VERSION = "1.0.0"
include_frames_in_json = False

BINARY_LABELS = {
    "mood_aggressive": ["aggressive", "not_aggressive"],
    "mood_happy": ["happy", "not_happy"],
    "mood_sad": ["sad", "not_sad"],
    "mood_relaxed": ["relaxed", "not_relaxed"],
    "mood_party": ["party", "not_party"],
    "mood_acoustic": ["acoustic", "not_acoustic"],
    "mood_electronic": ["electronic", "not_electronic"],
    "voice_instrumental": ["voice", "instrumental"],
    "gender": ["female", "male"],
    "timbre": ["bright", "dark"],
    "tonal_atonal": ["tonal", "atonal"],
    "danceability": ["danceable", "not_danceable"],
    "nsynth_acoustic_electronic": ["acoustic", "electronic"],
}

CLASSIFIERS = {
    "genre_discogs400": ("classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1", False),
    "mood_aggressive": ("classification-heads/mood_aggressive/mood_aggressive-discogs-effnet-1", True),
    "mood_happy": ("classification-heads/mood_happy/mood_happy-discogs-effnet-1", True),
    "mood_sad": ("classification-heads/mood_sad/mood_sad-discogs-effnet-1", True),
    "mood_relaxed": ("classification-heads/mood_relaxed/mood_relaxed-discogs-effnet-1", True),
    "mood_party": ("classification-heads/mood_party/mood_party-discogs-effnet-1", True),
    "mood_acoustic": ("classification-heads/mood_acoustic/mood_acoustic-discogs-effnet-1", True),
    "mood_electronic": ("classification-heads/mood_electronic/mood_electronic-discogs-effnet-1", True),
    "voice_instrumental": ("classification-heads/voice_instrumental/voice_instrumental-discogs-effnet-1", True),
    "gender": ("classification-heads/gender/gender-discogs-effnet-1", True),
    "timbre": ("classification-heads/timbre/timbre-discogs-effnet-1", True),
    "danceability": ("classification-heads/danceability/danceability-discogs-effnet-1", True),
    "tonal_atonal": ("classification-heads/tonal_atonal/tonal_atonal-discogs-effnet-1", True),
    "approachability": ("classification-heads/approachability/approachability_regression-discogs-effnet-1", False),
    "engagement": ("classification-heads/engagement/engagement_regression-discogs-effnet-1", False),
    "mtg_jamendo_instrument": ("classification-heads/mtg_jamendo_instrument/mtg_jamendo_instrument-discogs-effnet-1", False),
    "mtg_jamendo_moodtheme": ("classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-discogs-effnet-1", False),
    "mtg_jamendo_top50tags": ("classification-heads/mtg_jamendo_top50tags/mtg_jamendo_top50tags-discogs-effnet-1", False),
    "nsynth_acoustic_electronic": ("classification-heads/nsynth_acoustic_electronic/nsynth_acoustic_electronic-discogs-effnet-1", True),
    "nsynth_reverb": ("classification-heads/nsynth_reverb/nsynth_reverb-discogs-effnet-1", False),
    "nsynth_instrument": ("classification-heads/nsynth_instrument/nsynth_instrument-discogs-effnet-1", False),
    "nsynth_bright_dark": ("classification-heads/nsynth_bright_dark/nsynth_bright_dark-discogs-effnet-1", False),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Comprehensive local MP3/WAV analysis")
    parser.add_argument("audio", help="Path to MP3/WAV")
    parser.add_argument("--skip-demucs", action="store_true", help="Skip stem separation")
    parser.add_argument("--skip-beats", action="store_true", help="Skip beat detection")
    parser.add_argument("--skip-whisper", action="store_true", help="Skip transcription")
    parser.add_argument("--whisper-model", default="large-v2", help="Local Whisper model (default: large-v2)")
    parser.add_argument("--language", default="pl", help="Transcription language code, or auto (default: pl)")
    parser.add_argument("--skip-essentia", action="store_true", help="Skip essentia DSP analysis")
    parser.add_argument("--skip-ml", action="store_true", help="Skip ML classification")
    parser.add_argument("--skip-librosa", action="store_true", help="Skip librosa spectral analysis")
    parser.add_argument("--skip-pitch", action="store_true", help="Skip pitch extraction")
    parser.add_argument("--skip-sections", action="store_true", help="Skip section detection (allin1)")
    parser.add_argument("--skip-mastering", action="store_true", help="Skip mastering metrics")
    parser.add_argument("--skip-energy", action="store_true", help="Skip energy curve analysis")
    parser.add_argument("--skip-per-stem", action="store_true", help="Skip per-stem ML classification")
    parser.add_argument("--vocals-only", action="store_true", help="Demucs: only extract vocals (not all 4 stems)")
    parser.add_argument("--models-dir", default=str(Path(__file__).resolve().parents[1] / "models/essentia"), help="Path to essentia models (default: repository models/essentia/)")
    parser.add_argument("--include-frames", action="store_true", help="Include frame-level data in analysis-frames.npz")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    return parser.parse_args()


def resolve_paths(audio_arg: str, output_override: Optional[str], models_dir_arg: str) -> Dict[str, str]:
    audio_path = os.path.abspath(audio_arg)
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if output_override:
        output_dir = os.path.abspath(output_override)
    else:
        p = Path(audio_path)
        output_dir = str(p.parent.parent / "analysis" if p.parent.name == "audio"
                         else p.parent / f"{p.stem}.analysis")

    models_dir = os.path.abspath(models_dir_arg)
    os.makedirs(output_dir, exist_ok=True)
    return {"audio_path": audio_path, "output_dir": output_dir, "models_dir": models_dir}


def run_step_safe(step_name: str, func: Any, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    print(f"\n=== {step_name} ===")
    try:
        result = func(*args, **kwargs)
        print(f"  ✓ {step_name} completed")
        return {"status": "ok", "data": result}
    except ImportError as e:
        print(f"  ⚠ {step_name} SKIPPED — missing dependency: {e}")
        return {"status": "skipped", "reason": f"Missing dependency: {e}"}
    except Exception as e:
        print(f"  ✗ {step_name} FAILED — {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "reason": str(e)}


def make_skipped(reason: str) -> Dict[str, Any]:
    return {"status": "skipped", "reason": reason}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def get_duration(audio_path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", audio_path],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return float(result.stdout.strip())
    except (TypeError, ValueError):
        return 0.0


def format_time(seconds: float) -> str:
    m, s = divmod(max(0.0, float(seconds)), 60.0)
    return f"{int(m)}:{s:05.2f}"


def format_mmss(seconds: float) -> str:
    total = int(max(0.0, float(seconds)))
    m, s = divmod(total, 60)
    return f"{m}:{s:02d}"


def infer_track_name(audio_path: str) -> str:
    p = Path(audio_path)
    parts = list(p.parts)
    if "tracks" in parts:
        try:
            idx = parts.index("tracks")
            slug = parts[idx + 1]
            return slug.replace("-", " ").title()
        except (ValueError, IndexError):
            pass
    return p.stem.replace("-", " ").title()


def to_jsonable(obj: Any) -> Any:
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None and isinstance(obj, np.generic):
        return obj.item()
    if np is not None and isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def stat_summary(arr):
    import numpy as np
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def run_demucs(audio_path: str, output_dir: str, vocals_only: bool = False) -> str:
    import tempfile
    import shutil

    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [sys.executable, "-m", "demucs"]
        if vocals_only:
            cmd += ["--two-stems", "vocals"]
        cmd += ["-o", tmpdir, audio_path]
        subprocess.run(cmd, check=True)

        stem_name = Path(audio_path).stem
        demucs_out = os.path.join(tmpdir, "htdemucs", stem_name)

        stems_dir = os.path.join(output_dir, "stems")
        os.makedirs(stems_dir, exist_ok=True)

        for wav_file in Path(demucs_out).glob("*.wav"):
            shutil.copy2(str(wav_file), os.path.join(stems_dir, wav_file.name))

        vocals_src = os.path.join(demucs_out, "vocals.wav")
        vocals_dst = os.path.join(output_dir, "vocals.wav")
        if os.path.isfile(vocals_src):
            shutil.copy2(vocals_src, vocals_dst)

    return vocals_dst


def run_beat_this(audio_path: str, output_dir: str) -> Dict[str, Any]:
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


def run_whisper(vocals_path: str, output_dir: str, model_name: str = "large-v2", language: str = "pl", metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    print("\n=== Step 3/3: Lyrics transcription (whisper) ===")
    import whisper

    model = whisper.load_model(model_name)
    result = model.transcribe(
        vocals_path,
        language=None if language == "auto" else language,
        word_timestamps=True,
        condition_on_previous_text=True,
    )
    if metadata is not None:
        metadata.update(model=model_name, language=result.get("language"), requested_language=language)

    words = []
    segments = result.get("segments", []) if isinstance(result, dict) else []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        for w in segment.get("words", []):
            if not isinstance(w, dict):
                continue
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


def run_essentia_dsp(audio_path: str):
    es = importlib.import_module("essentia.standard")
    import numpy as np

    audio = es.MonoLoader(filename=audio_path, sampleRate=44100)()

    rhythm = es.RhythmExtractor2013(method="multifeature")
    bpm, beats, beats_confidence, _, beats_intervals = rhythm(audio)

    key_extractor = es.KeyExtractor()
    key, scale, key_strength = key_extractor(audio)

    frame_size = 4096
    hop_size = 2048
    hpcps = []
    for frame in es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size, startFromZero=True):
        spec = es.Spectrum()(es.Windowing(type="blackmanharris62")(frame))
        freqs, mags = es.SpectralPeaks(orderBy="magnitude", magnitudeThreshold=0.00001, minFrequency=20, maxFrequency=3500, maxPeaks=60)(spec)
        hpcp = es.HPCP(size=12, referenceFrequency=440, minFrequency=20, maxFrequency=3500)(freqs, mags)
        hpcps.append(hpcp)
    hpcp_matrix = np.array(hpcps)
    chords, chord_strengths = es.ChordsDetection(hopSize=hop_size, windowSize=2)(hpcp_matrix)

    chord_list = []
    frame_duration = hop_size / 44100.0
    for i, (chord, strength) in enumerate(zip(chords, chord_strengths)):
        chord_list.append({
            "time": round(i * frame_duration, 3),
            "label": chord,
            "strength": round(float(strength), 3)
        })

    onsets, onset_rate = es.OnsetRate()(audio)

    try:
        stereo_audio = es.AudioLoader(filename=audio_path)()[0]
        integrated_lufs, lra, momentary, short_term = es.LoudnessEBUR128()(stereo_audio)
    except Exception:
        integrated_lufs, lra, momentary, short_term = 0.0, 0.0, [], []
    rms = float(es.RMS()(audio))
    dynamic_complexity, avg_loudness = es.DynamicComplexity()(audio)
    replay_gain = float(es.ReplayGain()(audio))

    try:
        features, _ = es.MusicExtractor(
            lowlevelStats=["mean", "stdev", "min", "max"],
            rhythmStats=["mean", "stdev"],
            tonalStats=["mean", "stdev"]
        )(audio_path)

        music_extractor_data = {}
        for desc in features.descriptorNames():
            val = features[desc]
            if isinstance(val, (float, int, str)):
                music_extractor_data[desc] = val
            elif hasattr(val, 'tolist'):
                music_extractor_data[desc] = val.tolist()
    except Exception as e:
        music_extractor_data = {"error": str(e)}

    return {
        "bpm": float(bpm),
        "beats": [float(x) for x in beats],
        "beats_confidence": float(beats_confidence),
        "beats_intervals": [float(x) for x in beats_intervals],
        "key": str(key),
        "scale": str(scale),
        "key_strength": float(key_strength),
        "chords": chord_list,
        "onsets": [float(x) for x in onsets],
        "onset_rate": float(onset_rate),
        "loudness": {
            "integrated_lufs": safe_float(integrated_lufs),
            "lra": safe_float(lra),
            "rms": safe_float(rms),
            "dynamic_complexity": safe_float(dynamic_complexity),
            "avg_loudness": safe_float(avg_loudness),
            "replay_gain": safe_float(replay_gain),
        },
        "music_extractor": to_jsonable(music_extractor_data),
    }


def format_predictions(task_name, mean_preds, classes, is_binary):
    import numpy as np
    if is_binary:
        labels = BINARY_LABELS.get(task_name, ["class_0", "class_1"])
        winner_idx = int(np.argmax(mean_preds))
        return {
            "label": labels[winner_idx],
            "probability": round(float(mean_preds[winner_idx]), 4),
            "all_probabilities": {l: round(float(p), 4) for l, p in zip(labels, mean_preds)}
        }
    else:
        if classes:
            top_k = 10
            idx = np.argsort(mean_preds)[::-1][:top_k]
            return [{"label": classes[i], "probability": round(float(mean_preds[i]), 4)} for i in idx]
        else:
            return {"raw_probabilities": mean_preds.tolist()}


def load_model_metadata(json_path: str) -> Dict[str, Any]:
    if not os.path.exists(json_path):
        return {}
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_classes(json_path: str) -> Optional[List[str]]:
    meta = load_model_metadata(json_path)
    classes = meta.get("classes", None)
    if isinstance(classes, list):
        return [str(c) for c in classes]
    return None


def get_node_names(json_path: str, default_input: str, default_output: str) -> Tuple[str, str]:
    meta = load_model_metadata(json_path)
    schema = meta.get("schema", {})
    inputs = schema.get("inputs", [])
    outputs = schema.get("outputs", [])
    input_name = inputs[0]["name"] if inputs and "name" in inputs[0] else default_input
    output_name = default_output
    for o in outputs:
        if o.get("output_purpose") == "predictions" or o.get("op") in ("Softmax", "Sigmoid", "Identity"):
            output_name = o["name"]
            break
    if output_name == default_output and outputs and "name" in outputs[0]:
        output_name = outputs[0]["name"]
    return input_name, output_name


def run_essentia_ml(audio_path: str, models_dir: str):
    es = importlib.import_module("essentia.standard")
    has_essentia_tf = all(
        hasattr(es, x)
        for x in ["MonoLoader", "TensorflowPredictEffnetDiscogs", "TensorflowPredict2D"]
    )

    if has_essentia_tf:
        try:
            import numpy as np

            MonoLoader = getattr(es, "MonoLoader")
            TensorflowPredictEffnetDiscogs = getattr(es, "TensorflowPredictEffnetDiscogs")
            TensorflowPredict2D = getattr(es, "TensorflowPredict2D")

            audio = MonoLoader(filename=audio_path, sampleRate=16000, resampleQuality=4)()

            effnet_model = os.path.join(models_dir, "feature-extractors", "discogs-effnet-bs64-1.pb")
            if not os.path.exists(effnet_model):
                raise FileNotFoundError(f"Missing TF embedding model: {effnet_model}")

            embedding_model = TensorflowPredictEffnetDiscogs(
                graphFilename=effnet_model,
                output="PartitionedCall:1"
            )
            embeddings = embedding_model(audio)

            results = {}
            for task_name, (model_subpath, is_binary) in CLASSIFIERS.items():
                model_path = os.path.join(models_dir, f"{model_subpath}.pb")
                json_path = os.path.join(models_dir, f"{model_subpath}.json")

                if not os.path.exists(model_path):
                    continue

                try:
                    input_node, output_node = get_node_names(
                        json_path, "serving_default_model_Placeholder", "PartitionedCall:0"
                    )
                    classifier = TensorflowPredict2D(
                        graphFilename=model_path,
                        input=input_node,
                        output=output_node,
                    )
                    preds = classifier(embeddings)
                    mean_preds = preds.mean(axis=0)

                    classes = load_classes(json_path)
                    results[task_name] = format_predictions(task_name, mean_preds, classes, is_binary)
                except Exception as cls_err:
                    print(f"    ⚠ classifier {task_name} failed: {cls_err}")

            results["backend"] = "essentia-tensorflow"
            results["embedding_shape"] = list(np.asarray(embeddings).shape)
            return results
        except Exception as tf_error:
            print(f"  ⚠ essentia-tensorflow path failed, falling back to ONNX: {tf_error}")

    ort = importlib.import_module("onnxruntime")
    import numpy as np

    audio = es.MonoLoader(filename=audio_path, sampleRate=16000, resampleQuality=4)()

    effnet_onnx = os.path.join(models_dir, "feature-extractors", "discogs-effnet-bsdynamic-1.onnx")
    if not os.path.exists(effnet_onnx):
        raise FileNotFoundError(f"Missing ONNX embedding model: {effnet_onnx}")

    session = ort.InferenceSession(effnet_onnx)

    mel_bands_algo = es.MelBands(numberBands=96, sampleRate=16000, lowFrequencyBound=0, highFrequencyBound=8000)
    spec_algo = es.Spectrum(size=512)
    windowing = es.Windowing(type="hann", size=512)

    mel_frames = []
    for frame in es.FrameGenerator(audio, frameSize=512, hopSize=256, startFromZero=True):
        spec = spec_algo(windowing(frame))
        mel = mel_bands_algo(spec)
        mel_frames.append(np.log10(np.maximum(mel, 1e-10)))

    mel_frames = np.array(mel_frames)

    patch_size = 64
    patch_hop = 32
    patches = []
    for start in range(0, max(1, len(mel_frames) - patch_size + 1), patch_hop):
        patch = mel_frames[start:start + patch_size]
        if len(patch) < patch_size:
            patch = np.pad(patch, ((0, patch_size - len(patch)), (0, 0)))
        patches.append(patch)

    if not patches:
        patches = [np.zeros((patch_size, 96))]

    patches_array = np.array(patches, dtype=np.float32)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    embeddings = session.run([output_name], {input_name: patches_array})[0]

    results = {}
    for task_name, (model_subpath, is_binary) in CLASSIFIERS.items():
        onnx_path = os.path.join(models_dir, f"{model_subpath}.onnx")
        if not os.path.exists(onnx_path):
            continue
        json_path = os.path.join(models_dir, f"{model_subpath}.json")
        cls_session = ort.InferenceSession(onnx_path)
        cls_input = cls_session.get_inputs()[0].name
        cls_output = cls_session.get_outputs()[0].name
        preds = cls_session.run([cls_output], {cls_input: embeddings.astype(np.float32)})[0]
        mean_preds = preds.mean(axis=0)
        classes = load_classes(json_path)
        results[task_name] = format_predictions(task_name, mean_preds, classes, is_binary)

    results["backend"] = "onnx"
    results["embedding_shape"] = list(np.asarray(embeddings).shape)
    return results


def run_librosa_analysis(audio_path: str):
    librosa = importlib.import_module("librosa")
    import numpy as np

    y, sr = librosa.load(audio_path, sr=22050, mono=True)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmin=20, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    chroma = librosa.feature.chroma_cens(y=y, sr=sr)

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rms = librosa.feature.rms(y=y)[0]

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_mels=40)

    oenv = librosa.onset.onset_strength(y=y, sr=sr)
    tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr)
    tempo_bpm = float(librosa.feature.tempo(onset_envelope=oenv, sr=sr)[0])

    return {
        "mel_spectrogram_db": {"mean_per_band": mel_db.mean(axis=1).tolist(), "std_per_band": mel_db.std(axis=1).tolist()},
        "chroma_mean": chroma.mean(axis=1).tolist(),
        "spectral_centroid": stat_summary(centroid),
        "spectral_bandwidth": stat_summary(bandwidth),
        "spectral_rolloff": stat_summary(rolloff),
        "spectral_contrast": {"mean_per_band": contrast.mean(axis=1).tolist()},
        "zcr": stat_summary(zcr),
        "rms": stat_summary(rms),
        "mfcc": {"mean": mfcc.mean(axis=1).tolist(), "std": mfcc.std(axis=1).tolist()},
        "tempo_librosa_bpm": tempo_bpm,
        "_frames": {
            "mel_spectrogram": mel_db,
            "chromagram": chroma,
            "mfcc": mfcc,
            "spectral_centroid": centroid,
            "rms": rms,
            "tempogram": tempogram,
            "onset_env": oenv,
        },
    }


def pitch_frame_rms(samples, frame_count, hop_length=160, frame_length=1024):
    """Match torchcrepe's centered, zero-padded windows without a large frame matrix."""
    import numpy as np

    samples = np.asarray(samples).reshape(-1)
    rms = np.zeros(frame_count, dtype=np.float64)
    for offset in range(0, frame_count, 1024):
        centers = np.arange(offset, min(offset + 1024, frame_count)) * hop_length
        starts = np.clip(centers - frame_length // 2, 0, len(samples))
        ends = np.clip(centers + frame_length // 2, 0, len(samples))
        first, last = int(starts[0]), int(ends[-1])
        block = np.asarray(samples[first:last], dtype=np.float64)
        energy = np.concatenate(([0.0], np.cumsum(block * block)))
        power = (energy[ends - first] - energy[starts - first]) / frame_length
        rms[offset:offset + len(centers)] = np.sqrt(np.maximum(power, 0))
    return rms


def build_melody_timeline(pitch, confidence, rms, duration_seconds, hop_seconds,
                          source="vocals", max_points=6000):
    """Return a bounded monophonic contour and the full-resolution voiced mask."""
    import numpy as np

    pitch = np.asarray(pitch, dtype=np.float64).reshape(-1)
    confidence = np.asarray(confidence, dtype=np.float64).reshape(-1)
    rms = np.asarray(rms, dtype=np.float64).reshape(-1)
    if len(pitch) != len(confidence) or len(pitch) != len(rms):
        raise ValueError("Pitch, confidence and RMS frames must have equal lengths")
    if max_points < 2 or not np.isfinite(hop_seconds) or hop_seconds <= 0:
        raise ValueError("Melody requires a positive hop and at least two display points")
    if not np.isfinite(duration_seconds) or duration_seconds < 0:
        raise ValueError("Melody duration must be finite and nonnegative")

    threshold, silence_db = 0.5, -60.0
    times = np.arange(len(pitch), dtype=np.float64) * hop_seconds
    voiced = (np.isfinite(pitch) & (pitch >= 50) & (pitch <= 1000)
              & np.isfinite(confidence) & (confidence > threshold)
              & np.isfinite(rms) & (rms >= 10 ** (silence_db / 20))
              & (times <= duration_seconds + 1e-9))
    confidence = np.clip(np.nan_to_num(confidence, nan=0.0, posinf=0.0, neginf=0.0), 0, 1)
    valid_count = min(len(pitch), int(np.floor(duration_seconds / hop_seconds + 1e-9)) + 1)
    points = []
    if valid_count:
        # Every frame contributes to a display bucket. A bucket containing an
        # unvoiced frame stays a gap, so decimation cannot join separate phrases.
        starts = np.linspace(0, valid_count - 1, min(valid_count, max_points), dtype=int)
        ends = np.concatenate((starts[1:], [valid_count]))
        for start, end in zip(starts, ends):
            audible = bool(voiced[start:end].all())
            hz = float(np.median(pitch[start:end])) if audible else None
            points.append({
                "time": round(float(times[start]), 6),
                "hz": round(hz, 3) if hz is not None else None,
                "midi": round(float(69 + 12 * np.log2(hz / 440)), 3) if hz is not None else None,
                "confidence": round(float(confidence[start:end].min()), 4),
            })
    return {
        "source": source,
        "method": "torchcrepe",
        "kind": "monophonic_estimate",
        "duration_seconds": round(float(duration_seconds), 6),
        "hop_seconds": float(hop_seconds),
        "confidence_threshold": threshold,
        "silence_threshold_db": silence_db,
        "confidence_measure": "periodicity",
        "frame_count": valid_count,
        "voiced_frame_count": int(voiced.sum()),
        "points": points,
    }, voiced


def run_pitch_analysis(vocals_path, source="vocals"):
    torchcrepe = importlib.import_module("torchcrepe")
    import torchaudio
    import numpy as np
    librosa = importlib.import_module("librosa")

    audio, sr = torchaudio.load(vocals_path)
    duration_seconds = audio.shape[-1] / sr
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    # torchcrepe interprets hop_length in samples at the supplied sample rate,
    # then truncates it while resampling internally. Use its native rate so
    # frame centers are exactly 10 ms apart, independent of the source rate.
    pitch_rate, hop_length = 16000, 160
    if sr != pitch_rate:
        audio = torchaudio.functional.resample(audio, sr, pitch_rate)

    pitch, confidence = torchcrepe.predict(
        audio, pitch_rate,
        hop_length=hop_length,
        fmin=50, fmax=1000,
        model="full",
        batch_size=64,
        device="cpu",
        return_periodicity=True,
        pad=True,
    )

    pitch = pitch.detach().cpu().numpy().reshape(-1)
    confidence = confidence.detach().cpu().numpy().reshape(-1)
    rms = pitch_frame_rms(audio.numpy(), len(pitch), hop_length=hop_length)
    melody, voiced_mask = build_melody_timeline(
        pitch, confidence, rms, duration_seconds, hop_length / pitch_rate, source=source,
    )

    voiced_pitch = pitch[voiced_mask]

    if len(voiced_pitch) > 0:
        midi_notes = librosa.hz_to_midi(voiced_pitch)
        result = {
            "pitch_hz": stat_summary(voiced_pitch),
            "pitch_midi_min": librosa.midi_to_note(int(round(midi_notes.min()))),
            "pitch_midi_max": librosa.midi_to_note(int(round(midi_notes.max()))),
            "pitch_midi_mean": librosa.midi_to_note(int(round(np.mean(midi_notes)))),
            "vocal_range_semitones": int(round(midi_notes.max() - midi_notes.min())),
            "confidence_mean": float(confidence[voiced_mask].mean()),
            "voiced_ratio": float(voiced_mask.mean()),
        }
    else:
        result = {"error": "No voiced frames detected", "voiced_ratio": 0.0}
    result["melody"] = melody
    result["_frames"] = {"pitch": pitch, "pitch_confidence": confidence}
    return result


def run_sections(audio_path: str) -> Dict[str, Any]:
    import allin1

    def summarize_segments(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        by_label: Dict[str, Dict[str, Any]] = {}
        total_duration = 0.0
        for sec in segments:
            label = str(sec.get("label", "section"))
            start = safe_float(sec.get("start"), 0.0)
            end = safe_float(sec.get("end"), 0.0)
            duration = max(0.0, end - start)
            total_duration += duration
            if label not in by_label:
                by_label[label] = {"count": 0, "total_duration": 0.0}
            by_label[label]["count"] += 1
            by_label[label]["total_duration"] += duration
        ordered = {
            k: {
                "count": int(v["count"]),
                "total_duration": round(float(v["total_duration"]), 3),
                "mean_duration": round(float(v["total_duration"] / max(1, v["count"])), 3),
            }
            for k, v in sorted(by_label.items(), key=lambda item: item[0])
        }
        return {
            "count": len(segments),
            "total_duration": round(total_duration, 3),
            "by_label": ordered,
        }

    analysis_result = allin1.analyze(audio_path, device="cpu", keep_byproducts=False)
    result = analysis_result[0] if isinstance(analysis_result, list) and analysis_result else analysis_result
    if result is None:
        return {
            "bpm_allin1": 0.0,
            "segments": [],
            "beats_allin1": [],
            "downbeats_allin1": [],
            "summary": {"count": 0, "total_duration": 0.0, "by_label": {}},
        }
    segments = [
        {"start": float(s.start), "end": float(s.end), "label": str(s.label)}
        for s in getattr(result, "segments", [])
    ]
    return {
        "bpm_allin1": safe_float(getattr(result, "bpm", 0.0)),
        "segments": segments,
        "beats_allin1": [float(b) for b in getattr(result, "beats", [])],
        "downbeats_allin1": [float(d) for d in getattr(result, "downbeats", [])],
        "summary": summarize_segments(segments),
    }


def run_mastering(audio_path: str) -> Dict[str, Any]:
    import numpy as np
    import pyloudnorm as pyln
    import librosa
    from scipy.signal import resample_poly, welch

    y_stereo, sr = librosa.load(audio_path, sr=None, mono=False)
    y_stereo = y_stereo.T.astype(np.float64)
    y_mono = np.mean(y_stereo, axis=1) if y_stereo.ndim == 2 else y_stereo

    meter = pyln.Meter(sr)
    integrated_lufs = meter.integrated_loudness(y_stereo)
    lra = meter.loudness_range(y_stereo)

    def true_peak_dbtp(ch):
        up = resample_poly(ch, 4, 1)
        return 20 * np.log10(np.max(np.abs(up)) + 1e-10)

    if y_stereo.ndim == 2:
        tp = max(true_peak_dbtp(y_stereo[:, 0]), true_peak_dbtp(y_stereo[:, 1]))
    else:
        tp = true_peak_dbtp(y_stereo)

    peak = np.max(np.abs(y_mono))
    rms = np.sqrt(np.mean(y_mono ** 2))
    crest_db = 20 * np.log10(peak / (rms + 1e-10))

    win_samples = int(3.0 * sr)
    hop_samples = int(0.1 * sr)
    stl_values = []
    for start in range(0, len(y_stereo) - win_samples, hop_samples):
        chunk = y_stereo[start:start + win_samples]
        try:
            stl_values.append(meter.integrated_loudness(chunk))
        except Exception as stl_err:
            print(f"    ⚠ short-term loudness window failed at sample {start}: {stl_err}")
    stl_max = max(stl_values) if stl_values else float("-inf")
    psr = tp - stl_max

    if y_stereo.ndim == 2 and y_stereo.shape[1] >= 2:
        L, R = y_stereo[:, 0], y_stereo[:, 1]
        M = (L + R) / 2
        S = (L - R) / 2
        correlation = float(np.corrcoef(L, R)[0, 1])
        if not np.isfinite(correlation):
            correlation = 0.0
        energy_M = np.mean(M ** 2)
        energy_S = np.mean(S ** 2)
        stereo_width = float(energy_S / (energy_M + energy_S + 1e-10))
        ms_ratio_db = float(10 * np.log10((energy_M + 1e-10) / (energy_S + 1e-10)))
        energy_L = np.mean(L ** 2)
        energy_R = np.mean(R ** 2)
        lr_balance_db = float(10 * np.log10((energy_L + 1e-10) / (energy_R + 1e-10)))
    else:
        correlation = 1.0
        stereo_width = 0.0
        ms_ratio_db = float("inf")
        lr_balance_db = 0.0

    freqs, psd = welch(y_mono, fs=sr, nperseg=4096)

    def band_energy(f_low, f_high):
        mask = (freqs >= f_low) & (freqs < f_high)
        return float(np.sum(psd[mask]))

    bass = band_energy(20, 250)
    low_mid = band_energy(250, 2000)
    high_mid = band_energy(2000, 6000)
    treble = band_energy(6000, 20000)
    total = bass + low_mid + high_mid + treble + 1e-30

    valid = (freqs > 20) & (freqs < 20000) & (psd > 0)
    if np.any(valid):
        tilt = float(np.polyfit(np.log10(freqs[valid]), 10 * np.log10(psd[valid]), 1)[0])
    else:
        tilt = 0.0

    return {
        "integrated_lufs": round(float(integrated_lufs), 2),
        "loudness_range_lu": round(float(lra), 2),
        "true_peak_dbtp": round(float(tp), 2),
        "crest_factor_db": round(float(crest_db), 2),
        "psr_lu": round(float(psr), 2),
        "stereo": {
            "correlation": round(correlation, 4),
            "width": round(stereo_width, 4),
            "ms_ratio_db": round(ms_ratio_db, 2),
            "lr_balance_db": round(lr_balance_db, 2),
        },
        "spectral_balance": {
            "bass_ratio": round(bass / total, 4),
            "low_mid_ratio": round(low_mid / total, 4),
            "high_mid_ratio": round(high_mid / total, 4),
            "treble_ratio": round(treble / total, 4),
            "spectral_tilt_db_per_decade": round(tilt, 2),
        },
    }


def run_energy_curve(audio_path: str, sections_data: Dict[str, Any]) -> Dict[str, Any]:
    import librosa
    import numpy as np

    def summarize_section_energy(section_energy: List[Dict[str, Any]]) -> Dict[str, Any]:
        by_label: Dict[str, List[float]] = {}
        for row in section_energy:
            if not isinstance(row, dict):
                continue
            label = str(row.get("label", "section"))
            val = safe_float(row.get("rms_db"), 0.0)
            by_label.setdefault(label, []).append(val)
        label_summary: Dict[str, Dict[str, float]] = {}
        for label, vals in by_label.items():
            arr = np.asarray(vals, dtype=np.float64)
            label_summary[label] = {
                "mean_rms_db": round(float(arr.mean()), 3),
                "min_rms_db": round(float(arr.min()), 3),
                "max_rms_db": round(float(arr.max()), 3),
            }
        return {
            "section_count": len(section_energy),
            "labels": label_summary,
        }

    y, sr = librosa.load(audio_path, sr=22050, mono=True)

    hop_length = 512
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    times = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=hop_length)

    S = np.abs(librosa.stft(y, hop_length=hop_length))
    flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))
    flux = np.concatenate([[0], flux])

    sections = sections_data.get("segments", []) if isinstance(sections_data, dict) else []
    section_energy = []
    for sec in sections:
        if not isinstance(sec, dict):
            continue
        start = safe_float(sec.get("start"), 0.0)
        end = safe_float(sec.get("end"), 0.0)
        label = str(sec.get("label", "unknown"))
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        chunk = y[start_sample:end_sample]
        if len(chunk) > 0:
            chunk_rms = float(np.sqrt(np.mean(chunk ** 2)))
            section_energy.append(
                {
                    "label": label,
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "rms_mean": round(chunk_rms, 6),
                    "rms_db": round(float(20 * np.log10(chunk_rms + 1e-10)), 2),
                }
            )

    return {
        "rms_curve": {"times": times.tolist(), "values": rms.tolist()},
        "spectral_flux": {"times": times[: len(flux)].tolist(), "values": flux.tolist()},
        "rms_summary": stat_summary(rms) if len(rms) else {},
        "spectral_flux_summary": stat_summary(flux) if len(flux) else {},
        "section_energy": section_energy,
        "section_energy_summary": summarize_section_energy(section_energy),
    }


def run_per_stem_analysis(stems_dir: str, models_dir: str) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    stem_files = ["vocals.wav", "drums.wav", "bass.wav", "other.wav"]

    for stem_file in stem_files:
        stem_path = os.path.join(stems_dir, stem_file)
        if not os.path.isfile(stem_path):
            continue
        stem_name = stem_file.replace(".wav", "")
        try:
            stem_ml = run_essentia_ml(stem_path, models_dir)
            results[stem_name] = {
                "genre_top5": stem_ml.get("genre_discogs400", [])[:5] if isinstance(stem_ml.get("genre_discogs400"), list) else [],
                "mood_aggressive": stem_ml.get("mood_aggressive"),
                "mood_electronic": stem_ml.get("mood_electronic"),
                "timbre": stem_ml.get("timbre"),
                "voice_instrumental": stem_ml.get("voice_instrumental"),
            }
        except Exception as e:
            print(f"    ⚠ per-stem classifier failed for {stem_name}: {e}")
            results[stem_name] = {"error": str(e)}

    return results


def run_vggish_models(audio_path: str, models_dir: str) -> Dict[str, Any]:
    es = importlib.import_module("essentia.standard")
    import numpy as np

    has_vggish = hasattr(es, "TensorflowPredictVGGish")
    if not has_vggish:
        return {"error": "TensorflowPredictVGGish not available"}

    audio = es.MonoLoader(filename=audio_path, sampleRate=16000, resampleQuality=4)()

    vggish_model_path = os.path.join(models_dir, "feature-extractors", "audioset-vggish-3.pb")
    if not os.path.exists(vggish_model_path):
        return {"error": f"Missing VGGish model: {vggish_model_path}"}

    TensorflowPredictVGGish = getattr(es, "TensorflowPredictVGGish")
    TensorflowPredict2D = getattr(es, "TensorflowPredict2D")

    vggish = TensorflowPredictVGGish(
        graphFilename=vggish_model_path,
        output="model/vggish/embeddings",
    )
    embeddings = vggish(audio)

    VGGISH_CLASSIFIERS = {
        "deam_valence_arousal": "classification-heads/deam/deam-audioset-vggish-2",
        "moods_mirex": "classification-heads/moods_mirex/moods_mirex-audioset-vggish-1",
        "muse_valence_arousal": "classification-heads/muse/muse-audioset-vggish-2",
        "emomusic_valence_arousal": "classification-heads/emomusic/emomusic-audioset-vggish-2",
    }

    results: Dict[str, Any] = {}
    for task_name, model_subpath in VGGISH_CLASSIFIERS.items():
        model_path = os.path.join(models_dir, f"{model_subpath}.pb")
        json_path = os.path.join(models_dir, f"{model_subpath}.json")
        if not os.path.exists(model_path):
            continue
        try:
            input_node, output_node = get_node_names(json_path, "model/Placeholder", "model/Identity")
            classifier = TensorflowPredict2D(
                graphFilename=model_path,
                input=input_node,
                output=output_node,
            )
            preds = classifier(embeddings)
            mean_preds = np.asarray(preds).mean(axis=0)

            classes = load_classes(json_path)
            if classes:
                results[task_name] = {c: round(float(v), 4) for c, v in zip(classes, mean_preds)}
            else:
                results[task_name] = mean_preds.tolist()
        except Exception as e:
            print(f"    ⚠ VGGish classifier {task_name} failed: {e}")
            results[task_name] = {"error": str(e)}

    return results


def read_beats_if_present(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {"bpm": 0, "beats": [], "downbeats": []}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "bpm": safe_int(data.get("bpm", 0)),
        "beats": [safe_float(x) for x in data.get("beats", [])],
        "downbeats": [safe_float(x) for x in data.get("downbeats", [])],
    }


def read_lyrics_if_present(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        next(f, None)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                out.append({"start": safe_float(parts[0]), "end": safe_float(parts[1]), "word": parts[2]})
    return out


def pick_prob(bin_result: Any) -> Optional[float]:
    if isinstance(bin_result, dict) and "probability" in bin_result:
        try:
            return float(bin_result["probability"])
        except (TypeError, ValueError):
            return None
    return None


def derive_classification(ml: Dict[str, Any]) -> Dict[str, Any]:
    c: Dict[str, Any] = {
        "genre_top10": ml.get("genre_discogs400", []),
        "mood": {
            "aggressive": pick_prob(ml.get("mood_aggressive")),
            "happy": pick_prob(ml.get("mood_happy")),
            "sad": pick_prob(ml.get("mood_sad")),
            "relaxed": pick_prob(ml.get("mood_relaxed")),
            "party": pick_prob(ml.get("mood_party")),
            "acoustic": pick_prob(ml.get("mood_acoustic")),
            "electronic": pick_prob(ml.get("mood_electronic")),
        },
        "voice_instrumental": ml.get("voice_instrumental"),
        "voice_gender": ml.get("gender"),
        "timbre": ml.get("timbre"),
        "acoustic_electronic": ml.get("nsynth_acoustic_electronic"),
        "danceability": ml.get("danceability"),
        "tonal_atonal": ml.get("tonal_atonal"),
        "approachability": ml.get("approachability"),
        "engagement": ml.get("engagement"),
        "jamendo_instruments_top10": ml.get("mtg_jamendo_instrument", []),
        "jamendo_moodtheme_top10": ml.get("mtg_jamendo_moodtheme", []),
        "jamendo_top50tags_top10": ml.get("mtg_jamendo_top50tags", []),
        "nsynth_reverb": ml.get("nsynth_reverb"),
        "nsynth_instrument": ml.get("nsynth_instrument"),
        "nsynth_bright_dark": ml.get("nsynth_bright_dark"),
        "backend": ml.get("backend"),
    }
    for key in ["approachability", "engagement"]:
        val = c.get(key)
        if isinstance(val, list) and val and isinstance(val[0], dict):
            c[key] = safe_float(val[0].get("probability", 0.0))
    return c


def extract_step_lists(step_results: Dict[str, Dict[str, Any]]) -> Tuple[List[str], List[str], List[str]]:
    completed: List[str] = []
    failed: List[str] = []
    skipped: List[str] = []
    for name, result in step_results.items():
        st = result.get("status")
        if st == "ok":
            completed.append(name)
        elif st == "error":
            failed.append(name)
        else:
            skipped.append(name)
    return completed, failed, skipped


def generate_analysis_json(track_name: str, audio_path: str, duration_seconds: float, output_dir: str, step_results: Dict[str, Dict[str, Any]], words: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    beats_data = step_results["beats"].get("data", {})
    dsp_data = step_results["essentia_dsp"].get("data", {})
    ml_data = step_results["essentia_ml"].get("data", {})
    librosa_data = step_results["librosa"].get("data", {})
    pitch_data = step_results["pitch"].get("data", {})
    sections_data = step_results["sections"].get("data", {})
    mastering_data = step_results["mastering"].get("data", {})
    energy_data = step_results["energy"].get("data", {})
    per_stem_data = step_results["per_stem"].get("data", {})
    vggish_data = step_results["vggish_ml"].get("data", {})

    completed, failed, skipped = extract_step_lists(step_results)
    classification = derive_classification(ml_data if isinstance(ml_data, dict) else {})

    energy_payload = {
        "section_energy": energy_data.get("section_energy", []) if isinstance(energy_data, dict) else [],
        "section_energy_summary": energy_data.get("section_energy_summary", {}) if isinstance(energy_data, dict) else {},
        "rms_summary": energy_data.get("rms_summary", {}) if isinstance(energy_data, dict) else {},
        "spectral_flux_summary": energy_data.get("spectral_flux_summary", {}) if isinstance(energy_data, dict) else {},
    }
    if include_frames_in_json and isinstance(energy_data, dict):
        energy_payload["rms_curve"] = energy_data.get("rms_curve", {})
        energy_payload["spectral_flux"] = energy_data.get("spectral_flux", {})

    analysis = {
        "metadata": {
            "track_name": track_name,
            "source_file": os.path.basename(audio_path),
            "duration_seconds": round(float(duration_seconds), 3),
            "analyzed_at": datetime.now().isoformat(timespec="seconds"),
            "analyzer_version": ANALYZER_VERSION,
            "steps_completed": completed,
            "steps_failed": failed,
            "steps_skipped": skipped,
        },
        "rhythm": {
            "bpm_beat_this": beats_data.get("bpm"),
            "bpm_essentia": dsp_data.get("bpm") if isinstance(dsp_data, dict) else None,
            "bpm_librosa": librosa_data.get("tempo_librosa_bpm") if isinstance(librosa_data, dict) else None,
            "beats": beats_data.get("beats", []) if isinstance(beats_data, dict) else [],
            "downbeats": beats_data.get("downbeats", []) if isinstance(beats_data, dict) else [],
            "beat_count": len(beats_data.get("beats", [])) if isinstance(beats_data, dict) else 0,
            "onset_rate": dsp_data.get("onset_rate") if isinstance(dsp_data, dict) else None,
            "onset_positions": dsp_data.get("onsets", []) if isinstance(dsp_data, dict) else [],
            "danceability": classification.get("danceability"),
        },
        "tonality": {
            "key": dsp_data.get("key") if isinstance(dsp_data, dict) else None,
            "scale": dsp_data.get("scale") if isinstance(dsp_data, dict) else None,
            "key_strength": dsp_data.get("key_strength") if isinstance(dsp_data, dict) else None,
            "chords": dsp_data.get("chords", []) if isinstance(dsp_data, dict) else [],
            "tonal_atonal": classification.get("tonal_atonal"),
        },
        "loudness": {
            "integrated_lufs": dsp_data.get("loudness", {}).get("integrated_lufs") if isinstance(dsp_data, dict) else None,
            "loudness_range_lu": dsp_data.get("loudness", {}).get("lra") if isinstance(dsp_data, dict) else None,
            "rms_db": dsp_data.get("loudness", {}).get("rms") if isinstance(dsp_data, dict) else None,
            "dynamic_complexity": dsp_data.get("loudness", {}).get("dynamic_complexity") if isinstance(dsp_data, dict) else None,
            "replay_gain_db": dsp_data.get("loudness", {}).get("replay_gain") if isinstance(dsp_data, dict) else None,
        },
        "spectral": {
            "centroid": librosa_data.get("spectral_centroid") if isinstance(librosa_data, dict) else None,
            "bandwidth": librosa_data.get("spectral_bandwidth") if isinstance(librosa_data, dict) else None,
            "rolloff": librosa_data.get("spectral_rolloff") if isinstance(librosa_data, dict) else None,
            "contrast": librosa_data.get("spectral_contrast") if isinstance(librosa_data, dict) else None,
            "zcr": librosa_data.get("zcr") if isinstance(librosa_data, dict) else None,
            "rms": librosa_data.get("rms") if isinstance(librosa_data, dict) else None,
            "mfcc": librosa_data.get("mfcc") if isinstance(librosa_data, dict) else None,
            "chroma_mean": librosa_data.get("chroma_mean") if isinstance(librosa_data, dict) else None,
        },
        "classification": classification,
        "sections": sections_data if isinstance(sections_data, dict) else {},
        "mastering": mastering_data if isinstance(mastering_data, dict) else {},
        "energy": energy_payload,
        "per_stem_classification": per_stem_data if isinstance(per_stem_data, dict) else {},
        "vggish_classification": vggish_data if isinstance(vggish_data, dict) else {},
        "vocals": {
            "word_count": len(words),
            "language": step_results["whisper"].get("transcription", {}).get("language"),
            "pitch_hz": pitch_data.get("pitch_hz") if isinstance(pitch_data, dict) else None,
            "pitch_midi_min": pitch_data.get("pitch_midi_min") if isinstance(pitch_data, dict) else None,
            "pitch_midi_max": pitch_data.get("pitch_midi_max") if isinstance(pitch_data, dict) else None,
            "pitch_midi_mean": pitch_data.get("pitch_midi_mean") if isinstance(pitch_data, dict) else None,
            "vocal_range_semitones": pitch_data.get("vocal_range_semitones") if isinstance(pitch_data, dict) else None,
            "confidence_mean": pitch_data.get("confidence_mean") if isinstance(pitch_data, dict) else None,
            "voiced_ratio": pitch_data.get("voiced_ratio") if isinstance(pitch_data, dict) else None,
            "melody": pitch_data.get("melody") if isinstance(pitch_data, dict) else None,
        },
        "files": {
            "stems_dir": "stems/",
            "vocals": "vocals.wav",
            "beats": "beats.json",
            "lyrics": "lyrics.tsv",
            "song_map": "song-map.md"
        }
    }

    path = os.path.join(output_dir, "analysis.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(analysis), f, ensure_ascii=False, indent=2)
    return path, to_jsonable(analysis)


def percent(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.1f}%"


def scale_short(scale: Any) -> str:
    s = str(scale or "").lower()
    if s.startswith("maj"):
        return "maj"
    if s.startswith("min"):
        return "min"
    return s


def top_labels(lst: Any, k: int = 5) -> str:
    if not isinstance(lst, list) or not lst:
        return "n/a"
    out = []
    for item in lst[:k]:
        if isinstance(item, dict):
            label = str(item.get("label", "?"))
            prob = item.get("probability")
            if prob is not None:
                out.append(f"{label} ({percent(safe_float(prob))})")
            else:
                out.append(label)
        else:
            out.append(str(item))
    return ", ".join(out)


def mood_line(mood: Any) -> str:
    if not isinstance(mood, dict):
        return "n/a"
    pairs: List[Tuple[str, float]] = []
    for k, v in mood.items():
        try:
            pairs.append((str(k), float(v)))
        except (TypeError, ValueError):
            continue
    pairs.sort(key=lambda x: x[1], reverse=True)
    return ", ".join([f"{k} {percent(v)}" for k, v in pairs]) if pairs else "n/a"


def chord_rows(chords: Any) -> List[str]:
    if not isinstance(chords, list) or not chords:
        return ["| - | - | - |"]
    rows = []
    for c in chords[:500]:
        if not isinstance(c, dict):
            continue
        t = format_mmss(safe_float(c.get("time"), 0.0))
        label = str(c.get("label", "?"))
        conf = safe_float(c.get("strength"), 0.0)
        rows.append(f"| {t} | {label} | {conf:.2f} |")
    return rows or ["| - | - | - |"]


def beat_rows(beats: Any, downbeats: Any) -> List[str]:
    if not isinstance(beats, list) or not beats:
        return ["| - | - | - | - |"]
    ds = [safe_float(x) for x in downbeats] if isinstance(downbeats, list) else []
    rows = []
    bar = 0
    beat_in_bar = 0
    for i, t in enumerate(beats):
        tt = round(safe_float(t), 3)
        is_downbeat = any(abs(tt - d) < 0.05 for d in ds)
        if is_downbeat:
            bar += 1
            beat_in_bar = 1
        else:
            beat_in_bar += 1
        rows.append(f"| {i+1} | {tt:.3f} | {bar} | {'yes' if is_downbeat else ''} |")
    return rows


def section_rows(sections: Any) -> List[str]:
    if not isinstance(sections, list) or not sections:
        return ["| - | - | - | - |"]
    rows: List[str] = []
    for sec in sections:
        if not isinstance(sec, dict):
            continue
        start = safe_float(sec.get("start"), 0.0)
        end = safe_float(sec.get("end"), 0.0)
        duration = max(0.0, end - start)
        label = str(sec.get("label", "section"))
        rows.append(f"| {label} | {format_mmss(start)} | {format_mmss(end)} | {duration:.1f}s |")
    return rows or ["| - | - | - | - |"]


def section_summary_lines(summary: Any) -> List[str]:
    if not isinstance(summary, dict):
        return ["- n/a"]
    count = safe_int(summary.get("count"), 0)
    total_duration = safe_float(summary.get("total_duration"), 0.0)
    by_label = summary.get("by_label", {})
    lines = [f"- Sections detected: {count}", f"- Covered duration: {total_duration:.1f}s"]
    if isinstance(by_label, dict) and by_label:
        top = sorted(
            [
                (
                    str(label),
                    safe_int(stats.get("count"), 0) if isinstance(stats, dict) else 0,
                    safe_float(stats.get("total_duration"), 0.0) if isinstance(stats, dict) else 0.0,
                )
                for label, stats in by_label.items()
            ],
            key=lambda x: x[2],
            reverse=True,
        )
        for label, label_count, label_duration in top[:5]:
            lines.append(f"- {label}: {label_count}x / {label_duration:.1f}s")
    return lines


def generate_song_map(track_name: str, analysis: Dict[str, Any], output_dir: str) -> str:
    metadata = analysis.get("metadata", {})
    rhythm = analysis.get("rhythm", {})
    tonality = analysis.get("tonality", {})
    loudness = analysis.get("loudness", {})
    sections = analysis.get("sections", {})
    mastering = analysis.get("mastering", {})
    classification = analysis.get("classification", {})
    vocals = analysis.get("vocals", {})

    danceability = classification.get("danceability") if isinstance(classification, dict) else None
    voice_gender = classification.get("voice_gender") if isinstance(classification, dict) else None
    timbre = classification.get("timbre") if isinstance(classification, dict) else None
    mood = classification.get("mood") if isinstance(classification, dict) else None

    arousal = mood.get("aggressive") if isinstance(mood, dict) else None
    valence = mood.get("happy") if isinstance(mood, dict) else None

    lines = []
    lines.append(f"# {track_name} -- Song Map")
    lines.append("")
    lines.append(
        f"BPM: {rhythm.get('bpm_beat_this', 'n/a')} | Key: {tonality.get('key', 'n/a')}{scale_short(tonality.get('scale'))} | Duration: {format_time(safe_float(metadata.get('duration_seconds', 0.0)))} | Generated: {metadata.get('analyzed_at', datetime.now().isoformat(timespec='seconds'))}"
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Property | Value |")
    lines.append("|----------|-------|")
    lines.append(f"| Key / Scale | {tonality.get('key', 'n/a')} {tonality.get('scale', 'n/a')} (confidence: {percent(safe_float(tonality.get('key_strength')) if tonality.get('key_strength') is not None else None)}) |")
    lines.append(f"| BPM | {rhythm.get('bpm_beat_this')} (beat_this) / {rhythm.get('bpm_essentia')} (essentia) / {rhythm.get('bpm_librosa')} (librosa) |")
    lines.append(f"| Loudness | {loudness.get('integrated_lufs', 'n/a')} LUFS integrated |")
    lines.append(f"| True Peak | {mastering.get('true_peak_dbtp', 'n/a')} dBTP |")
    lines.append(f"| Integrated LUFS | {mastering.get('integrated_lufs', 'n/a')} LUFS |")
    lines.append(f"| PSR | {mastering.get('psr_lu', 'n/a')} LU |")
    lines.append(f"| Stereo Width | {percent(mastering.get('stereo', {}).get('width') if isinstance(mastering, dict) else None)} |")
    lines.append(f"| Danceability | {percent(safe_float(danceability.get('probability')) if isinstance(danceability, dict) else None)} |")
    lines.append(f"| Voice | {(voice_gender.get('label') if isinstance(voice_gender, dict) else 'n/a')}, {(timbre.get('label') if isinstance(timbre, dict) else 'n/a')} timbre |")
    lines.append(f"| Arousal / Valence | {percent(safe_float(arousal) if arousal is not None else None)} / {percent(safe_float(valence) if valence is not None else None)} |")
    lines.append("")
    lines.append(f"**Genre** (top 5): {top_labels(classification.get('genre_top10') if isinstance(classification, dict) else [], 5)}")
    lines.append(f"**Mood**: {mood_line(classification.get('mood') if isinstance(classification, dict) else {})}")
    lines.append(f"**Instruments**: {top_labels(classification.get('jamendo_instruments_top10') if isinstance(classification, dict) else [], 10)}")
    lines.append("")
    lines.append("## Sections")
    lines.append("")
    lines.append("| Section | Start | End | Duration |")
    lines.append("|---------|-------|-----|----------|")
    lines.extend(section_rows(sections.get("segments", []) if isinstance(sections, dict) else []))
    lines.append("")
    lines.extend(section_summary_lines(sections.get("summary", {}) if isinstance(sections, dict) else {}))
    lines.append("")
    lines.append("## Chord Progression")
    lines.append("")
    lines.append("| Time | Chord | Confidence |")
    lines.append("|------|-------|------------|")
    lines.extend(chord_rows(tonality.get("chords", [])))
    lines.append("")
    lines.append("## Beat Grid")
    lines.append("")
    lines.append("| Beat | Time | Bar | Downbeat |")
    lines.append("|------|------|-----|----------|")
    lines.extend(beat_rows(rhythm.get("beats", []), rhythm.get("downbeats", [])))
    lines.append("")
    lines.append("## Timestamped Lyrics")
    lines.append(f"See `lyrics.tsv` ({vocals.get('word_count', 0)} words)")
    lines.append("")
    lines.append("## Full Analysis")
    lines.append("See `analysis.json` for complete numerical data.")
    lines.append("")

    path = os.path.join(output_dir, "song-map.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def save_frames_if_requested(include_frames: bool, output_dir: str, librosa_step: Dict[str, Any], pitch_step: Dict[str, Any]) -> Optional[str]:
    if not include_frames:
        return None
    import numpy as np

    librosa_data = librosa_step.get("data", {}) if librosa_step.get("status") == "ok" else {}
    pitch_data = pitch_step.get("data", {}) if pitch_step.get("status") == "ok" else {}

    librosa_frames = librosa_data.get("_frames", {}) if isinstance(librosa_data, dict) else {}
    pitch_frames = pitch_data.get("_frames", {}) if isinstance(pitch_data, dict) else {}

    mel_db = librosa_frames.get("mel_spectrogram", np.array([], dtype=np.float32))
    chroma = librosa_frames.get("chromagram", np.array([], dtype=np.float32))
    mfcc = librosa_frames.get("mfcc", np.array([], dtype=np.float32))
    centroid = librosa_frames.get("spectral_centroid", np.array([], dtype=np.float32))
    rms = librosa_frames.get("rms", np.array([], dtype=np.float32))
    pitch_data_arr = pitch_frames.get("pitch", np.array([], dtype=np.float32))
    confidence_data = pitch_frames.get("pitch_confidence", np.array([], dtype=np.float32))

    out = os.path.join(output_dir, "analysis-frames.npz")
    np.savez_compressed(
        out,
        mel_spectrogram=mel_db,
        chromagram=chroma,
        mfcc=mfcc,
        spectral_centroid=centroid,
        rms=rms,
        pitch=pitch_data_arr,
        pitch_confidence=confidence_data,
    )
    return out


def ensure_step_keys(step_results: Dict[str, Dict[str, Any]]) -> None:
    for k in ["demucs", "beats", "whisper", "essentia_dsp", "essentia_ml", "librosa", "pitch", "sections", "mastering", "energy", "per_stem", "vggish_ml"]:
        if k not in step_results:
            step_results[k] = make_skipped("Not executed")


def print_step_overview(step_results: Dict[str, Dict[str, Any]]) -> None:
    print("\n=== Step Status Overview ===")
    for k, v in step_results.items():
        st = v.get("status", "unknown")
        rs = v.get("reason")
        if rs:
            print(f"- {k}: {st} ({rs})")
        else:
            print(f"- {k}: {st}")


def main():
    global include_frames_in_json
    args = parse_args()
    include_frames_in_json = args.include_frames
    try:
        paths = resolve_paths(args.audio, args.output_dir, args.models_dir)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    audio_path = paths["audio_path"]
    output_dir = paths["output_dir"]
    models_dir = paths["models_dir"]

    track_name = infer_track_name(audio_path)
    duration = get_duration(audio_path)

    print(f"Analyzing: {track_name}")
    print(f"Input: {audio_path}")
    print(f"Output: {output_dir}")
    print(f"Duration: {format_time(duration)}")

    step_results: Dict[str, Dict[str, Any]] = {}
    words: List[Dict[str, Any]] = []

    vocals_path = os.path.join(output_dir, "vocals.wav")
    if args.skip_demucs:
        step_results["demucs"] = make_skipped("Skipped by --skip-demucs")
        if not os.path.isfile(vocals_path):
            vocals_path = audio_path
    else:
        step_results["demucs"] = run_step_safe("Step 1: run_demucs", run_demucs, audio_path, output_dir, args.vocals_only)
        if step_results["demucs"].get("status") == "ok":
            vocals_path = str(step_results["demucs"]["data"])
        elif not os.path.isfile(vocals_path):
            vocals_path = audio_path

    beats_path = os.path.join(output_dir, "beats.json")
    if args.skip_beats:
        step_results["beats"] = make_skipped("Skipped by --skip-beats")
        step_results["beats"]["data"] = read_beats_if_present(beats_path)
    else:
        step_results["beats"] = run_step_safe("Step 2: run_beat_this", run_beat_this, audio_path, output_dir)
        if step_results["beats"].get("status") != "ok":
            step_results["beats"]["data"] = read_beats_if_present(beats_path)

    lyrics_path = os.path.join(output_dir, "lyrics.tsv")
    if args.skip_whisper:
        step_results["whisper"] = make_skipped("Skipped by --skip-whisper")
        words = read_lyrics_if_present(lyrics_path)
        step_results["whisper"]["data"] = words
    else:
        transcription_metadata: Dict[str, Any] = {}
        step_results["whisper"] = run_step_safe(
            "Step 3: run_whisper", run_whisper, vocals_path, output_dir,
            args.whisper_model, args.language, transcription_metadata,
        )
        step_results["whisper"]["transcription"] = transcription_metadata
        if step_results["whisper"].get("status") == "ok":
            words = step_results["whisper"]["data"]
        else:
            words = read_lyrics_if_present(lyrics_path)
            step_results["whisper"]["data"] = words

    if args.skip_essentia:
        step_results["essentia_dsp"] = make_skipped("Skipped by --skip-essentia")
    else:
        step_results["essentia_dsp"] = run_step_safe("Step 4: run_essentia_dsp", run_essentia_dsp, audio_path)

    if args.skip_ml:
        step_results["essentia_ml"] = make_skipped("Skipped by --skip-ml")
    else:
        step_results["essentia_ml"] = run_step_safe("Step 5: run_essentia_ml", run_essentia_ml, audio_path, models_dir)

    if args.skip_librosa:
        step_results["librosa"] = make_skipped("Skipped by --skip-librosa")
    else:
        step_results["librosa"] = run_step_safe("Step 6: run_librosa_analysis", run_librosa_analysis, audio_path)

    if args.skip_pitch:
        step_results["pitch"] = make_skipped("Skipped by --skip-pitch")
    else:
        pitch_source = "mix" if os.path.abspath(vocals_path) == audio_path else "vocals"
        step_results["pitch"] = run_step_safe("Step 7: run_pitch_analysis", run_pitch_analysis, vocals_path, pitch_source)

    if args.skip_sections:
        step_results["sections"] = make_skipped("Skipped by --skip-sections")
    else:
        step_results["sections"] = run_step_safe("Step 8: run_sections", run_sections, audio_path)

    if args.skip_mastering:
        step_results["mastering"] = make_skipped("Skipped by --skip-mastering")
    else:
        step_results["mastering"] = run_step_safe("Step 9: run_mastering", run_mastering, audio_path)

    sections_data = step_results["sections"].get("data", {}) if step_results["sections"].get("status") == "ok" else {}

    if args.skip_energy:
        step_results["energy"] = make_skipped("Skipped by --skip-energy")
    else:
        step_results["energy"] = run_step_safe("Step 10: run_energy_curve", run_energy_curve, audio_path, sections_data)

    stems_dir = os.path.join(output_dir, "stems")
    if args.skip_per_stem or not os.path.isdir(stems_dir):
        step_results["per_stem"] = make_skipped("Skipped or no stems")
    else:
        step_results["per_stem"] = run_step_safe("Step 11: run_per_stem_analysis", run_per_stem_analysis, stems_dir, models_dir)

    if not args.skip_ml:
        step_results["vggish_ml"] = run_step_safe("Step 12: run_vggish_models", run_vggish_models, audio_path, models_dir)
    else:
        step_results["vggish_ml"] = make_skipped("Skipped by --skip-ml")

    ensure_step_keys(step_results)

    analysis_path, analysis = generate_analysis_json(track_name, audio_path, duration, output_dir, step_results, words)
    song_map_path = generate_song_map(track_name, analysis, output_dir)

    frames_result = run_step_safe(
        "Frame export",
        save_frames_if_requested,
        args.include_frames,
        output_dir,
        step_results.get("librosa", make_skipped("missing")),
        step_results.get("pitch", make_skipped("missing")),
    )
    frames_path = frames_result.get("data") if frames_result.get("status") == "ok" else None

    print_step_overview(step_results)
    print("\n=== Output Files ===")
    print(f"- analysis.json: {analysis_path}")
    print(f"- song-map.md:  {song_map_path}")
    if frames_path:
        print(f"- analysis-frames.npz: {frames_path}")
    print("- stems/:       stems/")
    print("- vocals.wav:   vocals.wav")
    print("- beats.json:   beats.json")
    print("- lyrics.tsv:   lyrics.tsv")


if __name__ == "__main__":
    main()
