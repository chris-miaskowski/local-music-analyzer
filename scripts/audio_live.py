#!/usr/bin/env python3
"""Local Core Audio tap, bounded in-memory capture and music analysis API."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import math
import os
import queue
import re
import signal
import subprocess
import threading
import time
import uuid
import wave
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / ".tools/audio-live-data"
MAX_CAPTURE_SECONDS = 600
MAX_CAPTURE_BYTES = 128 * 1024 * 1024
LIVE_HISTORY_SECONDS = 30
ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
ARTIFACTS = {
    "analysis": "analysis.json", "song_map": "song-map.md",
    "beats": "beats.json", "log": "analysis.log",
    **{name: f"stems/{name}.wav" for name in ("vocals", "drums", "bass", "other")},
}


def dbfs(value: float) -> float:
    return round(max(-120.0, 20 * math.log10(max(float(value), 1e-6))), 2)


def decode_pcm(buffer) -> np.ndarray:
    """catap delivers immutable, packed, interleaved little-endian PCM."""
    fmt = buffer.format
    bits, channels = fmt.bits_per_sample, fmt.num_channels
    if not (1 <= channels <= 8 and 8000 <= fmt.sample_rate <= 384000):
        raise ValueError("Unsupported channel count or sample rate")
    expected = buffer.frame_count * channels * (bits // 8)
    if len(buffer.data) != expected:
        raise ValueError("PCM buffer byte count does not match its format")
    if fmt.sample_type == "float" and bits == 32:
        samples = np.frombuffer(buffer.data, dtype="<f4")
    elif fmt.sample_type == "signed_integer" and bits in (16, 32):
        samples = np.frombuffer(buffer.data, dtype=f"<i{bits // 8}").astype(np.float32)
        samples /= 2 ** (bits - 1)
    elif fmt.sample_type == "signed_integer" and bits == 24:
        packed = np.frombuffer(buffer.data, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        values = packed[:, 0] | (packed[:, 1] << 8) | (packed[:, 2] << 16)
        values = (values ^ 0x800000) - 0x800000
        samples = values.astype(np.float32) / 8388608
    else:
        raise ValueError(f"Unsupported PCM: {fmt.sample_type} {bits}")
    if not np.isfinite(samples).all():
        raise ValueError("Audio buffer contains non-finite samples")
    return samples.reshape(-1, channels)


class LiveDSP:
    """Measured signal features. Onsets and tempo are estimates, not drum labels."""

    def __init__(self, sample_rate: float):
        self.sample_rate = sample_rate
        self.size = 4096
        self.hop = 1024
        self.window = np.hanning(self.size)
        self.frequency = np.fft.rfftfreq(self.size, 1 / sample_rate)
        self.pending = np.empty(0, dtype=np.float32)
        self.processed = 0
        self.previous = np.zeros(self.size // 2 + 1)
        self.fluxes = deque(maxlen=max(2, int(30 * sample_rate / self.hop)))
        self.onsets = deque(maxlen=6000)
        self.history = deque(maxlen=6000)
        self.last_history = -1.0
        self.last_onset = -1.0
        self.last_tempo = -1.0
        self.bpm = None
        self.confidence = 0.0
        self.latest = self.empty_metrics()

    @staticmethod
    def empty_metrics():
        return {"time": 0.0, "rms_dbfs": -120.0, "peak_dbfs": -120.0,
                "bpm": None, "bpm_confidence": 0.0, "bpm_warmup_seconds": 12,
                "waveform": [], "spectrum": [], "bands": {"bass": 0, "mid": 0, "treble": 0},
                "spectral_flux": 0.0}

    def feed(self, samples: np.ndarray):
        # Energy averaging across channels avoids cancelling antiphase stereo.
        mono = samples.mean(axis=1)
        if np.mean(mono * mono) < np.mean(samples * samples) * 0.01:
            mono = samples[:, 0]
        self.pending = np.concatenate((self.pending, mono))
        while len(self.pending) >= self.size:
            frame = self.pending[:self.size]
            self.pending = self.pending[self.hop:]
            self.processed += self.hop
            stamp = self.processed / self.sample_rate
            magnitude = np.abs(np.fft.rfft(frame * self.window)) * 2 / self.window.sum()
            flux = float(np.maximum(magnitude - self.previous, 0).sum())
            self.previous = magnitude
            recent = np.asarray(list(self.fluxes)[-int(self.sample_rate / self.hop):])
            threshold = max(0.02, float(np.median(recent) + 2 * np.std(recent))) if len(recent) > 4 else 0.02
            if flux > threshold and stamp - self.last_onset > 0.12:
                self.onsets.append(round(stamp, 3))
                self.last_onset = stamp
            self.fluxes.append(flux)
            if stamp - self.last_tempo >= 1:
                self._tempo(stamp)
                self.last_tempo = stamp
            power = magnitude * magnitude
            total = float(power.sum())
            bands = {name: round(float(power[(self.frequency >= lo) & (self.frequency < hi)].sum()) / max(total, 1e-20), 4)
                     for name, lo, hi in (("bass", 20, 250), ("mid", 250, 4000), ("treble", 4000, self.sample_rate / 2 + 1))}
            rms = dbfs(float(np.sqrt(np.mean(frame * frame))))
            peak = dbfs(float(np.abs(frame).max()))
            if stamp - self.last_history >= 0.1:
                self.history.append({"time": round(stamp, 3), "rms_dbfs": rms, "peak_dbfs": peak, **bands, "flux": round(flux, 5)})
                self.last_history = stamp
            edges = np.unique(np.geomspace(1, len(magnitude), 121).astype(int))
            indices = [left + int(np.argmax(magnitude[left:right])) for left, right in zip(edges, edges[1:])]
            waveform = frame[np.linspace(0, self.size - 1, 256).astype(int)]
            self.latest = {"time": round(stamp, 3), "rms_dbfs": rms, "peak_dbfs": peak,
                           "bpm": self.bpm, "bpm_confidence": self.confidence,
                           "bpm_warmup_seconds": round(max(0, 12 - stamp), 1),
                           "waveform": np.round(waveform, 4).tolist(),
                           "spectrum": [{"hz": round(float(self.frequency[i]), 1), "dbfs": dbfs(magnitude[i])} for i in indices],
                           "bands": bands, "spectral_flux": round(flux, 5)}
        # A short NumPy slice can otherwise retain an entire large input allocation.
        self.pending = self.pending.copy()

    def _tempo(self, stamp):
        self.bpm, self.confidence = None, 0.0
        if stamp < 12 or len(self.onsets) < 8:
            return
        intervals = np.diff([x for x in self.onsets if x > stamp - 30])
        intervals = intervals[(intervals >= 0.18) & (intervals <= 1.5)]
        if len(intervals) < 7:
            return
        median = float(np.median(intervals))
        consistency = float(np.mean(np.abs(intervals - median) < median * 0.12))
        if consistency < 0.55:
            return
        tempo = 60 / median
        while tempo > 180:
            tempo /= 2
        while tempo < 60:
            tempo *= 2
        self.bpm = round(tempo, 1)
        # Heuristic stability score, deliberately capped; half/double tempo is ambiguous.
        self.confidence = round(min(0.85, consistency * min(1, len(intervals) / 24)), 2)


def native_capture(callback):
    from catap import record_system_audio
    return record_system_audio(on_buffer=callback, max_pending_buffers=128, mute=False)


def list_output_processes():
    from catap import list_audio_processes
    return [{"name": p.name, "pid": p.pid, "bundle_id": p.bundle_id}
            for p in list_audio_processes() if p.is_outputting]


class AudioService:
    def __init__(self, data_dir=DATA, capture_factory=native_capture, project=PROJECT):
        self.data_dir = Path(data_dir)
        self.project = Path(project)
        self.capture_factory = capture_factory
        self.lock = threading.RLock()
        self.lifecycle = threading.Lock()
        self.session = None
        self.worker = None
        self.state = "idle"
        self.error = None
        self.queue = queue.Queue(maxsize=256)
        self.stop_event = threading.Event()
        self.started_at = 0.0
        self.last_buffer_at = 0.0
        self.dropped = 0
        self.dsp = None
        self.ring = deque()
        self.ring_frames = 0
        self.total_frames = 0
        self.sample_rate = 0
        self.channels = 0
        self.saved_frames = 0
        self.captures = {}
        self.job = None
        self.job_process = None
        self.job_thread = None
        self.job_cancel = threading.Event()
        self.closing = False
        self._load_captures()

    def _load_captures(self):
        if not self.data_dir.exists():
            return
        for path in self.data_dir.glob("*/capture.json"):
            if not ID_PATTERN.fullmatch(path.parent.name) or path.is_symlink():
                continue
            try:
                item = json.loads(path.read_text())
                if item.get("id") == path.parent.name and (path.parent / "capture.wav").is_file():
                    self.captures[item["id"]] = item
            except (OSError, ValueError):
                continue

    def _enqueue(self, buffer):
        if self.stop_event.is_set():
            return
        try:
            self.queue.put_nowait(buffer)
        except queue.Full:
            self.dropped += 1

    def start(self):
        with self.lifecycle:
            with self.lock:
                if self.state in ("capturing", "starting"):
                    return self.snapshot()
                if self.session is not None:
                    raise RuntimeError("Tap cleanup is incomplete; retry Stop before starting")
                if self.total_frames > self.saved_frames:
                    raise ValueError("Save the stopped capture before starting a new one")
                self.state, self.error = "starting", None
                self.queue = queue.Queue(maxsize=256)
                self.stop_event.clear()
                self.ring.clear()
                self.ring_frames = self.total_frames = self.saved_frames = self.dropped = 0
                self.sample_rate = self.channels = 0
                self.dsp = None
                self.started_at = time.monotonic()
                self.last_buffer_at = 0
            try:
                self.session = self.capture_factory(self._enqueue)
                self.session.start()
                with self.lock:
                    self.state = "capturing"
                self.worker = threading.Thread(target=self._consume, name="audio-live-dsp", daemon=True)
                self.worker.start()
            except Exception as exc:
                self._release_session()
                with self.lock:
                    self.state, self.error = "error", str(exc)
                raise RuntimeError(str(exc)) from exc
        return self.snapshot()

    def _release_session(self):
        if self.session is None:
            return None
        error = None
        # catap explicitly supports a cleanup retry when stop raised.
        for _ in range(2):
            try:
                self.session.stop()
            except Exception as exc:
                error = str(exc)
            if not self.session.is_recording and not self.session.needs_cleanup:
                self.session = None
                break
        return error

    def stop(self):
        with self.lifecycle:
            self.stop_event.set()
            error = self._release_session()
            if self.worker and self.worker is not threading.current_thread():
                self.worker.join(timeout=5)
                if self.worker.is_alive():
                    error = error or "Audio processing did not stop cleanly"
            with self.lock:
                if error:
                    self.error = error
                self.state = "error" if self.error else "stopped"
        return self.snapshot()

    def _consume(self):
        try:
            while not self.stop_event.is_set() or not self.queue.empty():
                if self.dropped:
                    raise RuntimeError(f"Capture overflow: {self.dropped} audio buffers dropped")
                session = self.session
                if session is not None and session.capture_failed:
                    raise RuntimeError("Core Audio reported a capture failure")
                try:
                    buffer = self.queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                samples = decode_pcm(buffer)
                rate, channels = buffer.format.sample_rate, buffer.format.num_channels
                with self.lock:
                    if not self.sample_rate:
                        self.sample_rate, self.channels = rate, channels
                        self.dsp = LiveDSP(rate)
                    if self.sample_rate != rate or self.channels != channels:
                        raise RuntimeError("Output audio format changed; stop and start a new capture")
                    pcm = np.round(np.clip(samples, -1, 32767 / 32768) * 32768).astype("<i2")
                    self.ring.append(pcm)
                    self.ring_frames += len(pcm)
                    self.total_frames += len(pcm)
                    max_frames = min(int(rate * MAX_CAPTURE_SECONDS), MAX_CAPTURE_BYTES // (2 * channels))
                    while self.ring_frames > max_frames:
                        trim = min(len(self.ring[0]), self.ring_frames - max_frames)
                        first = self.ring.popleft()
                        if trim < len(first):
                            self.ring.appendleft(first[trim:].copy())
                        self.ring_frames -= trim
                    self.last_buffer_at = time.monotonic()
                    self.dsp.feed(samples)
        except Exception as exc:
            with self.lock:
                self.error, self.state = str(exc), "error"
            self.stop_event.set()
            # A different owner performs native cleanup, never a native callback.
            threading.Thread(target=self.stop, name="audio-live-cleanup", daemon=True).start()

    def snapshot(self, compact=False):
        with self.lock:
            metrics = copy.deepcopy(self.dsp.latest if self.dsp else LiveDSP.empty_metrics())
            signal_state = "waiting"
            if self.last_buffer_at:
                signal_state = "active" if metrics["peak_dbfs"] > -90 else "silent"
                if self.state == "capturing" and time.monotonic() - self.last_buffer_at > 3:
                    signal_state = "no_signal"
            elif self.started_at and time.monotonic() - self.started_at > 3:
                signal_state = "no_signal"
            history = list(self.dsp.history) if self.dsp else []
            onsets = list(self.dsp.onsets) if self.dsp else []
            if compact:
                cutoff = metrics["time"] - LIVE_HISTORY_SECONDS
                history = [point for point in history if point["time"] >= cutoff]
                onsets = [stamp for stamp in onsets if stamp >= cutoff]
            if compact and self.job:
                job = {key: self.job.get(key) for key in ("id", "capture_id", "status", "error")}
            else:
                job = copy.deepcopy(self.job)
            if job:
                job["details_available"] = self.job["status"] != "running" and not (self.job_thread and self.job_thread.is_alive())
            return {"state": self.state, "signal": signal_state, "error": self.error,
                    "source": "system_output", "elapsed_seconds": round(self.total_frames / (self.sample_rate or 1), 3),
                    "buffer_seconds": round(self.ring_frames / (self.sample_rate or 1), 3),
                    "sample_rate": self.sample_rate, "channels": self.channels,
                    "dropped_buffers": self.dropped, "unsaved": self.total_frames > self.saved_frames,
                    "metrics": metrics, "history": history, "onsets": onsets,
                    "captures": sorted(self.captures.values(), key=lambda x: x["created_at"], reverse=True),
                    "job": job}

    def live_signature(self):
        """Avoid rebuilding and sending an unchanged stopped capture indefinitely."""
        with self.lock:
            no_signal = self.state == "capturing" and time.monotonic() - (self.last_buffer_at or self.started_at) > 3
            job = self.job or {}
            return (self.state, self.total_frames, self.saved_frames, self.dropped, self.error,
                    no_signal, len(self.captures), job.get("id"), job.get("status"), job.get("error"),
                    bool(self.job_thread and self.job_thread.is_alive()))

    def save(self):
        with self.lock:
            if not self.ring_frames:
                raise ValueError("No audio has been captured")
            chunks = tuple(self.ring)
            frames, rate, channels, total = self.ring_frames, int(self.sample_rate), self.channels, self.total_frames
        capture_id = uuid.uuid4().hex
        folder = self.data_dir / capture_id
        folder.mkdir(parents=True, mode=0o700)
        os.chmod(folder, 0o700)
        path = folder / "capture.wav"
        with path.open("xb") as raw:
            os.chmod(path, 0o600)
            with wave.open(raw, "wb") as target:
                target.setnchannels(channels)
                target.setsampwidth(2)
                target.setframerate(rate)
                for chunk in chunks:
                    target.writeframesraw(chunk.tobytes())
        capture = {"id": capture_id, "created_at": datetime.now(timezone.utc).isoformat(),
                   "duration_seconds": round(frames / rate, 3), "sample_rate": rate, "channels": channels,
                   "source": "system_output", "offset_seconds": round((total - frames) / rate, 3),
                   "audio_url": f"/api/captures/{capture_id}/audio"}
        (folder / "capture.json").write_text(json.dumps(capture, indent=2))
        with self.lock:
            self.captures[capture_id] = capture
            self.saved_frames = total
        return {"capture": capture, **self.snapshot()}

    def discard(self):
        with self.lifecycle, self.lock:
            if self.session is not None or self.state in ("capturing", "starting"):
                raise ValueError("Stop capture before discarding the in-memory buffer")
            self.ring.clear()
            self.ring_frames = self.total_frames = self.saved_frames = 0
            self.sample_rate = self.channels = self.dropped = 0
            self.started_at = self.last_buffer_at = 0
            self.dsp = None
            self.error = None
            self.state = "idle"
        return self.snapshot()

    def capture_path(self, capture_id):
        if not ID_PATTERN.fullmatch(capture_id) or capture_id not in self.captures:
            raise KeyError("Unknown capture")
        path = self.data_dir / capture_id / "capture.wav"
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(self.data_dir.resolve()):
            raise KeyError("Capture file is unavailable")
        return path

    def analyze(self, capture_id):
        audio = self.capture_path(capture_id)
        with self.lock:
            if self.job_thread and self.job_thread.is_alive():
                raise ValueError("An analysis is already running")
            if not (self.project / ".venv/bin/python").exists():
                raise ValueError("Analyzer runtime missing: run scripts/install-analysis-deps.sh")
            job_id = uuid.uuid4().hex
            self.job = {"id": job_id, "capture_id": capture_id, "status": "running", "error": None,
                        "analysis": None, "artifacts": {}, "stems": {}}
            self.job_cancel.clear()
            self.job_thread = threading.Thread(target=self._analyze, args=(job_id, audio), name="audio-live-analysis", daemon=True)
            self.job_thread.start()
            return {"job": copy.deepcopy(self.job)}

    def _analyze(self, job_id, audio):
        folder = self.data_dir / "jobs" / job_id
        cache = self.data_dir / "cache"
        command = ["bash", str(self.project / "scripts/analyze-song.sh"), "--full", str(audio),
                   "--output-dir", str(folder), "--skip-whisper"]
        environment = {**os.environ, "PYTHONUNBUFFERED": "1", "MPLCONFIGDIR": str(cache / "matplotlib"),
                       "NUMBA_CACHE_DIR": str(cache / "numba")}
        try:
            folder.mkdir(parents=True, mode=0o700)
            cache.mkdir(exist_ok=True)
            with (folder / "analysis.log").open("w") as log:
                with self.lock:
                    if self.closing or self.job_cancel.is_set():
                        raise RuntimeError("Analysis cancelled")
                    process = subprocess.Popen(command, cwd=self.project, env=environment, stdout=log,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                    self.job_process = process
                try:
                    code = process.wait(timeout=7200)
                except subprocess.TimeoutExpired:
                    self._terminate_job()
                    raise RuntimeError("Analysis exceeded the two-hour time limit")
            if self.job_cancel.is_set():
                raise RuntimeError("Analysis cancelled")
            if code:
                raise RuntimeError(f"Analyzer exited with code {code}; inspect the local log")
            result = json.loads((folder / "analysis.json").read_text())
            metadata = result.get("metadata", {})
            unavailable_steps = set(metadata.get("steps_skipped") or ()) - {"whisper"}
            status = "partial" if metadata.get("steps_failed") or unavailable_steps else "completed"
            with self.lock:
                self.job.update(status=status, analysis=result)
        except Exception as exc:
            with self.lock:
                self.job.update(status="error", error=str(exc))
        finally:
            with self.lock:
                self.job_process = None
                self.job["artifacts"] = {name: f"/api/jobs/{job_id}/artifacts/{name}" for name, file in ARTIFACTS.items() if (folder / file).is_file()}
                self.job["stems"] = {name: self.job["artifacts"][name] for name in ("vocals", "drums", "bass", "other") if name in self.job["artifacts"]}
                try:
                    (folder / "job.json").write_text(json.dumps(self.job, indent=2))
                except OSError as exc:
                    self.job.update(status="error", error=f"Could not save analysis status: {exc}")

    def _terminate_job(self):
        with self.lock:
            process = self.job_process
        if process and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            except ProcessLookupError:
                pass

    def cancel(self):
        self.job_cancel.set()
        self._terminate_job()
        return self.snapshot()

    def close(self):
        self.closing = True
        self.stop()
        self.cancel()
        if self.job_thread:
            self.job_thread.join(timeout=7)


class EmptyBody(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AnalyzeBody(EmptyBody):
    capture_id: str


def valid_local_request(headers, port, require_origin=False):
    allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
    host = headers.get("host", "").lower()
    if host not in allowed:
        return False
    origin = headers.get("origin")
    if not origin:
        return not require_origin and headers.get("sec-fetch-site", "") not in ("cross-site", "same-site")
    parsed = urlsplit(origin)
    return parsed.scheme == "http" and parsed.netloc.lower() == host and not parsed.path


def create_app(service=None, port=8766):
    service = service or AudioService()

    @asynccontextmanager
    async def lifespan(app):
        yield
        await asyncio.to_thread(service.close)

    app = FastAPI(title="Local Music Analyzer", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.audio = service

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        if not valid_local_request(request.headers, port):
            return JSONResponse({"detail": "Only same-origin localhost requests are allowed"}, status_code=403)
        if request.method == "POST" and request.headers.get("content-type", "").split(";")[0] != "application/json":
            return JSONResponse({"detail": "JSON body required"}, status_code=415)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'; media-src 'self' blob:; img-src 'self' data:; frame-ancestors 'none'"
        return response

    @app.get("/")
    def index():
        return FileResponse(PROJECT / "web/audio-live/index.html")

    @app.get("/api/state")
    @app.get("/api/status")
    def status():
        return service.snapshot()

    @app.get("/api/devices")
    def devices():
        try:
            return {"source": "system_output", "processes": list_output_processes()}
        except Exception as exc:
            return {"source": "system_output", "processes": [], "error": str(exc)}

    def perform(action, *args):
        try:
            return action(*args)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(500, str(exc)) from exc

    @app.post("/api/start")
    def start(body: EmptyBody):
        return perform(service.start)

    @app.post("/api/stop")
    def stop(body: EmptyBody):
        return perform(service.stop)

    @app.post("/api/save")
    def save(body: EmptyBody):
        return perform(service.save)

    @app.post("/api/analyze")
    def analyze(body: AnalyzeBody):
        return perform(service.analyze, body.capture_id)

    @app.post("/api/discard")
    def discard(body: EmptyBody):
        return perform(service.discard)

    @app.post("/api/cancel")
    def cancel(body: EmptyBody):
        return perform(service.cancel)

    @app.get("/api/captures/{capture_id}/audio")
    def audio(capture_id: str):
        return FileResponse(perform(service.capture_path, capture_id), media_type="audio/wav")

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        snapshot = service.snapshot()["job"]
        if snapshot and snapshot["id"] == job_id:
            return snapshot
        if not ID_PATTERN.fullmatch(job_id):
            raise HTTPException(404)
        path = service.data_dir / "jobs" / job_id / "job.json"
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(service.data_dir.resolve()):
            raise HTTPException(404)
        return json.loads(path.read_text())

    @app.get("/api/jobs/{job_id}/artifacts/{name}")
    def artifact(job_id: str, name: str):
        if not ID_PATTERN.fullmatch(job_id) or name not in ARTIFACTS:
            raise HTTPException(404)
        path = service.data_dir / "jobs" / job_id / ARTIFACTS[name]
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(service.data_dir.resolve()):
            raise HTTPException(404)
        return FileResponse(path)

    @app.websocket("/ws")
    async def websocket(socket: WebSocket):
        if not valid_local_request(socket.headers, port, require_origin=True):
            await socket.close(code=1008)
            return
        await socket.accept()
        previous = None
        try:
            while True:
                signature = service.live_signature()
                if signature != previous:
                    await asyncio.wait_for(socket.send_json(service.snapshot(compact=True)), timeout=3)
                    previous = signature
                try:
                    event = await asyncio.wait_for(socket.receive(), timeout=0.2)
                    if event["type"] == "websocket.disconnect":
                        return
                except TimeoutError:
                    pass
        except WebSocketDisconnect:
            pass
        except TimeoutError:
            await socket.close(code=1013, reason="Client is not consuming live updates")

    return app


def main():
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("port must be between 1024 and 65535")
    uvicorn.run(create_app(port=args.port), host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
