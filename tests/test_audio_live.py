"""No hardware or model downloads: signal fixtures and fake capture lifecycles."""

import importlib.util
import json
import tempfile
import time
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("audio_live", PROJECT / "scripts/audio_live.py")
audio = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audio)


def buffer(samples, rate=48000, bits=32, sample_type="float", channels=1):
    return SimpleNamespace(data=samples, frame_count=len(samples) // (channels * bits // 8),
                           format=SimpleNamespace(sample_rate=rate, bits_per_sample=bits,
                                                  sample_type=sample_type, num_channels=channels))


class FakeCapture:
    def __init__(self, callback):
        self.callback = callback
        self.is_recording = False
        self.needs_cleanup = False
        self.capture_failed = False
        self.stop_calls = 0

    def start(self):
        self.is_recording = True

    def stop(self):
        self.stop_calls += 1
        self.is_recording = False

    def emit(self, samples):
        self.callback(buffer(np.asarray(samples, dtype="<f4").tobytes()))


class SignalTests(unittest.TestCase):
    def test_pcm_formats_and_stereo_order(self):
        expected = np.array([[-1, 0], [0.5, -0.5]], dtype=np.float32)
        decoded = audio.decode_pcm(buffer(expected.tobytes(), channels=2))
        np.testing.assert_array_equal(decoded, expected)
        for bits in (16, 24, 32):
            values = np.array([-2 ** (bits - 1), 0, 2 ** (bits - 2)], dtype=np.int64)
            if bits == 24:
                raw = b"".join(int(x & 0xFFFFFF).to_bytes(3, "little") for x in values)
            else:
                raw = values.astype(f"<i{bits // 8}").tobytes()
            decoded = audio.decode_pcm(buffer(raw, bits=bits, sample_type="signed_integer"))
            np.testing.assert_allclose(decoded[:, 0], [-1, 0, 0.5])

    def test_rejects_corrupt_or_nonfinite_audio(self):
        item = buffer(b"\0" * 4)
        item.frame_count = 4
        with self.assertRaisesRegex(ValueError, "byte count"):
            audio.decode_pcm(item)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            audio.decode_pcm(buffer(np.array([np.nan], dtype="<f4").tobytes()))

    def test_tone_level_spectrum_and_silence(self):
        rate = 48000
        dsp = audio.LiveDSP(rate)
        tone = 0.5 * np.sin(2 * np.pi * 1000 * np.arange(rate) / rate)
        dsp.feed(tone.astype(np.float32).reshape(-1, 1))
        metrics = dsp.latest
        self.assertAlmostEqual(metrics["rms_dbfs"], -9.03, delta=0.15)
        self.assertAlmostEqual(metrics["peak_dbfs"], -6.02, delta=0.05)
        self.assertGreater(metrics["bands"]["mid"], 0.99)
        peak_hz = float(dsp.frequency[np.argmax(dsp.previous)])
        self.assertAlmostEqual(peak_hz, 1000, delta=12)
        displayed_peak = max(metrics["spectrum"], key=lambda x: x["dbfs"])
        self.assertAlmostEqual(displayed_peak["hz"], 1000, delta=12)
        self.assertIsNone(metrics["bpm"])
        dsp.feed(np.zeros((rate, 1), dtype=np.float32))
        self.assertEqual(dsp.latest["peak_dbfs"], -120)
        self.assertTrue(all(x == 0 for x in dsp.latest["waveform"]))

    def test_regular_pulses_produce_tentative_tempo(self):
        rate = 12000
        samples = np.zeros(20 * rate, dtype=np.float32)
        rng = np.random.default_rng(5)
        for index in range(1, 40):
            start = index * rate // 2
            samples[start:start + 240] = rng.normal(0, 0.3, 240) * np.hanning(240)
        dsp = audio.LiveDSP(rate)
        dsp.feed(samples.reshape(-1, 1))
        self.assertGreater(len(dsp.onsets), 25)
        self.assertAlmostEqual(dsp.latest["bpm"], 120, delta=7)
        self.assertLessEqual(dsp.latest["bpm_confidence"], 0.85)
        self.assertGreater(dsp.latest["bpm_confidence"], 0.4)
        self.assertTrue(all(a < b for a, b in zip(dsp.onsets, list(dsp.onsets)[1:])))


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=PROJECT / "tmp")
        self.service = audio.AudioService(data_dir=Path(self.tmp.name), capture_factory=FakeCapture)

    def tearDown(self):
        self.service.close()
        self.tmp.cleanup()

    def wait_frames(self, count):
        until = time.monotonic() + 3
        while self.service.total_frames < count and time.monotonic() < until:
            time.sleep(0.01)
        self.assertEqual(self.service.total_frames, count)

    def test_stop_releases_tap_save_preserves_frames_and_restart_needs_save(self):
        self.service.start()
        capture = self.service.session
        capture.emit(np.full(12000, 0.25))
        self.wait_frames(12000)
        self.service.stop()
        self.assertFalse(capture.is_recording)
        self.assertGreater(capture.stop_calls, 0)
        self.assertIsNone(self.service.session)
        with self.assertRaisesRegex(ValueError, "Save"):
            self.service.start()
        saved = self.service.save()["capture"]
        path = self.service.capture_path(saved["id"])
        with wave.open(str(path)) as wav:
            self.assertEqual(wav.getnframes(), 12000)
            self.assertEqual(wav.getframerate(), 48000)
            self.assertEqual(wav.getnchannels(), 1)
            self.assertEqual(np.frombuffer(wav.readframes(12000), dtype="<i2")[0], 8192)
        self.service.start()
        self.assertEqual(self.service.snapshot()["metrics"]["peak_dbfs"], -120)
        self.assertEqual(self.service.total_frames, 0)
        self.assertTrue(path.is_file())
        reloaded = audio.AudioService(data_dir=Path(self.tmp.name), capture_factory=FakeCapture)
        self.assertIn(saved["id"], reloaded.captures)
        with self.assertRaises(KeyError):
            self.service.capture_path("../../README.md")

    def test_capture_failure_cleans_tap_and_reports_error(self):
        self.service.start()
        session = self.service.session
        session.capture_failed = True
        until = time.monotonic() + 3
        while self.service.session and time.monotonic() < until:
            time.sleep(0.01)
        self.assertIsNone(self.service.session)
        self.assertEqual(self.service.state, "error")
        self.assertIn("capture failure", self.service.error)

    def test_discard_requires_stop_and_never_deletes_saved_audio(self):
        self.service.start()
        with self.assertRaisesRegex(ValueError, "Stop"):
            self.service.discard()
        self.service.session.emit(np.zeros(4800))
        self.wait_frames(4800)
        saved = self.service.save()["capture"]
        self.service.stop()
        self.service.discard()
        self.assertEqual(self.service.total_frames, 0)
        self.assertFalse(self.service.snapshot()["unsaved"])
        self.assertIsNone(self.service.dsp)
        self.assertTrue(self.service.capture_path(saved["id"]).is_file())

    def test_rolling_buffer_retains_exact_tail(self):
        with patch.object(audio, "MAX_CAPTURE_SECONDS", 0.5):
            self.service.start()
            self.service.session.emit(np.full(20000, 0.25))
            self.wait_frames(20000)
            self.service.session.emit(np.full(20000, 0.5))
            self.wait_frames(40000)
            self.service.stop()
        self.assertEqual(self.service.ring_frames, 24000)
        saved = self.service.save()["capture"]
        self.assertEqual(saved["duration_seconds"], 0.5)
        self.assertAlmostEqual(saved["offset_seconds"], 16000 / 48000, delta=0.001)
        with wave.open(str(self.service.capture_path(saved["id"]))) as wav:
            pcm = np.frombuffer(wav.readframes(24000), dtype="<i2")
        np.testing.assert_array_equal(pcm[:4000], np.full(4000, 8192))
        np.testing.assert_array_equal(pcm[4000:], np.full(20000, 16384))

    def test_live_payload_excludes_full_analysis_and_bounds_history(self):
        self.service.dsp = audio.LiveDSP(48000)
        dsp = self.service.dsp
        dsp.latest["time"] = 1200
        for index in range(12000):
            stamp = index / 10
            dsp.history.append({"time": stamp, "rms_dbfs": -20, "peak_dbfs": -10,
                                "bass": 0.2, "mid": 0.7, "treble": 0.1, "flux": 0.01})
            dsp.onsets.append(stamp)
        self.service.job = {"id": "a" * 32, "capture_id": "b" * 32, "status": "completed", "error": None,
                            "analysis": {"large_result": ["payload" * 100] * 10000},
                            "words": [{"word": "large word data"}] * 10000, "stems": {"vocals": "/audio"}}
        state = self.service.snapshot(compact=True)
        self.assertLessEqual(len(state["history"]), 301)
        self.assertLessEqual(len(state["onsets"]), 301)
        self.assertNotIn("analysis", state["job"])
        self.assertNotIn("words", state["job"])
        self.assertTrue(state["job"]["details_available"])
        self.assertLess(len(json.dumps(state)), 60000)
        self.assertEqual(self.service.live_signature(), self.service.live_signature())
        self.assertEqual(len(dsp.history), 6000)
        self.assertEqual(len(self.service.job["analysis"]["large_result"]), 10000)
        self.service.job_thread = MagicMock()
        self.service.job_thread.is_alive.return_value = True
        self.assertFalse(self.service.snapshot()["job"]["details_available"])
        self.assertFalse(self.service.snapshot(compact=True)["job"]["details_available"])

    def test_dsp_retained_arrays_stay_bounded_after_large_input(self):
        dsp = audio.LiveDSP(8000)
        for _ in range(3):
            dsp.feed(np.zeros((80000, 2), dtype=np.float32))
            self.assertLess(len(dsp.pending), dsp.size)
            self.assertIsNone(dsp.pending.base)
            self.assertLessEqual(len(dsp.fluxes), int(30 * 8000 / dsp.hop))
            self.assertLessEqual(len(dsp.history), 6000)
            self.assertLessEqual(len(dsp.onsets), 6000)

    def test_no_signal_differs_from_silence_and_permission_failure(self):
        self.service.start()
        self.service.started_at -= 4
        self.assertEqual(self.service.snapshot()["signal"], "no_signal")
        self.service.session.emit(np.zeros(4800))
        self.wait_frames(4800)
        self.assertEqual(self.service.snapshot()["signal"], "silent")
        self.assertIsNone(self.service.error)

    def test_start_permission_error_does_not_claim_recording(self):
        def refused(callback):
            raise PermissionError("System Audio Recording permission denied")
        self.service.capture_factory = refused
        with self.assertRaisesRegex(RuntimeError, "permission denied"):
            self.service.start()
        self.assertEqual(self.service.state, "error")
        self.assertIsNone(self.service.session)

    def test_analysis_omits_whisper_and_only_marks_unavailable_music_steps_partial(self):
        project = Path(self.tmp.name) / "project"
        (project / ".venv/bin").mkdir(parents=True)
        (project / ".venv/bin/python").touch()
        self.service.project = project
        self.service.start()
        self.service.session.emit(np.zeros(4800))
        self.wait_frames(4800)
        self.service.stop()
        capture_id = self.service.save()["capture"]["id"]

        metadata = {"steps_completed": ["beats"], "steps_skipped": ["whisper"]}

        def fake_popen(command, **kwargs):
            self.assertNotIn("shell", kwargs)
            self.assertTrue(kwargs["start_new_session"])
            self.assertIn("--skip-whisper", command)
            self.assertNotIn("--language", command)
            self.assertNotIn("--whisper-model", command)
            folder = Path(command[command.index("--output-dir") + 1])
            (folder / "analysis.json").write_text(json.dumps({"metadata": metadata}))
            (folder / "lyrics.tsv").write_text("start\tend\tword\n0.1\t0.5\thello\n")
            process = MagicMock()
            process.wait.return_value = 0
            return process

        with patch.object(audio.subprocess, "Popen", side_effect=fake_popen):
            for skipped, failed, expected in ((["whisper"], [], "completed"),
                                               (["whisper", "pitch"], [], "partial"),
                                               (["whisper"], ["beats"], "partial")):
                metadata.update(steps_skipped=skipped, steps_failed=failed)
                self.service.analyze(capture_id)
                self.service.job_thread.join(timeout=3)
                job = self.service.snapshot()["job"]
                self.assertEqual(job["status"], expected)
                self.assertNotIn("words", job)
                self.assertNotIn("lyrics", job["artifacts"])
                self.assertIn("analysis", job["artifacts"])
                self.assertTrue((Path(self.tmp.name) / "jobs" / job["id"] / "job.json").is_file())


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=PROJECT / "tmp")
        self.service = audio.AudioService(data_dir=self.tmp.name, capture_factory=FakeCapture)
        self.client = TestClient(audio.create_app(self.service), base_url="http://127.0.0.1:8766")

    def tearDown(self):
        self.client.close()
        self.service.close()
        self.tmp.cleanup()

    def test_json_and_origin_guards(self):
        self.assertEqual(self.client.get("/api/status").status_code, 200)
        self.assertEqual(self.client.get("/api/status", headers={"host": "evil.example:8766"}).status_code, 403)
        self.assertEqual(self.client.post("/api/start", json={}, headers={"origin": "https://evil.example"}).status_code, 403)
        self.assertEqual(self.client.post("/api/start", content="{}").status_code, 415)
        self.assertEqual(self.client.post("/api/start", json={"path": "/private/data"}).status_code, 422)
        self.assertEqual(self.client.post("/api/start", json={}, headers={"origin": "http://127.0.0.1:8766"}).status_code, 200)
        self.assertEqual(self.client.post("/api/stop", json={}).status_code, 200)
        self.assertEqual(self.client.post("/api/save", json={}).status_code, 409)
        self.assertEqual(self.client.get("/api/captures/not-a-capture/audio").status_code, 404)
        self.assertEqual(self.client.get("/api/jobs/" + "a" * 32 + "/artifacts/env").status_code, 404)

    def test_websocket_requires_same_origin(self):
        with self.assertRaises(WebSocketDisconnect):
            with self.client.websocket_connect("ws://127.0.0.1:8766/ws", headers={"origin": "https://evil.example"}):
                self.fail("Cross-origin socket unexpectedly accepted")
        with self.client.websocket_connect("ws://127.0.0.1:8766/ws", headers={"origin": "http://127.0.0.1:8766"}) as socket:
            state = socket.receive_json()
            self.assertEqual(state["state"], "idle")
            self.assertEqual(state["source"], "system_output")


if __name__ == "__main__":
    unittest.main()
