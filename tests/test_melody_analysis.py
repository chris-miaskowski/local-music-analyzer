"""Pitch contour checks with synthetic audio and predictions; no model downloads."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("song_analysis", PROJECT / "scripts/analyze-song-full.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class Tensor:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.float32)
        self.shape = self.data.shape

    def mean(self, dim, keepdim):
        return Tensor(self.data.mean(axis=dim, keepdims=keepdim))

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.data


class MelodyTests(unittest.TestCase):
    def test_pitch_and_midi_follow_center_times_without_stretching(self):
        melody, voiced = analysis.build_melody_timeline(
            [220, 440, 880, 440], [0.9] * 4, [0.1] * 4, 0.035, 0.01,
        )
        self.assertEqual([point["time"] for point in melody["points"]], [0, 0.01, 0.02, 0.03])
        self.assertEqual([point["midi"] for point in melody["points"]], [57, 69, 81, 69])
        self.assertEqual(melody["duration_seconds"], 0.035)
        self.assertEqual(melody["kind"], "monophonic_estimate")
        self.assertTrue(voiced.all())

    def test_silence_low_confidence_and_invalid_predictions_are_gaps(self):
        melody, voiced = analysis.build_melody_timeline(
            [440, 440, 440, np.nan, 440, 0, 2000],
            [0.9, 0.99, 0.5, 0.9, np.inf, 0.9, 0.9],
            [0.1, 0.00001, 0.1, 0.1, 0.1, 0.1, 0.1], 0.07, 0.01,
        )
        self.assertEqual(voiced.tolist(), [True, False, False, False, False, False, False])
        self.assertEqual(melody["voiced_frame_count"], 1)
        for point in melody["points"][1:]:
            self.assertIsNone(point["hz"])
            self.assertIsNone(point["midi"])
        json.dumps(melody, allow_nan=False)

    def test_centered_rms_matches_padding_silence_and_chunk_boundaries(self):
        samples = np.ones(200000, dtype=np.float32)
        samples[16000:32000] = 0
        rms = analysis.pitch_frame_rms(samples, 1251)
        self.assertAlmostEqual(rms[0], np.sqrt(0.5))
        self.assertEqual(rms[110], 0)
        self.assertEqual(rms[1023], 1)
        self.assertEqual(rms[1024], 1)
        self.assertAlmostEqual(rms[-1], np.sqrt(0.5))

    def test_decimation_is_bounded_and_keeps_short_gaps_and_end_time(self):
        frames = 12001
        confidence = np.full(frames, 0.9)
        confidence[1] = 0
        melody, _ = analysis.build_melody_timeline(
            np.full(frames, 440), confidence, np.full(frames, 0.1), 120.007, 0.01,
        )
        self.assertEqual(len(melody["points"]), 6000)
        self.assertIsNone(melody["points"][0]["hz"])
        self.assertEqual(melody["points"][0]["time"], 0)
        self.assertEqual(melody["points"][-1]["time"], 120)
        self.assertEqual(melody["duration_seconds"], 120.007)
        self.assertEqual(melody["hop_seconds"], 0.01)

    def test_empty_single_frame_and_frames_beyond_duration(self):
        empty, voiced = analysis.build_melody_timeline([], [], [], 0, 0.01)
        self.assertEqual(empty["points"], [])
        self.assertEqual(len(voiced), 0)
        single, _ = analysis.build_melody_timeline([440], [0.9], [0.1], 0.003, 0.01)
        self.assertEqual(single["points"][0]["time"], 0)
        clipped, voiced = analysis.build_melody_timeline([440] * 3, [0.9] * 3, [0.1] * 3, 0.011, 0.01)
        self.assertEqual(len(clipped["points"]), 2)
        self.assertFalse(voiced[-1])
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            analysis.build_melody_timeline([440], [], [0.1], 1, 0.01)

    def test_stereo_48khz_is_resampled_before_predict_and_silence_gated(self):
        samples = np.concatenate((np.full(24000, 0.1), np.zeros(24000)))
        input_audio = Tensor(np.stack((samples, samples)))
        calls = {}

        def resample(audio, source_rate, target_rate):
            calls["resample"] = (source_rate, target_rate, audio.shape)
            return Tensor(audio.data[:, ::3])

        def predict(audio, sample_rate, **kwargs):
            calls["predict"] = (sample_rate, audio.shape, kwargs)
            frames = 1 + audio.shape[-1] // kwargs["hop_length"]
            return Tensor([[440] * frames]), Tensor([[0.9] * frames])

        fake_audio = SimpleNamespace(load=lambda _: (input_audio, 48000),
                                     functional=SimpleNamespace(resample=resample))
        modules = {
            "torchaudio": fake_audio,
            "torchcrepe": SimpleNamespace(predict=predict),
            "librosa": SimpleNamespace(hz_to_midi=lambda hz: 69 + 12 * np.log2(hz / 440),
                                        midi_to_note=lambda midi: str(midi)),
        }
        with patch.dict(sys.modules, modules):
            result = analysis.run_pitch_analysis("fixture.wav", source="mix")
        self.assertEqual(calls["resample"], (48000, 16000, (1, 48000)))
        rate, shape, options = calls["predict"]
        self.assertEqual((rate, shape), (16000, (1, 16000)))
        self.assertEqual(options["hop_length"], 160)
        self.assertLessEqual(options["batch_size"], 64)
        self.assertTrue(options["pad"])
        self.assertEqual(result["melody"]["points"][1]["time"], 0.01)
        self.assertEqual(result["melody"]["points"][-1]["time"], 1)
        self.assertEqual(result["melody"]["source"], "mix")
        self.assertIsNone(result["melody"]["points"][-1]["hz"])
        self.assertLess(result["voiced_ratio"], 0.6)

    def test_contour_is_written_to_default_analysis_without_frame_export(self):
        melody, _ = analysis.build_melody_timeline([440, 440], [0.9, 0.2], [0.1, 0.1], 0.02, 0.01)
        steps = {"pitch": {"status": "ok", "data": {"melody": melody}}}
        analysis.ensure_step_keys(steps)
        with tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as output:
            filename, _ = analysis.generate_analysis_json("test", "fixture.wav", 0.02, output, steps, [])
            saved = json.loads(Path(filename).read_text())
        self.assertEqual(saved["vocals"]["melody"], melody)
        self.assertIsNone(saved["vocals"]["melody"]["points"][1]["midi"])
        self.assertNotIn("_frames", saved["vocals"])


if __name__ == "__main__":
    unittest.main()
