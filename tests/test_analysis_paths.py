import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("path_analysis", ROOT / "scripts/analyze-song-full.py")
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


class StandalonePathsTest(unittest.TestCase):
    def test_input_in_arbitrary_directory_has_its_own_adjacent_report(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
            source = Path(directory) / "recording.wav"
            source.touch()
            result = analysis.resolve_paths(str(source), None, str(ROOT / "models/essentia"))
            self.assertEqual(Path(result["output_dir"]), source.parent / "recording.analysis")
            self.assertTrue(Path(result["output_dir"]).is_dir())

    def test_audio_subdirectory_keeps_existing_layout(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
            source = Path(directory) / "track/audio/song.wav"
            source.parent.mkdir(parents=True)
            source.touch()
            result = analysis.resolve_paths(str(source), None, str(ROOT / "models/essentia"))
            self.assertEqual(Path(result["output_dir"]), source.parent.parent / "analysis")

    def test_explicit_output_wins(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
            source = Path(directory) / "song.wav"
            source.touch()
            target = Path(directory) / "custom-report"
            result = analysis.resolve_paths(str(source), str(target), str(ROOT / "models/essentia"))
            self.assertEqual(Path(result["output_dir"]), target)

    def test_default_models_do_not_follow_the_callers_directory(self):
        original = Path.cwd()
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
            try:
                os.chdir(directory)
                with patch.object(sys, "argv", ["analyze-song-full.py", "song.wav"]):
                    args = analysis.parse_args()
                self.assertEqual(Path(args.models_dir), ROOT / "models/essentia")
            finally:
                os.chdir(original)
