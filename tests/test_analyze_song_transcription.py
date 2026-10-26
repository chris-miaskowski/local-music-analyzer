import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("full_analysis", ROOT / "scripts/analyze-song-full.py")
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


class TranscriptionOptionsTest(unittest.TestCase):
    def test_local_reference_can_detect_language_without_changing_album_defaults(self):
        for options, expected_model, expected_language in (
            ({}, "large-v2", "pl"),
            ({"model_name": "small", "language": "auto"}, "small", None),
        ):
            with self.subTest(options=options):
                model = Mock()
                model.transcribe.return_value = {"language": "en", "segments": [{"words": [
                    {"start": 0.1, "end": 0.5, "word": " test"}
                ]}]}
                whisper = Mock()
                whisper.load_model.return_value = model
                with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as output:
                    metadata = {}
                    with patch.dict(sys.modules, {"whisper": whisper}):
                        words = analysis.run_whisper("local.wav", output, metadata=metadata, **options)
                    whisper.load_model.assert_called_once_with(expected_model)
                    self.assertEqual(model.transcribe.call_args.kwargs["language"], expected_language)
                    self.assertEqual(words[0]["word"], "test")
                    self.assertEqual(metadata["language"], "en")
                    self.assertTrue((Path(output) / "lyrics.tsv").is_file())


if __name__ == "__main__":
    unittest.main()
