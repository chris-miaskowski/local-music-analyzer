#!/usr/bin/env bash
set -euo pipefail
umask 077

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$PROJECT_DIR/.tools/audio-live/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "Install the capture environment first; see README.md: Live system audio." >&2
  exit 1
fi

cd "$PROJECT_DIR"
exec "$PYTHON" -u scripts/audio_live.py "$@"
