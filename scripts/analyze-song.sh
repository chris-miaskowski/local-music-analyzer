#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="$PROJECT_DIR/.venv"

if [ ! -d "$VENV" ]; then
  echo "Error: Virtual environment not found at $VENV"
  echo "Run: bash scripts/install-analysis-deps.sh"
  exit 1
fi

source "$VENV/bin/activate"

if [[ "${1:-}" == "--full" ]]; then
  shift
  python "$SCRIPT_DIR/analyze-song-full.py" "$@"
else
  python "$SCRIPT_DIR/analyze-song.py" "$@"
fi
