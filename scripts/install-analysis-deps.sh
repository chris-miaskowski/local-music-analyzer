#!/usr/bin/env bash
# Install all dependencies for comprehensive audio analysis.
# Run from project root: bash scripts/install-analysis-deps.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="$PROJECT_DIR/.venv"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok() { echo -e "${GREEN}✓${NC} $1"; }
warn() { echo -e "${YELLOW}⚠${NC} $1"; }
fail() { echo -e "${RED}✗${NC} $1"; }

echo "=== Local Music Analyzer Dependencies Installer ==="
echo ""

# --- 1. Ensure venv exists ---
if [ ! -d "$VENV" ]; then
    echo "Creating virtual environment at $VENV ..."
    "${ANALYZER_PYTHON:-python3.12}" -m venv "$VENV"
    ok "Virtual environment created"
fi

source "$VENV/bin/activate"
ok "Activated venv: $(python --version) at $(which python)"
echo ""

# --- 2. Upgrade pip ---
pip install --upgrade pip --quiet
ok "pip upgraded"

# --- 3. Install core dependencies (if missing) ---
echo ""
echo "--- Core dependencies ---"
pip install --quiet "torch==2.10.0" "torchaudio==2.10.0" "torchcodec==0.10.0" "demucs>=4.0.1" "openai-whisper>=20250625" "beat-this>=0.1"
ok "Core deps (demucs, whisper, beat-this) ensured"

# --- 4. Install essentia-tensorflow (primary) or essentia (fallback) ---
echo ""
echo "--- Essentia ---"
ESSENTIA_STATUS="none"

echo "Trying essentia-tensorflow (bundles TF internally)..."
if pip install --quiet "essentia-tensorflow>=2.1b6.dev1389" 2>&1; then
    # Verify it actually works at runtime
    if python -c "from essentia.standard import TensorflowPredictEffnetDiscogs; print('TF predict OK')" 2>/dev/null; then
        ok "essentia-tensorflow installed and TF predict works"
        ESSENTIA_STATUS="tensorflow"
    else
        warn "essentia-tensorflow installed but TF predict failed at runtime"
        warn "Falling back to essentia base + onnxruntime"
        pip install --quiet essentia 2>&1 || true
        ESSENTIA_STATUS="base"
    fi
else
    warn "essentia-tensorflow failed to install"
    echo "Trying essentia base package..."
    if pip install --quiet essentia 2>&1; then
        ok "essentia (base, no TF) installed"
        ESSENTIA_STATUS="base"
    else
        fail "essentia failed to install — DSP analysis will be unavailable"
        ESSENTIA_STATUS="none"
    fi
fi

# --- 5. Install librosa ---
echo ""
echo "--- librosa ---"
if pip install --quiet "librosa>=0.10.2" 2>&1; then
    ok "librosa installed"
else
    fail "librosa failed to install"
fi

# --- 6. Install onnxruntime (ONNX fallback for ML models) ---
echo ""
echo "--- onnxruntime ---"
if pip install --quiet "onnxruntime>=1.21.1" 2>&1; then
    ok "onnxruntime installed"
else
    warn "onnxruntime failed — ML classification requires essentia-tensorflow"
fi

# --- 7. Install torchcrepe (pitch tracking) ---
echo ""
echo "--- torchcrepe ---"
if pip install --quiet "torchcrepe>=0.0.22" 2>&1; then
    ok "torchcrepe installed"
else
    fail "torchcrepe failed to install — pitch analysis will be unavailable"
fi

# --- 8. Verify all imports ---
echo ""
echo "--- Structure and mastering ---"
pip install "allin1>=1.1.0" || warn "allin1 unavailable; structure analysis will report a failed stage"
pip install "pyloudnorm>=0.2.0" || warn "pyloudnorm unavailable; mastering analysis will report a failed stage"

echo ""
echo "=== Verifying imports ==="

PASS=0
TOTAL=0

verify_import() {
    local name="$1"
    local import_stmt="$2"
    TOTAL=$((TOTAL + 1))
    if python -c "$import_stmt" 2>/dev/null; then
        ok "$name"
        PASS=$((PASS + 1))
    else
        fail "$name — import failed"
    fi
}

verify_import "demucs"         "import demucs.api"
verify_import "whisper"        "import whisper"
verify_import "beat_this"      "from beat_this.inference import File2Beats"
verify_import "essentia"       "import essentia.standard"
verify_import "librosa"        "import librosa"
verify_import "onnxruntime"    "import onnxruntime"
verify_import "torchcrepe"     "import torchcrepe"
verify_import "torchcodec"     "import torchcodec"
verify_import "allin1"         "import allin1"
verify_import "pyloudnorm"     "import pyloudnorm"
verify_import "torch"          "import torch"
verify_import "torchaudio"     "import torchaudio"
verify_import "numpy"          "import numpy"

echo ""
echo "=== Summary ==="
echo "$PASS/$TOTAL packages verified"
echo "Essentia mode: $ESSENTIA_STATUS"

if [ "$ESSENTIA_STATUS" = "tensorflow" ]; then
    ok "Full ML classification available via essentia-tensorflow"
elif [ "$ESSENTIA_STATUS" = "base" ]; then
    warn "ML classification requires ONNX models (run scripts/download-models.sh)"
    warn "DSP analysis (BPM, key, chords, loudness) fully available"
else
    fail "No essentia — DSP analysis unavailable"
fi

echo ""
echo "Next step: bash scripts/download-models.sh"
