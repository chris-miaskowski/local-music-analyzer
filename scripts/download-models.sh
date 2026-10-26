#!/usr/bin/env bash
# Download Essentia ML models for audio classification.
# Run from project root: bash scripts/download-models.sh
#
# Downloads both .pb (TensorFlow) and .onnx (ONNX Runtime) versions,
# plus .json metadata with class labels.
#
# Total download: ~80-100MB
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODELS_DIR="${1:-$PROJECT_DIR/models/essentia}"

BASE_URL="https://essentia.upf.edu/models"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

ok() { echo -e "${GREEN}✓${NC} $1"; }
warn() { echo -e "${YELLOW}⚠${NC} $1"; }
fail() { echo -e "${RED}✗${NC} $1"; }
info() { echo -e "${CYAN}→${NC} $1"; }

echo "=== Essentia Model Downloader ==="
echo "Target: $MODELS_DIR"
echo ""

# --- Create directory structure ---
mkdir -p "$MODELS_DIR/feature-extractors"
mkdir -p "$MODELS_DIR/classification-heads"

DOWNLOADED=0
SKIPPED=0
FAILED=0

# Download a file if it doesn't exist or is incomplete
download() {
    local url="$1"
    local dest="$2"
    local name
    name="$(basename "$dest")"

    if [ -f "$dest" ] && [ -s "$dest" ]; then
        SKIPPED=$((SKIPPED + 1))
        return 0
    fi

    mkdir -p "$(dirname "$dest")"

    if curl -sSL --fail -o "$dest" "$url" 2>/dev/null; then
        DOWNLOADED=$((DOWNLOADED + 1))
        return 0
    else
        rm -f "$dest"
        fail "Failed: $name"
        FAILED=$((FAILED + 1))
        return 1
    fi
}

# ============================================================
# 1. EMBEDDING MODELS (feature extractors)
# ============================================================
echo "--- Embedding Models ---"

# EffNet-Discogs (used by all classification heads)
info "discogs-effnet-bs64-1 (TF .pb)"
download "$BASE_URL/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb" \
         "$MODELS_DIR/feature-extractors/discogs-effnet-bs64-1.pb"

info "discogs-effnet-bs64-1 (metadata)"
download "$BASE_URL/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.json" \
         "$MODELS_DIR/feature-extractors/discogs-effnet-bs64-1.json"

# EffNet dynamic batch ONNX (for onnxruntime fallback)
info "discogs-effnet-bsdynamic-1 (ONNX)"
download "$BASE_URL/feature-extractors/discogs-effnet/discogs-effnet-bsdynamic-1.onnx" \
         "$MODELS_DIR/feature-extractors/discogs-effnet-bsdynamic-1.onnx"

echo ""

# ============================================================
# 2. CLASSIFICATION HEADS
# ============================================================
echo "--- Classification Heads ---"

# Each classifier: download .pb (TF), .onnx (ONNX), .json (metadata)
# Using classification-heads/ URL path for all

CLASSIFIERS=(
    # Genre
    "genre_discogs400:classification-heads/genre_discogs400:genre_discogs400-discogs-effnet-1"

    # Mood (binary)
    "mood_aggressive:classification-heads/mood_aggressive:mood_aggressive-discogs-effnet-1"
    "mood_happy:classification-heads/mood_happy:mood_happy-discogs-effnet-1"
    "mood_sad:classification-heads/mood_sad:mood_sad-discogs-effnet-1"
    "mood_relaxed:classification-heads/mood_relaxed:mood_relaxed-discogs-effnet-1"
    "mood_party:classification-heads/mood_party:mood_party-discogs-effnet-1"
    "mood_acoustic:classification-heads/mood_acoustic:mood_acoustic-discogs-effnet-1"
    "mood_electronic:classification-heads/mood_electronic:mood_electronic-discogs-effnet-1"

    # Voice & Timbre
    "voice_instrumental:classification-heads/voice_instrumental:voice_instrumental-discogs-effnet-1"
    "gender:classification-heads/gender:gender-discogs-effnet-1"
    "timbre:classification-heads/timbre:timbre-discogs-effnet-1"

    # Characteristics
    "danceability:classification-heads/danceability:danceability-discogs-effnet-1"
    "tonal_atonal:classification-heads/tonal_atonal:tonal_atonal-discogs-effnet-1"
    "approachability:classification-heads/approachability:approachability_regression-discogs-effnet-1"
    "engagement:classification-heads/engagement:engagement_regression-discogs-effnet-1"

    # Multi-label (MTG-Jamendo)
    "mtg_jamendo_instrument:classification-heads/mtg_jamendo_instrument:mtg_jamendo_instrument-discogs-effnet-1"
    "mtg_jamendo_moodtheme:classification-heads/mtg_jamendo_moodtheme:mtg_jamendo_moodtheme-discogs-effnet-1"

    # NSynth
    "nsynth_acoustic_electronic:classification-heads/nsynth_acoustic_electronic:nsynth_acoustic_electronic-discogs-effnet-1"
)

for entry in "${CLASSIFIERS[@]}"; do
    IFS=':' read -r name url_path file_base <<< "$entry"

    info "$name"

    download "$BASE_URL/$url_path/$file_base.pb" \
             "$MODELS_DIR/$url_path/$file_base.pb"

    download "$BASE_URL/$url_path/$file_base.json" \
             "$MODELS_DIR/$url_path/$file_base.json"
done

echo ""

# ============================================================
# 3. Summary
# ============================================================
echo "=== Download Summary ==="
ok "Downloaded: $DOWNLOADED files"
if [ "$SKIPPED" -gt 0 ]; then
    info "Already present: $SKIPPED files"
fi
if [ "$FAILED" -gt 0 ]; then
    fail "Failed: $FAILED files"
fi

# Show total size
TOTAL_SIZE=$(du -sh "$MODELS_DIR" 2>/dev/null | cut -f1)
echo ""
echo "Models directory: $MODELS_DIR ($TOTAL_SIZE)"
echo ""

# Verify model count
PB_COUNT=$(find "$MODELS_DIR" -name "*.pb" 2>/dev/null | wc -l | tr -d ' ')
ONNX_COUNT=$(find "$MODELS_DIR" -name "*.onnx" 2>/dev/null | wc -l | tr -d ' ')
JSON_COUNT=$(find "$MODELS_DIR" -name "*.json" 2>/dev/null | wc -l | tr -d ' ')

echo "Files: $PB_COUNT .pb, $ONNX_COUNT .onnx, $JSON_COUNT .json"

if [ "$FAILED" -gt 0 ]; then
    echo ""
    warn "Some downloads failed. Re-run this script to retry."
    exit 1
fi

ok "All models ready. Run: bash scripts/analyze-song.sh --full <audio_file>"
