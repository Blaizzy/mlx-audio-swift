#!/bin/bash
#
# Generate reference audio with Chatterbox Turbo using the Swift MLX Audio library.
# Mirrors chatterbox_reference.py for side-by-side comparison.
#
# Usage: ./chatterbox_swift_test.sh
#
# Pipeline overview (what the Swift code does):
#   1. Load model from HuggingFace: mlx-community/chatterbox-turbo-fp16
#   2. Tokenize text with GPT-2 tokenizer (raw IDs, no SOT/EOT wrapping for Turbo)
#   3. T3 (GPT-2 Medium): text tokens → speech tokens (autoregressive, temp=0.8, top_k=1000, top_p=0.95)
#   4. S3Gen: speech tokens → mel spectrogram (Conformer encoder + flow matching decoder, 2 Euler steps)
#   5. HiFi-GAN vocoder: mel → waveform at 24kHz
#   6. Apply trim_fade: 960-sample fade-in window (480 zeros + 480 cosine ramp)
#   7. Save as WAV
#
# Source: Tests/MLXAudioSmokeTests.swift → chatterboxTurboDebugPipeline()
# Text: "Testing one, two, three."
#

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_FILE="$SCRIPT_DIR/chatterbox_swift_output.wav"

echo "=== Chatterbox Turbo Swift Test ==="
echo "Model: mlx-community/chatterbox-turbo-fp16"
echo "Text:  \"Testing one, two, three.\""
echo ""
echo "Pipeline:"
echo "  1. Load model from HuggingFace"
echo "  2. Tokenize with GPT-2 (raw IDs, no SOT/EOT for Turbo)"
echo "  3. T3 GPT-2 Medium: text → speech tokens (temp=0.8, top_k=1000, top_p=0.95)"
echo "  4. S3Gen: speech tokens → mel (Conformer + flow matching, 2 Euler steps)"
echo "  5. HiFi-GAN: mel → waveform @ 24kHz"
echo "  6. trim_fade: 960-sample fade-in"
echo "  7. Save WAV"
echo ""

cd "$SCRIPT_DIR"

# Capture test output to extract the WAV path
LOGFILE=$(mktemp /tmp/chatterbox_swift_test.XXXXXX.log)

echo "Running chatterboxTurboDebugPipeline test via xcodebuild..."
echo "(Full log: $LOGFILE)"
echo ""

xcodebuild test \
    -scheme MLXAudio-Package \
    -only-testing:'MLXAudioTests/SmokeTests/TTSSmokeTests/chatterboxTurboDebugPipeline()' \
    -destination 'platform=macOS' \
    -skipPackagePluginValidation \
    -skipMacroValidation \
    > "$LOGFILE" 2>&1 || true

# Show the interesting parts of the output
echo "--- Test Output ---"
grep --color=never -E '(Conditioning|S3Gen Pipeline|HiFi-GAN|Full generate|text_tokens|prompt_token |prompt_feat|embedding|spk_emb|encoder_|mel_len|conds_signal|flow_matching|feat_gen|vocoder|full_generate|Saved|shape=|range=)' "$LOGFILE" || true
echo "---"
echo ""

# Check if test passed
if grep -q "Test Suite.*passed" "$LOGFILE"; then
    echo "Test: PASSED"
elif grep -q "TEST SUCCEEDED" "$LOGFILE"; then
    echo "Test: PASSED"
else
    echo "Test: FAILED (check $LOGFILE for details)"
    # Show errors
    grep -E "(error:|failed|FAIL)" "$LOGFILE" | tail -10
    exit 1
fi

# Extract the saved WAV path from test output
WAV_PATH=$(grep -o 'Saved full generate audio to.*: .*\.wav' "$LOGFILE" | sed 's/.*: //' | tail -1)

if [ -n "$WAV_PATH" ] && [ -f "$WAV_PATH" ]; then
    cp "$WAV_PATH" "$OUTPUT_FILE"
    echo ""
    echo "SUCCESS: Copied WAV to $OUTPUT_FILE"
    FILE_SIZE=$(ls -lh "$OUTPUT_FILE" | awk '{print $5}')
    echo "File size: $FILE_SIZE"

    if command -v afinfo &> /dev/null; then
        echo ""
        echo "Audio info:"
        afinfo "$OUTPUT_FILE" 2>/dev/null | grep -E "duration|sample rate|channels|format" || true
    fi
else
    echo ""
    echo "WARNING: Could not find generated WAV."
    echo "Looking for path in log..."
    grep "Saved" "$LOGFILE" || echo "No 'Saved' lines found in output."
    echo "Check $LOGFILE for full output."
fi

rm -f "$LOGFILE"
