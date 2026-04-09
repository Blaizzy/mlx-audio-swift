#!/bin/bash
# Quick test for OmniVoice - generates audio using all three modes

set -e

echo "=========================================="
echo "OmniVoice Quick Test"
echo "=========================================="
echo ""

MODEL="mlx-community/OmniVoice-bf16"
OUTPUT_DIR="/tmp/omnivoice_tests"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Model: $MODEL"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Check if swift is available
if ! command -v swift &> /dev/null; then
    echo "Error: swift command not found"
    exit 1
fi

# Test 1: Auto Voice
echo "=========================================="
echo "Test 1: Auto Voice Mode"
echo "=========================================="
echo "Generating speech with default voice..."
echo ""

swift run mlx-audio-swift-tts \
    --model "$MODEL" \
    --text "Hello, this is a test of OmniVoice auto voice mode." \
    --output "$OUTPUT_DIR/auto_voice.wav" 2>&1 | tee "$OUTPUT_DIR/auto_voice.log"

echo ""
echo "✓ Auto voice test complete"
echo ""

# Test 2: Voice Design
echo "=========================================="
echo "Test 2: Voice Design Mode"
echo "=========================================="
echo "Generating speech with voice instruction..."
echo ""

swift run mlx-audio-swift-tts \
    --model "$MODEL" \
    --text "Hello, this is a voice design test." \
    --instruct "male, British accent" \
    --output "$OUTPUT_DIR/voice_design.wav" \
    --num_step 32 \
    --guidance_scale 2.0 2>&1 | tee "$OUTPUT_DIR/voice_design.log"

echo ""
echo "✓ Voice design test complete"
echo ""

# Test 3: Voice Cloning (if reference audio exists)
REF_AUDIO="$(dirname "$0")/Tests/media/intention.wav"
if [ -f "$REF_AUDIO" ]; then
    echo "=========================================="
    echo "Test 3: Voice Cloning Mode"
    echo "=========================================="
    echo "Generating speech with voice cloning..."
    echo ""

    swift run mlx-audio-swift-tts \
        --model "$MODEL" \
        --text "This is a voice cloning test." \
        --ref_audio "$REF_AUDIO" \
        --ref_text "intention" \
        --output "$OUTPUT_DIR/voice_cloning.wav" 2>&1 | tee "$OUTPUT_DIR/voice_cloning.log"

    echo ""
    echo "✓ Voice cloning test complete"
    echo ""
else
    echo "⚠ Skipping voice cloning test - reference audio not found"
    echo ""
fi

# Summary
echo "=========================================="
echo "All Tests Complete!"
echo "=========================================="
echo ""
echo "Generated files:"
ls -lh "$OUTPUT_DIR"/*.wav 2>/dev/null || echo "No WAV files found"
echo ""
echo "To play the audio files on macOS:"
echo "  afplay $OUTPUT_DIR/auto_voice.wav"
echo "  afplay $OUTPUT_DIR/voice_design.wav"
echo "  afplay $OUTPUT_DIR/voice_cloning.wav"
echo ""
