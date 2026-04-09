#!/bin/bash
# Test script for OmniVoice model
# This will download and test the mlx-community/OmniVoice-bf16 model

set -e

echo "=========================================="
echo "OmniVoice Model Test"
echo "=========================================="
echo ""

# Set environment variables
export MLXAUDIO_ENABLE_NETWORK_TESTS=1
export MLXAUDIO_OMNIVOICE_REPO="mlx-community/OmniVoice-bf16"

echo "Testing OmniVoice model from: $MLXAUDIO_OMNIVOICE_REPO"
echo ""

# Check if swift is available
if ! command -v swift &> /dev/null; then
    echo "Error: swift command not found"
    exit 1
fi

# Run the tests
echo "Running OmniVoice tests..."
echo ""

swift test --filter OmniVoiceTests 2>&1 | tee omnivoice_test_output.log

echo ""
echo "=========================================="
echo "Test Complete!"
echo "=========================================="
echo ""
echo "Check the output above for test results."
echo "Generated audio files are saved in /tmp/omnivoice_*_test.wav"
echo ""
