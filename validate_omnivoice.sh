#!/bin/bash
# Validate OmniVoice implementation without running tests

echo "=========================================="
echo "OmniVoice Implementation Validation"
echo "=========================================="
echo ""

PASS=0
FAIL=0

# Function to check a file exists
check_file() {
    if [ -f "$1" ]; then
        echo "✓ Found: $1"
        PASS=$((PASS + 1))
    else
        echo "✗ Missing: $1"
        FAIL=$((FAIL + 1))
    fi
}

# Function to check file contains pattern
check_contains() {
    if grep -q "$2" "$1" 2>/dev/null; then
        echo "  ✓ Contains: $2"
        PASS=$((PASS + 1))
    else
        echo "  ✗ Missing: $2"
        FAIL=$((FAIL + 1))
    fi
}

echo "Checking file structure..."
echo ""

# Check OmniVoice model files
echo "1. OmniVoice Model Files:"
check_file "Sources/MLXAudioTTS/Models/OmniVoice/OmniVoiceConfig.swift"
check_file "Sources/MLXAudioTTS/Models/OmniVoice/OmniVoice.swift"
check_file "Sources/MLXAudioTTS/Models/OmniVoice/OmniVoiceGenerateParameters.swift"
check_file "Sources/MLXAudioTTS/Models/OmniVoice/README.md"
echo ""

# Check config contents
echo "2. Config Validation:"
check_contains "Sources/MLXAudioTTS/Models/OmniVoice/OmniVoiceConfig.swift" "OmniVoiceConfig"
check_contains "Sources/MLXAudioTTS/Models/OmniVoice/OmniVoiceConfig.swift" "OmniVoiceLLMConfig"
check_contains "Sources/MLXAudioTTS/Models/OmniVoice/OmniVoiceConfig.swift" "OmniVoiceAudioTokenizerConfig"
echo ""

# Check model contents
echo "3. Model Validation:"
check_contains "Sources/MLXAudioTTS/Models/OmniVoice/OmniVoice.swift" "OmniVoiceModel"
check_contains "Sources/MLXAudioTTS/Models/OmniVoice/OmniVoice.swift" "SpeechGenerationModel"
check_contains "Sources/MLXAudioTTS/Models/OmniVoice/OmniVoice.swift" "generate("
check_contains "Sources/MLXAudioTTS/Models/OmniVoice/OmniVoice.swift" "generateStream("
check_contains "Sources/MLXAudioTTS/Models/OmniVoice/OmniVoice.swift" "setGenerationConfig("
echo ""

# Check parameters contents
echo "4. Parameters Validation:"
check_contains "Sources/MLXAudioTTS/Models/OmniVoice/OmniVoiceGenerateParameters.swift" "OmniVoiceGenerateParameters"
check_contains "Sources/MLXAudioTTS/Models/OmniVoice/OmniVoiceGenerateParameters.swift" "numStep"
check_contains "Sources/MLXAudioTTS/Models/OmniVoice/OmniVoiceGenerateParameters.swift" "guidanceScale"
check_contains "Sources/MLXAudioTTS/Models/OmniVoice/OmniVoiceGenerateParameters.swift" "speed"
check_contains "Sources/MLXAudioTTS/Models/OmniVoice/OmniVoiceGenerateParameters.swift" "duration"
check_contains "Sources/MLXAudioTTS/Models/OmniVoice/OmniVoiceGenerateParameters.swift" "tShift"
check_contains "Sources/MLXAudioTTS/Models/OmniVoice/OmniVoiceGenerateParameters.swift" "denoise"
check_contains "Sources/MLXAudioTTS/Models/OmniVoice/OmniVoiceGenerateParameters.swift" "layerPenaltyFactor"
check_contains "Sources/MLXAudioTTS/Models/OmniVoice/OmniVoiceGenerateParameters.swift" "positionTemperature"
check_contains "Sources/MLXAudioTTS/Models/OmniVoice/OmniVoiceGenerateParameters.swift" "classTemperature"
echo ""

# Check TTS factory integration
echo "5. TTS Factory Integration:"
check_contains "Sources/MLXAudioTTS/TTSModel.swift" "case \"omnivoice\""
check_contains "Sources/MLXAudioTTS/TTSModel.swift" "OmniVoiceModel.fromPretrained"
check_contains "Sources/MLXAudioTTS/TTSModel.swift" "lower.contains(\"omnivoice\")"
echo ""

# Check CLI integration
echo "6. CLI Integration:"
check_contains "Sources/Tools/mlx-audio-swift-tts/App.swift" "--instruct"
check_contains "Sources/Tools/mlx-audio-swift-tts/App.swift" "--num_step"
check_contains "Sources/Tools/mlx-audio-swift-tts/App.swift" "--guidance_scale"
check_contains "Sources/Tools/mlx-audio-swift-tts/App.swift" "--speed"
check_contains "Sources/Tools/mlx-audio-swift-tts/App.swift" "--duration"
check_contains "Sources/Tools/mlx-audio-swift-tts/App.swift" "--t_shift"
check_contains "Sources/Tools/mlx-audio-swift-tts/App.swift" "--denoise"
check_contains "Sources/Tools/mlx-audio-swift-tts/App.swift" "--postprocess_output"
check_contains "Sources/Tools/mlx-audio-swift-tts/App.swift" "--layer_penalty_factor"
check_contains "Sources/Tools/mlx-audio-swift-tts/App.swift" "--position_temperature"
check_contains "Sources/Tools/mlx-audio-swift-tts/App.swift" "--class_temperature"
check_contains "Sources/Tools/mlx-audio-swift-tts/App.swift" "OmniVoiceModel"
echo ""

# Check test file
echo "7. Test File:"
check_file "Tests/OmniVoiceTests.swift"
check_contains "Tests/OmniVoiceTests.swift" "OmniVoiceConfigTests"
check_contains "Tests/OmniVoiceTests.swift" "OmniVoiceFactoryTests"
check_contains "Tests/OmniVoiceTests.swift" "OmniVoiceModelTests"
echo ""

# Syntax validation
echo "8. Syntax Validation:"
if swiftc -parse Sources/MLXAudioTTS/Models/OmniVoice/OmniVoiceConfig.swift 2>/dev/null; then
    echo "  ✓ OmniVoiceConfig.swift syntax valid"
    PASS=$((PASS + 1))
else
    echo "  ✗ OmniVoiceConfig.swift syntax error"
    FAIL=$((FAIL + 1))
fi

if swiftc -parse Sources/MLXAudioTTS/Models/OmniVoice/OmniVoice.swift 2>/dev/null; then
    echo "  ✓ OmniVoice.swift syntax valid"
    PASS=$((PASS + 1))
else
    echo "  ✗ OmniVoice.swift syntax error"
    FAIL=$((FAIL + 1))
fi

if swiftc -parse Sources/MLXAudioTTS/Models/OmniVoice/OmniVoiceGenerateParameters.swift 2>/dev/null; then
    echo "  ✓ OmniVoiceGenerateParameters.swift syntax valid"
    PASS=$((PASS + 1))
else
    echo "  ✗ OmniVoiceGenerateParameters.swift syntax error"
    FAIL=$((FAIL + 1))
fi

if swiftc -parse Sources/Tools/mlx-audio-swift-tts/App.swift 2>/dev/null; then
    echo "  ✓ CLI App.swift syntax valid"
    PASS=$((PASS + 1))
else
    echo "  ✗ CLI App.swift syntax error"
    FAIL=$((FAIL + 1))
fi

if swiftc -parse Tests/OmniVoiceTests.swift 2>/dev/null; then
    echo "  ✓ OmniVoiceTests.swift syntax valid"
    PASS=$((PASS + 1))
else
    echo "  ✗ OmniVoiceTests.swift syntax error"
    FAIL=$((FAIL + 1))
fi
echo ""

# Summary
echo "=========================================="
echo "Validation Summary"
echo "=========================================="
echo ""
echo "Passed: $PASS"
echo "Failed: $FAIL"
echo ""

if [ $FAIL -eq 0 ]; then
    echo "✅ All checks passed!"
    echo ""
    echo "Next steps:"
    echo "1. Wait for dependencies to download (if not done)"
    echo "2. Run config tests: swift test --filter OmniVoiceConfigTests"
    echo "3. Run factory tests: swift test --filter OmniVoiceFactoryTests"
    echo "4. Run full tests: MLXAUDIO_ENABLE_NETWORK_TESTS=1 swift test --filter OmniVoiceTests"
    echo ""
    echo "Or use the quick test script:"
    echo "  ./quick_test_omnivoice.sh"
    exit 0
else
    echo "❌ Some checks failed. Please review the output above."
    exit 1
fi
