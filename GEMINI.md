# Gemini Instructions

Read and follow all instructions in [AGENTS.md](AGENTS.md) before starting any task.

## Additional Rules for Gemini

- **Never use `swift build` or `swift test`**. Always use `xcodebuild` with `-scheme MLXAudio-Package -destination 'platform=macOS'`.
- **Never expose secrets**. Do not echo environment variables, API keys, or `.env` file contents. Use existence checks only (`test -n "$VAR"`).
- **CI runner must be `macos-26`** or later. Never use `macos-latest`, `macos-15`, etc.
- **Commit to `development`**, never directly to `main`. PRs go `development` → `main`.
- When modifying CI workflows, update branch protection rules in concert using `gh api`.
- Run unit tests (no model downloads) after code changes to verify nothing is broken:
  ```bash
  xcodebuild test -scheme MLXAudio-Package -destination 'platform=macOS' \
    -only-testing:MLXAudioTests/VocosTests \
    -only-testing:MLXAudioTests/EncodecTests \
    -only-testing:MLXAudioTests/DACVAETests \
    -only-testing:MLXAudioTests/GLMASRModuleSetupTests \
    -only-testing:MLXAudioTests/Qwen3ASRModuleSetupTests \
    -only-testing:MLXAudioTests/ForceAlignProcessorTests \
    -only-testing:MLXAudioTests/ForcedAlignResultTests \
    -only-testing:MLXAudioTests/Qwen3ASRHelperTests \
    -only-testing:MLXAudioTests/SplitAudioIntoChunksTests \
    -only-testing:MLXAudioTests/Qwen3TTSSpeechTokenizerTests \
    -only-testing:MLXAudioTests/Qwen3TTSSpeechTokenizerEncodeTests \
    -only-testing:MLXAudioTests/Qwen3TTSSpeechTokenizerWeightTests \
    -only-testing:MLXAudioTests/Qwen3TTSLanguageTests \
    -only-testing:MLXAudioTests/Qwen3TTSConfigTests \
    -only-testing:MLXAudioTests/Qwen3TTSRoutingTests \
    -only-testing:MLXAudioTests/Qwen3TTSPrepareBaseInputsTests \
    -only-testing:MLXAudioTests/Qwen3TTSGenerateCustomVoiceTests \
    -only-testing:MLXAudioTests/Qwen3TTSSpeakerEncoderTests \
    -only-testing:MLXAudioTests/Qwen3TTSSpeakerEncoderWeightTests \
    -only-testing:MLXAudioTests/Qwen3TTSSpeakerEmbeddingTests \
    -only-testing:MLXAudioTests/Qwen3TTSPrepareICLInputsTests \
    -only-testing:MLXAudioTests/Qwen3TTSGenerateICLTests \
    -only-testing:MLXAudioTests/Qwen3TTSSpeakerEncoderSmokeTests \
    CODE_SIGNING_ALLOWED=NO
  ```
