# Whisper Short Audio Padding Research

**Date**: 2026-01-09
**Problem**: Whisper outputs "..." and EOT for audio <10s padded to 30s

## Executive Summary

Whisper's silence detection mechanism causes transcription failure for short audio (<10s) padded to 30s. The model interprets excessive padding as "no speech" and triggers early termination. **The proven solution is VAD-based chunking** - not padding strategies.

**Confidence Level**: High (multiple production implementations confirm this approach)

---

## Root Cause Analysis

### Why Padding Fails

1. **Fixed 30s Context Window**: Whisper expects 30s of mel spectrogram input (3000 frames at 10ms intervals)

2. **Silence Detection**: The `SuppressBlank` filter zeroes probability for EOT when no real content has been sampled. However, when the model's attention sees mostly zeros (silence), it predicts high `no_speech_prob`

3. **The "..." Pattern**: When decoder sees speech followed by extended silence:
   - Cross-attention weights focus on real audio region
   - After content exhausted, model detects "no more speech"
   - Token 1097 ("...") indicates continuation/trailing content
   - EOT follows immediately

4. **Loop Detection**: Whisper suppresses repetitive content (compression ratio > 2.4). Our repeat-padding strategy triggers this.

### Evidence from Production Systems

> "If the model detects silence within one of these 30s segments the transcription will be terminated for that segment" - [OpenAI Community](https://community.openai.com/t/whisper-asr-model-skipping-chunks-in-audio-transcription/1067744)

> "When the last chunk is short...the model seems to have a problem with disambiguation and starts 'seeing things'" - [GitHub Discussion #679](https://github.com/openai/whisper/discussions/679)

---

## Production Solutions

### 1. VAD-Based Chunking (Recommended)

**Approach**: Don't pad short audio - only transcribe speech segments detected by VAD

**Implementations**:
- [WhisperX + Silero-VAD](https://github.com/m-bain/whisperX/pull/888)
- [faster-whisper batched](https://github.com/SYSTRAN/faster-whisper)
- [WhisperWithVAD](https://github.com/ANonEntity/WhisperWithVAD)

**Key Parameters**:
```python
vad_parameters=dict(
    min_silence_duration_ms=500,  # Merge gaps < 500ms
    min_speech_duration_ms=250,   # Ignore speech < 250ms
    max_speech_duration_s=30,     # Split at 30s boundaries
)
```

**How It Works**:
1. Run Silero-VAD to detect speech boundaries
2. Extract speech segments with ~100ms padding on each side
3. Concatenate segments until reaching ~30s
4. Send concatenated chunks to Whisper (mostly speech, minimal silence)
5. Map predicted timestamps back to original audio positions

### 2. Attention Mask (Partial Solution)

**Approach**: Tell the model which positions contain real data

```python
processor(
    audio_sample["array"],
    sampling_rate=16000,
    padding="max_length",
    return_attention_mask=True  # CRITICAL
)
```

> "For Whisper models, attention_mask should always be passed for batched inference" - [HuggingFace Docs](https://huggingface.co/docs/transformers/en/model_doc/whisper)

**Limitation**: Helps encoder focus but doesn't fully prevent decoder from seeing "no speech" in padded region

### 3. Decoder Parameter Tuning (Mitigation Only)

**Recommended Settings**:
```python
transcribe(
    condition_on_previous_text=False,  # Disable context carryover
    no_speech_threshold=0.1,           # More sensitive silence detection
    compression_ratio_threshold=1.35,  # Catch repetitive hallucinations
    temperature=0,                      # Deterministic output
)
```

### 4. Silence Removal Pre-processing

**ffmpeg Approach**:
```bash
ffmpeg -i input.wav -af silenceremove=start_periods=1:stop_periods=-1:start_threshold=-50dB:stop_threshold=-50dB output.wav
```

**Limitation**: Loses timing information, not suitable for real-time streaming

---

## Why Our Current Approaches Failed

| Approach | Problem |
|----------|---------|
| Zero padding | High `no_speech_prob` → "..." + EOT |
| Repeat padding (38x) | Loop detection → suppression |
| Repeat with decay | Spectral fingerprint preserved → still detected as loop |
| Repeat 3x + noise | Noise interpreted as speech → hallucinations |

**The fundamental issue**: We're trying to make Whisper accept input it was never designed for. Whisper expects either:
- 30s of continuous speech (training data format)
- VAD-segmented speech chunks with minimal padding

---

## Recommended Solution for MLX Audio Swift

### Option A: Integrate Silero-VAD (Best)

1. **Port or wrap Silero-VAD for Swift/MLX**
2. **Pre-segment audio before WhisperSession**
3. **Only send speech segments to Whisper**

**Benefits**:
- Matches production implementations
- No Whisper model changes needed
- Works for both short and long audio
- Reduces compute (no wasted inference on silence)

### Option B: Use Attention Mask in Encoder

1. **Compute attention mask based on actual audio length**
2. **Pass mask through encoder**
3. **Use mask to guide cross-attention**

**Implementation in MLX**:
```swift
// In AudioEncoder
let audioLength = audio.shape[0]
let melFrames = audioLength / AudioConstants.hopLength
let mask = MLXArray(0..<3000).map { $0 < melFrames ? 1.0 : 0.0 }
// Apply mask to encoder output before decoder
```

**Limitation**: Requires model architecture changes, may not fully prevent decoder issues

### Option C: WhisperKit-style No-Speech Detection

1. **Sample `no_speech_prob` after initial tokens**
2. **If `no_speech_prob > 0.6`, return empty result early**
3. **Don't force transcription of detected silence**

See: [WhisperKit Issue #27](https://github.com/argmaxinc/WhisperKit/issues/27)

---

## Implementation Recommendation

**For MLX Audio Swift streaming STT**:

1. **Short-term**: Use attention mask + `no_speech_prob` detection
2. **Medium-term**: Integrate lightweight VAD (even energy-based)
3. **Long-term**: Port Silero-VAD to Swift/MLX

**For immediate fix**: Don't force transcription of audio where >50% is padding. Return early with appropriate status.

---

## Sources

1. [HuggingFace Whisper Documentation](https://huggingface.co/docs/transformers/en/model_doc/whisper)
2. [OpenAI Whisper GitHub - Discussion #679](https://github.com/openai/whisper/discussions/679)
3. [OpenAI Community - Skipping Chunks](https://community.openai.com/t/whisper-asr-model-skipping-chunks-in-audio-transcription/1067744)
4. [WhisperX Silero-VAD PR](https://github.com/m-bain/whisperX/pull/888)
5. [faster-whisper VAD Implementation](https://github.com/SYSTRAN/faster-whisper)
6. [Simul-Whisper Paper](https://arxiv.org/html/2406.10052v1)
7. [WhisperKit No-Speech Detection](https://github.com/argmaxinc/WhisperKit/issues/27)
8. [Whisper Encoder Analysis](https://gattanasio.cc/post/whisper-encoder/)
