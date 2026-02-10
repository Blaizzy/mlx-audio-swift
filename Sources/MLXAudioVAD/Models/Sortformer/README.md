# Sortformer

A speaker diarization model that detects who is speaking when in audio, supporting up to 4 simultaneous speakers.

[Hugging Face Model Repo](https://huggingface.co/mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16)

## Swift Example

### Offline Diarization

```swift
import MLXAudioCore
import MLXAudioVAD

// Load audio
let (sampleRate, audio) = try loadAudioArray(from: audioURL)

// Load model
let model = try await SortformerModel.fromPretrained(
    "mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16"
)

// Run diarization
let output = model.generate(audio: audio, threshold: 0.5, verbose: true)

for segment in output.segments {
    print("Speaker \(segment.speaker): \(segment.start)s - \(segment.end)s")
}
```

### Streaming Diarization

```swift
let stream = model.generateStream(
    audio: audio,
    chunkDuration: 5.0,
    threshold: 0.5
)

for try await output in stream {
    for segment in output.segments {
        print("Speaker \(segment.speaker): \(segment.start)s - \(segment.end)s")
    }
}
```

### Low-Level Streaming (Real-Time)

```swift
var state = model.initStreamingState()

// Feed audio chunks as they arrive
let (result, newState) = model.feed(
    chunk: audioChunk,
    state: state,
    threshold: 0.5
)
state = newState

for segment in result.segments {
    print("Speaker \(segment.speaker): \(segment.start)s - \(segment.end)s")
}
```

## Output Format

The model outputs `DiarizationOutput` containing:
- `segments` - Array of `DiarizationSegment` (start time, end time, speaker ID)
- `numSpeakers` - Number of detected speakers
- `text` - RTTM-formatted output string

## Notes

- Supports up to **4 simultaneous speakers**
- Offline and streaming modes produce consistent results
- Streaming uses AOSC (Adaptive Online Speaker Cache) compression for long audio
- Audio is automatically trimmed of leading/trailing silence
- Default threshold of `0.5` works well for most audio; lower values increase sensitivity
