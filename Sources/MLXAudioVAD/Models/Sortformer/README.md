# Sortformer Speaker Diarization

Swift port of NVIDIA's Sortformer speaker diarization model. Sortformer predicts "who spoke when" by outputting per-frame speaker activity probabilities for up to 4 speakers.

[Hugging Face Model Repo](https://huggingface.co/mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16)

## Architecture

1. **FastConformer Encoder** — Conv subsampling (8x) + Conformer layers with relative positional attention
2. **Transformer Encoder** — BART-style post-LN encoder layers with positional embeddings
3. **Sortformer Modules** — Linear projection + feedforward + sigmoid output for 4 speakers

## Quick Start

```swift
import MLXAudioCore
import MLXAudioVAD

let model = try await SortformerModel.fromPretrained(
    "mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16"
)

let result = try await model.generate(audio: audioData, threshold: 0.5, verbose: true)
print(result.text)
```

## API

### `model.generate()`

Offline inference on a full audio file.

```swift
let result = try await model.generate(
    audio: audioData,         // MLXArray — 1-D audio samples
    sampleRate: 16000,        // sample rate of input audio
    threshold: 0.5,           // speaker activity threshold (0–1)
    minDuration: 0.0,         // minimum segment duration in seconds
    mergeGap: 0.0,            // max gap (seconds) to merge consecutive segments
    verbose: false            // print progress info
)
```

**Returns** a `DiarizationOutput` with:

| Field | Type | Description |
|-------|------|-------------|
| `segments` | `[DiarizationSegment]` | Speaker segments with `start`, `end`, `speaker` |
| `speakerProbs` | `MLXArray?` | Per-frame speaker probabilities `(numFrames, 4)` |
| `numSpeakers` | `Int` | Number of detected active speakers |
| `text` | `String` | RTTM-formatted output |

### `model.generateStream()`

Streaming inference that processes audio in chunks.

```swift
for try await result in model.generateStream(
    audio: audioData,         // MLXArray — full audio (chunked internally)
    chunkDuration: 5.0,       // seconds per chunk
    threshold: 0.5,
    minDuration: 0.0,
    mergeGap: 0.0,
    spkcacheMax: 188,         // max speaker cache size (diarization frames)
    fifoMax: 188,             // max FIFO buffer size (diarization frames)
    verbose: false
) {
    // each result contains segments for that chunk
}
```

### `model.feed()`

Low-level single-chunk API for real-time streaming.

```swift
var state = model.initStreamingState()
let (result, state) = try await model.feed(
    chunk: audioChunk,        // MLXArray — 1-D audio samples
    state: state,             // StreamingState
    sampleRate: 16000,
    threshold: 0.5,
    spkcacheMax: 188,
    fifoMax: 188
)
```

## Examples

### Basic diarization

```swift
import MLXAudioCore
import MLXAudioVAD

let (_, audio) = try loadAudioArray(from: audioURL)
let model = try await SortformerModel.fromPretrained(
    "mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16"
)
let result = try await model.generate(audio: audio, threshold: 0.5)

for seg in result.segments {
    print("Speaker \(seg.speaker): \(seg.start)s - \(seg.end)s")
}
```

### With post-processing

```swift
let result = try await model.generate(
    audio: audio,
    threshold: 0.4,
    minDuration: 0.25,   // ignore segments shorter than 250ms
    mergeGap: 0.5        // merge segments within 500ms of each other
)
```

### Streaming from a file

```swift
for try await result in model.generateStream(
    audio: audio,
    chunkDuration: 5.0,
    verbose: true
) {
    for seg in result.segments {
        print("Speaker \(seg.speaker): \(seg.start)s - \(seg.end)s")
    }
}
```

### Streaming from chunks

```swift
let chunkSize = Int(5.0 * Float(sampleRate))
var state = model.initStreamingState()

for start in stride(from: 0, to: audio.dim(0), by: chunkSize) {
    let end = min(start + chunkSize, audio.dim(0))
    let chunk = audio[start..<end]

    let (result, newState) = try await model.feed(
        chunk: chunk,
        state: state,
        threshold: 0.5
    )
    state = newState

    for seg in result.segments {
        print("Speaker \(seg.speaker): \(seg.start)s - \(seg.end)s")
    }
}
```

### Real-time streaming (e.g. microphone)

```swift
var state = model.initStreamingState()

for try await chunk in microphoneStream {
    let (result, newState) = try await model.feed(
        chunk: chunk,
        state: state,
        threshold: 0.5
    )
    state = newState

    for seg in result.segments {
        print("Speaker \(seg.speaker): \(seg.start)s - \(seg.end)s")
    }
}
```

### RTTM output

```swift
let result = try await model.generate(audio: audio, threshold: 0.5)
print(result.text)
// SPEAKER audio 1 0.000 3.200 <NA> <NA> speaker_0 <NA> <NA>
// SPEAKER audio 1 3.520 5.120 <NA> <NA> speaker_1 <NA> <NA>
```

## Streaming Architecture

The streaming pipeline maintains two buffers of pre-encoded embeddings:

```
[spkcache | fifo | left_ctx | new_chunk | right_ctx]
     ^         ^        ^          ^            ^
  long-term  recent  overlap    current      look-ahead
  context    context  from fifo  audio       (file mode)
```

- **Speaker Cache (spkcache)** — Long-term context, compressed when full to retain the most informative frames
- **FIFO** — Recent context buffer. Oldest frames roll into the speaker cache when the FIFO overflows
- **Left/Right Context** — Overlap frames from adjacent chunks for better boundary handling

Each streaming step encodes the full assembled sequence through the Conformer + Transformer encoders, but only emits predictions for the new chunk.

### AOSC Compression

When the speaker cache overflows, AOSC (Arrival-Order Speaker Cache) intelligently selects which frames to keep:

1. **Score** each frame per speaker using a log-likelihood ratio
2. **Filter** non-speech and overlapped-speech frames
3. **Boost** recent frames to add a recency bias
4. **Strong boost** top frames per speaker to guarantee minimum representation
5. **Weak boost** additional frames to prevent single-speaker dominance
6. **Pad** with silence slots to ensure silence is represented in the cache
7. **Select** top-K frames globally across all speakers
8. **Gather** selected embeddings, filling disabled slots with the running mean silence embedding

### Streaming Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunkDuration` | `5.0` | Seconds per chunk (file mode) |
| `spkcacheMax` | `188` | Max speaker cache size (diarization frames) |
| `fifoMax` | `188` | Max FIFO buffer size (diarization frames) |

## Notes

- Input audio is automatically converted to mono and peak-normalized
- Supports up to **4 simultaneous speakers**
- Lower `threshold` values detect more speaker activity (more sensitive, possibly noisier)
- Use `minDuration` and `mergeGap` to clean up fragmented segments
- Leading/trailing silence is automatically trimmed in offline mode
- Ported from [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) `SortformerEncLabelModel`
