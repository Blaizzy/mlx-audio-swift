# SAMAudio (Swift) Usage Guide

Swift/MLX implementation of SAM-Audio (Segment Anything Model for Audio) for text-guided source separation.

## Quick Start

```swift
import Foundation
import MLXAudioSTS

@main
struct Demo {
    static func main() async throws {
        let model = try await SAMAudio.fromPretrained("mlx-community/sam-audio-large")

        let result = try await model.separate(
            audioPaths: ["input.wav"],
            descriptions: ["speech"]
        )

        print("sampleRate:", model.sampleRate)
        print("target shape:", result.target[0].shape)
        print("residual shape:", result.residual[0].shape)
        print("peak memory (GB):", result.peakMemoryGB ?? -1)
    }
}
```

## Loading Models

`SAMAudio.fromPretrained(...)` accepts either:

- A Hugging Face repo, e.g. `mlx-community/sam-audio-large`
- A local directory containing `config.json` and one or more `.safetensors` files

```swift
let model = try await SAMAudio.fromPretrained("mlx-community/sam-audio-large")
```

For gated repos, pass a token:

```swift
let model = try await SAMAudio.fromPretrained(
    "facebook/sam-audio-large",
    hfToken: ProcessInfo.processInfo.environment["HF_TOKEN"]
)
```

## Separation APIs

### 1) `separate(...)` for short audio

Best for short clips when memory is not a concern.

```swift
let result = try await model.separate(
    audioPaths: ["input.wav"],
    descriptions: ["speech"],
    anchors: [[("+", 1.5, 3.0)]], // optional
    ode: SAMAudioODEOptions(method: .midpoint, stepSize: 2.0 / 32.0)
)
```

You can also pass pre-batched waveforms:

```swift
let result = try await model.separate(
    audios: batchedAudio, // shape: (B, 1, T)
    descriptions: prompts
)
```

### 2) `separateLong(...)` for long audio

Chunked inference with cosine crossfade stitching.

```swift
let result = try await model.separateLong(
    audioPaths: ["long_input.wav"],
    descriptions: ["speech"],
    chunkSeconds: 10.0,
    overlapSeconds: 3.0,
    ode: SAMAudioODEOptions(method: .euler, stepSize: 2.0 / 32.0)
)
```

### 3) `separateStreaming(...)` for incremental output

Generator-style:

```swift
let stream = try model.separateStreaming(
    audioPaths: ["input.wav"],
    descriptions: ["speech"],
    chunkSeconds: 10.0,
    overlapSeconds: 3.0
)

for try await chunk in stream {
    print(chunk.chunkIndex, chunk.target.shape, chunk.isLastChunk)
}
```

Callback-style:

```swift
let count = try await model.separateStreaming(
    audios: batchedAudio,
    descriptions: prompts,
    targetCallback: { audioChunk, idx, isLast in
        print("target chunk", idx, audioChunk.shape, isLast)
    }
)
print("total samples emitted:", count)
```

## Temporal Anchors

Anchor format:

- `SAMAudioAnchor = (token: String, startTime: Float, endTime: Float)`

Token meanings:

- `"+"`: target sound is present in this span
- `"-"`: target sound is not present in this span

Example:

```swift
anchors: [[("+", 1.0, 2.5), ("-", 4.0, 6.0)]]
```

## ODE Options

`SAMAudioODEOptions` controls quality/speed:

- `.midpoint` is slower and usually higher quality
- `.euler` is faster
- `stepSize` must be `0 < stepSize < 1` (default is `2/32`)

```swift
let ode = SAMAudioODEOptions(method: .midpoint, stepSize: 2.0 / 32.0)
```

## Current Limitations

- `separateLong(...)` and chunked streaming currently require batch size `1`
- Chunked methods do not currently support anchors (`chunkedAnchorsNotSupported`)
- Output arrays are mono waveforms per sample (`(samples, 1)`)

## Testing

Local integration test (no network):

```bash
swift test --filter fromPretrainedLoadsLocalFixture
```

Network-enabled integration test:

```bash
MLXAUDIO_ENABLE_NETWORK_TESTS=1 \
MLXAUDIO_SAMAUDIO_REPO=mlx-community/sam-audio-large \
swift test --filter fromPretrainedLoadsRealWeightsNetwork
```

