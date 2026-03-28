# DeepFilterNet

MLX Swift implementation of [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet), a real-time speech enhancement model that removes background noise while preserving speech quality. Supports offline (batch) and low-latency streaming modes.

## Available Models

| Model | Versions | Description |
|-------|----------|-------------|
| [mlx-community/DeepFilterNet-mlx](https://huggingface.co/mlx-community/DeepFilterNet-mlx) | V1, V2, V3 | Contains v1/, v2/, v3/ subfolders. V3 (default) recommended. |

V2/V3 support both offline and streaming modes. V1 supports offline only.

Each version subfolder contains `config.json` and `model.safetensors` in [MLX safetensors format](https://ml-explore.github.io/mlx/build/html/usage/weights.html).

## Swift Usage

### Offline Enhancement

```swift
import MLXAudioCore
import MLXAudioSTS

let (_, audio) = try loadAudioArray(from: audioURL, sampleRate: 48_000)

// Default: loads V3 from mlx-community/DeepFilterNet-mlx
let model = try await DeepFilterNetModel.fromPretrained()

// Or specify a version
let modelV2 = try await DeepFilterNetModel.fromPretrained(subfolder: "v2")

// Or load from a local path
let modelLocal = try await DeepFilterNetModel.fromPretrained("/path/to/DeepFilterNet3")
let enhanced = try model.enhance(audio)
// enhanced is a mono MLXArray of shape [samples] in [-1, 1]
```

### Streaming Enhancement

Process audio in real-time, one hop (10ms) at a time:

```swift
let streamer = model.createStreamer(
    config: DeepFilterNetStreamingConfig(
        padEndFrames: 3,
        compensateDelay: true
    )
)

// Feed chunks as they arrive (any size, internally buffered to hop-sized frames)
let out1 = try streamer.processChunk(chunk1)
let out2 = try streamer.processChunk(chunk2)
// ...
let tail = try streamer.flush()  // flush remaining samples
```

### Streaming via AsyncThrowingStream

```swift
for try await chunk in model.enhanceStreaming(audio) {
    // chunk.audio: MLXArray of enhanced samples
    // chunk.chunkIndex: sequential index
    // chunk.isLastChunk: true for final chunk
    playAudio(chunk.audio)
}
```

### Streaming Configuration

```swift
DeepFilterNetStreamingConfig(
    padEndFrames: 3,           // zero-pad end to flush pipeline
    compensateDelay: true,     // trim algorithmic delay from output
    enableStageSkipping: false, // LSNR-based stage skipping (experimental)
    materializeEveryHops: 512  // force eval() interval to bound lazy graph
)
```

## CLI Usage

```bash
# Offline enhancement
mlx-audio-swift-sts --model /path/to/DeepFilterNet3-MLX \
    --audio noisy.wav -o enhanced.wav --mode short

# Streaming enhancement
mlx-audio-swift-sts --model /path/to/DeepFilterNet3-MLX \
    --audio noisy.wav -o enhanced.wav --mode stream

# From HuggingFace
mlx-audio-swift-sts --model mlx-community/DeepFilterNet-mlx \
    --audio noisy.wav -o enhanced.wav
```

## Architecture

DeepFilterNet uses a dual-pathway encoder-decoder architecture:

- **ERB pathway** (32 bands): Processes broadband spectral envelope via 4 encoder conv blocks, a squeezed GRU bottleneck, and 4 decoder transpose-conv blocks. Outputs a per-band gain mask.
- **DF pathway** (96 bins): Processes low-frequency complex spectrum via 2 encoder conv blocks, a GRU, and a deep-filtering decoder. Outputs complex filter coefficients applied over a sliding window of `dfOrder` frames.
- **Deep filtering**: Applies learned complex FIR coefficients to the low-frequency spectrum for fine-grained noise suppression.
- **GRU optimization**: Recurrent layers use Accelerate-optimized CPU inference (`vDSP_mmul`) with batch GPU input projection, avoiding Metal kernel dispatch overhead in the sequential hidden-state loop.

Input: 48kHz mono audio. Hop size: 480 samples (10ms). FFT size: 960.

## Performance

On Apple M-series silicon with 10s of 48kHz audio:

| Mode | Latency | Real-time Factor |
|------|---------|-----------------|
| Offline | ~0.23s | ~43x real-time |
| Streaming | ~4.8ms/hop | ~2x headroom within 10ms budget |

Streaming peak memory: ~80MB RSS.

## License

DeepFilterNet model weights are subject to the [original DeepFilterNet license](https://github.com/Rikorose/DeepFilterNet/blob/main/LICENSE). This Swift implementation is part of mlx-audio-swift.
