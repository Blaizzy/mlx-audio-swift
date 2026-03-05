# DeepFilterNet (Swift / MLX)

Native Swift MLX inference runtime for DeepFilterNet speech enhancement.

## Model Files

Use a model directory containing:

- `config.json`
- `model.safetensors` (or any `.safetensors` file)

## Usage

```swift
let model = try await DeepFilterNetModel.fromPretrained("/path/to/DeepFilterNet3")
let enhanced = try model.enhance(audio)
```

`audio` is a mono `MLXArray` of shape `[samples]` in `[-1, 1]`.

Streaming API:

```swift
let streamer = model.createStreamer(
    config: DeepFilterNetStreamingConfig(padEndFrames: 3, compensateDelay: true)
)

let outA = try streamer.processChunk(chunkA)
let outB = try streamer.processChunk(chunkB)
let tail = try streamer.flush()
```
