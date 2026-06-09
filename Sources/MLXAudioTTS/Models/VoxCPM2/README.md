# VoxCPM2

OpenBMB VoxCPM2 text-to-speech support for MLX Swift.

The implementation is adapted from the VoxCPM2TTS module in
[soniqo/speech-swift](https://github.com/soniqo/speech-swift), which is
licensed under Apache-2.0, and integrated with this package's model loading
and generation APIs.

## Usage

```swift
import MLXAudioCore
import MLXAudioTTS

let model = try await TTS.loadModel(
    modelRepo: "aufklarer/VoxCPM2-MLX-bf16"
)

let audio = try await model.generate(
    text: "Hello from VoxCPM2.",
    voice: nil,
    refAudio: nil,
    refText: nil,
    language: nil
)

try AudioFileUtils.save(
    audio: audio,
    sampleRate: model.sampleRate,
    fileURL: URL(fileURLWithPath: "/tmp/voxcpm2.wav")
)
```

Local MLX checkpoints are supported as well:

```swift
let model = try await TTS.loadModel(
    modelRepo: "/path/to/VoxCPM2-8bit"
)
```

## Notes

- `voice` is passed as VoxCPM2 instruction text.
- `refAudio` and `refText` are mapped to VoxCPM2 reference conditioning.
- VoxCPM2 checkpoints are detected from `model_type`, `architecture`, or repository/path names containing `voxcpm2`.
