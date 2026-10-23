# Nemotron VoiceChat Codec

The neural audio codec used by NVIDIA NemotronLabs VoiceChat encodes and
reconstructs mono audio at 22,050 Hz using 31 residual codebooks. Each code frame
represents 80 ms of audio. It can run independently of the VoiceChat language
model and supports incremental waveform decoding.

## Supported models

- `mlx-community/NemotronLabs-VoiceChat-11B-4bit`
- `mlx-community/NemotronLabs-VoiceChat-11B-8bit`
- `mlx-community/NemotronLabs-VoiceChat-11B-bf16`
- `nvidia/NVIDIA-NemotronLabs-VoiceChat-11B`

The codec retains the precision stored in each checkpoint. The 4-bit and 8-bit
names describe the parent language model. Loading a codec selects only its
tensors; it does not load or run the language model. Local codec-only directories
and parent checkpoint directories are also supported.

## Swift

```swift
import Foundation
import MLX
import MLXAudioCodecs
import MLXAudioCore

let codec = try await NemotronVoiceChatCodec.fromPretrained(
    "mlx-community/NemotronLabs-VoiceChat-11B-4bit"
)

let (_, audio) = try loadAudioArray(
    from: URL(fileURLWithPath: "input.wav"),
    sampleRate: codec.sampleRate
)
let waveform = audio.reshaped([1, -1, 1])
let encoded = codec.encodeAudio(waveform)
let reconstructed = codec.decodeAudio(encoded)
// reconstructed has the original waveform length and layout.
```

For incremental decoding, keep one cache per audio stream. Native codes have
shape `(batch, codebooks, frames)` and native output is `(batch, 1, samples)`:

```swift
let cache = NemotronVoiceChatCodecCache()
let frames = encoded.codes.dim(2)
var chunks: [MLXArray] = []
for frame in 0..<frames {
    chunks.append(codec.decodeStep(
        encoded.codes[0..., 0..., frame..<(frame + 1)],
        cache: cache,
        flush: frame == frames - 1
    ))
}
let streamedAudio = MLX.concatenated(chunks, axis: -1)
cache.reset()
```

Set `flush: true` on the final nonempty code chunk to emit the final overlap
tail. The streaming waveform has an initial eight-sample lookahead delay;
the final flush adds eight samples. `streamingLookaheadSamples` reports the delay
for the loaded configuration. Reset the cache before starting another utterance.
Streaming encoding is not supported.

## CLI

```sh
mlx-audio-swift-codec \
  --model mlx-community/NemotronLabs-VoiceChat-11B-4bit \
  --audio input.wav \
  --output reconstructed.wav
```

## License and attribution

Model weights are distributed under the [Open Model, Data & Weights License 1.1](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B/blob/main/LICENSE).
See the [NVIDIA model card](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B)
for model attribution and terms. The Swift implementation follows the
[mlx-audio codec](https://github.com/Blaizzy/mlx-audio/tree/main/mlx_audio/codec/models/nemotron_voicechat).
