# Audio8 TTS

Audio8 0.1B and 0.6B generate multilingual speech at 44.1 kHz and support voice cloning from a mono reference recording and its matching transcript.

Supported models:

| Model | Repository |
| --- | --- |
| Audio8 0.1B, Falcon-H1 hybrid | [mlx-community/Audio8-TTS-Preview-0.1b-bf16](https://huggingface.co/mlx-community/Audio8-TTS-Preview-0.1b-bf16) |
| Audio8 0.6B, attention | [mlx-community/Audio8-TTS-Preview-0.6b-bf16](https://huggingface.co/mlx-community/Audio8-TTS-Preview-0.6b-bf16) |

```swift
import Foundation
import MLXAudioCore
import MLXAudioTTS

let model = try await TTS.loadModel(
    modelRepo: "mlx-community/Audio8-TTS-Preview-0.1b-bf16"
)
let audio = try await model.generate(
    text: "Hello from Audio8.",
    voice: nil, refAudio: nil, refText: nil, language: nil
)
```

For voice cloning, load reference audio at `model.sampleRate`, pass its mono waveform as `refAudio`, and supply the exact spoken words as `refText`. Keep speaker tags such as `<|speaker:0|>` when present in the transcript.

```swift
let (_, reference) = try loadAudioArray(
    from: URL(fileURLWithPath: "/path/to/reference.wav"),
    sampleRate: model.sampleRate
)
let cloned = try await model.generate(
    text: "Today is a wonderful day.",
    voice: nil, refAudio: reference,
    refText: "The words spoken in the reference recording.", language: nil
)
```

`GenerateParameters` controls the seed, temperature, top-p, positive top-k, and maximum audio frames. `Audio8Model.generate(..., options: Audio8GenerationOptions(doSample: false))` selects greedy decoding. `generateStream` emits the completed waveform and generation statistics.

```bash
mlx-audio-swift-tts \
  --model mlx-community/Audio8-TTS-Preview-0.1b-bf16 \
  --text "Hello from Audio8." \
  --output /tmp/audio8.wav
```

The 0.1B language model uses the [Audio8 Community License v1.0](https://huggingface.co/mlx-community/Audio8-TTS-Preview-0.1b-bf16/blob/main/LICENSE). The [Audio8 0.6B model and bundled codec](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.6b) are released under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0). This implementation follows the [mlx-audio Audio8 implementation](https://github.com/Blaizzy/mlx-audio/tree/main/mlx_audio/tts/models/arktts), under the [MIT license](https://github.com/Blaizzy/mlx-audio/blob/main/LICENSE).
