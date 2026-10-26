# VibeVoice TTS

Full VibeVoice turns text into 24 kHz mono speech. Use a reference recording to condition the speaker's voice. This is the full-model edition, distinct from VibeVoice Realtime.

Text-only generation is also available, with more variable pronunciation and voice quality.

Supported checkpoint: [microsoft/VibeVoice-1.5B](https://huggingface.co/microsoft/VibeVoice-1.5B).

```swift
import Foundation
import MLXAudioCore
import MLXAudioTTS

let model = try await TTS.loadModel(modelRepo: "microsoft/VibeVoice-1.5B")
let (_, referenceAudio) = try loadAudioArray(
    from: URL(fileURLWithPath: "reference.wav"), sampleRate: model.sampleRate
)
let audio = try await model.generate(
    text: "Native speech is working.", voice: nil,
    refAudio: referenceAudio, refText: nil, language: nil,
    generationParameters: GenerateParameters(maxTokens: 300, seed: 42)
)
```

The shared `refAudio` argument accepts mono samples at 24 kHz. The concrete model also accepts a reference sample rate and resamples internally:

```swift
let model = try await VibeVoiceTTSModel.fromPretrained("microsoft/VibeVoice-1.5B")
let (referenceRate, referenceSamples) = try loadAudioArray(from: URL(fileURLWithPath: "reference.wav"))
let audio = try await model.generate(
    text: "A familiar voice reads this sentence.",
    referenceAudio: [referenceSamples], referenceSampleRate: referenceRate,
    options: .init(guidanceScale: 1.5, diffusionSteps: 20),
    generationParameters: .init(maxTokens: 300, seed: 42)
)
```

Scripts may use `Speaker 1:` and `Speaker 2:` labels. Supply one reference recording per speaker, ordered by first appearance.

`generateStream` returns one completed waveform for this full model.

```bash
mlx-audio-swift-tts --model microsoft/VibeVoice-1.5B --text "Native speech is working." --ref_audio reference.wav --output hello.wav
```

VibeVoice code and weights use the [MIT license](https://github.com/microsoft/VibeVoice/blob/main/LICENSE). This implementation follows [mlx-audio](https://github.com/Blaizzy/mlx-audio), also [MIT licensed](https://github.com/Blaizzy/mlx-audio/blob/main/LICENSE). The language tokenizer comes from [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B), under [Apache 2.0](https://huggingface.co/Qwen/Qwen2.5-1.5B/blob/main/LICENSE).
