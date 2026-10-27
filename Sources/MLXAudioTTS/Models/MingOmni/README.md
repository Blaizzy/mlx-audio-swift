# Ming Omni speech and audio generation

Ming Omni 16.8B/A3B uses the original Bailing mixture-of-experts backbone and produces 44.1 kHz mono audio. It supports speech, reference speaker conditioning, acoustic continuation, delivery instructions, sound events, and music descriptions.

Supported checkpoint:

- [mlx-community/Ming-omni-tts-16.8B-A3B-4bit](https://huggingface.co/mlx-community/Ming-omni-tts-16.8B-A3B-4bit)

```bash
mlx-audio-swift-tts \
  --model mlx-community/Ming-omni-tts-16.8B-A3B-4bit \
  --text "Hello world." --output ming-omni.wav
```

```swift
import MLXAudioCore
import MLXAudioTTS

let model = try await TTS.loadModel(
    modelRepo: "mlx-community/Ming-omni-tts-16.8B-A3B-4bit")
let audio = try await model.generate(
    text: "Hello world.", voice: nil,
    refAudio: nil, refText: nil, language: "en",
    generationParameters: GenerateParameters(temperature: 0, seed: 42))
```

Use `MingOmniModel` for model controls. Reference audio must be mono at 44.1 kHz. Supply a clean recording and its exact transcript for acoustic continuation. Speaker conditioning is optional: set `automaticSpeakerEmbedding = true` to extract a speaker vector, or supply `speakerEmbedding` explicitly. Speaker-conditioned generation can repeat phrases or lose text adherence; assess the output for your reference.

```swift
import Foundation

let model = try await MingOmniModel.fromPretrained(
    "mlx-community/Ming-omni-tts-16.8B-A3B-4bit")
model.options.instruction = "Speak warmly, with medium pitch."
let (_, reference) = try loadAudioArray(
    from: URL(fileURLWithPath: "reference.wav"), sampleRate: model.sampleRate)
let audio = try await model.generate(
    text: "Good morning.", voice: nil,
    refAudio: reference, refText: "The exact words spoken in the reference recording.", language: "en",
    generationParameters: GenerateParameters(temperature: 0, seed: 42))
```

`options` also exposes `cfgScale`, `sigma`, `flowSteps`, a precomputed 192-value `speakerEmbedding`, and `zeroSpeakerEmbedding`. For sound events, set `options.prompt` to `"Please generate audio events based on given text.\n"`. For music, use `"Please generate music based on the following description.\n"` and text containing `Genre:`, `Mood:`, `Instrument:`, `Theme:`, and `Duration:` fields. These controls guide the learned model; requested duration is approximate.

`generateStream` yields successive waveform chunks with interpolation lookahead. `maxTokens` bounds generated latent patches. Model instances require serialized use; create a fresh reference tensor for each generation call. Load a downloaded model folder through `TTS.loadModel(modelRepo:)` or `MingOmniModel.fromModelDirectory(_:)`.

The model weights use [Apache 2.0](https://huggingface.co/inclusionAI/Ming-omni-tts-16.8B-A3B). See the [Ming project](https://github.com/inclusionAI/Ming), [MLX-Audio](https://github.com/Blaizzy/mlx-audio), and [CAMPPlus / 3D-Speaker](https://github.com/modelscope/3D-Speaker) for model and component attribution.
