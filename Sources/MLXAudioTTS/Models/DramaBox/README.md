# DramaBox

Resemble AI [DramaBox](https://huggingface.co/ResembleAI/Dramabox) as a `SpeechGenerationModel`: a scene prompt is encoded by a **headless Gemma 3 12B** backbone (all hidden states, no chat template), then an LTX flow-matching DiT, audio VAE, and BigVGAN+BWE vocoder render **48 kHz stereo**.

This is not “Gemma generates tokens, then a vocoder speaks.” The Gemma checkpoint is an encoder only.

## Weights

Supported pair (App Automaton MLX conversion). These are **not** interchangeable with `mlx-community/ResembleAI-Dramabox`.

| Role | Hugging Face repo |
|------|-------------------|
| Audio stack (DiT + VAE + vocoder + connector) | [`appautomaton/dramabox-tts-3.3b-bf16-mlx`](https://huggingface.co/appautomaton/dramabox-tts-3.3b-bf16-mlx) |
| Text encoder | [`appautomaton/gemma-3-12b-it-backbone-4bit-mlx`](https://huggingface.co/appautomaton/gemma-3-12b-it-backbone-4bit-mlx) |

Loaders accept a local directory or a Hugging Face repo id. If a Hub snapshot is already on disk (`~/.cache/huggingface/hub/models--org--name/snapshots/...`), that is used. Downloads go into that same Hub cache.

## License

DramaBox weights are under the **LTX-2 Community License**. The Gemma 3 backbone is under Google Gemma terms. Do not relicense either as MIT.

RE-USE / SEMamba voice-ref denoising (`denoiseRef = true`) is **out of scope for v1**.

## Memory

mlx-speech documents ~15.7 GB persistent / ~17.3 GB peak. Target: macOS Apple Silicon with ~32 GB unified memory. English only. 48 kHz stereo `[2, T]`.

## Prompt style

Stage directions outside quotes; spoken text inside quotes:

```
A woman speaks clearly, "The weather today will be sunny."
```

Optional ~10 s voice reference for cloning.

## Swift

```swift
import MLXAudioTTS
import MLXAudioCore

let model = try await TTS.loadModel(
    modelRepo: "appautomaton/dramabox-tts-3.3b-bf16-mlx"
)

let audio = try await model.generate(
    text: #"A woman speaks clearly, "The weather today will be sunny.""#,
    voice: nil,
    refAudio: nil,
    refText: nil,
    language: "English",
    generationParameters: GenerateParameters()
)
// audio shape [2, T] @ 48 kHz — do not downmix
try saveAudioArray(audio, sampleRate: Double(model.sampleRate), to: outputURL)
```

Rich path on the concrete type:

```swift
let model = try await DramaBoxModel.fromPretrained()
let result = try await model.generate(
    prompt: #"A woman speaks clearly, "The weather today will be sunny.""#,
    referenceAudio: nil,
    config: DramaBoxGenerateConfig(durationSeconds: 5.0)
)
```

Local directories (e.g. from mlx-speech testing):

```swift
try await DramaBoxModel.fromModelDirectory(audioDir, gemmaDir: gemmaDir)
```

### Defaults (match mlx-speech)

| Knob | Default |
|------|---------|
| `cfgScale` | 2.5 |
| `stgScale` | 1.5 (block 29; `0` skips the extra forward) |
| `rescaleScale` | `nil` = auto → 0.3 when cfg=2.5 |
| `modalityScale` | 1.0 |
| `steps` | 30 |
| `seed` | 42 |
| `denoiseRef` | false |
| output | 48 kHz stereo |

`GenerateParameters` AR fields (`maxTokens`, `temperature`, `topP`, …) are ignored. Do not overload `maxTokens` as duration.

## CLI

```bash
swift run mlx-audio-swift-tts \
  --model appautomaton/dramabox-tts-3.3b-bf16-mlx \
  --text 'A woman speaks clearly, "The weather today will be sunny."' \
  --output dramabox.wav
```
