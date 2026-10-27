# TADA

[TADA](https://huggingface.co/collections/HumeAI/tada) generates 24 kHz speech conditioned on a reference recording and its transcript. It combines a Llama backbone with a diffusion head that predicts acoustic features and token durations.

Supported checkpoints:

- [HumeAI/mlx-tada-1b](https://huggingface.co/HumeAI/mlx-tada-1b)
- [HumeAI/mlx-tada-3b](https://huggingface.co/HumeAI/mlx-tada-3b)

The compatible [Llama 3.2 tokenizer](https://huggingface.co/mlx-community/Llama-3.2-1B-Instruct-bf16) is downloaded automatically into the caller's Hugging Face cache. A local tokenizer directory can also be supplied to `fromModelDirectory`.

```sh
mlx-audio-swift-tts --model HumeAI/mlx-tada-1b \
  --text "Hello, welcome to TADA." \
  --ref_audio reference.wav --ref_text "The exact reference transcript." \
  --output hello.wav

mlx-audio-swift-tts --model HumeAI/mlx-tada-3b \
  --text "This sentence uses the reference voice." \
  --ref_audio reference.wav --ref_text "The exact reference transcript." \
  --output conditioned.wav
```

TADA requires reference audio and its matching transcript for synthesis. Use a clean recording of one speaker, preferably several seconds long. The CLI resamples reference files automatically. Swift waveform tensors passed through `SpeechGenerationModel` must be mono at 24 kHz; `encodeReference` accepts an explicit source sample rate.

```swift
import Foundation
import MLXAudioCore
import MLXAudioTTS

let model = try await TADAModel.fromPretrained("HumeAI/mlx-tada-1b")
var options = TADAGenerationOptions()
options.flowSteps = 20
options.acousticGuidance = 1.6
options.speed = 1.2

let (_, referenceAudio) = try loadAudioArray(
    from: URL(fileURLWithPath: "reference.wav"), sampleRate: 24000)
let reference = try model.encodeReference(
    audio: referenceAudio, text: "The exact reference transcript.")
let conditioned = try await model.generate(
    text: "This sentence uses the reference voice.", reference: reference,
    parameters: .init(maxTokens: 1024, seed: 42), options: options)

for try await event in model.generateStream(
    text: "Streaming speech arrives in consecutive audio chunks.",
    reference: reference, options: options, streamingInterval: 1) {
    if case .audio(let chunk) = event {
        // Consume this mono 24 kHz chunk.
    }
}
```

The generic `TTS.loadModel(modelRepo:)` API accepts these repository IDs and local nested checkpoint directories. Flow steps, acoustic and duration guidance, guidance/time schedules, initial-noise temperature, transition steps, and two-pass speed control are available through `TADAGenerationOptions`.

The original [Hume AI TADA implementation](https://github.com/HumeAI/tada/blob/main/LICENSE_CODE) and codec are available under the MIT license. Model weights follow the [Llama 3.2 Community License](https://huggingface.co/meta-llama/Llama-3.2-1B/blob/main/LICENSE). See the [TADA model card](https://huggingface.co/HumeAI/mlx-tada-1b) and [codec model card](https://huggingface.co/HumeAI/tada-codec) for their respective terms.
