# LongCat AudioDiT

LongCat AudioDiT generates mono speech at 24 kHz with a UMT5 text encoder, conditional flow matching, and a waveform VAE. It supports English and Chinese text, voice conditioning from a mono reference recording, classifier-free guidance (CFG), adaptive projected guidance (APG), and overlapping audio chunks during decoding. Audio generation is limited to 30 seconds including the reference recording.

## Supported models

- [mlx-community/LongCat-AudioDiT-1B-bf16](https://huggingface.co/mlx-community/LongCat-AudioDiT-1B-bf16)
- [mlx-community/LongCat-AudioDiT-1B-4bit](https://huggingface.co/mlx-community/LongCat-AudioDiT-1B-4bit)
- [mlx-community/LongCat-AudioDiT-1B-mxfp4](https://huggingface.co/mlx-community/LongCat-AudioDiT-1B-mxfp4)

The [google/umt5-base tokenizer](https://huggingface.co/google/umt5-base) is downloaded automatically when the model directory does not include `tokenizer.json` or `tokenizer/tokenizer.json`. Local model directories with that file also work offline.

## Swift

```swift
import MLXAudioCore
import MLXAudioTTS

let model = try await TTS.loadModel(
    modelRepo: "mlx-community/LongCat-AudioDiT-1B-bf16"
)
let audio = try await model.generate(
    text: "Hello, this is Long Cat.",
    voice: nil, refAudio: nil, refText: nil, language: nil,
    generationParameters: GenerateParameters(seed: 1024)
)
// Mono samples at model.sampleRate (24,000 Hz).
```

Use the model API to choose diffusion steps and guidance. Reference audio must contain mono samples at the model's sample rate; supply its transcript to align the text conditioning.

```swift
let model = try await LongCatModel.fromPretrained(
    "mlx-community/LongCat-AudioDiT-1B-bf16"
)
let (_, reference) = try loadAudioArray(from: referenceURL, sampleRate: model.sampleRate)
let audio = try await model.generate(
    text: "Today is warm turning to rain.",
    refAudio: reference,
    refText: "The words spoken in the reference recording.",
    options: LongCatGenerationOptions(steps: 16, guidanceScale: 4, guidance: .apg)
)
```

`generateStream` runs diffusion first, then yields overlapping VAE audio chunks and generation information. Chunk duration and overlap can be set through `LongCatGenerationOptions`.

```swift
for try await event in model.generateStream(
    text: "This speech is decoded in overlapping chunks.",
    options: LongCatGenerationOptions(chunkSeconds: 2, overlapSeconds: 0.5)
) {
    if case .audio(let chunk) = event {
        // Queue chunk for playback at model.sampleRate.
    }
}
```

## CLI

```bash
mlx-audio-swift-tts --model mlx-community/LongCat-AudioDiT-1B-bf16 \
  --text "Hello, this is Long Cat." --output longcat.wav

mlx-audio-swift-tts --model mlx-community/LongCat-AudioDiT-1B-bf16 \
  --text "Today is warm turning to rain." \
  --ref_audio reference.wav --ref_text "The words in the recording." \
  --output cloned.wav
```

## License and attribution

LongCat AudioDiT is developed by [Meituan LongCat](https://github.com/meituan-longcat/LongCat-AudioDiT). The original implementation is Copyright (c) 2026 Meituan. Its [model and code use the MIT license](https://github.com/meituan-longcat/LongCat-AudioDiT/blob/main/LICENSE). UMT5 is developed by Google and distributed under [Apache 2.0](https://huggingface.co/google/umt5-base). The Swift implementation follows the [MLX Audio reference](https://github.com/Blaizzy/mlx-audio/tree/main/mlx_audio/tts/models/longcat_audiodit).
