# Higgs Audio v3 TTS

Higgs TTS 3 generates expressive, multilingual speech at 24 kHz. It supports voice references, reference transcripts, and inline emotion, style, sound-effect, and prosody controls.

Supported checkpoint: [bosonai/higgs-tts-3-4b](https://huggingface.co/bosonai/higgs-tts-3-4b). The earlier `bosonai/higgs-audio-v3-tts-4b` name redirects to this repository. Local checkpoint directories work with the same loader.

```swift
import MLXAudioCore
import MLXAudioTTS

let model = try await TTS.loadModel(modelRepo: "bosonai/higgs-tts-3-4b")
let audio = try await model.generate(
    text: "<|emotion:enthusiasm|>Welcome! It's great to hear from you.",
    voice: nil, refAudio: nil, refText: nil, language: nil,
    generationParameters: .init(maxTokens: 512, temperature: 0.7, topP: 0.95)
)
```

Supply mono reference samples at 24 kHz through `refAudio`, with the matching transcript in `refText`. `HiggsAudio3TTSModel.encodeReferenceAudio(_:)` returns reusable delayed codes; its `generate(text:references:generationParameters:)` method accepts multiple `HiggsV3Reference` values. Model instances require serialized use.

`generateStream` returns one completed audio chunk. See Boson AI's [prompting guide](https://huggingface.co/bosonai/higgs-tts-3-4b/blob/main/PROMPTING.md) for inline controls.

```bash
swift run mlx-audio-swift-tts \
  --model bosonai/higgs-tts-3-4b \
  --text "Hello! It's good to hear from you." \
  --output speech.wav
```

The model and bundled audio tokenizer are published by Boson AI under the [Higgs TTS 3 Research and Non-Commercial License](https://huggingface.co/bosonai/higgs-tts-3-4b/blob/main/LICENSE). Review its terms, including the Creator Use Grant and attribution requirements. The backbone uses [Qwen3](https://huggingface.co/Qwen/Qwen3-4B-Base), and the semantic encoder uses [HuBERT](https://huggingface.co/facebook/hubert-base-ls960).
