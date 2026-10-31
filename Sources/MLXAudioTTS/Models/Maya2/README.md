# Maya-2 Native

[Maya-2 Native](https://huggingface.co/maya-research/maya2-native) generates 24 kHz mono speech in 12 languages with 20 named voices, incremental audio output, and reference-audio voice cloning.

Supported model:

- [maya-research/maya2-native](https://huggingface.co/maya-research/maya2-native)

Supported language codes: `as`, `bn`, `en`, `gu`, `hi`, `kn`, `ml`, `mr`, `or`, `pa`, `ta`, `te`.

```bash
mlx-audio-swift-tts \
  --model maya-research/maya2-native \
  --text "Hello from Maya." \
  --voice Siddhant --language en \
  --output speech.wav
```

```swift
import Foundation
import MLXAudioCore
import MLXAudioTTS

let model = try await TTS.loadModel(modelRepo: "maya-research/maya2-native")
let audio = try await model.generate(
    text: "Hello from Maya.", voice: "Siddhant",
    refAudio: nil, refText: nil, language: "en"
)
try AudioUtils.writeWavFile(
    samples: audio.asArray(Float.self), sampleRate: model.sampleRate,
    fileURL: URL(fileURLWithPath: "speech.wav")
)
```

The shared loader accepts repository IDs and local checkpoint directories. Mimi audio weights download on first model load; the speaker encoder downloads when speaker-conditioned voice cloning is first used.

## Voices and streaming

The default voice is `Ahana`. Voice names are case-insensitive: Ahana, Vaidehi, Malini, Charvi, Jahnavi, Anirban, Nakul, Vasant, Siddhant, Satvik, Abhay, Nivedita, Ishwari, Ritvik, Vrinda, Sreekar, Kayal, Shaunak, Akshat, Tanushree.

```swift
let maya = try await Maya2Model.fromPretrained("maya-research/maya2-native")
for try await event in maya.generateStream(
    text: "नमस्ते, आप कैसे हैं?", voice: "Ahana", language: "hi",
    streamingInterval: 0.32
) {
    if case .audio(let samples) = event {
        // Play or concatenate the 24 kHz mono samples.
    }
}
```

`maxTokens` limits generated audio frames; each frame represents 80 ms. `GenerateParameters` controls temperature, top-k, top-p, and seed. Use one active generation per instance; calls through the generation API are serialized.

## Voice cloning

Use a clean 3–10 second recording of one speaker. Load mono reference audio at the model's sample rate and pass its exact transcript. The reference and output languages may differ.

```swift
let (_, reference) = try loadAudioArray(
    from: URL(fileURLWithPath: "/path/to/reference.wav"), sampleRate: maya.sampleRate
)
var options = Maya2GenerationOptions()
options.referenceLanguage = "en"
let cloned = try await maya.generate(
    text: "வணக்கம், எப்படி இருக்கிறீர்கள்?", refAudio: reference,
    refText: "The exact words in the reference recording.",
    language: "ta", options: options
)
```

Omit `voice` when passing reference audio. The default `.inContextAndSpeaker` conditioning uses both speech codes and speaker identity. `.inContext` uses speech codes alone; `.speaker` uses identity alone and does not require a reference transcript. These controls are available in `Maya2GenerationOptions`, alongside acoustic sampling, presence penalty, minimum frames, and text normalization.

Text normalization is enabled by default for all 12 languages and expands numbers, decimals, currency, dates, times, and common identifiers. Set `options.normalizeText = false` for text already prepared for speech. Numeric dates use day/month/year; ISO dates and month names are also supported. Invalid dates and times are preserved. Reference transcripts are kept unchanged.

Maya-2 and Qwen3 weights use [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0). [Kyutai Mimi](https://huggingface.co/kyutai/mimi) weights use [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The [SpeechBrain ECAPA-TDNN speaker encoder](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) uses Apache 2.0. This implementation follows [mlx-audio](https://github.com/Blaizzy/mlx-audio/tree/main/mlx_audio/tts/models/maya2), under the [MIT license](https://github.com/Blaizzy/mlx-audio/blob/main/LICENSE). Normalization rules and tables derive from [Maya Research](https://github.com/MayaResearch/maya2-native), under Apache 2.0; Assamese number data includes [AI4Bhārat indic-numtowords](https://github.com/AI4Bharat/indic-numtowords), Copyright (c) 2022 AI4Bhārat, under the [MIT license](https://github.com/AI4Bharat/indic-numtowords/blob/master/LICENSE).
