# ZONOS2

ZONOS2 generates mono speech at 44.1 kHz with optional voice cloning, speaking-rate buckets, and recording-quality controls. It supports full-waveform generation, incremental audio chunks, and serial batch generation.

Supported repository:

- [amal-david/Zyphra-ZONOS2-4bit](https://huggingface.co/amal-david/Zyphra-ZONOS2-4bit), with mixed 4-bit and 8-bit weights.

The model downloads the [Descript DAC codec](https://huggingface.co/mlx-community/descript-audio-codec-44khz) when synthesis first needs it. Speaker extraction uses the encoder bundled with the checkpoint.

```sh
mlx-audio-swift-tts --model amal-david/Zyphra-ZONOS2-4bit \
  --text "Hello, this is Zonos two." --output speech.wav

mlx-audio-swift-tts --model amal-david/Zyphra-ZONOS2-4bit \
  --text "This sentence uses the reference speaker." \
  --ref_audio reference.wav --output cloned.wav
```

```swift
import Foundation
import MLX
import MLXAudioCore
import MLXAudioTTS

let model = try await Zonos2Model.fromPretrained("amal-david/Zyphra-ZONOS2-4bit")
let audio = try await model.generate(text: "Hello, this is Zonos two.")
try AudioUtils.writeWavFile(
    samples: audio.asArray(Float.self), sampleRate: model.sampleRate,
    fileURL: URL(fileURLWithPath: "speech.wav"))

// Reference samples passed to generate default to the model's 44.1 kHz rate.
let (rate, reference) = try loadAudioArray(from: URL(fileURLWithPath: "reference.wav"))
var options = Zonos2Options()
options.referenceSampleRate = rate
options.speakingRateBucket = 3
options.cleanSpeakerBackground = true
options.qualityBuckets = ["trailing_silence_s": 3, "estimated_snr": 8]
let cloned = try await model.generate(
    text: "This uses the reference speaker.", refAudio: reference,
    options: options, generationParameters: GenerateParameters(
        maxTokens: 512, temperature: 1.15, topK: 106, minP: 0.18,
        repetitionPenalty: 1.2, repetitionContextSize: 50, seed: 42))
```

`extractSpeakerEmbedding(_:sampleRate:)` accepts mono or two-dimensional channel/sample audio and returns a reusable 2048-dimensional speaker vector. Pass a fresh tensor to `speakerEmbedding:` for each call. Supply either reference audio or a precomputed embedding.

```swift
for try await event in model.generateStream(
    text: "Audio arrives in successive chunks.", streamingInterval: 0.5
) {
    if case .audio(let chunk) = event {
        // Play or collect the successive 44.1 kHz mono samples.
    }
}

let waveforms = try await model.generateBatch(texts: ["Hello.", "Good morning."])
```

`generateBatch` executes sequences serially and preserves input order. It accepts a shared `refAudio:` or `speakerEmbedding:`, or per-sequence `refAudios:` or `speakerEmbeddings:`. `referenceSampleRates:` supplies one sample rate for each per-sequence reference.

`Zonos2Options` also controls accurate mode, text normalization, repetition codebooks, and EOS handling. Speaking-rate buckets run from 0 through 7. Quality feature names and available bucket counts are exposed through `model.config.qualityFeatures` and `qualityBucketCounts`; an empty quality dictionary omits these markers. English normalization handles numbers, dates, times, currencies, units, fractions, and acronyms. Other languages retain their UTF-8 text.

The common `TTS.loadModel(modelRepo:)` API accepts this repository and local model directories. Model instances require serialized use.

ZONOS2 uses the [Apache 2.0 model license](https://huggingface.co/Zyphra/ZONOS2). The [speaker encoder](https://huggingface.co/marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B) uses Apache 2.0. Descript DAC code uses the [MIT license](https://github.com/descriptinc/descript-audio-codec/blob/main/LICENSE); see the [codec model card](https://huggingface.co/descript/dac_44khz) for the published component.
