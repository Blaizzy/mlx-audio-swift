# MeloTTS English

MeloTTS generates 44.1 kHz English speech using preset voices, BERT prosody features, VITS synthesis, and a HiFi-GAN vocoder. It supports speed and noise controls and audio streaming.

Supported checkpoint: [mlx-community/MeloTTS-English-MLX](https://huggingface.co/mlx-community/MeloTTS-English-MLX).

Voices: `EN-US`, `EN-BR`, `EN_INDIA`, `EN-AU`, and `EN-Default`.

```sh
mlx-audio-swift-tts --model mlx-community/MeloTTS-English-MLX \
  --text "Hello, this is Melo speaking." --voice EN-US --output hello.wav
```

```swift
import Foundation
import MLXAudioCore
import MLXAudioTTS

let model = try await MeloTTSModel.fromPretrained("mlx-community/MeloTTS-English-MLX")
let audio = try await model.generate(
    text: "Hello, this is Melo speaking.",
    voice: "EN-US",
    options: .init(speed: 1.1, noiseScale: 0.667, stochasticDurationRatio: 0.2),
    generationParameters: .init(seed: 42)
)
try AudioUtils.writeWavFile(
    samples: audio.asArray(Float.self), sampleRate: model.sampleRate,
    fileURL: URL(fileURLWithPath: "hello.wav")
)
```

The shared loader and streaming API are also available:

```swift
let model = try await TTS.loadModel(modelRepo: "mlx-community/MeloTTS-English-MLX")
for try await event in model.generateStream(
    text: "Streaming English speech.", voice: "EN-AU",
    refAudio: nil, refText: nil, language: "EN",
    generationParameters: .init(seed: 42), streamingInterval: 0.5
) {
    if case .audio(let chunk) = event {
        // Consume each chunk at model.sampleRate.
    }
}
```

English frontend resources download into the caller's Hugging Face cache on first load. For offline use, pass `frontendDirectory` to `fromModelDirectory`, or place `vocab.txt`, `checkpoint20.npz`, `homographs.en`, `cmudict.zip`, and `averaged_perceptron_tagger_eng.zip` in the model's `frontend` folder. Input text must fit within 510 BERT wordpieces. The model uses preset voices; voice cloning is unavailable.

MeloTTS [code](https://github.com/myshell-ai/MeloTTS/blob/main/LICENSE) and the [published weights](https://huggingface.co/myshell-ai/MeloTTS-English) use the MIT license. The native port follows [mlx-audio](https://github.com/Blaizzy/mlx-audio/blob/main/LICENSE). English pronunciation follows Kyubyong Park and Jongseok Kim's [g2p_en](https://github.com/Kyubyong/g2p/blob/master/LICENSE), with the [CMU dictionary](https://github.com/nltk/nltk_data/blob/gh-pages/packages/corpora/cmudict.xml), [NLTK English tagger](https://github.com/nltk/nltk_data/blob/gh-pages/packages/taggers/averaged_perceptron_tagger_eng.xml), and [BERT base uncased](https://huggingface.co/google-bert/bert-base-uncased) resources under their respective licenses.
