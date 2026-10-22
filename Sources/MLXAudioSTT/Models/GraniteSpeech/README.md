# Granite Speech

IBM Granite Speech transcribes speech and translates spoken audio into text. Granite Speech 4.1 Plus also produces speaker-attributed transcripts and word timestamps. These models accept mono audio at 16 kHz.

## Supported models

| Repository ID | Features |
| --- | --- |
| [`ibm-granite/granite-speech-5.0-2b`](https://huggingface.co/ibm-granite/granite-speech-5.0-2b) | Transcription and speech translation |
| [`ibm-granite/granite-speech-4.1-2b`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b) | Transcription and speech translation |
| [`ibm-granite/granite-speech-4.1-2b-plus`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-plus) | Transcription, speech translation, speaker attribution, and word timestamps |
| [`mlx-community/granite-4.0-1b-speech-5bit`](https://huggingface.co/mlx-community/granite-4.0-1b-speech-5bit) | Transcription and speech translation, with lower memory use |

For the non-autoregressive 4.1 model, see [Granite Speech NAR](../GraniteSpeechNAR/README.md). For the encoder-only 5.0 model, see [Granite Speech TurboCTC](../GraniteSpeech5CTC/README.md).

## CLI

```bash
mlx-audio-swift-stt --model ibm-granite/granite-speech-5.0-2b --audio speech.wav --output-path transcript --format txt

mlx-audio-swift-stt --model ibm-granite/granite-speech-4.1-2b-plus --audio speech.wav --stream --output-path transcript --format txt

# Translate the speech to French.
mlx-audio-swift-stt --model ibm-granite/granite-speech-5.0-2b --audio speech.wav --language fr --output-path transcript --format txt
```

You can also pass a local model directory to `--model`.

## Swift

```swift
import Foundation
import MLXAudioCore
import MLXAudioSTT

let audioURL = URL(fileURLWithPath: "speech.wav")
let (_, audio) = try loadAudioArray(from: audioURL, sampleRate: 16000)
let model = try await GraniteSpeechModel.fromPretrained(
    "ibm-granite/granite-speech-5.0-2b"
)

let output = model.generate(audio: audio)
print(output.text)

let translation = model.generate(audio: audio, language: "fr")
print(translation.text)
```

To guide transcription, supply a source language, keywords, or a custom prompt:

```swift
let output = model.generate(
    audio: audio,
    sourceLanguage: "English",
    hotwords: ["MLX", "Granite"],
    punctuate: true
)
print(output.text)
```

### Speaker attribution and timestamps

Use Granite Speech 4.1 Plus for these tasks. Speaker tags identify turns within the clip; they do not identify real people.

```swift
let plus = try await GraniteSpeechModel.fromPretrained(
    "ibm-granite/granite-speech-4.1-2b-plus"
)

let speakers = plus.generate(audio: audio, task: .speakerAttribution)
print(speakers.text)
for segment in speakers.segments ?? [] {
    print(segment["speaker_id"]!, segment["text"]!)
}

let timed = plus.generate(audio: audio, task: .timestamps)
for segment in timed.segments ?? [] {
    for word in segment["words"] as? [[String: Any]] ?? [] {
        print(word["word"]!, word["start"]!, word["end"]!)
    }
}
```

Choose a structured task on its own, without a custom prompt, language, keywords, or punctuation controls. Word times are in seconds. The tagged text is retained in `output.text`; `output.segments` is populated when the generated tags can be parsed completely.

### Streaming

```swift
for try await event in model.generateStream(audio: audio) {
    switch event {
    case .token(let text):
        print(text, terminator: "")
    case .result(let output):
        print("\n\(output.text)")
    case .info:
        break
    }
}
```

## License

IBM Granite Speech models are released under [Apache 2.0](https://huggingface.co/ibm-granite/granite-speech-5.0-2b/blob/main/LICENSE). See each model repository for its model card and license.
