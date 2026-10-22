# Granite Speech NAR

IBM Granite Speech 4.1 NAR transcribes a complete audio clip using a non-autoregressive transcript editor. It produces text in a single editing pass and accepts mono audio at 16 kHz. Use [Granite Speech](../GraniteSpeech/README.md) for speech translation, speaker attribution, or word timestamps.

## Supported models

| Repository ID | Description |
| --- | --- |
| [`ibm-granite/granite-speech-4.1-2b-nar`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-nar) | Original IBM checkpoint |
| [`mlx-community/granite-speech-4.1-2b-nar-mlx`](https://huggingface.co/mlx-community/granite-speech-4.1-2b-nar-mlx) | MLX checkpoint |
| [`mlx-community/granite-speech-4.1-2b-nar-mlx-5bit`](https://huggingface.co/mlx-community/granite-speech-4.1-2b-nar-mlx-5bit) | Lower-memory MLX checkpoint |

## CLI

```bash
mlx-audio-swift-stt --model ibm-granite/granite-speech-4.1-2b-nar --audio speech.wav --output-path transcript --format txt

# Local model directories are also supported.
mlx-audio-swift-stt --model /path/to/granite-speech-4.1-2b-nar --audio speech.wav --output-path transcript --format txt
```

## Swift

```swift
import Foundation
import MLXAudioCore
import MLXAudioSTT

let audioURL = URL(fileURLWithPath: "speech.wav")
let (_, audio) = try loadAudioArray(from: audioURL, sampleRate: 16000)
let model = try await GraniteSpeechNARModel.fromPretrained(
    "ibm-granite/granite-speech-4.1-2b-nar"
)
let output = model.generate(audio: audio)
print(output.text)
```

The shared loader selects the model from its configuration:

```swift
let model = try await STT.loadModel(
    modelRepo: "mlx-community/granite-speech-4.1-2b-nar-mlx-5bit"
)
print(model.generate(audio: audio).text)
```

`generateStream(audio:)` emits the completed transcript after the editing pass. NAR does not generate text token by token, and autoregressive controls such as temperature and maximum generated tokens do not change its decoding.

## License

Granite Speech NAR is released under [Apache 2.0](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-nar). See the model repository for its model card and license.
