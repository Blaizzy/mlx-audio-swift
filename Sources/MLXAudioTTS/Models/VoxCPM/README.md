# VoxCPM 1 / 1.5

VoxCPM synthesizes English and Chinese speech as mono audio at 44.1 kHz. It supports generation with a sampled voice and voice cloning from a reference recording and its transcript.

## Supported models

- [mlx-community/VoxCPM1.5-fp16](https://huggingface.co/mlx-community/VoxCPM1.5-fp16)
- [mlx-community/VoxCPM1.5-4bit](https://huggingface.co/mlx-community/VoxCPM1.5-4bit)
- [mlx-community/VoxCPM1.5-8bit](https://huggingface.co/mlx-community/VoxCPM1.5-8bit)

The supported published releases are VoxCPM 1.5.

## Swift

```swift
import MLXAudioCore
import MLXAudioTTS

let model = try await TTS.loadModel(modelRepo: "mlx-community/VoxCPM1.5-4bit")
let audio = try await model.generate(
    text: "Hello, this is Vox CPM.",
    voice: nil, refAudio: nil, refText: nil, language: nil,
    generationParameters: GenerateParameters(maxTokens: 128)
)
// `audio` contains mono samples at model.sampleRate (44,100 Hz).
```

Voice cloning requires both reference audio and its exact transcript. Load the recording at the model's sample rate:

```swift
let model = try await VoxCPMModel.fromPretrained("mlx-community/VoxCPM1.5-4bit")
let (_, reference) = try loadAudioArray(from: referenceURL, sampleRate: model.sampleRate)
let audio = try await model.generate(
    text: "This sentence uses the reference voice.",
    refAudio: reference,
    refText: "The words spoken in the reference recording.",
    parameters: VoxCPMGenerationParameters(
        maxTokens: 128, inferenceTimesteps: 10, cfgValue: 2
    )
)
```

The shared `generateStream` API emits one complete audio event followed by generation information. VoxCPM does not emit incremental waveform chunks.

## CLI

```bash
mlx-audio-swift-tts --model mlx-community/VoxCPM1.5-4bit \
  --text "Hello, this is Vox CPM." --max_tokens 128 --output voxcpm.wav

mlx-audio-swift-tts --model mlx-community/VoxCPM1.5-4bit \
  --text "This sentence uses the reference voice." \
  --ref_audio reference.wav --ref_text "The words in the reference recording." \
  --max_tokens 128 --output cloned.wav
```

## License and attribution

VoxCPM is developed by [OpenBMB and the VoxCPM contributors](https://github.com/OpenBMB/VoxCPM). The model and reference code use the [Apache 2.0 license](https://github.com/OpenBMB/VoxCPM/blob/main/LICENSE); see the [VoxCPM 1.5 model card](https://huggingface.co/openbmb/VoxCPM1.5) and [VoxCPM 1 model card](https://huggingface.co/openbmb/VoxCPM-0.5B) for model attribution and terms.
