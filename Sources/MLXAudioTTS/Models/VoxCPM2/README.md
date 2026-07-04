# VoxCPM2

OpenBMB VoxCPM2 text-to-speech support for MLX Swift.

The implementation is adapted from the VoxCPM2TTS module in
[soniqo/speech-swift](https://github.com/soniqo/speech-swift), which is
licensed under Apache-2.0, and integrated with this package's model loading
and generation APIs.

## Usage

```swift
import MLXAudioCore
import MLXAudioTTS

let model = try await TTS.loadModel(
    modelRepo: "mlx-community/VoxCPM2-8bit"
)

let audio = try await model.generate(
    text: "Hello from VoxCPM2.",
    voice: nil,
    refAudio: nil,
    refText: nil,
    language: nil
)

try AudioFileUtils.save(
    audio: audio,
    sampleRate: model.sampleRate,
    fileURL: URL(fileURLWithPath: "/tmp/voxcpm2.wav")
)
```

Local MLX checkpoints are supported as well:

```swift
let model = try await TTS.loadModel(
    modelRepo: "/path/to/VoxCPM2-8bit"
)
```

## Advanced Features

### Text Normalization

Enable the `normalize` parameter to expand numbers, dates, currency amounts,
and common abbreviations into spoken form before synthesis:

```swift
let audio = try await voxModel.generateVoxCPM2(
    text: "It costs $50.25 — estimated delivery Feb 3rd.",
    normalize: true
)
// Text becomes: "It costs fifty dollars and twenty-five cents — estimated delivery February third"
```

### Bad-case Retry

The `retryBadcase` parameter automatically retries when the generated audio is
abnormally short or long relative to the input text:

```swift
let audio = try await voxModel.generateVoxCPM2(
    text: "Hello world.",
    retryBadcase: true,
    retryBadcaseMaxTimes: 3
)
```

### Reference vs. Continuation Prompt

`refAudio` and `refText` are speaker reference conditioning. They describe
the voice to clone and should not be treated as text/audio to continue.

`promptText` and `promptAudio` are continuation context. Pass them only when
the next utterance should continue from a previous spoken segment:

```swift
guard let voxModel = model as? VoxCPM2Model else { return }

let stream = voxModel.generateStream(
    text: "This is the next sentence.",
    voice: nil,
    refAudio: referenceAudio,
    refText: "This is what the reference speaker said.",
    promptText: previousText,
    promptAudio: previousAudio,
    promptAudioSampleRate: 16_000,
    language: "English",
    generationParameters: voxModel.defaultGenerationParameters
)
```

If `promptText` / `promptAudio` are omitted, streaming generation uses only
speaker reference conditioning and will not replay or continue `refText`.

### LoRA Fine-Tuning

Load a model with LoRA adapters for fine-tuned voice adaptation:

```swift
let config = LoRAConfig(enableLM: true, enableDiT: true, r: 8, alpha: 16)
let model = try await VoxCPM2Model.fromModelDirectory(
    modelDir,
    loraConfig: config
)
try model.loadLoRA(weightsPath: "/path/to/lora_weights.safetensors")
model.setLoRAEnabled(true)

let audio = try await model.generateVoxCPM2(text: "Fine-tuned voice output.")
```

For training, the LoRA matrices (`loraA`, `loraB`) can be updated on each
`LoRALinear` module. Use `getLoRAStateDict()` to export weights for saving:

```swift
let state = model.getLoRAStateDict()
// Save `state` as a safetensors file for future inference
```

## Notes

- `voice` is passed as VoxCPM2 instruction text.
- `refAudio` and `refText` are mapped to VoxCPM2 reference conditioning.
- VoxCPM2-specific streaming can accept explicit `promptText` / `promptAudio`
  for continuation; the generic protocol path leaves them empty.
- VoxCPM2 checkpoints are detected from `model_type`, `architecture`, or repository/path names containing `voxcpm2`.
- LoRA requires injecting adapters at load time via `VoxCPM2Model.fromModelDirectory(_:loraConfig:)`.
  The generic `TTS.loadModel()` path does not yet support LoRA configuration.
- The ZipEnhancer denoiser (`denoise: true`) is not yet implemented in MLX.
  Reference audio should be clean for best results.
