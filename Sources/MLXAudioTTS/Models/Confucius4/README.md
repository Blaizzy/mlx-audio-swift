# Confucius4 TTS

Confucius4 is a multilingual speech model with zero-shot and cross-language voice cloning. It combines a W2V-BERT reference encoder, CAMPPlus speaker conditioning, a text-to-semantic language model, a flow-matching acoustic model, and BigVGAN. Output is mono audio at 22,050 Hz.

Supported checkpoint: [mlx-community/Confucius4-TTS-mlx-int8](https://huggingface.co/mlx-community/Confucius4-TTS-mlx-int8).

A reference audio clip is required. A reference transcript is optional and is not used by this model. Supported language prompts are `zh`, `en`, `vi`, `ja`, `ko`, and `th`; the default is Vietnamese. Unknown language codes use the English prompt. The stream API emits one completed waveform per request.

```bash
mlx-audio-swift-tts --model mlx-community/Confucius4-TTS-mlx-int8 \
  --text "Hello, welcome to our demonstration." \
  --ref_audio reference.wav --language en --output speech.wav
```

```swift
import MLX
import MLXAudioCore
import MLXAudioTTS

let model = try await Confucius4Model.fromPretrained()
// reference16k contains a mono Float waveform sampled at 16,000 Hz.
let audio = try await model.generate(
    text: "Hello, welcome to our demonstration.",
    refAudio: MLXArray(reference16k),
    language: "en",
    generationParameters: .init(
        maxTokens: 512, temperature: 0.8, topP: 0.8, topK: 30,
        repetitionPenalty: 10, seed: 0),
    options: .init(referenceSampleRate: 16000, diffusionSteps: 25, guidance: 0.7)
)
```

`TTS.loadModel(modelRepo:)` also accepts the repository ID or a local checkpoint directory. Its reference tensors use the model sample rate, 22,050 Hz. Model instances require serialized access. Stochastic seeds are reproducible within Swift; sampling draws differ from NumPy's random generator.

The checkpoint uses [Apache 2.0](https://huggingface.co/mlx-community/Confucius4-TTS-mlx-int8). See the [Confucius4 project](https://huggingface.co/netease-youdao/Confucius4-TTS), [MLX Audio](https://github.com/Blaizzy/mlx-audio), [W2V-BERT 2.0](https://huggingface.co/facebook/w2v-bert-2.0), [CAMPPlus](https://github.com/modelscope/3D-Speaker), and [BigVGAN](https://github.com/NVIDIA/BigVGAN) for source and license attribution.
