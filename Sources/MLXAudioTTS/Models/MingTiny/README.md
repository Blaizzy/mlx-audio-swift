# Ming-3.0 TTS Tiny

Ming Tiny generates 44.1 kHz mono speech with built-in voices, reference voice cloning, acoustic continuation, multilingual speech and voice controls. It uses a 7.9B mixture-of-experts language model.

Supported checkpoints:

- [mlx-community/Ming-3.0-tts-tiny-4bit](https://huggingface.co/mlx-community/Ming-3.0-tts-tiny-4bit)
- [mlx-community/Ming-3.0-tts-tiny-8bit](https://huggingface.co/mlx-community/Ming-3.0-tts-tiny-8bit)
- [mlx-community/Ming-3.0-tts-tiny-bf16](https://huggingface.co/mlx-community/Ming-3.0-tts-tiny-bf16)

```swift
import MLXAudioCore
import MLXAudioTTS

let model = try await TTS.loadModel(
    modelRepo: "mlx-community/Ming-3.0-tts-tiny-4bit")
let audio = try await model.generate(
    text: "Hello from Ming three.", voice: "US_James",
    refAudio: nil, refText: nil, language: "en",
    generationParameters: GenerateParameters(maxTokens: 250, temperature: 0, seed: 42))
```

Other built-in voices include `灵小柔` and `灵楷`. Supply reference audio and its transcript to continue in that speaker's voice. Array references must contain mono audio at the model's sample rate.

```swift
let (_, reference) = try loadAudioArray(
    from: URL(fileURLWithPath: "/path/to/reference.wav"), sampleRate: model.sampleRate)
let cloned = try await model.generate(
    text: "This sentence uses the reference speaker.", voice: nil,
    refAudio: reference, refText: "Words spoken in the reference.", language: "en",
    generationParameters: GenerateParameters(maxTokens: 250, temperature: 0, seed: 42))
```

Reference cloning supports Chinese and English, plus German, Spanish, French, Korean and Japanese through the multilingual task. Native CAMPPlus speaker weights are included in these checkpoints. Use the concrete model's options for trained controls:

```swift
let ming = try await MingTinyModel.fromPretrained(
    "mlx-community/Ming-3.0-tts-tiny-4bit")
var options = MingTinyOptions()
options.task = .voiceDesign
options.style = "A cheerful young woman speaking clear English."
let designed = try await ming.generate(
    text: "Today feels like a wonderful day.", options: options,
    generationParameters: GenerateParameters(temperature: 0, seed: 42))
```

`MingTinyOptions` exposes voice design, freeform instructions, emotion, dialect, basic speed/pitch/volume and paralinguistic tasks. Emotion accepts `happy`, `sad`, `angry`, `neutral`, `surprised`, `fearful` or `disgusted`. Dialect accepts `cantonese` or `sichuanese`. The basic task requires all three categorical controls: speed `慢速`/`中速`/`快速`, pitch `低`/`中`/`高`, and volume `低`/`中`/`高`. These tasks use reference speakers except voice design and built-in voices. A 192-dimensional speaker embedding can replace speaker extraction.

For podcast continuation, set `.podcast`, pass ordered `references` and `transcripts`, and label each target dialogue line `speaker_1:`, `speaker_2:`, etc. Include every reference speaker in the dialogue. Cloning and podcast tasks require reference audio even when speaker embeddings are supplied.

```bash
mlx-audio-swift-tts --model mlx-community/Ming-3.0-tts-tiny-4bit \
  --text "Hello from Ming three." --voice US_James --output speech.wav
```

Generation is offline. `generateStream` yields one completed waveform. The default limit is 250 acoustic steps, approximately 20 seconds, within the 4096-position context budget. Set `maxTokens` for longer speech. Text is spoken as supplied; spell out numbers and abbreviations when a particular reading is required.

Model weights and MLX adaptation: [Ming Tiny model card](https://huggingface.co/mlx-community/Ming-3.0-tts-tiny-4bit) (MIT). Original model (MIT) and architecture: [inclusionAI/Ming-3.0-tts-tiny](https://huggingface.co/inclusionAI/Ming-3.0-tts-tiny). Backbone adaptation: [mlx-audio](https://github.com/Blaizzy/mlx-audio) and inclusionAI's Apache-2.0 BailingMoE V3. Native speaker encoder (Apache-2.0): [Step-Audio-2-token2wav](https://huggingface.co/mlx-community/Step-Audio-2-token2wav), with [CAMPPlus attribution](https://github.com/alibaba-damo-academy/3D-Speaker).
