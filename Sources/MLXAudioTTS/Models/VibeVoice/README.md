# VibeVoice Realtime

VibeVoice Realtime synthesizes 24 kHz mono speech with streaming output and 25 preset voices. The default voice is `en-Carter_man`. Voice names such as `en-Emma_woman`, `fr-Spk0_man`, and `jp-Spk1_woman` select the published speaker presets.

Supported checkpoints:

- [VibeVoice-Realtime-0.5B-8bit](https://huggingface.co/mlx-community/VibeVoice-Realtime-0.5B-8bit)
- [VibeVoice-Realtime-0.5B-4bit](https://huggingface.co/mlx-community/VibeVoice-Realtime-0.5B-4bit)
- [VibeVoice-Realtime-0.5B-fp16](https://huggingface.co/mlx-community/VibeVoice-Realtime-0.5B-fp16)

```swift
import MLXAudioCore
import MLXAudioTTS

let model = try await TTS.loadModel(
    modelRepo: "mlx-community/VibeVoice-Realtime-0.5B-8bit")
let audio = try await model.generate(
    text: "Hello from VibeVoice.", voice: "en-Carter_man",
    refAudio: nil, refText: nil, language: nil)

for try await event in model.generateStream(
    text: "Speech can arrive in short chunks.", voice: "en-Emma_woman",
    refAudio: nil, refText: nil, language: nil,
    generationParameters: GenerateParameters(maxTokens: 512),
    streamingInterval: 0.5)
{
    if case .audio(let chunk) = event {
        // Play or append this chunk.
    }
}
```

Use `VibeVoiceRealtimeModel.fromPretrained` for the concrete model, its `availableVoices` list, and `options` controls for guidance strength and diffusion steps. Realtime uses preset voices; it does not accept reference audio.

```bash
mlx-audio-swift-tts \
  --model mlx-community/VibeVoice-Realtime-0.5B-8bit \
  --text "Hello from VibeVoice." \
  --voice en-Carter_man \
  --output speech.wav
```

The [Microsoft VibeVoice project](https://github.com/microsoft/VibeVoice) and [Realtime model](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B) use the [MIT license](https://github.com/microsoft/VibeVoice/blob/main/LICENSE). The [MLX implementation](https://github.com/Blaizzy/mlx-audio/tree/main/mlx_audio/tts/models/vibevoice) uses [MIT](https://github.com/Blaizzy/mlx-audio/blob/main/LICENSE). The [Qwen2.5 tokenizer](https://huggingface.co/Qwen/Qwen2.5-0.5B) uses [Apache 2.0](https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/main/LICENSE).
