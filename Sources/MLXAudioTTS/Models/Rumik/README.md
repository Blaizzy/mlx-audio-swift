# rumik-oss 1

Rumik synthesizes 24 kHz mono speech in Indic languages and English. It provides four voices (Ira, Aisha, Siya, and Zoya), delivery descriptions, inline `<laugh>`, `<chuckle>`, and `<sigh>` vocalizations, and incremental audio streaming.

Supported repositories:

- [rumik-ai/rumik-oss-1-mlx-8bit](https://huggingface.co/rumik-ai/rumik-oss-1-mlx-8bit)
- [rumik-ai/rumik-oss-1-mlx-4bit](https://huggingface.co/rumik-ai/rumik-oss-1-mlx-4bit)
- [rumik-ai/rumik-oss-1](https://huggingface.co/rumik-ai/rumik-oss-1)

```swift
import MLXAudioCore
import MLXAudioTTS

let model = try await RumikModel.fromPretrained("rumik-ai/rumik-oss-1-mlx-8bit")
let audio = try await model.generate(
    text: "नमस्ते! आपका स्वागत है। <chuckle>",
    voice: "Ira",
    options: .init(description: "happy, Hindi accent, steady pace"),
    parameters: GenerateParameters(maxTokens: 2048, temperature: 0.8, topK: 30)
)

for try await event in model.generateStream(
    text: "Hello, welcome to the demo.",
    voice: "Aisha",
    options: .init(instruct: "calm, steady pace"),
    streamingInterval: 0.5
) {
    if case .audio(let chunk) = event {
        // Play or collect each successive 24 kHz mono chunk.
    }
}
```

The common `TTS.loadModel(modelRepo:)` API also loads Rumik. Inline descriptions such as `<description="excited, fast pace">` work through the common API and CLI. Reference-audio voice cloning is not supported.

```sh
mlx-audio-swift-tts --model rumik-ai/rumik-oss-1-mlx-8bit \
  --text '<description="happy, steady pace"> नमस्ते! आपका स्वागत है।' \
  --voice Ira --output speech.wav
```

The [Rumik model license](https://huggingface.co/rumik-ai/rumik-oss-1/blob/main/LICENSE) contains CC BY-NC 4.0 terms and the Cohere addendum; see its [attribution notice](https://huggingface.co/rumik-ai/rumik-oss-1/blob/main/NOTICE). The separately downloaded [Mimi codec](https://huggingface.co/kyutai/moshiko-pytorch-bf16) has [CC BY 4.0 weights](https://creativecommons.org/licenses/by/4.0/). The backbone follows [Cohere2 in MLX LM](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cohere2.py), and the codec follows [Kyutai Mimi](https://github.com/kyutai-labs/moshi).
