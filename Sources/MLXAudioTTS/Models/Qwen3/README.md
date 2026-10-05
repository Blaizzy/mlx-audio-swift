# VyvoTTS

VyvoTTS is a text-to-speech model by Vyvo team using Qwen3 architecture.

[Hugging Face Model Repo](https://huggingface.co/mlx-community/VyvoTTS-EN-Beta-4bit)

## CLI Example

```bash
mlx-audio-swift-tts --model mlx-community/VyvoTTS-EN-Beta-4bit --text "Hello world."
```

## Swift Example

```swift
import MLXAudioTTS

let model = try await Qwen3Model.fromPretrained("mlx-community/VyvoTTS-EN-Beta-4bit")
let audio = try await model.generate(
    text: "Hello world.",
    voice: "en-us-1",
    parameters: GenerateParameters()
)
```

## Reusable KV cache

Pass a `TTSGenerationCache` to `generate` or `generateStream` to retain KV state
across calls. Requests sharing the container run sequentially, and cancellation
keeps exclusive access until the generation worker exits.

```swift
let cache = TTSGenerationCache()
let audio = try await model.generate(text: "Hello world.", cache: cache)
try await cache.reset() // Discard context before an independent utterance.
```

The container binds to one model instance. It preserves KV state but does not
manage conversation text or automatically match prompt prefixes. Reset after
failed or cancelled requests if their partial prefix should be discarded.
For migration from a raw cache array, use `TTSGenerationCache(model.makeCache())`
and stop accessing the transferred raw cache and its aliases. Calls without a
`cache` argument still use a fresh cache each time. Different containers do not
serialize concurrent use of the same model instance.
