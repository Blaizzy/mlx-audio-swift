# Llama TTS (Orpheus)

Orpheus TTS is a Llama-based Speech-LLM designed for high-quality, empathetic text-to-speech generation.

[Hugging Face Model Repo](https://huggingface.co/mlx-community/orpheus-3b-0.1-ft-bf16)

## Suggested Voices

- `tara`
- `leah`
- `jess`
- `leo`
- `dan`
- `mia`
- `zac`
- `zoe`

## CLI Example

```bash
mlx-audio-swift-tts --model mlx-community/orpheus-3b-0.1-ft-bf16 --voice tara --text "Hello world."
```

## Swift Example

```swift
import MLXAudioTTS

let model = try await LlamaTTSModel.fromPretrained("mlx-community/orpheus-3b-0.1-ft-bf16")
let audio = try await model.generate(
    text: "Hello world.",
    voice: "tara",
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
