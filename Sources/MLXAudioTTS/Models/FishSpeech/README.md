# Fish Audio S2 Pro

Fish Audio S2 Pro is Fish Audio's dual-autoregressive text-to-speech model with reference voice cloning, multi-speaker tags, and automatic long-form batching.

## Swift Example

```swift
import Foundation
import MLXAudioCore
import MLXAudioTTS

let model = try await TTS.loadModel(modelRepo: "mlx-community/fish-audio-s2-pro-8bit")

let audio = try await model.generate(
    text: "Hello from Fish Speech.",
    voice: nil,
    refAudio: nil,
    refText: nil,
    language: nil
)

try AudioUtils.writeWavFile(
    samples: audio.asArray(Float.self),
    sampleRate: model.sampleRate,
    fileURL: URL(fileURLWithPath: "/tmp/output.wav")
)
```

`voice` and `language` are currently unused for Fish Speech in the Swift API, so pass `nil`.

## Available Models

- [mlx-community/fish-audio-s2-pro-bf16](https://huggingface.co/mlx-community/fish-audio-s2-pro-bf16)
- [mlx-community/fish-audio-s2-pro-8bit](https://huggingface.co/mlx-community/fish-audio-s2-pro-8bit)

## Voice Cloning

Clone a voice from reference audio by providing the waveform and its transcript:

```swift
import Foundation
import MLXAudioCore
import MLXAudioTTS

let model = try await TTS.loadModel(modelRepo: "mlx-community/fish-audio-s2-pro-8bit")

let (_, refAudio) = try loadAudioArray(
    from: URL(fileURLWithPath: "sample_audio.wav"),
    sampleRate: model.sampleRate
)

let cloned = try await model.generate(
    text: "Hello from Fish Speech.",
    voice: nil,
    refAudio: refAudio,
    refText: "This is what my voice sounds like.",
    language: nil
)
```

For best cloning results, keep `refText` closely aligned with the spoken content in the reference clip.

## Fine-Grained Inline Control

S2 Pro supports localized control with `[tag]` syntax embedded directly in the text. It accepts free-form textual descriptions for word-level expression control. Common examples include:

`[pause]` `[emphasis]` `[laughing]` `[inhale]` `[chuckle]` `[tsk]` `[singing]` `[excited]` `[laughing tone]` `[interrupting]` `[chuckling]` `[excited tone]` `[volume up]` `[echo]` `[angry]` `[low volume]` `[sigh]` `[low voice]` `[whisper]` `[screaming]` `[shouting]` `[loud]` `[surprised]` `[short pause]` `[exhale]` `[delight]` `[panting]` `[audience laughter]` `[with strong accent]` `[volume down]` `[clearing throat]` `[sad]` `[moaning]` `[shocked]`

```swift
let expressive = try await model.generate(
    text: "[whisper] Keep this between us. [pause] Now say it clearly.",
    voice: nil,
    refAudio: nil,
    refText: nil,
    language: nil
)
```

## Multi-Speaker Tags

Fish Speech supports inline speaker tags such as `<|speaker:0|>` and `<|speaker:1|>`:

```swift
let dialogue = try await model.generate(
    text: """
    <|speaker:0|>Welcome everyone.
    <|speaker:1|>Thanks, it's good to be here.
    """,
    voice: nil,
    refAudio: nil,
    refText: nil,
    language: nil
)
```

## Long-Form Generation

Long text is batched internally while preserving the running conversation context, including inline speaker turns. In Swift, chunking is automatic, so you can pass the full passage directly:

```swift
let longPassage = """
Fish Speech can synthesize longer passages without forcing you to split the text yourself.
The Swift port keeps the running conversation state internally so multi-sentence narration and
speaker-tagged dialogue stay coherent across batches.
"""

let longForm = try await model.generate(
    text: longPassage,
    voice: nil,
    refAudio: nil,
    refText: nil,
    language: nil
)
```

If you need generation statistics for a long run, use `generateStream(...)` and inspect the `.info` event after synthesis completes.
