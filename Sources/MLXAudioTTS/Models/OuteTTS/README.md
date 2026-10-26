# OuteTTS

OuteTTS 1.0 generates multilingual speech at 24 kHz, with a built-in voice, portable speaker profiles, and streaming audio. The supported checkpoint is [mlx-community/OuteTTS-1.0-0.6B-fp16](https://huggingface.co/mlx-community/OuteTTS-1.0-0.6B-fp16). Its speech codec is [mlx-community/dac-speech-24khz-1.5kbps](https://huggingface.co/mlx-community/dac-speech-24khz-1.5kbps).

```swift
import MLXAudioCore
import MLXAudioTTS

let model = try await TTS.loadModel(
    modelRepo: "mlx-community/OuteTTS-1.0-0.6B-fp16"
)
let audio = try await model.generate(
    text: "Hello from OuteTTS.",
    voice: nil,
    refAudio: nil,
    refText: nil,
    language: nil
)

for try await event in model.generateStream(
    text: "This speech arrives in chunks.",
    voice: nil, refAudio: nil, refText: nil, language: nil,
    generationParameters: model.defaultGenerationParameters,
    streamingInterval: 1.0
) {
    if case .audio(let chunk) = event {
        // Play or save successive mono chunks at model.sampleRate.
    }
}
```

Use an OuteTTS speaker JSON path as `voice`, or assign a profile directly:

```swift
let model = try await OuteTTSModel.fromPretrained(
    "mlx-community/OuteTTS-1.0-0.6B-fp16"
)
model.speaker = try OuteTTSSpeaker.load(from: speakerURL)
```

To create a profile, provide several seconds of clear mono 24 kHz speech, its matching transcript, and ordered word boundaries in seconds. Word alignment is supplied by your application; no transcription model is downloaded automatically. The shared `refAudio`/`refText` arguments require this explicit profile workflow.

```swift
let profile = try model.createSpeaker(
    audio: referenceAudio,
    text: referenceTranscript,
    words: alignedWords // [OuteTTSAlignedWord], with start/end in seconds
)
try profile.save(to: speakerURL)
model.speaker = profile
```

```bash
mlx-audio-swift-tts \
  --model mlx-community/OuteTTS-1.0-0.6B-fp16 \
  --text "Hello from OuteTTS." \
  --output outetts.wav

mlx-audio-swift-tts \
  --model mlx-community/OuteTTS-1.0-0.6B-fp16 \
  --voice /path/to/speaker.json \
  --text "Speaking with a saved voice." \
  --output voice.wav
```

OuteTTS is by [OuteAI](https://github.com/edwko/OuteTTS). The [OuteTTS model](https://huggingface.co/OuteAI/OuteTTS-1.0-0.6B) and [MLX Audio reference implementation](https://github.com/Blaizzy/mlx-audio) use the Apache 2.0 license. The [IBM speech DAC weights](https://huggingface.co/ibm-research/DAC.speech.v1.0) use CDLA Permissive 2.0; the [Descript codec implementation](https://github.com/descriptinc/descript-audio-codec) uses MIT.
