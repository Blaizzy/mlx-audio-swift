# VoxtralApp

Continuous speech-to-text demo using the Voxtral Mini 4B Realtime model with energy-based voice activity detection.

## Prerequisites

- macOS 14+
- Apple Silicon (M1 or later)
- ~5 GB RAM for the 4-bit quantized model
- Microphone access

## Build & Run

```bash
cd Examples/VoxtralApp
swift build
swift run VoxtralApp
```

Or open `Package.swift` in Xcode and run the `VoxtralApp` target.

## How It Works

1. **Model Loading** — On launch, downloads the [Voxtral Mini 4B Realtime](https://huggingface.co/mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit) model (~2.5 GB, cached after first run).
2. **Listening** — Press Start to begin capturing audio from the microphone at 16kHz.
3. **VAD** — Energy-based voice activity detection monitors RMS levels. When speech energy exceeds the threshold, audio is accumulated.
4. **Transcription** — After speech ends (hangtime expires), the accumulated audio is sent to the Voxtral model for streaming transcription.
5. **Display** — Tokens appear in real-time as they're generated. Completed segments are added to the history.

## Settings

- **Energy Threshold** — RMS level to detect speech (lower = more sensitive).
- **Hang Time** — Seconds of silence before finalizing a speech segment (higher = fewer segments).
