# mlx-audio-swift-stt

Command-line tool for speech-to-text and forced alignment with `MLXAudioSTT` models.

## Build and Run

```bash
swift run mlx-audio-swift-stt \
  --audio /path/to/audio.wav \
  --output-path /tmp/transcript
```

## ASR Example

```bash
swift run mlx-audio-swift-stt \
  --model mlx-community/Qwen3-ASR-0.6B-4bit \
  --audio /path/to/audio.wav \
  --output-path /tmp/transcript \
  --format json \
  --language English
```

## Streaming Example

```bash
swift run mlx-audio-swift-stt \
  --model mlx-community/Qwen3-ASR-0.6B-4bit \
  --audio /path/to/audio.wav \
  --output-path /tmp/transcript \
  --stream
```

## Forced Aligner Example

```bash
swift run mlx-audio-swift-stt \
  --model mlx-community/Qwen3-ForcedAligner-0.6B-4bit \
  --audio /path/to/audio.wav \
  --text "The transcript to align" \
  --language English \
  --output-path /tmp/alignment \
  --format json
```

## Options

- `--model`: Model repo id
- `--audio`: Input audio path (required)
- `--output-path`: Output path stem (required)
- `--format`: `txt | srt | vtt | json`
- `--verbose`: Verbose logging
- `--max-tokens`: Max generated tokens
- `--language`: Language hint/code
- `--chunk-duration`: Chunk size in seconds
- `--stream`: Stream generated tokens
- `--gen-kwargs`: Extra generation kwargs JSON
- `--text`: Required for forced aligner models
- `--frame-threshold`: Compatibility flag (currently unused)
- `--context`: Compatibility flag (currently unused)
- `--prefill-step-size`: Compatibility flag (currently unused)
- `--help`, `-h`: Show help
