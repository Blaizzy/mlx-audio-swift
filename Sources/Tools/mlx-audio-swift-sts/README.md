# mlx-audio-swift-sts

Command-line tool for Speech-to-Speech tasks using models from the `MLXAudioSTS` module.

Supports two model families:
- **LFM2.5-Audio**: Multimodal generation (text-to-text, text-to-speech, speech-to-text, speech-to-speech)
- **SAM Audio**: Source separation

## Build and Run

```bash
swift run mlx-audio-swift-sts --help
```

## LFM2.5-Audio Examples

### Text-to-Text
```bash
swift run mlx-audio-swift-sts \
  --model mlx-community/LFM2.5-Audio-1.5B-6bit \
  --mode t2t \
  --text "What is 2 + 2?" \
  --system "Answer briefly." \
  --stream
```

### Text-to-Speech
```bash
swift run mlx-audio-swift-sts \
  --model mlx-community/LFM2.5-Audio-1.5B-6bit \
  --mode tts \
  --text "Hello, welcome to MLX Audio!" \
  --system "Perform TTS. Use a UK male voice." \
  -o /tmp/lfm_tts.wav
```

### Speech-to-Text
```bash
swift run mlx-audio-swift-sts \
  --model mlx-community/LFM2.5-Audio-1.5B-6bit \
  --mode stt \
  --audio /path/to/audio.wav \
  --stream
```

### Speech-to-Speech
```bash
swift run mlx-audio-swift-sts \
  --model mlx-community/LFM2.5-Audio-1.5B-6bit \
  --mode sts \
  --audio /path/to/audio.wav \
  -o /tmp/lfm_sts.wav
```

## SAM Audio Examples

### Short Mode (default)
```bash
swift run mlx-audio-swift-sts \
  --audio /path/to/mix.wav \
  --description speech \
  --mode short \
  --output-target /tmp/target.wav
```

### Streaming Mode
```bash
swift run mlx-audio-swift-sts \
  --audio /path/to/mix.wav \
  --mode stream \
  --chunk-seconds 10 \
  --overlap-seconds 3
```

## Options

### LFM2.5-Audio
- `--model`: Model repo (must contain "lfm" to auto-detect)
- `--mode`: `t2t | tts | stt | sts`
- `--text`, `-t`: Input text
- `--audio`, `-i`: Input audio path
- `--system`: System prompt (overrides per-mode defaults: t2t="You are a helpful assistant.", tts="Perform TTS.", stt="You are a helpful assistant that transcribes audio.", sts="Respond to the user with interleaved text and speech audio.")
- `--max-new-tokens`: Max generation tokens (default: 512)
- `--temperature`: Text temperature (default: 0.8)
- `--top-k`: Text top-K (default: 50)
- `--audio-temperature`: Audio temperature (default: 0.7)
- `--audio-top-k`: Audio top-K (default: 30)
- `--stream`: Stream text to stdout
- `--output-target`, `-o`: Audio WAV output
- `--output-text`: Text output file

### SAM Audio
- `--audio`, `-i`: Input audio path (required)
- `--model`: Model repo or local path
- `--description`, `--prompt`, `-d`: Target description text
- `--mode`: `short | long | stream`
- `--output-target`, `-o`: Target WAV output path
- `--output-residual`: Residual WAV output path
- `--no-residual`: Skip residual write
- `--chunk-seconds`: Chunk length for `long`/`stream`
- `--overlap-seconds`: Chunk overlap for `long`/`stream`
- `--ode-method`: `midpoint | euler`
- `--step-size`: ODE step size
- `--anchor`: Anchor rule (`+|-:start:end`), repeatable, `short` mode only
- `--strict`: Strict weight loading

### Common
- `--hf-token`: Hugging Face token (or use `HF_TOKEN` env var)
- `--help`, `-h`: Show help
