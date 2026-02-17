# mlx-audio-swift-tts

Command-line tool for text-to-speech generation with `MLXAudioTTS` models.

## Build and Run

```bash
swift run mlx-audio-swift-tts --text "Hello world"
```

## Example

```bash
swift run mlx-audio-swift-tts \
  --model mlx-community/VyvoTTS-EN-Beta-4bit \
  --text "Hello from MLX Audio" \
  --voice en-us-1 \
  --output /tmp/tts.wav
```

## Reference Audio Example

```bash
swift run mlx-audio-swift-tts \
  --model Marvis-AI/marvis-tts-250m-v0.2-MLX-8bit \
  --text "Hello from MLX Audio" \
  --ref_audio /path/to/reference.wav \
  --ref_text "Reference transcript" \
  --output /tmp/tts_ref.wav
```

## Options

- `--text`, `-t`: Text to synthesize (required)
- `--voice`, `-v`: Voice id
- `--model`: Hugging Face repo id
- `--output`, `-o`: Output WAV path (default: `./output.wav`)
- `--ref_audio`: Reference audio path
- `--ref_text`: Reference transcript
- `--max_tokens`: Override generation max tokens
- `--temperature`: Override sampling temperature
- `--top_p`: Override top-p
- `--help`, `-h`: Show help
