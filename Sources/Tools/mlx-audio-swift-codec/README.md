# mlx-audio-swift-codec

Command-line tool for codec roundtrip reconstruction (`encode -> decode`) with `MLXAudioCodecs` models.

## Build and Run

```bash
swift run mlx-audio-swift-codec --audio /path/to/input.wav
```

## Example

```bash
swift run mlx-audio-swift-codec \
  --model mlx-community/dacvae-watermarked \
  --audio /path/to/input.wav \
  --output /tmp/reconstructed.wav
```

## Supported Model Patterns

- `dacvae`
- `encodec`
- `snac`
- `mimi`

## Options

- `--audio`, `-i`: Input audio path (required)
- `--model`: Hugging Face repo id (default: `mlx-community/dacvae-watermarked`)
- `--output`, `-o`: Output WAV path (default: `<input_stem>.reconstructed.wav`)
- `--help`, `-h`: Show help
