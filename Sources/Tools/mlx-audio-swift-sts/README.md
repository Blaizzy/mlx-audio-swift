# mlx-audio-swift-sts

Command-line tool for speech/source separation using SAM Audio (`MLXAudioSTS`).

## Build and Run

```bash
swift run mlx-audio-swift-sts --audio /path/to/input.wav
```

## Example

```bash
swift run mlx-audio-swift-sts \
  --model mlx-community/SAM-48k2-v1.5 \
  --audio /path/to/mix.wav \
  --description speech \
  --mode short \
  --output-target /tmp/target.wav \
  --output-residual /tmp/residual.wav
```

## Streaming Mode Example

```bash
swift run mlx-audio-swift-sts \
  --audio /path/to/mix.wav \
  --mode stream \
  --chunk-seconds 10 \
  --overlap-seconds 3
```

## Options

- `--audio`, `-i`: Input audio path (required)
- `--model`: Model repo id or local path
- `--description`, `--prompt`, `-d`: Target description text
- `--mode`: `short | long | stream`
- `--output-target`, `-o`: Target WAV output path
- `--output-residual`: Residual WAV output path
- `--no-residual`: Skip residual write
- `--chunk-seconds`: Chunk length for `long`/`stream`
- `--overlap-seconds`: Chunk overlap for `long`/`stream`
- `--ode-method`: `midpoint | euler`
- `--step-size`: ODE step size
- `--decode-chunk-size`: Optional decoder chunk size
- `--anchor`: Anchor rule (`+| - :start:end`), repeatable, `short` mode only
- `--strict`: Strict weight loading
- `--hf-token`: Hugging Face token (or use `HF_TOKEN` env var)
- `--help`, `-h`: Show help
