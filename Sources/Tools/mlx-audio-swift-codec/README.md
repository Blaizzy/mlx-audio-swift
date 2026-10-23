# mlx-audio-swift-codec

Command-line tool for codec roundtrip reconstruction (`encode -> decode`) with
`MLXAudioCodecs` models. Accepts a model repository ID or local checkpoint
directory. Input audio is resampled to the codec's sample rate; multichannel
input is averaged for mono codecs. Output WAV files use the codec's sample rate
and channel count.

## Build and Run

```bash
xcodebuild -scheme MLXAudio-Package -destination 'platform=macOS' \
  -derivedDataPath .build/codec build CODE_SIGNING_ALLOWED=NO MTL_FAST_MATH=NO

./.build/codec/Build/Products/Debug/mlx-audio-swift-codec --audio /path/to/input.wav
```

## Example

```bash
./.build/codec/Build/Products/Debug/mlx-audio-swift-codec \
  --model mlx-community/dacvae-watermarked \
  --audio /path/to/input.wav \
  --output /tmp/reconstructed.wav
```

## Supported Roundtrip Codecs

- `dacvae`
- `encodec`
- `snac`
- `mimi`
- Descript DAC
- Fish S1 DAC
- MiMo Audio Tokenizer (`XiaomiMiMo/MiMo-Audio-Tokenizer`)
- Nemotron VoiceChat Codec (`mlx-community/NemotronLabs-VoiceChat-11B-4bit`)

Vocoder and encode-only tokenizer operations are available through their Swift
APIs. Use `AudioCodecs.loadModel(modelRepo:)` for the same roundtrip loading in
Swift, including arbitrary local directory names.

## Options

- `--audio`, `-i`: Input audio path (required)
- `--model`: Hugging Face repository ID or local checkpoint directory (default: `mlx-community/dacvae-watermarked`)
- `--output`, `-o`: Output WAV path (default: `<input_stem>.reconstructed.wav`)
- `--help`, `-h`: Show help
