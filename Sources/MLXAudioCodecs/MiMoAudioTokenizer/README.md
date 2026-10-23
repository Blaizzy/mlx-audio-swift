# MiMo Audio Tokenizer

MiMo Audio Tokenizer encodes mono audio into discrete speech codes and decodes
them into audio at 24 kHz. It supports all twenty codebooks, including the first
eight used by MiMo speech models, and can decode any nonempty codebook prefix.

Supported model: [`XiaomiMiMo/MiMo-Audio-Tokenizer`](https://huggingface.co/XiaomiMiMo/MiMo-Audio-Tokenizer).

```swift
import Foundation
import MLXAudioCodecs

let audioURL = URL(fileURLWithPath: "input.wav")
let codec = try await MiMoAudioTokenizer.fromPretrained()
let codes = try codec.encode(audioURL)
let reconstructed = try codec.decode(codes)

// Use an eight-codebook representation.
let speechCodes = try codec.encode(audioURL, numQuantizers: 8)
let speech = try codec.decode(speechCodes)
```

Reconstruct an audio file with the codec CLI:

```bash
mlx-audio-swift-codec --model XiaomiMiMo/MiMo-Audio-Tokenizer \
  --audio input.wav --output reconstructed.wav
```

In-memory waveforms use `(samples,)` or `(samples, channels)` with an explicit
sample rate. Channels are averaged to mono and audio is resampled when needed.
Codes use `(codebooks, frames)`; decoded audio is a one-dimensional mono array.
Pass an original sample count at 24 kHz as `length:` to `decode` to trim framing
padding. For an in-memory waveform already at 24 kHz, that count is
`waveform.shape[0]`. This codec processes audio offline.

The model weights are released under the [MIT license](https://huggingface.co/XiaomiMiMo/MiMo-Audio-Tokenizer).
The [MiMo-Audio reference implementation](https://github.com/XiaomiMiMo/MiMo-Audio)
is licensed under Apache-2.0.
