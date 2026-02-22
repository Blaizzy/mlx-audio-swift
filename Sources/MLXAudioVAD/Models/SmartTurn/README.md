# Smart Turn v3 Endpoint Detection

Smart Turn v3.2 endpoint detection for conversational turns.

Model: [mlx-community/smart-turn-v3](https://huggingface.co/mlx-community/smart-turn-v3)

## Quick Start

```swift
import MLX
import MLXAudioVAD

let model = try await SmartTurnModel.fromPretrained("mlx-community/smart-turn-v3")
let audio = MLXArray.zeros([16000], type: Float.self)
let result = try model.predictEndpoint(audio, sampleRate: 16000, threshold: 0.5)

print(result.prediction)   // 0 or 1
print(result.probability)  // sigmoid probability
```

## Notes

- Input audio is expected to be mono and will be resampled to 16 kHz.
- Audio shorter than 8 seconds is left-padded with zeros.
- Audio longer than 8 seconds uses the latest 8 seconds.

## License

This code is licensed under the BSD 2-Clause "Simplified" License. See the [LICENSE](https://github.com/pipecat-ai/smart-turn/blob/main/LICENSE) for more information.
