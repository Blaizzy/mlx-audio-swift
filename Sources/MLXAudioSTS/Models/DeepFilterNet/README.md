# DeepFilterNet (Swift / MLX)

Native Swift MLX inference runtime for DeepFilterNet speech enhancement.

## Model Files

Use a model directory containing:

- `config.json`
- `model.safetensors` (or any `.safetensors` file)

## Usage

```swift
let model = try await DeepFilterNetModel.fromPretrained("/path/to/DeepFilterNet3")
let enhanced = try model.enhance(audio)
```

`audio` is a mono `MLXArray` of shape `[samples]` in `[-1, 1]`.
