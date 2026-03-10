# mlx-audio-swift-lid

CLI demo for `MLXAudioLID`.

## Usage

```bash
swift run mlx-audio-swift-lid --audio Tests/media/intention.wav
```

Use MMS-LID-256 instead of the default ECAPA model:

```bash
swift run mlx-audio-swift-lid \
  --audio Tests/media/intention.wav \
  --model facebook/mms-lid-256 \
  --top-k 3
```

Save the result as JSON:

```bash
swift run mlx-audio-swift-lid \
  --audio Tests/media/intention.wav \
  --output-path lid-output.json
```

## Command-Line MLX Runtime

When you run the tool from the shell, `mlx-swift` must be able to find its metal shader resources.
If the CLI reports that the MLX runtime is not configured, either:

```bash
export DYLD_FRAMEWORK_PATH="$(swift build --show-bin-path)"
swift run mlx-audio-swift-lid --audio Tests/media/intention.wav
```

or run the executable from Xcode, which sets the resource lookup path automatically.
