# Consumer Graph Fixture

This fixture reproduces package-graph behavior that does not show up when building `mlx-audio-swift` by itself.

It depends on:

- the local package via `.package(path: "../..")`
- `swift-transformers` from `1.3.0`
- `mlx-swift-lm` from the `main` branch

It intentionally depends on `MLXAudioTTS`, not just `MLXAudioCodecs`, because the current consumer-graph failure shows up while compiling the higher-level TTS target against that newer shared stack.

Build it locally with:

```sh
swift build --package-path Integration/ConsumerGraphFixture
```

That same command is intended to run in CI as a regression check.
