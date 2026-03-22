# AGENTS.md — Project Rules for AI Agents

## Build & Verification

**ALWAYS use `xcodebuild` for build verification, NOT `swift build` / `swift test`.**

```bash
# Build a specific target
xcodebuild build -scheme <SchemeName> -destination 'platform=macOS'

# Build all
xcodebuild build -scheme MLXAudio-Package -destination 'platform=macOS'

# Run tests
xcodebuild test -scheme MLXAudio-Package -destination 'platform=macOS' -only-testing:MLXAudioTests
```

Available schemes: MLXAudio, MLXAudio-Package, MLXAudioCodecs, MLXAudioCore,
MLXAudioG2P, MLXAudioModules, MLXAudioLID, MLXAudioSTS, MLXAudioSTT,
MLXAudioTTS, MLXAudioUI, MLXAudioVAD

**Do NOT run multiple `xcodebuild` commands in parallel** — they share
DerivedData and will fail with "database is locked". Run sequentially.

## Known Pre-existing Issues

- `swift test` cannot run Metal-dependent tests — SPM does not compile
  Metal shaders, so MLX fails with `Failed to load the default metallib`.
  **Use `xcodebuild test` instead.**
- `swift test` may show Sendable warnings in upstream dependencies — ignore.

## CI Testing

Swift Testing uses **struct names**, not file names, for `-only-testing:`.
CI must use real struct names (e.g., `MLXAudioTests/SharedDSPTests`) or
run all tests without `-only-testing:` filter.

**ALWAYS skip SmokeTests in CI** — they download multi-GB models and run
inference, which exceeds the CI runner memory limit (OOM → runner crash).
Use `-skip-testing:'MLXAudioTests/SmokeTests'`. SmokeTests are designed
for local runs only (see `Tests/MLXAudioSmokeTests.swift` header).

## Testing Framework

Tests use **Swift Testing** (`import Testing`), NOT XCTest.

```swift
import Testing

struct MyTests {
    @Test func myTest() {
        #expect(value == expected)
    }
}
```

## Package Structure

- Products and targets defined in `Package.swift`
- Sources: `Sources/<TargetName>/`
- Tests: `Tests/` (single unified directory)
- New targets: add product, target, umbrella entry, and test dependency
