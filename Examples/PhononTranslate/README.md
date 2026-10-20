# Phonon Translate

A native SwiftUI example for iPhone, iPad, and Apple Silicon Mac. Type in English,
translate into French with a small local language model, then listen to either
language with Phonon. Audio streams into `AVAudioEngine` as it is generated.

| Role | Model |
|---|---|
| English → French | [Qwen2.5 0.5B Instruct, 4-bit](https://huggingface.co/mlx-community/Qwen2.5-0.5B-Instruct-4bit) |
| English speech | [Phonon 40M English](https://huggingface.co/gradium/phonon-40m-en) |
| French speech | [Phonon 40M French](https://huggingface.co/gradium/phonon-40m-fr) |

Requires Xcode 26 or later, XcodeGen 2.45+, and iOS 18+ or macOS 15+.
Use a physical Apple Silicon device for MLX inference; Simulator builds are useful
for checking the UI.

## Generate and run

From this directory:

```sh
xcodegen generate
open PhononTranslate.xcodeproj
```

Select the `PhononTranslate` scheme and a Mac or iOS device. For device signing,
copy `Config/Local.xcconfig.template` to `Config/Local.xcconfig` and set your
bundle ID and development team. The generated Xcode project and local settings
are ignored by Git; `project.yml` is the source of truth.

Command-line macOS build:

```sh
xcodebuild -project PhononTranslate.xcodeproj -scheme PhononTranslate \
  -destination 'platform=macOS,arch=arm64' CODE_SIGNING_ALLOWED=NO build
```

iOS Simulator build:

```sh
xcodebuild -project PhononTranslate.xcodeproj -scheme PhononTranslate \
  -destination 'generic/platform=iOS Simulator' CODE_SIGNING_ALLOWED=NO build
```

## Voice model access

1. Accept the access terms on the [English](https://huggingface.co/gradium/phonon-40m-en)
   and [French](https://huggingface.co/gradium/phonon-40m-fr) Phonon model pages.
2. Open **Models & access** in the app and save a Hugging Face read token.
   It is stored in this device’s Keychain. You can also set `HF_TOKEN` in the
   Xcode scheme’s Run environment.
3. Translate a phrase and tap **Speak** on either card.

Models download on first use (about 534 MB total) and remain cached for offline
inference. The text and audio stay on device. Loading errors include a retry path;
you can cancel downloads, translation, or speech. French speech is disabled while
the translation is incomplete or the English text has changed.

## Real-model integration check

The hosted test loads all three models, translates a phrase, and verifies finite,
non-silent streamed audio from both speech models. It skips model downloads unless
`PHONON_TRANSLATE_MODEL_TESTS=1` is set in the test process environment. Supply
`HF_TOKEN` for gated voice model access. For command-line Xcode tests, prefix
test environment variables with `TEST_RUNNER_`:

```sh
TEST_RUNNER_PHONON_TRANSLATE_MODEL_TESTS=1 xcodebuild test \
  -project PhononTranslate.xcodeproj -scheme PhononTranslate \
  -destination 'platform=macOS,arch=arm64' CODE_SIGNING_ALLOWED=NO
```

Optionally set `PHONON_TRANSLATE_TEST_OUTPUT` to save the generated WAV files and
translation. Previews show ready, translated, and loading states without
loading models or accessing the network.
