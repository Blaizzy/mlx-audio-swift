# Contributing to MLX Audio Swift

Thanks for contributing to MLX Audio Swift.

We welcome model ports, bug fixes, performance work, and documentation
improvements.

For large API changes or new features, open an issue first so the approach can
be discussed before implementation starts.

## Reporting Bugs

Search [existing issues](https://github.com/Neywa-Labs/mlx-audio-swift/issues) before
opening a new one. When reporting a bug, include:

- macOS version and Apple Silicon chip (e.g., macOS 15.3, M3 Pro)
- Xcode version (`xcodebuild -version`)
- Full error output or crash log
- Minimal reproducible snippet

## Security Vulnerabilities

Do **not** open a public issue for security vulnerabilities. Use
[GitHub Security Advisories](https://github.com/Neywa-Labs/mlx-audio-swift/security/advisories/new)
to report privately.

## Development Setup

```bash
git clone git@github.com:<you>/mlx-audio-swift.git
cd mlx-audio-swift
open Package.swift
```

## Pull Requests

- Open pull requests against `Neywa-Labs/mlx-audio-swift:main`.
- If you are contributing from a fork, make sure the base repo is
  `Neywa-Labs/mlx-audio-swift` and the base branch is `main`.
- Keep pull requests focused. Include tests and documentation updates when
  behavior changes.
- Keep PRs atomic and touch the smallest possible amount of code. This helps
  reviewers evaluate and merge changes faster and with higher confidence.
- Run local build and test checks before opening a PR.

If you want to mirror CI more closely, install the Metal toolchain first:

```bash
xcodebuild -downloadComponent MetalToolchain
```

Then run the usual local checks:

```bash
xcodebuild build-for-testing \
  -scheme MLXAudio-Package \
  -destination 'platform=macOS' \
  MACOSX_DEPLOYMENT_TARGET=14.0 \
  CODE_SIGNING_ALLOWED=NO

xcodebuild test-without-building \
  -scheme MLXAudio-Package \
  -destination 'platform=macOS' \
  -skip-testing:'MLXAudioTests/SmokeTests' \
  -parallel-testing-enabled NO \
  CODE_SIGNING_ALLOWED=NO
```

## Adding a New Model

See [ADDING_A_MODEL.md](ADDING_A_MODEL.md) for the short checklist.

## Keeping the repository clean

Do not commit personal or temporary files. Before opening a PR, make sure your
diff does not include:

- Planning docs, notes, or spec files
- Test scripts, scratch playgrounds, or one-off debug files
- Local model weights or large binaries
- Changes to `.gitignore` that only cover your personal setup

## Good First Issues

Issues labeled [`good first issue`](https://github.com/Neywa-Labs/mlx-audio-swift/contribute)
are a good starting point if you are new to the codebase.

## Code of Conduct

This project follows the [Code of Conduct](CODE_OF_CONDUCT.md). By
participating, you agree to uphold it.

## Commit Signing and Account Security

Please sign commits submitted to this repository.

- Any GitHub-supported signing method is fine: GPG, SSH, or S/MIME.
- Enable GitHub vigilant mode.
- Enable two-factor authentication on your GitHub account.

## References

- [About commit signature verification](https://docs.github.com/en/authentication/managing-commit-signature-verification/about-commit-signature-verification)
- [Enable vigilant mode](https://docs.github.com/en/authentication/managing-commit-signature-verification/displaying-verification-statuses-for-all-of-your-commits#enabling-vigilant-mode)
