# Contributing to MLX Audio Swift

Thanks for contributing to MLX Audio Swift.

## Scope

We welcome:
- New model ports (TTS, STT, STS, codecs)
- Bug fixes and performance improvements
- Documentation and example improvements

For large API changes or new features, **open an issue first** to discuss the
approach before starting implementation. This avoids wasted effort on PRs that
won't be accepted.

## Reporting Bugs

Search [existing issues](https://github.com/Blaizzy/mlx-audio-swift/issues) before
opening a new one. When reporting a bug, include:

- macOS version and Apple Silicon chip (e.g., macOS 15.3, M3 Pro)
- Xcode version (`xcodebuild -version`)
- Full error output or crash log
- Minimal reproducible snippet

## Security Vulnerabilities

Do **not** open a public issue for security vulnerabilities. Use
[GitHub Security Advisories](https://github.com/Blaizzy/mlx-audio-swift/security/advisories/new)
to report privately.

## Development Setup

```bash
# Clone your fork
git clone git@github.com:<you>/mlx-audio-swift.git
cd mlx-audio-swift

# Open in Xcode or build from CLI
open Package.swift
```

## Pull Requests

- Open pull requests against `Blaizzy/mlx-audio-swift:main`.
- If you are contributing from a fork, make sure the base repository is
  `Blaizzy/mlx-audio-swift` and the base branch is `main`.
- Keep pull requests focused. Include tests and documentation updates when
  behavior changes.
- Keep PRs atomic and touch the smallest possible amount of code. This helps
  reviewers evaluate and merge changes faster and with higher confidence.
- Run local build and test checks before opening a PR.

The current CI workflow on `main` uses `xcodebuild` on macOS:

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

See [ADDING_A_MODEL.md](ADDING_A_MODEL.md) for the full guide — module layout,
required protocols, configuration structs, factory registration, weight loading,
mlx-community naming convention, tests, and PR checklist.

## Keeping the repository clean

Do not commit personal or temporary files. Before opening a PR, make sure your
diff does not include:

- Planning docs, notes, or spec files
- Test scripts, scratch playgrounds, or one-off debug files
- Local model weights or large binaries
- Changes to `.gitignore` that only cover your personal setup

**Use a global gitignore for personal patterns** so you never have to touch the
project's `.gitignore`:

```bash
# Create (or append to) your global gitignore
echo "*.local.*" >> ~/.gitignore_global
echo ".env.local" >> ~/.gitignore_global

# Register it with git (one-time setup)
git config --global core.excludesFile ~/.gitignore_global
```

The `*.local.*` pattern (e.g., `notes.local.md`, `config.local.json`) is a
useful convention for files that should always stay local.

## Good First Issues

Issues labeled [`good first issue`](https://github.com/Blaizzy/mlx-audio-swift/contribute)
are a good starting point if you are new to the codebase.

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).
By participating, you agree to uphold it.

## Commit Signing and Account Security

To improve commit provenance and reduce supply chain risk, please sign commits
submitted to this repository. This is a one-time setup on your machine.

- Any GitHub-supported signing method is fine: GPG, SSH, or S/MIME.
- Enable GitHub vigilant mode so commits and tags always show a verification
  status.
- Enable two-factor authentication on your GitHub account. Passkeys are
  preferred when available.

## References

- [About commit signature verification](https://docs.github.com/en/authentication/managing-commit-signature-verification/about-commit-signature-verification)
- [Displaying verification statuses for all of your commits](https://docs.github.com/en/authentication/managing-commit-signature-verification/displaying-verification-statuses-for-all-of-your-commits)
- [Enable vigilant mode](https://docs.github.com/en/authentication/managing-commit-signature-verification/displaying-verification-statuses-for-all-of-your-commits#enabling-vigilant-mode)
- [GPG setup walkthrough](https://docs.github.com/en/authentication/managing-commit-signature-verification/telling-git-about-your-signing-key)
