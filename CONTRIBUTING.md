# Contributing to MLX Audio Swift

Thanks for contributing to MLX Audio Swift.

## Pull Requests

- Open pull requests against `Blaizzy/mlx-audio-swift:main`.
- If you are contributing from a fork, make sure the base repository is
  `Blaizzy/mlx-audio-swift` and the base branch is `main`.
- Keep pull requests focused. Include tests and documentation updates when
  behavior changes.
- Keep PRs atomic and touch the smallest possible amount of code. This helps
  reviewers evaluate and merge changes faster and with higher confidence.
- Run local build and test checks before opening a PR when applicable.
- The current CI workflow on `main` uses `xcodebuild` on macOS:

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
