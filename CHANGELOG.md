# Changelog

All notable changes to mlx-audio-swift are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Audio Component Registry (P1 Models)**: SNAC and Mimi audio codecs now register with SwiftAcervo Component Registry at module initialization via `ComponentDescriptor`. This enables:
  - Declarative model metadata (file lists, memory requirements, codec parameters)
  - Intelligent downloads by Acervo (knows exact files and size before model code runs)
  - Shared model caching across all intrusive-memory projects at `~/Library/SharedModels/`
  - Automatic verification of required files before inference

- **Acervo CDN Integration**: Audio models now integrate seamlessly with SwiftAcervo's unified model resolution system:
  - Automatic caching at `~/Library/SharedModels/<namespace>_<repo>/`
  - Legacy path auto-migration from `~/Library/Caches/intrusive-memory/Models/`
  - Offline availability (models persist across app restarts)
  - Shared cache accelerates repeated model loading

- **ComponentDescriptor Pattern Documentation**: Comprehensive guide in `AGENTS.md` explains:
  - Pattern structure (enums, file lists, descriptors, registration triggers)
  - Usage in model loading workflows
  - Benefits (declarative, discoverable, testable, resilient)
  - Full implementation examples

### Changed

- **README.md Audio Model Management**: New section documents:
  - Automatic downloads on first use
  - Storage location (`~/Library/SharedModels/`)
  - P1 models (SNAC, Mimi) with component IDs
  - Acervo integration benefits

### Performance Improvements

- **Offline availability**: Models cached locally after first download, eliminating network dependencies for repeated use
- **Faster startup**: ComponentDescriptor metadata available before model inference, enabling optimized loading UI
- **Shared cache efficiency**: Multiple apps reuse the same model files, reducing disk usage and bandwidth

---

## [Previous Versions]

For information on earlier features and implementations, see the git log and project issues.
