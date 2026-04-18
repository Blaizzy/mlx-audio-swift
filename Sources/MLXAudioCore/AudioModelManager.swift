//
//  AudioModelManager.swift
//  MLXAudioCore
//
//  Manages audio model component registration and download lifecycle via SwiftAcervo.
//  This module serves as the central registry for:
//    - P1 (priority 1) audio models: SNAC, Mimi codecs (hard-coded in apps)
//    - P2 (priority 2) audio models: VyvoTTS, Orpheus, Soprano, MarvisTTS, PocketTTS
//
//  All required files are pre-declared so Acervo can download and verify before inference.
//
//  Pattern: ComponentDescriptor with lazy module-level registration (following SwiftVoxAlta).
//

import Foundation
import SwiftAcervo

// MARK: - Supported Audio Models

/// Known audio model repos supported by mlx-audio-swift.
///
/// Covers both P1 (stable codec models: SNAC, Mimi) and P2 (extended audio codecs and TTS models)
/// suitable for ComponentDescriptor registration. Dynamic models (user-specified variants) use
/// HuggingFace discovery via ModelResolver.
public enum AudioModelRepo: String, CaseIterable, Sendable {
  // MARK: - Codecs (P1 — Stable)

  /// SNAC 24 kHz neural audio codec.
  /// Compresses 24 kHz audio to discrete codes at 0.98 kbps.
  /// Used by: Qwen3-TTS for audio encoding during inference.
  case snac24kHz = "mlx-community/snac_24khz"

  /// Mimi PyTorch BF16 neural audio codec.
  /// High-quality 24 kHz audio compression via 32 codebooks.
  /// Used by: MarvisTTS as encoder/decoder, other generative models.
  case mimiPyTorchBF16 = "kyutai/moshiko-pytorch-bf16"

  // MARK: - TTS Models (P2 — Extended)

  /// VyvoTTS / Qwen3 TTS (4-bit quantized).
  /// Fast, multi-language TTS model for real-time synthesis.
  case vyvoTTSBeta4bit = "mlx-community/VyvoTTS-EN-Beta-4bit"

  /// Orpheus / LlamaTTS (3B params, bfloat16).
  /// High-quality generative TTS based on Llama architecture.
  case orpheusTTS = "mlx-community/orpheus-3b-0.1-ft-bf16"

  /// Soprano TTS (80M params, bfloat16).
  /// Lightweight, fast TTS model for Apple Silicon.
  case sopranoTTS = "mlx-community/Soprano-80M-bf16"

  /// MarvisTTS (250M params, 8-bit quantized).
  /// Premium quality TTS with speaker control and voice design.
  case marvisTTS = "Marvis-AI/marvis-tts-250m-v0.2-MLX-8bit"

  /// PocketTTS (small, multi-voice).
  /// Compact TTS model for on-device synthesis.
  case pocketTTS = "mlx-community/pocket-tts"

  /// Human-readable display name for the model variant.
  public var displayName: String {
    switch self {
    case .snac24kHz:
      return "SNAC 24 kHz Audio Codec"
    case .mimiPyTorchBF16:
      return "Mimi Audio Codec (PyTorch BF16)"
    case .vyvoTTSBeta4bit:
      return "VyvoTTS / Qwen3 TTS (4-bit)"
    case .orpheusTTS:
      return "Orpheus TTS (LlamaTTS, BF16)"
    case .sopranoTTS:
      return "Soprano TTS (80M, BF16)"
    case .marvisTTS:
      return "MarvisTTS (250M, 8-bit)"
    case .pocketTTS:
      return "PocketTTS (Multi-voice)"
    }
  }

  /// The Acervo component ID for this model variant.
  ///
  /// Used to look up, download, and check availability of this model
  /// via the SwiftAcervo Component Registry.
  public var componentId: String {
    switch self {
    case .snac24kHz:
      return "snac-24khz"
    case .mimiPyTorchBF16:
      return "mimi-pytorch-bf16"
    case .vyvoTTSBeta4bit:
      return "vyvo-tts-beta-4bit"
    case .orpheusTTS:
      return "orpheus-tts-3b"
    case .sopranoTTS:
      return "soprano-tts-80m"
    case .marvisTTS:
      return "marvis-tts-250m-8bit"
    case .pocketTTS:
      return "pocket-tts"
    }
  }

  /// The corresponding Acervo component type.
  ///
  /// Codecs are decoders (latent-to-audio conversion).
  /// TTS models are language models (autoregressive generative models).
  public var componentType: ComponentType {
    switch self {
    case .snac24kHz, .mimiPyTorchBF16:
      return .decoder
    case .vyvoTTSBeta4bit, .orpheusTTS, .sopranoTTS, .marvisTTS, .pocketTTS:
      return .languageModel
    }
  }
}

// MARK: - Acervo Component Registration

/// Required files for the SNAC 24 kHz model variant.
///
/// These files are declared so that `Acervo.ensureComponentReady()` knows exactly
/// what to download before any codec operation is attempted.
private let snac24kHzRequiredFiles: [ComponentFile] = [
  ComponentFile(relativePath: "config.json"),
  ComponentFile(relativePath: "model.safetensors"),
]

/// Required files for the Mimi PyTorch BF16 model variant.
///
/// Single-file model: only the tokenizer checkpoint is required.
private let mimiPyTorchBF16RequiredFiles: [ComponentFile] = [
  ComponentFile(relativePath: "tokenizer-e351c8d8-checkpoint125.safetensors"),
]

// MARK: - P2 TTS Model Files

/// Required files for VyvoTTS / Qwen3 TTS (4-bit quantized variant).
///
/// Lightweight multi-language TTS with quantized weights for faster inference.
private let vyvoTTSBeta4bitRequiredFiles: [ComponentFile] = [
  ComponentFile(relativePath: "config.json"),
  ComponentFile(relativePath: "model.safetensors"),
  ComponentFile(relativePath: "tokenizer.json"),
]

/// Required files for Orpheus / LlamaTTS (3B params, bfloat16).
///
/// Generative TTS based on Llama architecture with high-quality synthesis.
private let orpheusTTSRequiredFiles: [ComponentFile] = [
  ComponentFile(relativePath: "config.json"),
  ComponentFile(relativePath: "model.safetensors"),
]

/// Required files for Soprano TTS (80M params, bfloat16).
///
/// Lightweight model optimized for Apple Silicon with single-file weight distribution.
private let sopranoTTSRequiredFiles: [ComponentFile] = [
  ComponentFile(relativePath: "config.json"),
  ComponentFile(relativePath: "model.safetensors"),
  ComponentFile(relativePath: "tokenizer.json"),
]

/// Required files for MarvisTTS (250M params, 8-bit quantized).
///
/// Premium quality TTS with speaker control, voice design, and multi-speaker synthesis.
private let marvisTTSRequiredFiles: [ComponentFile] = [
  ComponentFile(relativePath: "config.json"),
  ComponentFile(relativePath: "model.safetensors"),
  ComponentFile(relativePath: "tokenizer.json"),
]

/// Required files for PocketTTS (small, multi-voice).
///
/// Compact TTS model for on-device synthesis with multiple voice options.
private let pocketTTSRequiredFiles: [ComponentFile] = [
  ComponentFile(relativePath: "config.json"),
  ComponentFile(relativePath: "model.safetensors"),
  ComponentFile(relativePath: "tokenizer.json"),
]

/// All audio model component descriptors (P1 + P2).
///
/// P1 models: Stable codec models (SNAC, Mimi) hard-coded in applications.
/// P2 models: Extended TTS models for text-to-speech synthesis.
///
/// Registered at module initialization so the Acervo Component Registry
/// is populated before any model loading or download is attempted.
private let audioComponentDescriptors: [ComponentDescriptor] = [
  // MARK: - P1 Codecs (Stable)

  ComponentDescriptor(
    id: AudioModelRepo.snac24kHz.componentId,
    type: .decoder,
    displayName: "SNAC 24 kHz Audio Codec",
    repoId: AudioModelRepo.snac24kHz.rawValue,
    files: snac24kHzRequiredFiles,
    estimatedSizeBytes: 158_809_902,
    minimumMemoryBytes: 200_000_000,
    metadata: [
      "sampleRate": "24000",
      "bitrate": "0.98 kbps",
      "rvqLevels": "3",
      "modelType": "codec",
      "stage": "P1",
    ]
  ),
  ComponentDescriptor(
    id: AudioModelRepo.mimiPyTorchBF16.componentId,
    type: .decoder,
    displayName: "Mimi Audio Codec (PyTorch BF16)",
    repoId: AudioModelRepo.mimiPyTorchBF16.rawValue,
    files: mimiPyTorchBF16RequiredFiles,
    estimatedSizeBytes: 403_931_136,
    minimumMemoryBytes: 500_000_000,
    metadata: [
      "sampleRate": "24000",
      "numCodebooks": "32",
      "modelType": "codec",
      "stage": "P1",
    ]
  ),

  // MARK: - P2 TTS Models (Extended)

  ComponentDescriptor(
    id: AudioModelRepo.vyvoTTSBeta4bit.componentId,
    type: .languageModel,
    displayName: "VyvoTTS / Qwen3 TTS (4-bit)",
    repoId: AudioModelRepo.vyvoTTSBeta4bit.rawValue,
    files: vyvoTTSBeta4bitRequiredFiles,
    estimatedSizeBytes: 1_200_000_000,
    minimumMemoryBytes: 2_000_000_000,
    metadata: [
      "sampleRate": "22050",
      "quantization": "4-bit",
      "languages": "multi",
      "modelType": "tts",
      "stage": "P2",
    ]
  ),
  ComponentDescriptor(
    id: AudioModelRepo.orpheusTTS.componentId,
    type: .languageModel,
    displayName: "Orpheus TTS (LlamaTTS, BF16)",
    repoId: AudioModelRepo.orpheusTTS.rawValue,
    files: orpheusTTSRequiredFiles,
    estimatedSizeBytes: 6_500_000_000,
    minimumMemoryBytes: 8_000_000_000,
    metadata: [
      "sampleRate": "22050",
      "params": "3B",
      "dtype": "bfloat16",
      "modelType": "tts",
      "stage": "P2",
    ]
  ),
  ComponentDescriptor(
    id: AudioModelRepo.sopranoTTS.componentId,
    type: .languageModel,
    displayName: "Soprano TTS (80M, BF16)",
    repoId: AudioModelRepo.sopranoTTS.rawValue,
    files: sopranoTTSRequiredFiles,
    estimatedSizeBytes: 350_000_000,
    minimumMemoryBytes: 1_000_000_000,
    metadata: [
      "sampleRate": "22050",
      "params": "80M",
      "dtype": "bfloat16",
      "modelType": "tts",
      "stage": "P2",
    ]
  ),
  ComponentDescriptor(
    id: AudioModelRepo.marvisTTS.componentId,
    type: .languageModel,
    displayName: "MarvisTTS (250M, 8-bit)",
    repoId: AudioModelRepo.marvisTTS.rawValue,
    files: marvisTTSRequiredFiles,
    estimatedSizeBytes: 1_000_000_000,
    minimumMemoryBytes: 2_500_000_000,
    metadata: [
      "sampleRate": "22050",
      "params": "250M",
      "quantization": "8-bit",
      "speakers": "multi",
      "modelType": "tts",
      "stage": "P2",
    ]
  ),
  ComponentDescriptor(
    id: AudioModelRepo.pocketTTS.componentId,
    type: .languageModel,
    displayName: "PocketTTS (Multi-voice)",
    repoId: AudioModelRepo.pocketTTS.rawValue,
    files: pocketTTSRequiredFiles,
    estimatedSizeBytes: 600_000_000,
    minimumMemoryBytes: 1_500_000_000,
    metadata: [
      "sampleRate": "22050",
      "voices": "multi",
      "modelType": "tts",
      "stage": "P2",
    ]
  ),
]

/// Module-level registration trigger.
///
/// This `let` is evaluated once (lazily) on first access, registering all
/// P1 (codecs: SNAC, Mimi) and P2 (TTS: VyvoTTS, Orpheus, Soprano, MarvisTTS, PocketTTS)
/// audio component descriptors with the SwiftAcervo Component Registry.
/// Called automatically by the AudioModelManager or consumers that need
/// model availability checks.
private let _registerAudioComponents: Void = {
  Acervo.register(audioComponentDescriptors)
}()

// MARK: - AudioModelManager

/// Manager for audio model lifecycle and Acervo component integration.
///
/// Provides convenient methods to:
/// - Ensure component readiness (downloads if needed)
/// - Check if a model is available locally
/// - Get component metadata
///
/// All Acervo component registration happens at module init via the
/// lazy `_registerAudioComponents` initializer.
public enum AudioModelManager {
  /// Trigger lazy registration of all P1 and P2 audio components.
  ///
  /// This is called automatically on first use, but can be invoked
  /// explicitly to ensure early registration before any model loading.
  /// Registration includes both codec models (P1) and TTS models (P2).
  public static func ensureComponentsRegistered() {
    _ = _registerAudioComponents
  }

  /// Look up a registered audio model by its HuggingFace repo ID.
  ///
  /// - Parameter modelId: HuggingFace repo ID (e.g., `"mlx-community/snac_24khz"`)
  /// - Returns: The corresponding `AudioModelRepo`, or `nil` if not registered.
  public static func repo(for modelId: String) -> AudioModelRepo? {
    AudioModelRepo.allCases.first { $0.rawValue == modelId }
  }

  /// Get the component ID for a registered model by its HuggingFace repo ID.
  ///
  /// - Parameter modelId: HuggingFace repo ID (e.g., `"mlx-community/snac_24khz"`)
  /// - Returns: The Acervo component ID, or `nil` if not registered.
  public static func componentId(for modelId: String) -> String? {
    repo(for: modelId)?.componentId
  }

  /// Ensure a specific audio model component is ready for use.
  ///
  /// Downloads missing files from HuggingFace via SwiftAcervo if needed.
  /// Verification includes checksum validation for data integrity.
  ///
  /// - Parameters:
  ///   - modelRepo: The audio model variant to prepare.
  ///   - progress: Optional callback for download progress updates.
  /// - Throws: `AcervoError` if download or verification fails.
  public static func ensureModelReady(
    _ modelRepo: AudioModelRepo,
    progress: (@Sendable (AcervoDownloadProgress) -> Void)? = nil
  ) async throws {
    // Trigger registration if not already done
    ensureComponentsRegistered()

    // Use Acervo to ensure component is downloaded and ready
    try await Acervo.ensureComponentReady(
      modelRepo.componentId,
      progress: progress
    )
  }

  /// Check whether a model is available in Acervo's shared directory.
  ///
  /// - Parameter modelRepo: The audio model variant to check.
  /// - Returns: `true` if model is cached locally, `false` otherwise.
  public static func isModelAvailable(_ modelRepo: AudioModelRepo) -> Bool {
    Acervo.isModelAvailable(modelRepo.rawValue)
  }

  /// Get the local directory path for a model if it's cached.
  ///
  /// - Parameter modelRepo: The audio model variant to locate.
  /// - Returns: Local directory URL, or `nil` if not cached.
  public static func modelDirectory(
    for modelRepo: AudioModelRepo
  ) -> URL? {
    try? Acervo.modelDirectory(for: modelRepo.rawValue)
  }

  /// Get the Acervo component descriptor for a model.
  ///
  /// - Parameter modelRepo: The audio model variant to describe.
  /// - Returns: Component descriptor with metadata, or `nil` if unregistered.
  public static func component(
    for modelRepo: AudioModelRepo
  ) -> ComponentDescriptor? {
    Acervo.component(modelRepo.componentId)
  }
}

// MARK: - Extension: AudioModelRepo convenience

extension AudioModelRepo {
  /// Ensure this model is ready via AudioModelManager.
  public func ensureReady(
    progress: (@Sendable (AcervoDownloadProgress) -> Void)? = nil
  ) async throws {
    try await AudioModelManager.ensureModelReady(self, progress: progress)
  }

  /// Check if this model is available locally.
  public var isAvailable: Bool {
    AudioModelManager.isModelAvailable(self)
  }

  /// Get the local directory for this model if cached.
  public var localDirectory: URL? {
    AudioModelManager.modelDirectory(for: self)
  }
}
