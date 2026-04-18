//
//  MimiModelManager.swift
//  MLXAudioCodecs
//
//  Audio codec model lifecycle and Acervo component registration.
//  Manages Mimi audio codec model discovery, download, and availability.
//

import Foundation
import SwiftAcervo

// MARK: - Supported Mimi Models

/// Known Mimi model variants on HuggingFace.
public enum MimiModelRepo: String, CaseIterable, Sendable {
  /// Mimi PyTorch BF16 checkpoint for 24 kHz audio compression.
  /// High-quality neural audio codec optimized for real-time inference.
  case mimiPyTorchBF16 = "kyutai/moshiko-pytorch-bf16"

  /// Human-readable display name for the model variant.
  public var displayName: String {
    switch self {
    case .mimiPyTorchBF16:
      return "Mimi Audio Codec (PyTorch BF16)"
    }
  }

  /// The Acervo component ID for this model variant.
  ///
  /// Used to look up, download, and check availability of this model
  /// via the SwiftAcervo Component Registry.
  public var componentId: String {
    switch self {
    case .mimiPyTorchBF16:
      return "mimi-pytorch-bf16"
    }
  }
}

// MARK: - Acervo Component Registration

/// Required files for the Mimi PyTorch BF16 model variant.
///
/// These files are declared in each `ComponentDescriptor` so that
/// `Acervo.ensureComponentReady()` knows exactly what to download.
private let mimiPyTorchBF16RequiredFiles: [ComponentFile] = [
  ComponentFile(relativePath: "tokenizer-e351c8d8-checkpoint125.safetensors"),
]

/// The Mimi component descriptor.
///
/// Registered at module initialization so the Acervo Component Registry
/// is populated before any model loading or download is attempted.
private let mimiComponentDescriptors: [ComponentDescriptor] = [
  ComponentDescriptor(
    id: MimiModelRepo.mimiPyTorchBF16.componentId,
    type: .decoder,
    displayName: "Mimi Audio Codec (PyTorch BF16)",
    repoId: MimiModelRepo.mimiPyTorchBF16.rawValue,
    files: mimiPyTorchBF16RequiredFiles,
    estimatedSizeBytes: 403_931_136,
    minimumMemoryBytes: 500_000_000,
    metadata: ["sampleRate": "24000", "numCodebooks": "32"]
  ),
]

/// Module-level registration trigger.
///
/// This `let` is evaluated once (lazily) on first access, registering all
/// Mimi component descriptors with the SwiftAcervo Component Registry.
/// `Mimi.fromPretrained()` references this to ensure registration happens
/// before any model loading or download call.
private let _registerMimiComponents: Void = {
  Acervo.register(mimiComponentDescriptors)
}()

// MARK: - Model Loading Integration

extension Mimi {
  /// Ensure Mimi components are registered with Acervo before use.
  ///
  /// This method is called automatically by `fromPretrained()`,
  /// but can be invoked explicitly to trigger early registration.
  public static func ensureComponentsRegistered() {
    _ = _registerMimiComponents
  }
}
