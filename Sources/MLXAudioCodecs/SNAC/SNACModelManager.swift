//
//  SNACModelManager.swift
//  MLXAudioCodecs
//
//  Audio codec model lifecycle and Acervo component registration.
//  Manages SNAC (Scalable Neural Audio Codec) decoder model discovery,
//  download, and availability.
//

import Foundation
import SwiftAcervo

// MARK: - Supported SNAC Models

/// Known SNAC model variants on HuggingFace.
public enum SNACModelRepo: String, CaseIterable, Sendable {
  /// 24 kHz audio codec (primary variant for TTS applications).
  /// Compresses 24 kHz audio to discrete codes at 0.98 kbps using 3 RVQ levels.
  /// Token rates: 12, 23, and 47 Hz.
  case snac24kHz = "mlx-community/snac_24khz"

  /// Human-readable display name for the model variant.
  public var displayName: String {
    switch self {
    case .snac24kHz:
      return "SNAC 24 kHz Audio Codec"
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
    }
  }
}

// MARK: - Acervo Component Registration

/// Required files for the SNAC 24 kHz model variant.
///
/// These files are declared in each `ComponentDescriptor` so that
/// `Acervo.ensureComponentReady()` knows exactly what to download.
private let snac24kHzRequiredFiles: [ComponentFile] = [
  ComponentFile(relativePath: "config.json"),
  ComponentFile(relativePath: "model.safetensors"),
]

/// The SNAC 24 kHz component descriptor.
///
/// Registered at module initialization so the Acervo Component Registry
/// is populated before any model loading or download is attempted.
private let snacComponentDescriptors: [ComponentDescriptor] = [
  ComponentDescriptor(
    id: SNACModelRepo.snac24kHz.componentId,
    type: .decoder,
    displayName: "SNAC 24 kHz Audio Codec",
    repoId: SNACModelRepo.snac24kHz.rawValue,
    files: snac24kHzRequiredFiles,
    estimatedSizeBytes: 158_809_902,
    minimumMemoryBytes: 200_000_000,
    metadata: ["sampleRate": "24000", "bitrate": "0.98 kbps", "rvqLevels": "3"]
  ),
]

/// Module-level registration trigger.
///
/// This `let` is evaluated once (lazily) on first access, registering all
/// SNAC component descriptors with the SwiftAcervo Component Registry.
/// `SNAC.fromPretrained()` references this to ensure registration happens
/// before any model loading or download call.
private let _registerSNACComponents: Void = {
  Acervo.register(snacComponentDescriptors)
}()

// MARK: - Model Loading Integration

extension SNAC {
  /// Ensure SNAC components are registered with Acervo before use.
  ///
  /// This method is called automatically by `fromPretrained()`,
  /// but can be invoked explicitly to trigger early registration.
  public static func ensureComponentsRegistered() {
    _ = _registerSNACComponents
  }
}
