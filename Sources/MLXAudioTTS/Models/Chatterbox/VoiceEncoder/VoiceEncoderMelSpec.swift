//
//  VoiceEncoderMelSpec.swift
//  MLXAudio
//
//  Mel spectrogram computation for VoiceEncoder.
//  Ported from mlx-audio Python: chatterbox/voice_encoder/melspec.py
//

import Foundation
import MLX
import MLXAudioCore

/// Compute mel spectrogram for VoiceEncoder from waveform.
///
/// Uses STFT with hann window and mel filterbank with optional power/dB conversion.
///
/// **Important**: The mel filterbank normalization differs between Regular and Turbo:
/// - Regular: slaney norm + slaney mel scale (matching Python's `mel_filters(norm="slaney")`)
/// - Turbo: no norm + HTK mel scale (matching Python's `librosa.filters.mel()` defaults)
///
/// - Parameters:
///   - wav: Waveform (T,) or (B, T) at 16kHz.
///   - hp: Voice encoder configuration.
///   - isTurbo: Whether to use Turbo mel filterbank (HTK, no norm) vs Regular (slaney).
/// - Returns: Mel spectrogram (M, T') or (B, M, T').
public func voiceEncoderMelSpectrogram(
    _ wav: MLXArray,
    hp: VoiceEncoderConfiguration = .default,
    isTurbo: Bool = false
) -> MLXArray {
    var audio = wav
    let was1D = audio.ndim == 1
    if was1D {
        audio = audio.expandedDimensions(axis: 0)
    }

    // Create hann window
    let window = hanningWindow(size: hp.winSize)

    // STFT per batch item
    var specs = [MLXArray]()
    for i in 0 ..< audio.dim(0) {
        let spec = stft(
            audio: audio[i],
            window: window,
            nFft: hp.nFft,
            hopLength: hp.hopSize
        )
        specs.append(spec)
    }

    // Stack: (B, T', F)
    var spec = MLX.stacked(specs, axis: 0)

    // Magnitudes
    var specMagnitudes = MLX.abs(spec)

    // Apply power
    if hp.melPower != 1.0 {
        specMagnitudes = MLX.pow(specMagnitudes, MLXArray(hp.melPower))
    }

    // Mel filterbank:
    // - Turbo uses librosa.filters.mel() defaults: norm=None, HTK mel scale
    // - Regular uses slaney norm + slaney mel scale
    let filters: MLXArray
    if isTurbo {
        filters = melFilters(
            sampleRate: hp.sampleRate,
            nFft: hp.nFft,
            nMels: hp.numMels,
            fMin: Float(hp.fmin),
            fMax: Float(hp.fmax),
            norm: nil,
            melScale: .htk
        )
    } else {
        filters = melFilters(
            sampleRate: hp.sampleRate,
            nFft: hp.nFft,
            nMels: hp.numMels,
            fMin: Float(hp.fmin),
            fMax: Float(hp.fmax),
            norm: "slaney",
            melScale: .slaney
        )
    }

    // Apply: (B, T', F) @ (F, M) → (B, T', M) → transpose → (B, M, T')
    // melFilters returns (nFreqs, nMels) = (F, M) — already correct for matmul
    var mel = matmul(specMagnitudes, filters)
    mel = mel.transposed(0, 2, 1)

    // Convert to dB if needed
    if hp.melType == "db" {
        mel = MLXArray(Float(20.0)) * MLX.log10(MLX.maximum(mel, MLXArray(hp.stftMagnitudeMin)))
    }

    return was1D ? mel.squeezed(axis: 0) : mel
}
