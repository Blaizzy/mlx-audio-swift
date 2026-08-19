//
//  MelFrontendReferenceTests.swift
//
//  Guards the Qwen3-ASR mel frontend against the reference implementation
//  (transformers' WhisperFeatureExtractor). The fixture below was generated
//  with WhisperFeatureExtractor(feature_size: 128, sample_rate: 16000,
//  n_fft: 400, hop_length: 160, dither: 0) on 0.1s of
//  0.5 * sin(2π·440·t) at 16 kHz (float32).
//
//  Regenerate with:
//    python -c "
//    import numpy as np
//    from transformers import WhisperFeatureExtractor
//    fx = WhisperFeatureExtractor(feature_size=128, sample_rate=16000, n_fft=400, hop_length=160, chunk_length=30, n_samples=480000, dither=0.0)
//    t = np.arange(1600) / 16000.0
//    sig = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
//    f = fx(sig, sampling_rate=16000, padding='do_not_pad', return_tensors='np').input_features[0]
//    for m in [0, 1, 2, 32, 64, 100]: print(f'bin{m}:', f[m, :4])
//    "
//

import XCTest

@testable import MLXAudioCore

final class MelFrontendReferenceTests: XCTestCase {

    private func referenceSignal() -> MLXArray {
        MLXArray((0..<1600).map { i in
            0.5 * sin(2 * Float.pi * 440.0 * Float(i) / 16000.0)
        })
    }

    /// Qwen3-ASR path (slaney mel + periodic hann) must reproduce the
    /// WhisperFeatureExtractor reference values.
    func testQwen3FrontendMatchesWhisperFeatureExtractor() throws {
        let mel = computeMelSpectrogram(
            audio: referenceSignal(), sampleRate: 16000, nFft: 400, hopLength: 160,
            nMels: 128, melScale: .slaney, hannPeriodic: true)
        // mel is [numFrames, nMels]; the reference layout is [nMels, numFrames].
        XCTAssertEqual(mel.dim(1), 128)
        let features = mel.transposed(1, 0).asArray(Float.self)
        let frames = mel.dim(0)
        XCTAssertGreaterThanOrEqual(frames, 10)

        let reference: [Int: [Float]] = [
            0: [0.907520, 0.395512, -0.514646, -0.514646],
            1: [1.005084, 0.493077, -0.514646, -0.514646],
            2: [0.986405, 0.475504, -0.514646, -0.514646],
            32: [0.821224, 0.330902, -0.514646, -0.514646],
            64: [0.413674, -0.094624, -0.514646, -0.514646],
            100: [0.065004, -0.444576, -0.514646, -0.514646],
        ]
        for (bin, values) in reference.sorted(by: { $0.key < $1.key }) {
            for (t, expected) in values.enumerated() {
                let actual = features[bin * frames + t]
                XCTAssertEqual(actual, expected, accuracy: 2e-3,
                               "mel bin \(bin) frame \(t) drifted from WhisperFeatureExtractor reference")
            }
        }
    }

    /// The legacy defaults (htk scale, symmetric window) must remain available
    /// and must differ from the Whisper-style path, so the opt-in parameters
    /// keep working for existing callers.
    func testMelScaleAndWindowOptionsChangeOutput() {
        let signal = referenceSignal()
        let whisper = computeMelSpectrogram(
            audio: signal, sampleRate: 16000, nFft: 400, hopLength: 160,
            nMels: 128, melScale: .slaney, hannPeriodic: true)
        let legacy = computeMelSpectrogram(
            audio: signal, sampleRate: 16000, nFft: 400, hopLength: 160,
            nMels: 128, melScale: .htk, hannPeriodic: false)
        XCTAssertNotEqual(whisper.max().item(Float.self),
                          legacy.max().item(Float.self),
                          accuracy: 1e-4)

        // torch.hann_window semantics: the symmetric window ends exactly at 0;
        // the periodic window ends at 0.5*(1-cos(2π(N-1)/N)) > 0, matching one
        // period of the underlying cosine.
        let n = 400
        let periodic = hanningWindow(size: n, periodic: true).asArray(Float.self)
        let periodicTail: Float = 0.5 * (1.0 - cos(2.0 * Float.pi * Float(n - 1) / Float(n)))
        XCTAssertEqual(periodic.last!, periodicTail, accuracy: 1e-6)
        let symmetric = hanningWindow(size: n, periodic: false).asArray(Float.self)
        XCTAssertEqual(symmetric.last!, 0, accuracy: 1e-6)
    }
}
