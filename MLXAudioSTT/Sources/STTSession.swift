//
//  STTSession.swift
//  MLXAudioSTT
//
//  Protocol for Speech-to-Text sessions.
//  API design aligned with MLX-Audio Swift SDK v1.
//

import Foundation
import MLX

// MARK: - STT Session Protocol

/// Protocol for speech-to-text sessions.
public protocol STTSessionProtocol: Sendable {
    /// Generate transcription from audio (blocking).
    ///
    /// - Parameters:
    ///   - audio: Audio samples as MLXArray (mono, float32)
    ///   - maxTokens: Maximum tokens to generate
    ///   - temperature: Sampling temperature (0 = greedy)
    /// - Returns: Transcription output with text and metadata
    func generate(
        audio: MLXArray,
        maxTokens: Int,
        temperature: Float
    ) async throws -> STTOutput

    /// Generate transcription from audio with streaming.
    ///
    /// - Parameters:
    ///   - audio: Audio samples as MLXArray (mono, float32)
    ///   - maxTokens: Maximum tokens to generate
    ///   - temperature: Sampling temperature (0 = greedy)
    /// - Returns: Async stream of generation events
    func generateStream(
        audio: MLXArray,
        maxTokens: Int,
        temperature: Float
    ) -> AsyncThrowingStream<STTGeneration, Error>
}

// MARK: - Default Implementations

extension STTSessionProtocol {
    /// Generate with default parameters.
    public func generate(audio: MLXArray) async throws -> STTOutput {
        try await generate(audio: audio, maxTokens: 448, temperature: 0.0)
    }

    /// Generate stream with default parameters.
    public func generateStream(audio: MLXArray) -> AsyncThrowingStream<STTGeneration, Error> {
        generateStream(audio: audio, maxTokens: 448, temperature: 0.0)
    }
}
