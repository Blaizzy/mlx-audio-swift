import Testing
import MLX
import Foundation
@testable import MLXAudioSTT

// MARK: - WAV Loading Helper

private func loadWAV(from url: URL) throws -> MLXArray {
    let data = try Data(contentsOf: url)

    // Parse WAV header (simplified - assumes 16-bit PCM)
    guard data.count > 44 else {
        throw NSError(domain: "WAV", code: 1, userInfo: [NSLocalizedDescriptionKey: "File too small"])
    }

    // Read sample rate from header (bytes 24-27)
    let sampleRate = data.withUnsafeBytes { ptr -> UInt32 in
        ptr.load(fromByteOffset: 24, as: UInt32.self)
    }

    guard sampleRate == 16000 else {
        throw NSError(domain: "WAV", code: 2, userInfo: [NSLocalizedDescriptionKey: "Expected 16kHz, got \(sampleRate)Hz"])
    }

    // Skip header (44 bytes) and read audio samples
    let audioData = data.subdata(in: 44..<data.count)

    // Convert Int16 samples to Float32 normalized to [-1, 1]
    let sampleCount = audioData.count / 2
    var floatSamples = [Float](repeating: 0, count: sampleCount)

    audioData.withUnsafeBytes { ptr in
        let int16Ptr = ptr.bindMemory(to: Int16.self)
        for i in 0..<sampleCount {
            floatSamples[i] = Float(int16Ptr[i]) / 32768.0
        }
    }

    return MLXArray(floatSamples)
}

// MARK: - Tests

struct WhisperSessionTests {
    @Test func transcribe_invalidSampleRate_throws() async throws {
        let session = try await WhisperSession.fromPretrained(model: .largeTurbo)
        let audio = MLXArray.zeros([44100])

        await #expect(throws: WhisperError.self) {
            _ = try await session.transcribe(audio, sampleRate: 44100)
        }
    }

    @Test func cancel_stopsTranscription() async throws {
        let session = try await WhisperSession.fromPretrained(model: .largeTurbo)
        let audio = MLXArray.zeros([AudioConstants.nSamples])

        // Use type annotation to get streaming version
        let stream: AsyncThrowingStream<StreamingResult, Error> = session.transcribe(audio, sampleRate: AudioConstants.sampleRate)

        session.cancel()

        var wasCancelled = false
        do {
            for try await _ in stream { }
        } catch let error as WhisperError {
            if case .cancelled = error {
                wasCancelled = true
            }
        }

        #expect(wasCancelled)
    }

    @Test func fromPretrained_createsSession() async throws {
        let session = try await WhisperSession.fromPretrained(
            model: .largeTurbo,
            streaming: .default
        )

        // Session created successfully (type is non-optional)
        _ = session
    }

    @Test func transcribe_validSampleRate_streams() async throws {
        let session = try await WhisperSession.fromPretrained(model: .largeTurbo)
        let audio = MLXArray.zeros([AudioConstants.nSamples])

        var results: [StreamingResult] = []
        let stream: AsyncThrowingStream<StreamingResult, Error> = session.transcribe(audio, sampleRate: AudioConstants.sampleRate)
        for try await result in stream {
            results.append(result)
        }

        #expect(!results.isEmpty)
        #expect(results.last?.isFinal == true)
    }

    @Test(.disabled("Requires network and model download"))
    func transcribe_silentAudio_returnsEmptyOrMinimal() async throws {
        // Given: A session with tiny model
        let session = try await WhisperSession.fromPretrained(model: .tiny)

        // Silent audio (1 second)
        let audio = MLXArray.zeros([AudioConstants.sampleRate])

        // When: Transcribing
        var results: [StreamingResult] = []
        for try await result in session.transcribe(audio, sampleRate: AudioConstants.sampleRate) {
            results.append(result)
        }

        // Then: Should complete (may have minimal output for silence)
        #expect(!results.isEmpty)
        #expect(results.last?.isFinal == true)
    }

    @Test(.disabled("Requires network and model download"))
    func transcribe_realSpeech_returnsExpectedText() async throws {
        // Given: A session with tiny model and real speech audio
        // Audio contains: "The quick brown fox jumps over the lazy dog. This is a test of the speech to text system."
        let session = try await WhisperSession.fromPretrained(model: .tiny)

        guard let audioURL = Bundle.module.url(forResource: "test_speech", withExtension: "wav", subdirectory: "Resources") else {
            throw NSError(domain: "Test", code: 1, userInfo: [NSLocalizedDescriptionKey: "test_speech.wav not found in bundle"])
        }
        let audio = try loadWAV(from: audioURL)

        // When: Transcribing
        var results: [StreamingResult] = []
        for try await result in session.transcribe(audio, sampleRate: AudioConstants.sampleRate) {
            results.append(result)
            print("Streaming: \(result.text) (final: \(result.isFinal))")
        }

        // Then: Should produce transcription containing expected words
        #expect(!results.isEmpty)
        #expect(results.last?.isFinal == true)

        let finalText = results.last?.text.lowercased() ?? ""
        // Check for key words from the audio (allowing for minor transcription variations)
        #expect(finalText.contains("quick") || finalText.contains("fox") || finalText.contains("dog") || finalText.contains("test"))
    }
}
