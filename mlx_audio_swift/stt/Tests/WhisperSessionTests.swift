import Testing
import MLX
@testable import MLXAudioSTT

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

        let stream = session.transcribe(audio, sampleRate: AudioConstants.sampleRate)

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

        #expect(session != nil)
    }

    @Test func transcribe_validSampleRate_streams() async throws {
        let session = try await WhisperSession.fromPretrained(model: .largeTurbo)
        let audio = MLXArray.zeros([AudioConstants.nSamples])

        var results: [StreamingResult] = []
        for try await result in session.transcribe(audio, sampleRate: AudioConstants.sampleRate) {
            results.append(result)
        }

        #expect(!results.isEmpty)
        #expect(results.last?.isFinal == true)
    }
}
