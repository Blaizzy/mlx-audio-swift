import Foundation
import Testing
@testable import MLXAudioSTT

struct WhisperConfigurationTests {
    @Test func largeTurboConfigValues() {
        let config = WhisperConfiguration.largeTurbo

        // Model dimensions
        #expect(config.nMels == 128)
        #expect(config.nAudioCtx == 1500)
        #expect(config.nAudioState == 1280)
        #expect(config.nAudioHead == 20)
        #expect(config.nAudioLayer == 32)

        // Decoder dimensions
        #expect(config.nVocab == 51866)
        #expect(config.nTextCtx == 448)
        #expect(config.nTextState == 1280)
        #expect(config.nTextHead == 20)
        #expect(config.nTextLayer == 4)
    }

    @Test func largeV3ConfigValues() {
        let config = WhisperConfiguration.largeV3

        #expect(config.nMels == 128)
        #expect(config.nAudioLayer == 32)
        #expect(config.nTextLayer == 32)  // Full decoder, not turbo
    }

    @Test func configHasAlignmentHeads() {
        let config = WhisperConfiguration.largeTurbo
        #expect(!config.alignmentHeads.isEmpty)
    }

    @Test func largeTurboHasCorrectAlignmentHeadCount() {
        let config = WhisperConfiguration.largeTurbo
        #expect(config.alignmentHeads.count == 6)
    }

    @Test func largeV3HasCorrectAlignmentHeadCount() {
        let config = WhisperConfiguration.largeV3
        #expect(config.alignmentHeads.count == 20)
    }

    @Test func jsonEncodingDecoding() throws {
        let original = WhisperConfiguration.largeTurbo
        let encoder = JSONEncoder()
        let decoder = JSONDecoder()

        let data = try encoder.encode(original)
        let decoded = try decoder.decode(WhisperConfiguration.self, from: data)

        #expect(decoded.nMels == original.nMels)
        #expect(decoded.nAudioCtx == original.nAudioCtx)
        #expect(decoded.nAudioState == original.nAudioState)
        #expect(decoded.nAudioHead == original.nAudioHead)
        #expect(decoded.nAudioLayer == original.nAudioLayer)
        #expect(decoded.nVocab == original.nVocab)
        #expect(decoded.nTextCtx == original.nTextCtx)
        #expect(decoded.nTextState == original.nTextState)
        #expect(decoded.nTextHead == original.nTextHead)
        #expect(decoded.nTextLayer == original.nTextLayer)
        #expect(decoded.alignmentHeads.count == original.alignmentHeads.count)
    }

    @Test func jsonUsesSnakeCaseKeys() throws {
        let config = WhisperConfiguration.largeTurbo
        let encoder = JSONEncoder()

        let data = try encoder.encode(config)
        let jsonString = String(data: data, encoding: .utf8)!

        #expect(jsonString.contains("n_mels"))
        #expect(jsonString.contains("n_audio_ctx"))
        #expect(jsonString.contains("n_audio_state"))
        #expect(jsonString.contains("alignment_heads"))
        #expect(!jsonString.contains("nMels"))
    }
}
