import Testing
import Foundation

@testable import MLXAudioCore
@testable import MLXAudioCodecs

extension SmokeTests.CodecsTests {

    @Suite("SNAC Tests", .serialized)
    struct SNACTests {

        @Test func snacEncodeDecodeCycle() async throws {
            testHeader("snacEncodeDecodeCycle")
            defer { testCleanup("snacEncodeDecodeCycle") }
            let audioURL = Bundle.module.url(forResource: "intention", withExtension: "wav", subdirectory: "media")!
            let (sampleRate, audioData) = try loadAudioArray(from: audioURL)
            print("Loaded audio: \(audioData.shape), sample rate: \(sampleRate)")

            print("\u{001B}[33mLoading SNAC model...\u{001B}[0m")
            let snac = try await SNAC.fromPretrained("mlx-community/snac_24khz")
            print("\u{001B}[32mSNAC model loaded!\u{001B}[0m")

            let audioInput = audioData.reshaped([1, 1, audioData.shape[0]])
            print("Audio input shape: \(audioInput.shape)")

            print("\u{001B}[33mEncoding audio...\u{001B}[0m")
            let codes = snac.encode(audioInput)
            print("Encoded to \(codes.count) codebook levels:")
            for (i, code) in codes.enumerated() {
                print("  Level \(i): \(code.shape)")
            }

            print("\u{001B}[33mDecoding audio...\u{001B}[0m")
            let reconstructed = snac.decode(codes)
            print("Reconstructed audio shape: \(reconstructed.shape)")

            let outputURL = temporaryTestOutputURL("intention_snac_reconstructed.wav")
            defer { removeTemporaryTestOutput(outputURL) }
            let outputAudio = reconstructed.squeezed()
            try saveAudioArray(outputAudio, sampleRate: Double(snac.samplingRate), to: outputURL)
            print("\u{001B}[32mSaved reconstructed audio to\u{001B}[0m: \(outputURL.path)")

            #expect(reconstructed.shape.last! > 0)
        }
    }
}
