import Testing
import Foundation

@testable import MLXAudioCore
@testable import MLXAudioCodecs

extension SmokeTests.CodecsTests {

    @Suite("Mimi Tests", .serialized)
    struct MimiTests {

        @Test func mimiEncodeDecodeCycle() async throws {
            testHeader("mimiEncodeDecodeCycle")
            defer { testCleanup("mimiEncodeDecodeCycle") }
            let audioURL = Bundle.module.url(forResource: "intention", withExtension: "wav", subdirectory: "media")!
            let (sampleRate, audioData) = try loadAudioArray(from: audioURL)
            print("Loaded audio: \(audioData.shape), sample rate: \(sampleRate)")

            print("\u{001B}[33mLoading Mimi model...\u{001B}[0m")
            let mimi = try await Mimi.fromPretrained(
                repoId: "kyutai/moshiko-pytorch-bf16",
                filename: "tokenizer-e351c8d8-checkpoint125.safetensors"
            ) { progress in
                print("Download progress: \(progress.fractionCompleted * 100)%")
            }
            print("\u{001B}[32mMimi model loaded!\u{001B}[0m")

            let audioInput = audioData.reshaped([1, 1, audioData.shape[0]])
            print("Audio input shape: \(audioInput.shape)")

            print("\u{001B}[33mEncoding audio...\u{001B}[0m")
            let codes = mimi.encode(audioInput)
            print("Encoded to codes shape: \(codes.shape)")

            print("\u{001B}[33mDecoding audio...\u{001B}[0m")
            let reconstructed = mimi.decode(codes)
            print("Reconstructed audio shape: \(reconstructed.shape)")

            let outputURL = temporaryTestOutputURL("intention_mimi_reconstructed.wav")
            defer { removeTemporaryTestOutput(outputURL) }
            let outputAudio = reconstructed.squeezed()
            try saveAudioArray(outputAudio, sampleRate: mimi.sampleRate, to: outputURL)
            print("\u{001B}[32mSaved reconstructed audio to\u{001B}[0m: \(outputURL.path)")

            #expect(reconstructed.shape.last! > 0)
        }
    }
}
