import Testing
import MLXLMCommon
import Foundation

@testable import MLXAudioCore
@testable import MLXAudioTTS

extension SmokeTests.TTSTests {

    @Suite("Pocket TTS Tests", .serialized)
    struct PocketTTSTests {

        @Test func pocketTTSGenerate() async throws {
            testHeader("pocketTTSGenerate")
            defer { testCleanup("pocketTTSGenerate") }
            print("\u{001B}[33mLoading PocketTTS model...\u{001B}[0m")
            let model = try await PocketTTSModel.fromPretrained("mlx-community/pocket-tts")
            print("\u{001B}[32mPocketTTS model loaded!\u{001B}[0m")

            let text = "Hello, this is a test of the PocketTTS model."
            print("\u{001B}[33mGenerating audio for: \"\(text)\"...\u{001B}[0m")

            let audio = try await model.generate(
                text: text,
                voice: "alba",
                generationParameters: GenerateParameters(temperature: 0.7)
            )

            print("\u{001B}[32mGenerated audio shape: \(audio.shape)\u{001B}[0m")
            #expect(audio.shape[0] > 0, "Audio should have samples")

            let outputURL = temporaryTestOutputURL("pocket_tts_test_output.wav")
            defer { removeTemporaryTestOutput(outputURL) }
            try saveAudioArray(audio, sampleRate: Double(model.sampleRate), to: outputURL)
            print("\u{001B}[32mSaved generated audio to\u{001B}[0m: \(outputURL.path)")
        }
    }
}
