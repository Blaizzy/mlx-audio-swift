import Testing
import MLX
import MLXLMCommon
import Foundation

@testable import MLXAudioCore
@testable import MLXAudioTTS

extension SmokeTests.TTSTests {

    @Suite("Llama TTS Tests", .serialized)
    struct LlamaTTSTests {

        @Test func llamaTTSGenerate() async throws {
            testHeader("llamaTTSGenerate")
            defer { testCleanup("llamaTTSGenerate") }
            print("\u{001B}[33mLoading LlamaTTS (Orpheus) model...\u{001B}[0m")
            let model = try await LlamaTTSModel.fromPretrained("mlx-community/orpheus-3b-0.1-ft-bf16")
            print("\u{001B}[32mLlamaTTS model loaded!\u{001B}[0m")

            let text = "Hello, this is a test of the Orpheus text to speech model."
            print("\u{001B}[33mGenerating audio for: \"\(text)\"...\u{001B}[0m")

            let parameters = GenerateParameters(
                maxTokens: 800,
                temperature: 0.7,
                topP: 0.8,
                repetitionPenalty: 1.3,
                repetitionContextSize: 20
            )

            let audio = try await model.generate(
                text: text,
                voice: "tara",
                parameters: parameters
            )

            print("\u{001B}[32mGenerated audio shape: \(audio.shape)\u{001B}[0m")
            #expect(audio.shape[0] > 0, "Audio should have samples")

            let outputURL = temporaryTestOutputURL("llama_tts_test_output.wav")
            defer { removeTemporaryTestOutput(outputURL) }
            try saveAudioArray(audio, sampleRate: Double(model.sampleRate), to: outputURL)
            print("\u{001B}[32mSaved generated audio to\u{001B}[0m: \(outputURL.path)")
        }

        @Test func llamaTTSGenerateStream() async throws {
            testHeader("llamaTTSGenerateStream")
            defer { testCleanup("llamaTTSGenerateStream") }
            print("\u{001B}[33mLoading LlamaTTS (Orpheus) model...\u{001B}[0m")
            let model = try await LlamaTTSModel.fromPretrained("mlx-community/orpheus-3b-0.1-ft-bf16")
            print("\u{001B}[32mLlamaTTS model loaded!\u{001B}[0m")

            let text = "Streaming test for Orpheus model."
            print("\u{001B}[33mStreaming generation for: \"\(text)\"...\u{001B}[0m")

            let parameters = GenerateParameters(
                maxTokens: 300,
                temperature: 0.6,
                topP: 0.8,
                repetitionPenalty: 1.3,
                repetitionContextSize: 20
            )

            var tokenCount = 0
            var finalAudio: MLXArray?
            var generationInfo: LlamaTTSGenerationInfo?

            for try await event in model.generateStream(text: text, voice: "tara", parameters: parameters) {
                switch event {
                case .token(_):
                    tokenCount += 1
                    if tokenCount % 50 == 0 {
                        print("  Generated \(tokenCount) tokens...")
                    }
                case .info(let info):
                    generationInfo = info
                    print("\u{001B}[36m\(info.summary)\u{001B}[0m")
                case .audio(let audio):
                    finalAudio = audio
                    print("\u{001B}[32mReceived final audio: \(audio.shape)\u{001B}[0m")
                }
            }

            #expect(tokenCount > 0, "Should have generated tokens")
            #expect(finalAudio != nil, "Should have received final audio")
            #expect(generationInfo != nil, "Should have received generation info")

            if let audio = finalAudio {
                #expect(audio.shape[0] > 0, "Audio should have samples")

                let outputURL = temporaryTestOutputURL("llama_tts_stream_test_output.wav")
                defer { removeTemporaryTestOutput(outputURL) }
                try saveAudioArray(audio, sampleRate: Double(model.sampleRate), to: outputURL)
                print("\u{001B}[32mSaved streamed audio to\u{001B}[0m: \(outputURL.path)")
            }
        }
    }
}
