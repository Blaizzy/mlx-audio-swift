import Testing
import MLX
import MLXLMCommon
import Foundation

@testable import MLXAudioCore
@testable import MLXAudioTTS

struct SopranoTextCleaningTests {

    @Test func testTextCleaning() {
        let text1 = "I have $100 and 50 cents."
        let cleaned1 = cleanTextForSoprano(text1)
        #expect(cleaned1.contains("one hundred dollars"), "Should expand dollar amounts")

        let text2 = "Dr. Smith went to the API conference."
        let cleaned2 = cleanTextForSoprano(text2)
        #expect(cleaned2.contains("doctor"), "Should expand Dr. to doctor")
        #expect(cleaned2.contains("a p i"), "Should expand API")

        let text3 = "This is the 1st and 2nd test."
        let cleaned3 = cleanTextForSoprano(text3)
        #expect(cleaned3.contains("first"), "Should expand 1st to first")
        #expect(cleaned3.contains("second"), "Should expand 2nd to second")

        print("\u{001B}[32mText cleaning tests passed!\u{001B}[0m")
    }
}

extension SmokeTests.TTSTests {

    @Suite("Soprano Tests", .serialized)
    struct SopranoTests {

        @Test func sopranoGenerate() async throws {
            testHeader("sopranoGenerate")
            defer { testCleanup("sopranoGenerate") }
            print("\u{001B}[33mLoading Soprano TTS model...\u{001B}[0m")
            let model = try await SopranoModel.fromPretrained("mlx-community/Soprano-1.1-80M-bf16")
            print("\u{001B}[32mSoprano model loaded!\u{001B}[0m")

            let text = "Performance Optimization: Automatic model quantization and hardware optimization that delivers 30%-100% faster inference than standard implementations."
            print("\u{001B}[33mGenerating audio for: \"\(text)\"...\u{001B}[0m")

            let parameters = GenerateParameters(
                maxTokens: 200,
                temperature: 0.3,
                topP: 0.95
            )

            let audio = try await model.generate(
                text: text,
                voice: nil,
                parameters: parameters
            )

            print("\u{001B}[32mGenerated audio shape: \(audio.shape)\u{001B}[0m")
            #expect(audio.shape[0] > 0, "Audio should have samples")

            let outputURL = temporaryTestOutputURL("soprano_test_output.wav")
            defer { removeTemporaryTestOutput(outputURL) }
            try saveAudioArray(audio, sampleRate: Double(model.sampleRate), to: outputURL)
            print("\u{001B}[32mSaved generated audio to\u{001B}[0m: \(outputURL.path)")
        }

        @Test func sopranoGenerateStream() async throws {
            testHeader("sopranoGenerateStream")
            defer { testCleanup("sopranoGenerateStream") }
            print("\u{001B}[33mLoading Soprano TTS model...\u{001B}[0m")
            let model = try await SopranoModel.fromPretrained("mlx-community/Soprano-80M-bf16")
            print("\u{001B}[32mSoprano model loaded!\u{001B}[0m")

            let text = "Streaming test for Soprano model. I think it's working."
            print("\u{001B}[33mStreaming generation for: \"\(text)\"...\u{001B}[0m")

            let parameters = GenerateParameters(
                maxTokens: 100,
                temperature: 0.3,
                topP: 1.0
            )

            var tokenCount = 0
            var finalAudio: MLXArray?
            var generationInfo: SopranoGenerationInfo?

            for try await event in model.generateStream(text: text, parameters: parameters) {
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

                let outputURL = temporaryTestOutputURL("soprano_stream_test_output.wav")
                defer { removeTemporaryTestOutput(outputURL) }
                try saveAudioArray(audio, sampleRate: Double(model.sampleRate), to: outputURL)
                print("\u{001B}[32mSaved streamed audio to\u{001B}[0m: \(outputURL.path)")
            }
        }
    }
}
