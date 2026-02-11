//
//  MLXAudioSmokeTests.swift
//  MLXAudioTests
//
//  End-to-end inference smoke tests that download models from HuggingFace and run generation.
//  These are intentionally separated from the fast unit tests so CI can skip them easily.
//
//  Run all smoke tests (serialized):
//    xcodebuild test \
//      -scheme MLXAudio-Package \
//      -destination 'platform=macOS' \
//      -only-testing:MLXAudioTests/Smoke \
//      CODE_SIGNING_ALLOWED=NO
//
//  Run a single category:
//    -only-testing:'MLXAudioTests/Smoke/CodecsSmokeTests'
//    -only-testing:'MLXAudioTests/Smoke/TTSSmokeTests'
//    -only-testing:'MLXAudioTests/Smoke/STTSmokeTests'
//    -only-testing:'MLXAudioTests/Smoke/VADSmokeTests'
//
//  Run a single test (note the trailing parentheses for Swift Testing):
//    -only-testing:'MLXAudioTests/Smoke/STTSmokeTests/qwen3ASRTranscribe()'
//
//  Filter test results:
//   2>&1 | grep --color=never -E '(^􀟈 |^􁁛 |^􀢄 |^\*\* TEST|\x1b\[1;35m|model loaded|Encoded to|Reconstructed audio|Generating audio|Generated audio|Generated [0-9]+ tokens|Streaming|Saved |Received final|Found [0-9]|Processing time|Streaming complete|Chunk [0-9]|  [Tt]ext:|  prompt_tokens|  generation_tokens|  total_tokens|  prompt_tps|  generation_tps|total_time| peak_memory|Peak Memory|Prompt:.*tokens/s|Prompt Tokens|Total Time|SPEAKER audio|Sortformer Output|Audio input shape|Loading.*model|Loaded audio|ForcedAligner|Running forced|\[.*s - .*s\])'

import Testing
import MLX
import MLXLMCommon
import Foundation

@testable import MLXAudioCore
@testable import MLXAudioCodecs
@testable import MLXAudioTTS
@testable import MLXAudioSTT
@testable import MLXAudioVAD


// MARK: - Helpers

private let delimiter = String(repeating: "=", count: 60)

private func testHeader(_ name: String) {
    // Free memory left over from the previous test (locals are now out of scope)
    GPU.clearCache()
    GPU.resetPeakMemory()
    print("\n\u{001B}[1;35m\(delimiter)\u{001B}[0m")
    print("\u{001B}[1;35m  \(name)\u{001B}[0m")
    print("\u{001B}[1;35m\(delimiter)\u{001B}[0m")
}

private func testCleanup(_ name: String) {
    let peak = Double(GPU.peakMemory) / 1_073_741_824
    print("\u{001B}[1;35m\(delimiter) \(name) done (peak: \(String(format: "%.2f", peak)) GB)\u{001B}[0m\n")
}

// MARK: - Top-level serialized wrapper (all suites run sequentially)

@Suite("SmokeTests", .serialized)
struct SmokeTests {

// MARK: - Codecs Smoke Tests

@Suite("Codecs Smoke Tests", .serialized)
struct CodecsSmokeTests {

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

        let outputURL = audioURL.deletingLastPathComponent().appendingPathComponent("intention_snac_reconstructed.wav")
        let outputAudio = reconstructed.squeezed()
        try saveAudioArray(outputAudio, sampleRate: Double(snac.samplingRate), to: outputURL)
        print("\u{001B}[32mSaved reconstructed audio to\u{001B}[0m: \(outputURL.path)")

        #expect(reconstructed.shape.last! > 0)
    }

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

        let outputURL = audioURL.deletingLastPathComponent().appendingPathComponent("intention_mimi_reconstructed.wav")
        let outputAudio = reconstructed.squeezed()
        try saveAudioArray(outputAudio, sampleRate: mimi.sampleRate, to: outputURL)
        print("\u{001B}[32mSaved reconstructed audio to\u{001B}[0m: \(outputURL.path)")

        #expect(reconstructed.shape.last! > 0)
    }
}


// MARK: - TTS Smoke Tests

@Suite("TTS Smoke Tests", .serialized)
struct TTSSmokeTests {

    @Test func qwen3Generate() async throws {
        testHeader("qwen3Generate")
        defer { testCleanup("qwen3Generate") }
        print("\u{001B}[33mLoading Qwen3 TTS model...\u{001B}[0m")
        let model = try await Qwen3Model.fromPretrained("mlx-community/VyvoTTS-EN-Beta-4bit")
        print("\u{001B}[32mQwen3 model loaded!\u{001B}[0m")

        let text = "Hello, this is a test of the Qwen3 text to speech model."
        print("\u{001B}[33mGenerating audio for: \"\(text)\"...\u{001B}[0m")

        let parameters = GenerateParameters(
            maxTokens: 500,
            temperature: 0.6,
            topP: 0.8,
            repetitionPenalty: 1.3,
            repetitionContextSize: 20
        )

        let audio = try await model.generate(
            text: text,
            voice: nil,
            parameters: parameters
        )

        print("\u{001B}[32mGenerated audio shape: \(audio.shape)\u{001B}[0m")
        #expect(audio.shape[0] > 0, "Audio should have samples")

        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen3_test_output.wav")
        try saveAudioArray(audio, sampleRate: Double(model.sampleRate), to: outputURL)
        print("\u{001B}[32mSaved generated audio to\u{001B}[0m: \(outputURL.path)")
    }

    @Test func qwen3GenerateStream() async throws {
        testHeader("qwen3GenerateStream")
        defer { testCleanup("qwen3GenerateStream") }
        print("\u{001B}[33mLoading Qwen3 TTS model...\u{001B}[0m")
        let model = try await Qwen3Model.fromPretrained("mlx-community/VyvoTTS-EN-Beta-4bit")
        print("\u{001B}[32mQwen3 model loaded!\u{001B}[0m")

        let text = "Streaming test for Qwen3 model."
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
        var generationInfo: Qwen3GenerationInfo?

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

            let outputURL = FileManager.default.temporaryDirectory
                .appendingPathComponent("qwen3_stream_test_output.wav")
            try saveAudioArray(audio, sampleRate: Double(model.sampleRate), to: outputURL)
            print("\u{001B}[32mSaved streamed audio to\u{001B}[0m: \(outputURL.path)")
        }
    }

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

        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("llama_tts_test_output.wav")
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

            let outputURL = FileManager.default.temporaryDirectory
                .appendingPathComponent("llama_tts_stream_test_output.wav")
            try saveAudioArray(audio, sampleRate: Double(model.sampleRate), to: outputURL)
            print("\u{001B}[32mSaved streamed audio to\u{001B}[0m: \(outputURL.path)")
        }
    }

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

        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("pocket_tts_test_output.wav")
        try saveAudioArray(audio, sampleRate: Double(model.sampleRate), to: outputURL)
        print("\u{001B}[32mSaved generated audio to\u{001B}[0m: \(outputURL.path)")
    }

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

        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("soprano_test_output.wav")
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

            let outputURL = FileManager.default.temporaryDirectory
                .appendingPathComponent("soprano_stream_test_output.wav")
            try saveAudioArray(audio, sampleRate: Double(model.sampleRate), to: outputURL)
            print("\u{001B}[32mSaved streamed audio to\u{001B}[0m: \(outputURL.path)")
        }
    }
}


// MARK: - STT Smoke Tests

@Suite("STT Smoke Tests", .serialized)
struct STTSmokeTests {

    @Test func qwen3ASRTranscribe() async throws {
        testHeader("qwen3ASRTranscribe")
        defer { testCleanup("qwen3ASRTranscribe") }
        let audioURL = Bundle.module.url(forResource: "conversational_a", withExtension: "wav", subdirectory: "media")!
        let (sampleRate, audioData) = try loadAudioArray(from: audioURL)
        print("\u{001B}[33mLoaded audio: \(audioData.shape), sample rate: \(sampleRate)\u{001B}[0m")

        print("\u{001B}[33mLoading Qwen3 ASR model...\u{001B}[0m")
        let model = try await Qwen3ASRModel.fromPretrained("mlx-community/Qwen3-ASR-0.6B-4bit")
        print("\u{001B}[32mQwen3 ASR model loaded!\u{001B}[0m")

        let output = model.generate(audio: audioData)
        print("\u{001B}[32m Qwen3 ASR Transcription: \(output.text)\u{001B}[0m")
        print("\u{001B}[32m Qwen3 ASR Generation Stats: \(output)\u{001B}[0m")

        #expect(!output.text.isEmpty, "Transcription text should not be empty")
        #expect(output.generationTokens > 0, "Generation tokens should be greater than 0")
    }

    @Test func qwen3ASRTranscribeStream() async throws {
        testHeader("qwen3ASRTranscribeStream")
        defer { testCleanup("qwen3ASRTranscribeStream") }
        let audioURL = Bundle.module.url(forResource: "conversational_a", withExtension: "wav", subdirectory: "media")!
        let (sampleRate, audioData) = try loadAudioArray(from: audioURL)
        print("\u{001B}[33mLoaded audio: \(audioData.shape), sample rate: \(sampleRate)\u{001B}[0m")

        print("\u{001B}[33mLoading Qwen3 ASR model...\u{001B}[0m")
        let model = try await Qwen3ASRModel.fromPretrained("mlx-community/Qwen3-ASR-0.6B-4bit")
        print("\u{001B}[32mQwen3 ASR model loaded!\u{001B}[0m")

        print("\u{001B}[33mStreaming transcription ...\u{001B}[0m")

        var tokenCount = 0
        var transcribedText = ""
        var finalOutput: STTOutput?
        var generationInfo: STTGenerationInfo?

        for try await event in model.generateStream(audio: audioData) {
            switch event {
            case .token(let token):
                tokenCount += 1
                transcribedText += token
            case .info(let info):
                generationInfo = info
                print("\n\u{001B}[36m\(info.summary)\u{001B}[0m")
            case .result(let output):
                finalOutput = output
                print("\u{001B}[32m Qwen3 ASR Streaming Transcription: \(output.text)\u{001B}[0m")
                print("\u{001B}[32m Qwen3 ASR Streaming Stats: \(output)\u{001B}[0m")
            }
        }

        #expect(tokenCount > 0, "Should have generated tokens")
        #expect(finalOutput != nil, "Should have received final output")
        #expect(generationInfo != nil, "Should have received generation info")

        if let output = finalOutput {
            #expect(!output.text.isEmpty, "Transcription text should not be empty")
            #expect(output.generationTokens > 0, "Generation tokens should be greater than 0")
            print("\u{001B}[32m\(output)\u{001B}[0m")
        }
    }

    @Test func qwen3ForcedAlignerAlign() async throws {
        testHeader("qwen3ForcedAlignerAlign")
        defer { testCleanup("qwen3ForcedAlignerAlign") }
        let audioURL = Bundle.module.url(forResource: "conversational_a", withExtension: "wav", subdirectory: "media")!
        let (sampleRate, audioData) = try loadAudioArray(from: audioURL)
        print("\u{001B}[33mLoaded audio: \(audioData.shape), sample rate: \(sampleRate)\u{001B}[0m")

        print("\u{001B}[33mLoading Qwen3 ForcedAligner model...\u{001B}[0m")
        let model = try await Qwen3ForcedAlignerModel.fromPretrained("mlx-community/Qwen3-ForcedAligner-0.6B-4bit")
        print("\u{001B}[32mQwen3 ForcedAligner model loaded!\u{001B}[0m")

        let transcript = "Coffee's story likely begins in Ethiopia, where legend tells of a goat herder named Kaldi, who notices goats became energetic after eating red berries from a particular bush. Curious, he tried them himself and felt invigorated."

        print("\u{001B}[33mRunning forced alignment...\u{001B}[0m")
        let result = model.generate(audio: audioData, text: transcript, language: "English")

        print("\u{001B}[32m Qwen3 ForcedAligner Result:\u{001B}[0m")
        for item in result.items {
            print("\u{001B}[32m  [\(String(format: "%.3f", item.startTime))s - \(String(format: "%.3f", item.endTime))s] \(item.text)\u{001B}[0m")
        }

        #expect(!result.items.isEmpty, "Alignment should produce items")
        #expect(!result.text.isEmpty, "Alignment text should not be empty")

        for item in result.items {
            #expect(item.startTime >= 0, "Start time should be non-negative")
            #expect(item.endTime >= item.startTime, "End time should be >= start time")
        }
        print("\u{001B}[32m Qwen3 ForcedAligner Summary:\u{001B}[0m")
        print("\u{001B}[32m  Text: \(result.text)\u{001B}[0m")
        print("\u{001B}[32m  Prompt Tokens: \(result.promptTokens)\u{001B}[0m")
        print("\u{001B}[32m  Total Time: \(String(format: "%.3f", result.totalTime))s\u{001B}[0m")
        print("\u{001B}[32m  Peak Memory: \(String(format: "%.2f", result.peakMemoryUsage))GB\u{001B}[0m")
    }

    @Test func glmASRTranscribe() async throws {
        testHeader("glmASRTranscribe")
        defer { testCleanup("glmASRTranscribe") }
        let audioURL = Bundle.module.url(forResource: "conversational_a", withExtension: "wav", subdirectory: "media")!
        let (sampleRate, audioData) = try loadAudioArray(from: audioURL)
        print("\u{001B}[33mLoaded audio: \(audioData.shape), sample rate: \(sampleRate)\u{001B}[0m")

        print("\u{001B}[33mLoading GLMASR model...\u{001B}[0m")
        let model = try await GLMASRModel.fromPretrained("mlx-community/GLM-ASR-Nano-2512-4bit")
        print("\u{001B}[32mGLMASR model loaded!\u{001B}[0m")

        let output = model.generate(audio: audioData)
        print("\u{001B}[32m GLMASR Transcription: \(output.text)\u{001B}[0m")
        print("\u{001B}[32m GLMASR Generation Stats: \(output)\u{001B}[0m")

        #expect(output.generationTokens > 0, "Generation tokens should be greater than 0")
    }

    @Test func glmASRTranscribeStream() async throws {
        testHeader("glmASRTranscribeStream")
        defer { testCleanup("glmASRTranscribeStream") }
        let audioURL = Bundle.module.url(forResource: "conversational_a", withExtension: "wav", subdirectory: "media")!
        let (sampleRate, audioData) = try loadAudioArray(from: audioURL)
        print("\u{001B}[33mLoaded audio: \(audioData.shape), sample rate: \(sampleRate)\u{001B}[0m")

        print("\u{001B}[33mLoading GLMASR model...\u{001B}[0m")
        let model = try await GLMASRModel.fromPretrained("mlx-community/GLM-ASR-Nano-2512-4bit")
        print("\u{001B}[32mGLMASR model loaded!\u{001B}[0m")

        print("\u{001B}[33mStreaming transcription ...\u{001B}[0m")

        var tokenCount = 0
        var transcribedText = ""
        var finalOutput: STTOutput?
        var generationInfo: STTGenerationInfo?

        for try await event in model.generateStream(audio: audioData) {
            switch event {
            case .token(let token):
                tokenCount += 1
                transcribedText += token
            case .info(let info):
                generationInfo = info
                print("\n\u{001B}[36m\(info.summary)\u{001B}[0m")
            case .result(let output):
                finalOutput = output
                print("\u{001B}[32m GLMASR Streaming Transcription: \(output.text)\u{001B}[0m")
                print("\u{001B}[32m GLMASR Streaming Stats: \(output)\u{001B}[0m")
            }
        }

        #expect(tokenCount > 0, "Should have generated tokens")
        #expect(finalOutput != nil, "Should have received final output")
        #expect(generationInfo != nil, "Should have received generation info")

        if let output = finalOutput {
            #expect(output.generationTokens > 0, "Generation tokens should be greater than 0")
            print("\u{001B}[32m\(output)\u{001B}[0m")
        }
    }
}


// MARK: - VAD Smoke Tests

@Suite("VAD Smoke Tests", .serialized)
struct VADSmokeTests {

    private static func saveSegmentsJSON(
        _ segments: [DiarizationSegment],
        to path: String,
        mode: String,
        audioDuration: Float,
        processingTime: Double
    ) throws {
        var jsonSegments = [[String: Any]]()
        for seg in segments {
            jsonSegments.append([
                "start": seg.start,
                "end": seg.end,
                "speaker": seg.speaker
            ])
        }

        let result: [String: Any] = [
            "mode": mode,
            "audio_duration": audioDuration,
            "processing_time": processingTime,
            "num_segments": segments.count,
            "num_speakers": Set(segments.map { $0.speaker }).count,
            "segments": jsonSegments
        ]

        let data = try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: URL(fileURLWithPath: path))
        print("\u{001B}[32mSaved results to \(path)\u{001B}[0m")
    }

    @Test func sortformerOfflineInference() async throws {
        testHeader("sortformerOfflineInference")
        defer { testCleanup("sortformerOfflineInference") }
        let audioURL = Bundle.module.url(forResource: "multi_speaker", withExtension: "wav", subdirectory: "media")!
        let (sampleRate, audioData) = try loadAudioArray(from: audioURL)
        let audioDuration = Float(audioData.dim(0)) / Float(sampleRate)
        print("\u{001B}[33mLoaded audio: \(audioData.shape), sample rate: \(sampleRate), duration: \(String(format: "%.1f", audioDuration))s\u{001B}[0m")

        print("\u{001B}[33mLoading Sortformer model...\u{001B}[0m")
        let model = try await SortformerModel.fromPretrained("mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16")
        print("\u{001B}[32mSortformer model loaded!\u{001B}[0m")

        let output = try await model.generate(audio: audioData, verbose: true)

        print("\u{001B}[32mSortformer Output:\u{001B}[0m")
        print("\u{001B}[36m\(output.text)\u{001B}[0m")
        print("\u{001B}[32mFound \(output.numSpeakers) speakers, \(output.segments.count) segments\u{001B}[0m")
        print("\u{001B}[32mProcessing time: \(String(format: "%.2f", output.totalTime))s\u{001B}[0m")

        let outputPath = "/tmp/sortformer_offline_results.json"
        try Self.saveSegmentsJSON(
            output.segments, to: outputPath, mode: "offline",
            audioDuration: audioDuration, processingTime: output.totalTime
        )

        #expect(output.segments.count > 0, "Should detect at least one segment")
        #expect(output.numSpeakers > 0, "Should detect at least one speaker")

        for seg in output.segments {
            #expect(seg.start >= 0, "Start time should be non-negative")
            #expect(seg.end > seg.start, "End time should be after start time")
            #expect(seg.speaker >= 0 && seg.speaker < 4, "Speaker ID should be in range [0, 4)")
        }
    }

    @Test func sortformerStreamingInference() async throws {
        testHeader("sortformerStreamingInference")
        defer { testCleanup("sortformerStreamingInference") }
        let audioURL = Bundle.module.url(forResource: "multi_speaker", withExtension: "wav", subdirectory: "media")!
        let (sampleRate, audioData) = try loadAudioArray(from: audioURL)
        let audioDuration = Float(audioData.dim(0)) / Float(sampleRate)
        print("\u{001B}[33mLoaded audio: \(audioData.shape), sample rate: \(sampleRate), duration: \(String(format: "%.1f", audioDuration))s\u{001B}[0m")

        print("\u{001B}[33mLoading Sortformer model...\u{001B}[0m")
        let model = try await SortformerModel.fromPretrained("mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16")
        print("\u{001B}[32mSortformer model loaded!\u{001B}[0m")

        print("\u{001B}[33mStreaming diarization...\u{001B}[0m")

        let startTime = CFAbsoluteTimeGetCurrent()
        var chunkCount = 0
        var allSegments = [DiarizationSegment]()

        for try await chunkOutput in model.generateStream(audio: audioData, verbose: true) {
            chunkCount += 1
            allSegments.append(contentsOf: chunkOutput.segments)

            print("\u{001B}[36m  Chunk \(chunkCount): \(chunkOutput.segments.count) segments, \(chunkOutput.numSpeakers) speakers\u{001B}[0m")
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        print("\u{001B}[32mStreaming complete: \(chunkCount) chunks, \(allSegments.count) total segments in \(String(format: "%.2f", elapsed))s\u{001B}[0m")

        let outputPath = "/tmp/sortformer_streaming_results.json"
        try Self.saveSegmentsJSON(
            allSegments, to: outputPath, mode: "streaming",
            audioDuration: audioDuration, processingTime: elapsed
        )

        #expect(chunkCount > 0, "Should process at least one chunk")
        #expect(allSegments.count > 0, "Should detect segments across chunks")
    }

    @Test func sortformerChunkedInference() async throws {
        testHeader("sortformerChunkedInference")
        defer { testCleanup("sortformerChunkedInference") }
        let audioURL = Bundle.module.url(forResource: "multi_speaker", withExtension: "wav", subdirectory: "media")!
        let (sampleRate, audioData) = try loadAudioArray(from: audioURL)
        let audio = audioData
        print("Loaded audio: \(audio.shape), sample rate: \(sampleRate)")

        let model = try await SortformerModel.fromPretrained("mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16")

        let chunkSize = Int(5.0 * Float(sampleRate))
        var state = model.initStreamingState()
        var allSegments = [DiarizationSegment]()

        for start in stride(from: 0, to: audio.dim(0), by: chunkSize) {
            let end = min(start + chunkSize, audio.dim(0))
            let chunk = audio[start..<end]

            let (result, newState) = try await model.feed(
                chunk: chunk,
                state: state,
                threshold: 0.5
            )
            state = newState

            allSegments.append(contentsOf: result.segments)
            for seg in result.segments {
                print("Speaker \(seg.speaker): \(String(format: "%.2f", seg.start))s - \(String(format: "%.2f", seg.end))s")
            }
        }

        print("Total segments: \(allSegments.count)")
        #expect(allSegments.count > 0, "Should detect segments from chunked feed")
        #expect(state.framesProcessed > 0, "State should track processed frames")

        for seg in allSegments {
            #expect(seg.start >= 0)
            #expect(seg.end > seg.start)
            #expect(seg.speaker >= 0 && seg.speaker < 4)
        }
    }
}

} // end Smoke
