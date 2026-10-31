import Foundation
import Testing
import MLX
@testable import MLXAudioCore
@testable import MLXAudioTTS

@Suite(.serialized)
struct Maya2IntegrationTests {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["RUN_MAYA2_INTEGRATION"] == "1"))
    func publishedCheckpointSynthesisAndStreaming() async throws {
        let repo = ProcessInfo.processInfo.environment["MAYA2_MODEL_PATH"] ?? "maya-research/maya2-native"
        let model = try #require(try await TTS.loadModel(modelRepo: repo) as? Maya2Model)
        var options = Maya2GenerationOptions(); options.depthTemperature = 0
        let parameters = GenerateParameters(maxTokens: 80, temperature: 0, topK: 64, seed: 42)
        let audio = try await model.generate(text: "Hello from Maya.", voice: "Siddhant", options: options, parameters: parameters)
        #expect(model.sampleRate == 24000)
        #expect(audio.size > model.sampleRate / 2)
        #expect(all(isFinite(audio)).item(Bool.self))
        var chunks: [MaterializedArray] = []
        for try await event in model.generateStream(text: "Hello from Maya.", voice: "Siddhant", options: options,
            parameters: parameters, streamingInterval: 0.32) {
            if case .audio(let samples) = event { chunks.append(samples) }
        }
        #expect(chunks.count > 1)
        let streamed = concatenated(chunks)
        #expect(streamed.shape == audio.shape)
        #expect(max(abs(streamed - audio)).item(Float.self) < 1e-6)
    }
}
