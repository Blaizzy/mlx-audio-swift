import Foundation
@preconcurrency import MLX
import MLXLMCommon
import Testing

@testable import MLXAudioTTS

/// End-to-end controllable-TTS smoke test: resolves Spark-TTS through the public
/// `TTS.loadModel` factory and verifies non-trivial audio is synthesized.
/// Skips if the checkpoint is unavailable.
@Test func sparkTTSGeneratesAudio() async throws {
    let model: SpeechGenerationModel
    do {
        model = try await TTS.loadModel(modelRepo: "mlx-community/Spark-TTS-0.5B-bf16")
    } catch {
        print("⚠️ skip: Spark-TTS checkpoint unavailable (\(error))")
        return
    }

    #expect(model is SparkModel)

    let audio = try await model.generate(
        text: "Hello world, this is a test.",
        voice: "female",
        refAudio: nil, refText: nil, language: nil,
        generationParameters: GenerateParameters(
            maxTokens: 2000, temperature: 0.8, topP: 0.95,
            repetitionPenalty: 1.3, repetitionContextSize: 20))
    eval(audio)

    let samples = audio.asType(.float32).asArray(Float.self)
    let maxAbs = samples.map { abs($0) }.max() ?? 0
    #expect(samples.count > 4000)
    #expect(maxAbs > 0.01)
}
