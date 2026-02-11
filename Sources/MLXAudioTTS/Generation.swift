@preconcurrency import MLX
@preconcurrency import MLXLMCommon
import MLXAudioCore

public protocol SpeechGenerationModel: AnyObject {
    var sampleRate: Int { get }

    func generate(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray

    func generateStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error>
}
