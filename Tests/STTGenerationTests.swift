import MLX
import MLXAudioSTT
import Testing

private final class LegacySTTModel: STTGenerationModel {
    let defaultGenerationParameters = STTGenerateParameters(maxTokens: 17, language: "English")
    var receivedParameters: STTGenerateParameters?
    var receivedAudioShape: [Int]?

    func generate(audio: MLXArray, generationParameters: STTGenerateParameters) -> STTOutput {
        receivedParameters = generationParameters
        receivedAudioShape = audio.shape
        return STTOutput(text: "Legacy transcription", language: generationParameters.language)
    }

    func generateStream(
        audio: MLXArray, generationParameters: STTGenerateParameters
    ) -> AsyncThrowingStream<STTGeneration, Error> {
        AsyncThrowingStream { $0.finish() }
    }
}

private enum STTTestError: Error { case rejected }

private final class ThrowingSTTModel: STTGenerationModel {
    let defaultGenerationParameters = STTGenerateParameters(maxTokens: 23)
    var receivedMaxTokens: Int?

    func generate(audio: MLXArray, generationParameters: STTGenerateParameters) -> STTOutput {
        STTOutput(text: "Nonthrowing transcription")
    }

    func generateThrowing(
        audio: MLXArray, generationParameters: STTGenerateParameters
    ) throws -> STTOutput {
        receivedMaxTokens = generationParameters.maxTokens
        throw STTTestError.rejected
    }

    func generateStream(
        audio: MLXArray, generationParameters: STTGenerateParameters
    ) -> AsyncThrowingStream<STTGeneration, Error> {
        AsyncThrowingStream { $0.finish() }
    }
}

struct STTGenerationTests {
    @Test func existingGenerateRemainsNonthrowing() {
        let model: any STTGenerationModel = LegacySTTModel()
        let audio = MLXArray.zeros([8])
        #expect(model.generate(audio: audio).text == "Legacy transcription")
        #expect(model.generate(audio: audio, generationParameters: .init()).text == "Legacy transcription")
    }

    @Test func throwingDefaultDelegatesToExistingGenerate() throws {
        let legacy = LegacySTTModel()
        let model: any STTGenerationModel = legacy
        let audio = MLXArray.zeros([8])
        let defaultOutput = try model.generateThrowing(audio: audio)
        #expect(defaultOutput.text == "Legacy transcription")
        #expect(defaultOutput.language == "English")
        #expect(legacy.receivedParameters?.maxTokens == 17)
        #expect(legacy.receivedAudioShape == [8])

        let explicitOutput = try model.generateThrowing(
            audio: audio, generationParameters: .init(maxTokens: 31, language: "French"))
        #expect(explicitOutput.language == "French")
        #expect(legacy.receivedParameters?.maxTokens == 31)
        #expect(legacy.receivedAudioShape == [8])
    }

    @Test func throwingOverridePropagatesThroughSharedInterface() {
        let throwing = ThrowingSTTModel()
        let model: any STTGenerationModel = throwing
        let audio = MLXArray.zeros([8])
        #expect(throws: STTTestError.rejected) {
            try model.generateThrowing(audio: audio)
        }
        #expect(throwing.receivedMaxTokens == 23)
        #expect(throws: STTTestError.rejected) {
            try model.generateThrowing(audio: audio, generationParameters: .init(maxTokens: 47))
        }
        #expect(throwing.receivedMaxTokens == 47)
        #expect(model.generate(audio: audio).text == "Nonthrowing transcription")
    }
}
