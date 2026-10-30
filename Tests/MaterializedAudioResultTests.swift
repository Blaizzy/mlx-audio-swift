import MLX
import MLXAudioCore
import MLXAudioSTS
import MLXAudioTTS
import MLXAudioVAD
import Testing
@testable import MLXAudioSTT

@Suite("Materialized audio results", .serialized)
struct MaterializedAudioResultTests {
    @Test func preparedVoiceReferencesCanBeSharedAndReused() async {
        var features = MLXArray([Float(1), 2, 3, 4]).reshaped([1, 2, 2])
        let reference = TADAReference(text: "hello", tokens: [1, 2], positions: [0, 1], features: features)
        let conditioning = DefaultConditioning(
            speakerEmb: features, condPromptSpeechTokens: MLXArray([Int32(1), 2]),
            emotionAdv: MLXArray(Float(0.5)), xVector: features,
            promptToken: MLXArray([Int32(3)]), promptTokenLen: MLXArray(Int32(1)),
            promptFeat: features, promptFeatLen: MLXArray(Int32(2)))
        features += 10
        #expect(await readOnWorker(reference) { $0.features.asArray(Float.self) } == [1, 2, 3, 4])
        #expect(await readOnWorker(conditioning) { $0.speakerEmb.asArray(Float.self) } == [1, 2, 3, 4])
        #expect(reference.features.shape == [1, 2, 2])
        #expect(conditioning.xVector.asArray(Float.self) == reference.features.asArray(Float.self))
        #expect(conditioning.condPromptSpeechTokens.dtype == .int32)
    }

    @Test(arguments: [0, 31, 10, 21])
    func preprocessedBatchesPreserveOptionalTensorPositions(mask: Int) async {
        let optionalTensors: [MLXArray?] = (0..<5).map { index in
            mask & (1 << index) == 0 ? nil : MLXArray(Int32(index + 1))
        }
        let batch = SAMAudioBatch(
            audios: MLXArray.zeros([1, 1, 8]), sizes: optionalTensors[0], wavSizes: optionalTensors[1],
            descriptions: ["speech"], anchorIDs: optionalTensors[2], anchorAlignment: optionalTensors[3],
            audioPadMask: optionalTensors[4])
        let actual = await readOnWorker(batch) {
            [$0.sizes, $0.wavSizes, $0.anchorIDs, $0.anchorAlignment, $0.audioPadMask]
                .map { $0?.item(Float.self) ?? -1 }
        }
        let expected = (0..<5).map { index in mask & (1 << index) == 0 ? Float(-1) : Float(index + 1) }
        #expect(actual == expected)
        #expect(batch.audios.shape == [1, 1, 8])
        #expect(batch.descriptions == ["speech"])
    }

    @Test func omniVoicePromptKeepsCodesAsASharedTensor() async {
        let codes = MLXArray([Int32(1), 2, 3, 4]).reshaped([1, 2, 2]).materialized()
        let prompt = OmniVoiceVoicePrompt(codes: codes, transcript: "hello")
        #expect(prompt.codes === codes)
        #expect(prompt.shape == codes.shape)
        #expect(await readOnWorker(prompt) { $0.codes.asArray(Float.self) } == [1, 2, 3, 4])
        #expect(prompt.codes.dtype == .int32)
        #expect(prompt.transcript == "hello")
        let converted = OmniVoiceVoicePrompt(codes: codes.asType(.int64), transcript: "again")
        #expect(converted.codes.dtype == .int32)
        #expect(converted.codes.asArray(Int32.self) == [1, 2, 3, 4])
    }

    @Test func whisperFilterbankCacheSharesMaterializedValues() async {
        async let first = Task.detached { WhisperAudio.melFilters(nMels: 80) }.value
        async let second = Task.detached { WhisperAudio.melFilters(nMels: 80) }.value
        let (a, b) = await (first, second)
        #expect(a === b)
        #expect(a.shape == [201, 80])
        #expect(a.asArray(Float.self) == b.asArray(Float.self))
        #expect(WhisperAudio.melFilters(nMels: 128).shape == [201, 128])
    }

    @Test func audioEventSnapshotsRemainStableAndCanBeShared() async throws {
        var source = MLXArray([Float(1), 2, 3, 4, 5, 6])
            .asType(.float16).reshaped([2, 3]).transposed(1, 0)
        let event = AudioGeneration.audio(source)
        guard case .audio(let snapshot) = event else {
            Issue.record("Expected an audio event")
            return
        }
        source += 10

        async let first = readOnWorker(event) { event in
            guard case .audio(let samples) = event else { return [] }
            return samples.asArray(Float.self)
        }
        async let second = readOnWorker(event) { event in
            guard case .audio(let samples) = event else { return [] }
            return samples.asArray(Float.self)
        }
        #expect(await first == [1, 4, 2, 5, 3, 6])
        #expect(await second == [1, 4, 2, 5, 3, 6])
        #expect(snapshot.shape == [3, 2])
        #expect(snapshot.dtype == .float16)
        #expect(source.asArray(Float.self) == [11, 14, 12, 15, 13, 16])
    }

    @Test func referenceConditioningSnapshotsAllTensors() async {
        var speaker = MLXArray([Float(0.25), 0.5])
        var codes = MLXArray([Int32(1), 2]).reshaped([1, 2])
        var text = MLXArray([Int32(3), 4]).reshaped([1, 2])
        let conditioning = Qwen3TTSModel.Qwen3TTSReferenceConditioning(
            speakerEmbedding: speaker, referenceSpeechCodes: codes,
            referenceTextTokenIDs: text, resolvedLanguage: "en", codecLanguageID: 0)
        speaker += 1
        codes += 10
        text += 20

        let samples = await readOnWorker(conditioning) {
            ($0.speakerEmbedding?.asArray(Float.self) ?? [])
                + $0.referenceSpeechCodes.asArray(Float.self)
                + $0.referenceTextTokenIDs.asArray(Float.self)
        }
        #expect(samples == [0.25, 0.5, 1, 2, 3, 4])
        #expect(conditioning.referenceSpeechCodes.dtype == .int32)
        #expect(conditioning.referenceSpeechCodes.shape == [1, 2])
        #expect(conditioning.speakerEmbedding?.asArray(Float.self) == [0.25, 0.5])
    }

    @Test func separationResultsPreserveEveryOutputAndOptionalNoise() async {
        var source = MLXArray([Float(1), 2, 3, 4, 5, 6])
        let result = SAMAudioSeparationResult(
            target: [source[0..<2], source[2..<4]], residual: [source[4..<6]],
            noise: source + 10, peakMemoryGB: 0.5)
        source *= 2

        let samples = await readOnWorker(result) {
            $0.target.flatMap { $0.asArray(Float.self) }
                + $0.residual.flatMap { $0.asArray(Float.self) }
                + ($0.noise?.asArray(Float.self) ?? [])
        }
        #expect(samples == [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16])
        #expect(result.target.count == 2)
        #expect(result.residual.count == 1)
        #expect(result.target[0].asArray(Float.self) == [1, 2])
        #expect(result.peakMemoryGB == 0.5)

        let empty = SAMAudioSeparationResult(target: [], residual: [])
        #expect(empty.target.isEmpty)
        #expect(empty.residual.isEmpty)
        #expect(empty.noise == nil)
    }

    @Test func streamingChunksAndDiarizationShareStableSnapshots() async {
        var source = MLXArray([Float(1), 2, 3])
        let enhanced = DeepFilterNetStreamingChunk(audio: source, chunkIndex: 2, isLastChunk: true)
        let separated = SAMAudioStreamingChunk(
            target: source, residual: source + 1, chunkIndex: 2, isLastChunk: true,
            noise: source + 2)
        let diarization = DiarizationOutput(segments: [], speakerProbs: source, numSpeakers: 2)
        source += 100

        #expect(await readOnWorker(enhanced) { $0.audio.asArray(Float.self) } == [1, 2, 3])
        #expect(await readOnWorker(separated) {
            $0.target.asArray(Float.self) + $0.residual.asArray(Float.self)
                + ($0.noise?.asArray(Float.self) ?? [])
        } == [1, 2, 3, 2, 3, 4, 3, 4, 5])
        #expect(await readOnWorker(diarization) { $0.speakerProbs?.asArray(Float.self) ?? [] } == [1, 2, 3])
        #expect(enhanced.audio.asArray(Float.self) == [1, 2, 3])
        #expect(enhanced.chunkIndex == 2)
        #expect(separated.isLastChunk)
        #expect(diarization.numSpeakers == 2)
    }

    @Test func absentConditioningAndChunkTensorsStayAbsent() async {
        let conditioning = Qwen3TTSModel.Qwen3TTSReferenceConditioning(
            speakerEmbedding: nil, referenceSpeechCodes: MLXArray.zeros([1, 0], type: Int32.self),
            referenceTextTokenIDs: MLXArray([Int32(3)]), resolvedLanguage: "en", codecLanguageID: nil)
        let chunk = SAMAudioStreamingChunk(
            target: MLXArray.zeros([0]), residual: MLXArray.zeros([0]), chunkIndex: 0, isLastChunk: true)
        #expect(await readOnWorker(conditioning) { $0.referenceTextTokenIDs.asArray(Float.self) } == [3])
        #expect(conditioning.speakerEmbedding == nil)
        #expect(conditioning.referenceSpeechCodes.size == 0)
        #expect(chunk.noise == nil)
        #expect(DiarizationOutput(segments: []).speakerProbs == nil)
    }
}

private func readOnWorker<Value: Sendable>(
    _ value: Value, read: @escaping @Sendable (Value) -> [Float]
) async -> [Float] {
    await Task.detached { read(value) }.value
}
