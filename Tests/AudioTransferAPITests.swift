import Foundation
import MLX
import MLXAudioCore
@testable import MLXAudioTTS
import MLXLMCommon
import MLXAudioVAD
import Testing

@Suite("Audio transfer API", .serialized)
struct AudioTransferAPITests {
    @MainActor
    @Test func indexTTSResultsTransferWithoutConsumingConditioning() async throws {
        let producer = IndexTTSProducer()
        let first = try await producer.prepare()
        let firstSamples = first.embeddings.asArray(Float.self)
        let second = try await producer.prepare()
        #expect(firstSamples == second.embeddings.asArray(Float.self))
        #expect(first.embeddings !== second.embeddings)

        let generated = try await producer.generate()
        #expect(generated.latentStates.ndim == 3)
        #expect(generated.latentStates.dim(2) == first.embeddings.dim(2))
        #expect(first.embeddings.asArray(Float.self) == firstSamples)
    }

    @MainActor
    @Test func sortformerFeedReturnsResultsToCaller() async throws {
        let model = SortformerModel(try tinySortformerConfig())
        let store = StreamingStateStore(model.initStreamingState())
        var previousFrames = 0
        for _ in 0..<3 {
            let state = await store.state
            let alias = state
            let (result, nextState) = try await model.feed(
                chunk: MLXArray(referenceSamples()), state: state,
                spkcacheMax: 2, fifoMax: 2)
            #expect(result.speakerProbs?.dim(1) == 2)
            #expect(nextState.framesProcessed > previousFrames)
            #expect(nextState.spkcacheLen <= 2)
            #expect(nextState.fifoLen <= 2)
            previousFrames = nextState.framesProcessed
            await store.save(nextState)
            do {
                _ = try await model.feed(chunk: MLXArray(referenceSamples()), state: alias)
                Issue.record("A copied, consumed handle was accepted")
            } catch {
                #expect(error as? StreamingState.ConsumptionError == .alreadyConsumed)
            }
        }
        let finalState = await store.state
        #expect(finalState.fifoLen > 0)
    }

    @MainActor
    @Test func sortformerStreamingStepConsumesHandle() throws {
        let config = try tinySortformerConfig()
        let model = SortformerModel(config)
        let state = model.initStreamingState()
        let features = MLXArray.zeros([1, config.processorConfig.featureSize, 32])
        let length = MLXArray([Int32(32)])
        let (first, nextState) = try model.streamingStep(
            chunkFeatures: features, chunkLength: length, state: state)
        let firstSamples = first.asArray(Float.self)
        #expect(nextState.framesProcessed > 0)
        #expect(throws: StreamingState.ConsumptionError.alreadyConsumed) {
            _ = try model.streamingStep(
                chunkFeatures: features, chunkLength: length, state: state)
        }
        let (_, finalState) = try model.streamingStep(
            chunkFeatures: features, chunkLength: length, state: nextState)
        #expect(finalState.framesProcessed > nextState.framesProcessed)
        #expect(first.asArray(Float.self) == firstSamples)
    }

    @MainActor
    @Test func sortformerStreamTransfersResultsAcrossActors() async throws {
        let producer = try DiarizationProducer()
        let stream = await producer.stream(samples: referenceSamples(count: 8_000))
        var first: MLXArray?
        var firstSamples: [Float] = []
        var chunks = 0
        for try await result in stream {
            let probabilities = try #require(result.speakerProbs)
            #expect(probabilities.dim(1) == 2)
            #expect(probabilities.asArray(Float.self).allSatisfy { $0.isFinite })
            if first == nil {
                first = probabilities
                firstSamples = probabilities.asArray(Float.self)
            }
            chunks += 1
        }
        #expect(chunks > 1)
        #expect(first?.asArray(Float.self) == firstSamples)
    }

    @MainActor
    @Test func loaderTransfersTensorAcrossActorBoundary() async throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString).appendingPathExtension("wav")
        defer { try? FileManager.default.removeItem(at: url) }
        let samples = (0..<320).map { Float(sin(Double($0) * 0.1)) * 0.25 }
        try AudioUtils.writeWavFile(samples: samples, sampleRate: 16_000, fileURL: url)

        let loader = AudioLoader()
        let (rate, loaded) = try await loader.load(url)
        #expect(rate == 16_000)
        #expect(loaded.asArray(Float.self) == samples)
        let (resampledRate, resampled) = try await loader.load(url, sampleRate: 8_000)
        #expect(resampledRate == 8_000)
        #expect(resampled.size == 160)
        #expect(resampled.asArray(Float.self).allSatisfy { $0.isFinite })
        #expect(throws: AudioUtils.AudioUtilsErrors.self) {
            _ = try loadAudioArray(from: url, sampleRate: 0)
        }
    }

    @MainActor
    @Test func protocolTransfersInputAndResultWithoutCopying() async throws {
        let model: any SpeechGenerationModel = TransferAPIProbe()
        let tensor = MLXArray([Float(1), 2, 3, 4, 5, 6])
            .asType(.float16).reshaped([2, 3]).transposed(1, 0)
        let identity = ObjectIdentifier(tensor)
        let audio = try await model.generate(
            text: "test", voice: nil, refAudio: tensor, refText: nil, language: nil)
        #expect(ObjectIdentifier(audio) == identity)
        #expect(audio.shape == [3, 2])
        #expect(audio.dtype == .float16)
        #expect(audio.asArray(Float.self) == [1, 4, 2, 5, 3, 6])

        let empty = try await model.generate(
            text: "test", voice: nil, refAudio: nil, refText: nil, language: nil)
        #expect(empty.size == 0)
    }

    @MainActor
    @Test func eventsTransferAcrossActorBoundaryWithoutCopying() async throws {
        let tensor = MLXArray([Float(1), 2, 3, 4, 5, 6])
            .asType(.float16).reshaped([2, 3]).transposed(1, 0)
        let identity = ObjectIdentifier(tensor)
        let producer = AudioEventProducer()
        let stream = await producer.events(refAudio: tensor)

        var kinds: [String] = []
        for try await event in stream {
            switch event {
            case .token(let token):
                kinds.append("token")
                #expect(token == 42)
            case .progress(let progress):
                kinds.append("progress")
                #expect(progress == 1)
            case .info(let info):
                kinds.append("info")
                #expect(info.generationTokenCount == 1)
            case .audio(let audio):
                kinds.append("audio")
                #expect(ObjectIdentifier(audio) == identity)
                #expect(audio.shape == [3, 2])
                #expect(audio.dtype == .float16)
                #expect(audio.asArray(Float.self) == [1, 4, 2, 5, 3, 6])
            }
        }
        #expect(kinds == ["token", "progress", "info", "audio"])
    }

    @MainActor
    @Test func streamTransfersReferenceAndHandlesNil() async throws {
        let model: any SpeechGenerationModel = TransferAPIProbe()
        let samples: [Float] = [0.125, -0.25, 0.5]
        var chunks: [[Float]] = []
        for try await chunk in model.generateSamplesStream(
            text: "test", voice: nil, refAudio: MLXArray(samples), refText: nil, language: nil)
        {
            chunks.append(chunk)
        }
        #expect(chunks == [samples])

        var emptyChunks: [[Float]] = []
        for try await chunk in model.generateSamplesStream(
            text: "test", voice: nil, refAudio: nil, refText: nil, language: nil)
        {
            emptyChunks.append(chunk)
        }
        #expect(emptyChunks == [[]])
    }
}

private func referenceSamples(count: Int = 4_000) -> [Float] {
    (0..<count).map { Float(sin(Double($0) * 0.1)) * 0.25 }
}

private func tinySortformerConfig() throws -> SortformerConfig {
    let json = #"""
    {
      "num_speakers": 2,
      "fc_encoder_config": {
        "hidden_size": 8, "num_hidden_layers": 1, "num_attention_heads": 2,
        "num_key_value_heads": 2, "intermediate_size": 16,
        "subsampling_conv_channels": 4, "conv_kernel_size": 3
      },
      "tf_encoder_config": {
        "d_model": 8, "encoder_layers": 1, "encoder_attention_heads": 2,
        "encoder_ffn_dim": 16, "max_source_positions": 128
      },
      "modules_config": {"num_speakers": 2, "fc_d_model": 8, "tf_d_model": 8},
      "processor_config": {}
    }
    """#
    return try JSONDecoder().decode(SortformerConfig.self, from: Data(json.utf8))
}

private actor StreamingStateStore {
    private(set) var state: StreamingState

    init(_ state: StreamingState) { self.state = state }

    func save(_ state: StreamingState) { self.state = state }
}

private actor DiarizationProducer {
    let model: SortformerModel

    init() throws {
        model = SortformerModel(try tinySortformerConfig())
    }

    func stream(samples: [Float]) -> sending AsyncThrowingStream<DiarizationOutput, Error> {
        model.generateStream(audio: MLXArray(samples), chunkDuration: 0.1)
    }
}

private actor IndexTTSProducer {
    let core: IndexTTSCore
    let conditioning: MLXArray

    init() {
        let config = IndexTTSConfig.tinyForTests()
        core = IndexTTSCore(config: config)
        conditioning = MLXArray.zeros([1, config.gpt.conditionNumLatent, config.gpt.modelDim])
    }

    func prepare() throws -> sending IndexTTSPreparedEmbedding {
        try core.prepareInputEmbedding(textTokenIDs: [1], conditioningLatents: conditioning)
    }

    func generate() throws -> sending IndexTTSMelGeneration {
        try core.generateMelTokens(textTokenIDs: [1], conditioningLatents: conditioning, maxTokens: 2)
    }
}

private actor AudioLoader {
    func load(_ url: URL, sampleRate: Int? = nil) throws -> sending (Int, MLXArray) {
        try loadAudioArray(from: url, sampleRate: sampleRate)
    }
}

private actor AudioEventProducer {
    func events(refAudio: sending MLXArray?) -> sending AsyncThrowingStream<AudioGeneration, Error> {
        TransferAPIProbe().generateStream(
            text: "test", voice: nil, refAudio: refAudio, refText: nil, language: nil,
            generationParameters: GenerateParameters(maxTokens: 1))
    }
}

private final class TransferAPIProbe: SpeechGenerationModel {
    let sampleRate = 16_000
    let defaultGenerationParameters = GenerateParameters(maxTokens: 1)

    func generate(
        text: String, voice: String?, refAudio: sending MLXArray?, refText: String?,
        language: String?, generationParameters: GenerateParameters
    ) async throws -> sending MLXArray {
        refAudio ?? MLXArray([Float]())
    }

    func generateStream(
        text: String, voice: String?, refAudio: sending MLXArray?, refText: String?,
        language: String?, generationParameters: GenerateParameters
    ) -> sending AsyncThrowingStream<AudioGeneration, Error> {
        let input = SendingBox(refAudio)
        let (stream, continuation) = AsyncThrowingStream<AudioGeneration, Error>.makeStream()
        let task = Task { @Sendable in
            let audio = input.take() ?? MLXArray([Float]())
            continuation.yield(.token(42))
            continuation.yield(.progress(1))
            continuation.yield(.info(AudioGenerationInfo(
                promptTokenCount: 1, generationTokenCount: 1, prefillTime: 0,
                generateTime: 1, tokensPerSecond: 1, peakMemoryUsage: 0)))
            continuation.yield(.audio(audio))
            continuation.finish()
        }
        continuation.onTermination = { @Sendable _ in task.cancel() }
        return stream
    }
}
