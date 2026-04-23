import Foundation
import MLX
import Testing
import Darwin

@testable import MLXAudioCore
@testable import MLXAudioSTT

@Suite("Parakeet Batch Parity Tests", .serialized)
struct ParakeetBatchParityTests {
    private func makeFixtureModel() throws -> ParakeetModel {
        let fixtureDir = try makeFixtureDirectory()
        return try ParakeetModel.fromDirectory(fixtureDir)
    }

    private func makeTDTFixtureModel(blankAsPad: Bool = true) throws -> ParakeetModel {
        let fixtureDir = try makeTDTFixtureDirectory(blankAsPad: blankAsPad)
        return try ParakeetModel.fromDirectory(fixtureDir)
    }

    @Test("generateBatch preserves order and text parity for chunk-sized audio")
    func generateBatchPreservesOrderAndTextParity() throws {
        let model = try makeFixtureModel()
        let audios = [
            makeChunkAudio(sampleCount: 3_200, frequency: 180),
            makeChunkAudio(sampleCount: 9_600, frequency: 260),
            makeChunkAudio(sampleCount: 16_000, frequency: 340),
        ]

        let singleOutputs = audios.map { model.generate(audio: $0) }
        let singleSignatures = singleOutputs.map(outputSignature)
        #expect(Set(singleSignatures).count == singleSignatures.count)

        let batchOutputs = try model.generateBatch(audios: audios)

        #expect(batchOutputs.count == audios.count)
        #expect(batchOutputs.map(outputSignature) == singleSignatures)
    }

    @Test("generateBatch rejects empty input")
    func generateBatchRejectsEmptyInput() throws {
        let model = try makeFixtureModel()

        #expect(throws: STTError.self) {
            _ = try model.generateBatch(audios: [])
        }
    }

    @Test("generateBatch matches single-item generate for singleton input")
    func generateBatchMatchesSingleItemGenerate() throws {
        let model = try makeFixtureModel()
        let audio = makeChunkAudio(sampleCount: 8_000, frequency: 220)

        let single = model.generate(audio: audio)
        let batch = try model.generateBatch(audios: [audio])

        #expect(batch.count == 1)
        #expect(outputSignature(batch[0]) == outputSignature(single))
    }

    @Test("TDT fixture can decode batched inputs")
    func tdtFixtureCanDecodeBatchedInputs() throws {
        let model = try makeTDTFixtureModel()
        let audios = [
            makeChunkAudio(sampleCount: 3_200, frequency: 180),
            makeChunkAudio(sampleCount: 9_600, frequency: 260),
            makeChunkAudio(sampleCount: 16_000, frequency: 340),
        ]
        let traceOracle = makeTDTTraceOracleScaffold(batchSize: audios.count, durations: model.durations)
        model.tdtDecoderImplementation = .serial
        model.tdtTraceEmitter = traceOracle.serialEmitter

        #expect(model.variant == .tdt)
        #expect(model.durations == [0, 1, 2])
        #expect(traceOracle.expectedBatchSize == audios.count)
        #expect(traceOracle.fixtureDurations == model.durations)

        let singleOutputs = audios.map { model.generate(audio: $0) }
        let batchOutputs = try model.generateBatch(audios: audios)

        #expect(batchOutputs.count == audios.count)
        #expect(batchOutputs.map(outputSignature) == singleOutputs.map(outputSignature))
        #expect(batchOutputs.allSatisfy { !$0.text.isEmpty })

        #expect(
            traceOracle.isReadyForComparison,
            "Trace hooks for old/new decodeTDT paths are not implemented yet"
        )
        #expect(!traceOracle.serialTrace.isEmpty)
    }

    @Test("old and new decodeTDT traces match on TDT fixture")
    func oldAndNewDecodeTDTTracesMatch() throws {
        let audios = [
            makeChunkAudio(sampleCount: 4_800, frequency: 180),
            makeChunkAudio(sampleCount: 8_000, frequency: 260),
            makeChunkAudio(sampleCount: 12_800, frequency: 340),
        ]

        let serialModel = try makeTDTFixtureModel()
        let traceOracle = makeTDTTraceOracleScaffold(batchSize: audios.count, durations: serialModel.durations)
        serialModel.tdtDecoderImplementation = .serial
        serialModel.tdtTraceEmitter = traceOracle.serialEmitter
        let serialOutputs = try serialModel.generateBatch(audios: audios)

        let hybridModel = try makeTDTFixtureModel()
        hybridModel.tdtDecoderImplementation = .hybrid
        hybridModel.tdtTraceEmitter = traceOracle.hybridEmitter
        let hybridOutputs = try hybridModel.generateBatch(audios: audios)

        #expect(serialOutputs.map(outputSignature) == hybridOutputs.map(outputSignature))
        #expect(groupTraceStepsByRow(traceOracle.serialTrace) == groupTraceStepsByRow(traceOracle.hybridTrace))
        #expect(!traceOracle.serialTrace.isEmpty)
    }

    @Test("TDT generateBatch defaults to hybrid for multi-item batches")
    func tdtGenerateBatchDefaultsToHybridForMultiItemBatches() throws {
        let audios = [
            makeChunkAudio(sampleCount: 4_800, frequency: 180),
            makeChunkAudio(sampleCount: 8_000, frequency: 260),
            makeChunkAudio(sampleCount: 12_800, frequency: 340),
        ]

        let defaultModel = try makeTDTFixtureModel()
        let defaultTrace = makeTDTTraceOracleScaffold(batchSize: audios.count, durations: defaultModel.durations)
        defaultModel.tdtTraceEmitter = defaultTrace.hybridEmitter
        let defaultOutputs = try defaultModel.generateBatch(audios: audios)

        let explicitHybridModel = try makeTDTFixtureModel()
        let hybridTrace = makeTDTTraceOracleScaffold(batchSize: audios.count, durations: explicitHybridModel.durations)
        explicitHybridModel.tdtDecoderImplementation = .hybrid
        explicitHybridModel.tdtTraceEmitter = hybridTrace.hybridEmitter
        let explicitHybridOutputs = try explicitHybridModel.generateBatch(audios: audios)

        #expect(defaultOutputs.map(outputSignature) == explicitHybridOutputs.map(outputSignature))
        #expect(groupTraceStepsByRow(defaultTrace.hybridTrace) == groupTraceStepsByRow(hybridTrace.hybridTrace))
        #expect(!defaultTrace.hybridTrace.isEmpty)
    }

    @Test("TDT generateBatch defaults to serial for singleton batches")
    func tdtGenerateBatchDefaultsToSerialForSingletonBatches() throws {
        let audio = makeChunkAudio(sampleCount: 8_000, frequency: 220)

        let defaultModel = try makeTDTFixtureModel()
        let defaultTrace = makeTDTTraceOracleScaffold(batchSize: 1, durations: defaultModel.durations)
        defaultModel.tdtTraceEmitter = defaultTrace.serialEmitter
        let defaultOutputs = try defaultModel.generateBatch(audios: [audio])

        let explicitSerialModel = try makeTDTFixtureModel()
        let serialTrace = makeTDTTraceOracleScaffold(batchSize: 1, durations: explicitSerialModel.durations)
        explicitSerialModel.tdtDecoderImplementation = .serial
        explicitSerialModel.tdtTraceEmitter = serialTrace.serialEmitter
        let explicitSerialOutputs = try explicitSerialModel.generateBatch(audios: [audio])

        #expect(defaultOutputs.map(outputSignature) == explicitSerialOutputs.map(outputSignature))
        #expect(defaultTrace.serialTrace == serialTrace.serialTrace)
        #expect(!defaultTrace.serialTrace.isEmpty)
    }

    @Test("Parakeet predictor accepts batched blank-masked token input")
    func predictorAcceptsBatchedTokenInput() throws {
        let model = try makeTDTFixtureModel()
        let blankToken = Int32(model.vocabulary.count)
        let state: ParakeetLSTMState = (
            hidden: MLXArray.zeros([1, 3, 8], type: Float.self),
            cell: MLXArray.zeros([1, 3, 8], type: Float.self)
        )
        let tokenIds = MLXArray([blankToken, Int32(1), blankToken]).reshaped([3, 1]).asType(.int32)

        let blankReference = try #require(model.predictTDTToken(nil, state: nil)).0.reshaped([8]).asArray(Float.self)
        let batched = try #require(model.predictTDTBatch(tokenIds, state: state, blankToken: blankToken))

        guard batched.0.shape == [3, 1, 8] else {
            Issue.record("unexpected batched predictor shape: \(batched.0.shape)")
            return
        }
        #expect(batched.1.hidden?.shape == [1, 3, 8])
        #expect(batched.1.cell?.shape == [1, 3, 8])

        let batchedPred = batched.0.reshaped([3, 8]).asArray(Float.self)

        let blankRow0 = Array(batchedPred[0..<8])
        let nonBlankRow = Array(batchedPred[8..<16])
        let blankRow2 = Array(batchedPred[16..<24])

        #expect(blankRow0 == blankReference)
        #expect(blankRow2 == blankReference)
        #expect(nonBlankRow != blankReference)
        #expect(blankRow0 == blankRow2)
    }

    @Test("Parakeet predictor batches blanks when blank_as_pad is false")
    func predictorBatchesBlanksWithoutBlankAsPad() throws {
        let model = try makeTDTFixtureModel(blankAsPad: false)
        let blankToken = Int32(model.vocabulary.count)
        let state: ParakeetLSTMState = (
            hidden: MLXArray.ones([1, 2, 8], type: Float.self),
            cell: MLXArray.ones([1, 2, 8], type: Float.self)
        )
        let tokenIds = MLXArray([blankToken, Int32(1)]).reshaped([2, 1]).asType(.int32)

        let batched = try #require(model.predictTDTBatch(tokenIds, state: state, blankToken: blankToken))

        #expect(batched.0.shape == [2, 1, 8])
        #expect(batched.1.hidden?.shape == [1, 2, 8])
        #expect(batched.1.cell?.shape == [1, 2, 8])
    }

    @Test("Compiled encoder matches uncompiled Parakeet encoder output")
    func compiledEncoderMatchesUncompiledParakeetEncoderOutput() throws {
        let model = try makeTDTFixtureModel()
        let audios = [
            makeChunkAudio(sampleCount: 4_800, frequency: 180),
            makeChunkAudio(sampleCount: 8_000, frequency: 260),
            makeChunkAudio(sampleCount: 12_800, frequency: 340),
        ]
        let batchFeatures = model.makeBatchFeatures(audios)

        model.encoderExecutionImplementation = .plain
        let plainEncoded = model.encodeBatchFeatures(batchFeatures.features, lengths: batchFeatures.lengths)
        eval(plainEncoded.0, plainEncoded.1)

        model.encoderExecutionImplementation = .compiled
        let compiledEncoded = model.encodeBatchFeatures(batchFeatures.features, lengths: batchFeatures.lengths)
        eval(compiledEncoded.0, compiledEncoded.1)

        #expect(plainEncoded.1.asArray(Int32.self) == compiledEncoded.1.asArray(Int32.self))
        #expect(plainEncoded.0.shape == compiledEncoded.0.shape)
        #expect(plainEncoded.0.asArray(Float.self) == compiledEncoded.0.asArray(Float.self))

        model.tdtDecoderImplementation = .serial
        let plainDecoded = model.decodeEncoded(batchFeatures: plainEncoded.0, lengths: plainEncoded.1)
        let compiledDecoded = model.decodeEncoded(batchFeatures: compiledEncoded.0, lengths: compiledEncoded.1)

        #expect(plainDecoded.map(alignedResultSignature) == compiledDecoded.map(alignedResultSignature))
    }

    @Test("Parakeet stage benchmark harness measures mel encoder decode and full batch")
    func parakeetStageBenchmarkHarnessMeasuresMelEncoderDecodeAndFullBatch() throws {
        let model = try makeTDTFixtureModel()
        let audios = [
            makeChunkAudio(sampleCount: 4_800, frequency: 180),
            makeChunkAudio(sampleCount: 8_000, frequency: 260),
            makeChunkAudio(sampleCount: 12_800, frequency: 340),
        ]

        let melTimed = measureWallClock {
            model.makeBatchFeatures(audios)
        }

        model.encoderExecutionImplementation = .plain
        let encoderTimed = measureWallClock {
            model.encodeBatchFeatures(melTimed.value.features, lengths: melTimed.value.lengths)
        }

        model.tdtDecoderImplementation = .serial
        let decodeTimed = measureWallClock {
            model.decodeEncoded(batchFeatures: encoderTimed.value.0, lengths: encoderTimed.value.1)
        }

        let fullTimed = try measureWallClock {
            try model.generateBatch(audios: audios)
        }

        let result = ParakeetStageBenchmarkResult(
            batchSize: audios.count,
            melWallClock: melTimed.wallClock,
            encoderWallClock: encoderTimed.wallClock,
            decodeWallClock: decodeTimed.wallClock,
            fullBatchWallClock: fullTimed.wallClock
        )

        #expect(result.batchSize == 3)
        #expect(result.melWallClock >= 0)
        #expect(result.encoderWallClock >= 0)
        #expect(result.decodeWallClock >= 0)
        #expect(result.fullBatchWallClock >= 0)
        #expect(!result.summary.isEmpty)
        #expect(decodeTimed.value.map(alignedResultSignature) == fullTimed.value.map(outputSignature))
    }

    @Test("generateBatch normalizes multichannel audio to mono before mel extraction")
    func generateBatchNormalizesMultichannelAudioToMono() throws {
        let model = try makeFixtureModel()
        let mono = makeChunkAudio(sampleCount: 12_000, frequency: 300)
        let stereo = makeStereoAudio(from: mono)

        let monoOutput = try model.generateBatch(audios: [mono])
        let stereoOutput = try model.generateBatch(audios: [stereo])

        #expect(monoOutput.count == 1)
        #expect(stereoOutput.count == 1)
        #expect(outputSignature(monoOutput[0]) == outputSignature(stereoOutput[0]))
    }

    @Test("benchmark metadata captures the batch contract")
    func benchmarkMetadataCapturesBatchContract() throws {
        let timed = measureWallClock {
            Thread.sleep(forTimeInterval: 0.001)
            return 42
        }

        #expect(timed.wallClock >= 0)
        #expect(timed.value == 42)

        let result = BatchBenchmarkResult(
            checkpoint: "mlx-community/parakeet-tdt-0.6b-v3",
            batchSize: 4,
            warmupRuns: 1,
            measuredRuns: 3,
            medianWallClock: median([0.9, 0.4, 0.7]),
            peakRSSBytes: captureCurrentRSSBytes(),
            maxFrameLength: 123
        )

        #expect(result.checkpoint == "mlx-community/parakeet-tdt-0.6b-v3")
        #expect(result.batchSize == 4)
        #expect(result.warmupRuns == 1)
        #expect(result.measuredRuns == 3)
        #expect(result.medianWallClock == 0.7)
        #expect(result.peakRSSBytes > 0)
        #expect(result.maxFrameLength == 123)
        #expect(result.summary.contains("parakeet-tdt-0.6b-v3"))
        #expect(result.summary.contains("batch_size=4"))
        #expect(result.summary.contains("warmup_runs=1"))
        #expect(result.summary.contains("measured_runs=3"))
        #expect(result.summary.contains("median_wall_clock=0.700000"))
        #expect(result.summary.contains("max_frame_length=123"))
    }

    @Test("benchmark harness measures single and batched Parakeet paths")
    func benchmarkHarnessMeasuresSingleAndBatchedPaths() throws {
        let model = try makeFixtureModel()
        let audios = [
            makeChunkAudio(sampleCount: 4_000, frequency: 180),
            makeChunkAudio(sampleCount: 8_000, frequency: 240),
        ]

        let singleTimed = measureWallClock {
            audios.map { model.generate(audio: $0) }
        }
        let batchTimed = try measureWallClock {
            try model.generateBatch(audios: audios)
        }

        #expect(singleTimed.value.map(outputSignature) == batchTimed.value.map(outputSignature))

        let maxFrameLength = audios
            .map { ParakeetAudio.logMelSpectrogram($0, config: model.preprocessConfig).shape[1] }
            .max() ?? 0

        let benchmark = BatchBenchmarkResult(
            checkpoint: "fixture/parakeet-ctc-batch-smoke",
            batchSize: audios.count,
            warmupRuns: 0,
            measuredRuns: 1,
            medianWallClock: batchTimed.wallClock,
            peakRSSBytes: captureCurrentRSSBytes(),
            maxFrameLength: maxFrameLength
        )

        #expect(singleTimed.wallClock >= 0)
        #expect(batchTimed.wallClock >= 0)
        #expect(benchmark.batchSize == 2)
        #expect(benchmark.maxFrameLength > 0)
        #expect(benchmark.peakRSSBytes > 0)
    }

    @Test("Parakeet pre-encoder preserves dModel for mixed-length batch")
    func parakeetPreEncoderPreservesDModelForMixedLengthBatch() throws {
        let configJSON = """
        {
          "feat_in": 128,
          "n_layers": 1,
          "d_model": 1024,
          "n_heads": 8,
          "ff_expansion_factor": 4,
          "subsampling_factor": 8,
          "self_attention_model": "rel_pos",
          "subsampling": "dw_striding",
          "conv_kernel_size": 9,
          "subsampling_conv_channels": 256,
          "pos_emb_max_len": 2048,
          "causal_downsampling": false,
          "use_bias": false,
          "xscaling": false,
          "subsampling_conv_chunking_factor": 1
        }
        """

        let config = try JSONDecoder().decode(ParakeetConformerConfig.self, from: Data(configJSON.utf8))
        let encoder = ParakeetConformer(args: config)
        let preEncoder = try #require(encoder.preEncodeDw)

        let lengths = MLXArray([1043, 1001, 977, 913, 881, 799, 643, 511]).asType(.int32)
        let maxFrames = 1043
        let batch = lengths.shape[0]
        let features = MLXArray.zeros([batch, maxFrames, 128], type: Float.self)

        let preEncoded = preEncoder(features, lengths: lengths)

        #expect(preEncoded.0.ndim == 3)
        #expect(preEncoded.0.shape[0] == batch)
        #expect(preEncoded.0.shape[2] == 1024)
        eval(preEncoded.0)
        let preEncodedValues = preEncoded.0.asArray(Float.self)
        #expect(preEncodedValues.count == batch * preEncoded.0.shape[1] * 1024)
    }

    @Test("Parakeet first-layer linearQ preserves dModel for mixed-length batch")
    func parakeetFirstLayerLinearQPreservesDModelForMixedLengthBatch() throws {
        let configJSON = """
        {
          "feat_in": 128,
          "n_layers": 1,
          "d_model": 1024,
          "n_heads": 8,
          "ff_expansion_factor": 4,
          "subsampling_factor": 8,
          "self_attention_model": "rel_pos",
          "subsampling": "dw_striding",
          "conv_kernel_size": 9,
          "subsampling_conv_channels": 256,
          "pos_emb_max_len": 2048,
          "causal_downsampling": false,
          "use_bias": false,
          "xscaling": false,
          "subsampling_conv_chunking_factor": 1
        }
        """

        let config = try JSONDecoder().decode(ParakeetConformerConfig.self, from: Data(configJSON.utf8))
        let encoder = ParakeetConformer(args: config)
        let preEncoder = try #require(encoder.preEncodeDw)

        let lengths = MLXArray([1043, 1001, 977, 913, 881, 799, 643, 511]).asType(.int32)
        let maxFrames = 1043
        let batch = lengths.shape[0]
        let features = MLXArray.zeros([batch, maxFrames, 128], type: Float.self)

        let preEncoded = preEncoder(features, lengths: lengths)

        let firstLayer = encoder.layers[0]
        let xNorm = firstLayer.normSelfAtt(preEncoded.0)
        #expect(xNorm.shape[2] == 1024)

        let relSelfAttn = try #require(firstLayer.relSelfAttn)
        let qProj = relSelfAttn.linearQ(xNorm)
        #expect(qProj.shape[2] == 1024)
        eval(qProj)
        let qProjValues = qProj.asArray(Float.self)
        #expect(qProjValues.count == batch * qProj.shape[1] * 1024)
    }

    @Test("Parakeet full conformer preserves dModel for mixed-length batch")
    func parakeetConformerPreservesDModelForMixedLengthBatch() throws {
        let configJSON = """
        {
          "feat_in": 128,
          "n_layers": 1,
          "d_model": 1024,
          "n_heads": 8,
          "ff_expansion_factor": 4,
          "subsampling_factor": 8,
          "self_attention_model": "rel_pos",
          "subsampling": "dw_striding",
          "conv_kernel_size": 9,
          "subsampling_conv_channels": 256,
          "pos_emb_max_len": 2048,
          "causal_downsampling": false,
          "use_bias": false,
          "xscaling": false,
          "subsampling_conv_chunking_factor": 1
        }
        """

        let config = try JSONDecoder().decode(ParakeetConformerConfig.self, from: Data(configJSON.utf8))
        let encoder = ParakeetConformer(args: config)

        let lengths = MLXArray([1043, 1001, 977, 913, 881, 799, 643, 511]).asType(.int32)
        let maxFrames = 1043
        let batch = lengths.shape[0]
        let features = MLXArray.zeros([batch, maxFrames, 128], type: Float.self)

        let encoded = encoder(features, lengths: lengths)

        #expect(encoded.0.ndim == 3)
        #expect(encoded.0.shape[0] == batch)
        #expect(encoded.0.shape[2] == 1024)
        #expect(encoded.1.shape[0] == batch)
    }
}

private struct BatchBenchmarkResult {
    let checkpoint: String
    let batchSize: Int
    let warmupRuns: Int
    let measuredRuns: Int
    let medianWallClock: TimeInterval
    let peakRSSBytes: UInt64
    let maxFrameLength: Int

    var summary: String {
        let formattedMedian = String(format: "%.6f", medianWallClock)
        return "checkpoint=\(checkpoint) batch_size=\(batchSize) warmup_runs=\(warmupRuns) measured_runs=\(measuredRuns) median_wall_clock=\(formattedMedian) peak_rss_bytes=\(peakRSSBytes) max_frame_length=\(maxFrameLength)"
    }
}

private struct ParakeetStageBenchmarkResult {
    let batchSize: Int
    let melWallClock: TimeInterval
    let encoderWallClock: TimeInterval
    let decodeWallClock: TimeInterval
    let fullBatchWallClock: TimeInterval

    var summary: String {
        let mel = String(format: "%.6f", melWallClock)
        let encoder = String(format: "%.6f", encoderWallClock)
        let decode = String(format: "%.6f", decodeWallClock)
        let fullBatch = String(format: "%.6f", fullBatchWallClock)
        return "batch_size=\(batchSize) mel=\(mel) encoder=\(encoder) decode=\(decode) full_batch=\(fullBatch)"
    }
}

private typealias TDTTraceStep = ParakeetModel.TDTTraceStep
private typealias TDTTraceEmitter = @Sendable (TDTTraceStep) -> Void

private struct TDTTraceOracleScaffold {
    let expectedBatchSize: Int
    let fixtureDurations: [Int]
    private let serialTraceStorage: TraceStorage
    private let hybridTraceStorage: TraceStorage

    init(expectedBatchSize: Int, fixtureDurations: [Int], serialTraceStorage: TraceStorage, hybridTraceStorage: TraceStorage) {
        self.expectedBatchSize = expectedBatchSize
        self.fixtureDurations = fixtureDurations
        self.serialTraceStorage = serialTraceStorage
        self.hybridTraceStorage = hybridTraceStorage
    }

    var serialTrace: [TDTTraceStep] { serialTraceStorage.steps }
    var hybridTrace: [TDTTraceStep] { hybridTraceStorage.steps }
    var serialEmitter: TDTTraceEmitter { serialTraceStorage.makeEmitter() }
    var hybridEmitter: TDTTraceEmitter { hybridTraceStorage.makeEmitter() }

    var isReadyForComparison: Bool {
        true
    }
}

private final class TraceStorage: @unchecked Sendable {
    private(set) var steps: [TDTTraceStep] = []

    func makeEmitter() -> TDTTraceEmitter {
        { [weak self] step in
            self?.steps.append(step)
        }
    }
}

private func makeFixtureDirectory() throws -> URL {
    let fixtureDir = FileManager.default.temporaryDirectory
        .appendingPathComponent("parakeet-batch-fixture-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: fixtureDir, withIntermediateDirectories: true)

    let configJSON = """
    {
      "target": "nemo.collections.asr.models.ctc_bpe_models.EncDecCTCModelBPE",
      "preprocessor": {
        "sample_rate": 16000,
        "normalize": "per_feature",
        "window_size": 0.02,
        "window_stride": 0.01,
        "window": "hann",
        "features": 80,
        "n_fft": 512,
        "dither": 0.0
      },
      "encoder": {
        "feat_in": 80,
        "n_layers": 0,
        "d_model": 16,
        "n_heads": 4,
        "ff_expansion_factor": 2,
        "subsampling_factor": 2,
        "self_attention_model": "abs_pos",
        "subsampling": "dw_striding",
        "conv_kernel_size": 15,
        "subsampling_conv_channels": 16,
        "pos_emb_max_len": 128
      },
      "decoder": {
        "feat_in": 16,
        "num_classes": 4,
        "vocabulary": ["▁", "a", "b", "."]
      },
      "decoding": {"greedy": {"max_symbols": 8}}
    }
    """
    try configJSON.write(
        to: fixtureDir.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    let weights: [String: MLXArray] = [
        "encoder.pre_encode.conv0.weight": MLXArray.zeros([16, 3, 3, 1], type: Float.self),
        "encoder.pre_encode.conv0.bias": MLXArray.zeros([16], type: Float.self),
        "encoder.pre_encode.out.weight": MLXArray.zeros([16, 640], type: Float.self),
        "encoder.pre_encode.out.bias": MLXArray.zeros([16], type: Float.self),
        "decoder.decoder_layers.0.weight": MLXArray.zeros([5, 1, 16], type: Float.self),
        "decoder.decoder_layers.0.bias": MLXArray([Float(0.3), 0.2, 0.1, -0.1, -0.5]),
    ]
    try MLX.save(arrays: weights, url: fixtureDir.appendingPathComponent("model.safetensors"))

    return fixtureDir
}

private func makeTDTTraceOracleScaffold(batchSize: Int, durations: [Int]) -> TDTTraceOracleScaffold {
    TDTTraceOracleScaffold(
        expectedBatchSize: batchSize,
        fixtureDurations: durations,
        serialTraceStorage: TraceStorage(),
        hybridTraceStorage: TraceStorage()
    )
}

private func groupTraceStepsByRow(_ steps: [TDTTraceStep]) -> [Int: [TDTTraceStep]] {
    Dictionary(grouping: steps, by: \ .row)
}

private func makeTDTFixtureDirectory(blankAsPad: Bool = true) throws -> URL {
    let fixtureDir = FileManager.default.temporaryDirectory
        .appendingPathComponent("parakeet-tdt-fixture-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: fixtureDir, withIntermediateDirectories: true)

    let configJSON = """
    {
      "target": "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel",
      "model_defaults": {"tdt_durations": [0, 1, 2]},
      "preprocessor": {
        "sample_rate": 16000,
        "normalize": "per_feature",
        "window_size": 0.02,
        "window_stride": 0.01,
        "window": "hann",
        "features": 80,
        "n_fft": 512,
        "dither": 0.0
      },
      "encoder": {
        "feat_in": 80,
        "n_layers": 0,
        "d_model": 16,
        "n_heads": 4,
        "ff_expansion_factor": 2,
        "subsampling_factor": 2,
        "self_attention_model": "abs_pos",
        "subsampling": "dw_striding",
        "conv_kernel_size": 15,
        "subsampling_conv_channels": 16,
        "pos_emb_max_len": 128
      },
      "decoder": {
        "blank_as_pad": \(blankAsPad ? "true" : "false"),
        "vocab_size": 4,
        "prednet": {
          "pred_hidden": 8,
          "pred_rnn_layers": 1,
          "rnn_hidden_size": 8
        }
      },
      "joint": {
        "num_classes": 4,
        "num_extra_outputs": 3,
        "vocabulary": ["▁", "a", "b", "."],
        "jointnet": {
          "joint_hidden": 8,
          "activation": "relu",
          "encoder_hidden": 16,
          "pred_hidden": 8
        }
      },
      "decoding": {
        "model_type": "tdt",
        "durations": [0, 1, 2],
        "greedy": {"max_symbols": 4}
      }
    }
    """
    try configJSON.write(
        to: fixtureDir.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    let weights: [String: MLXArray] = [
        "encoder.pre_encode.conv0.weight": MLXArray.zeros([16, 3, 3, 1], type: Float.self),
        "encoder.pre_encode.conv0.bias": MLXArray.zeros([16], type: Float.self),
        "encoder.pre_encode.out.weight": MLXArray.zeros([16, 640], type: Float.self),
        "encoder.pre_encode.out.bias": MLXArray.zeros([16], type: Float.self),
        "decoder.prediction.embed.weight": MLXArray(Array([
            Float(0.1), Float(0.0), Float(0.0), Float(0.0), Float(0.0), Float(0.0), Float(0.0), Float(0.0),
            Float(0.2), Float(0.1), Float(0.0), Float(0.0), Float(0.0), Float(0.0), Float(0.0), Float(0.0),
            Float(0.3), Float(0.1), Float(0.1), Float(0.0), Float(0.0), Float(0.0), Float(0.0), Float(0.0),
            Float(0.4), Float(0.2), Float(0.1), Float(0.1), Float(0.0), Float(0.0), Float(0.0), Float(0.0),
            Float(0.9), Float(0.9), Float(0.9), Float(0.9), Float(0.9), Float(0.9), Float(0.9), Float(0.9),
        ].prefix(blankAsPad ? 40 : 32))).reshaped([blankAsPad ? 5 : 4, 8]),
        "decoder.prediction.dec_rnn.lstm.0.Wx": MLXArray.ones([32, 8], type: Float.self),
        "decoder.prediction.dec_rnn.lstm.0.Wh": MLXArray.zeros([32, 8], type: Float.self),
        "decoder.prediction.dec_rnn.lstm.0.bias": MLXArray.zeros([32], type: Float.self),
        "joint.pred.weight": MLXArray.zeros([8, 8], type: Float.self),
        "joint.pred.bias": MLXArray.zeros([8], type: Float.self),
        "joint.enc.weight": MLXArray.zeros([8, 16], type: Float.self),
        "joint.enc.bias": MLXArray.zeros([8], type: Float.self),
        "joint.joint_net.weight": MLXArray.zeros([8, 8], type: Float.self),
        "joint.joint_net.bias": MLXArray([Float(0.0), 4.0, 1.0, -1.0, -3.0, 0.0, 5.0, 1.0]),
    ]
    try MLX.save(arrays: weights, url: fixtureDir.appendingPathComponent("model.safetensors"))

    return fixtureDir
}

private func makeChunkAudio(sampleCount: Int, frequency: Float) -> MLXArray {
    let sampleRate = 16_000.0
    let values = (0..<sampleCount).map { index in
        let phase = (2.0 * Double.pi * Double(frequency) * Double(index)) / sampleRate
        return Float(Darwin.sin(phase)) * Float(0.25)
    }
    return MLXArray(values)
}

private func makeStereoAudio(from mono: MLXArray) -> MLXArray {
    let left = mono.expandedDimensions(axis: 1)
    let right = mono.expandedDimensions(axis: 1)
    return MLX.concatenated([left, right], axis: 1)
}

private func outputSignature(_ output: STTOutput) -> String {
    let segments = (output.segments ?? []).map { segment -> String in
        let text = segment["text"] as? String ?? ""
        let start = segment["start"] as? Double ?? -1
        let end = segment["end"] as? Double ?? -1
        let formattedStart = String(format: "%.5f", start)
        let formattedEnd = String(format: "%.5f", end)
        return "\(text)@\(formattedStart)-\(formattedEnd)"
    }
    return "\(output.text)|\(segments.joined(separator: ","))"
}

private func alignedResultSignature(_ result: ParakeetAlignedResult) -> String {
    let segments = (result.segments).map { segment -> String in
        let text = segment["text"] as? String ?? ""
        let start = segment["start"] as? Double ?? -1
        let end = segment["end"] as? Double ?? -1
        let formattedStart = String(format: "%.5f", start)
        let formattedEnd = String(format: "%.5f", end)
        return "\(text)@\(formattedStart)-\(formattedEnd)"
    }
    return "\(result.text)|\(segments.joined(separator: ","))"
}

private func measureWallClock<T>(_ body: () throws -> T) rethrows -> (value: T, wallClock: TimeInterval) {
    let start = CFAbsoluteTimeGetCurrent()
    let value = try body()
    return (value, CFAbsoluteTimeGetCurrent() - start)
}

private func median(_ values: [TimeInterval]) -> TimeInterval {
    precondition(!values.isEmpty)
    let sorted = values.sorted()
    let midpoint = sorted.count / 2
    if sorted.count.isMultiple(of: 2) {
        return (sorted[midpoint - 1] + sorted[midpoint]) / 2
    }
    return sorted[midpoint]
}

private func captureCurrentRSSBytes() -> UInt64 {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4

    let result: kern_return_t = withUnsafeMutablePointer(to: &info) { pointer in
        pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
            task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), rebound, &count)
        }
    }

    guard result == KERN_SUCCESS else {
        return 0
    }
    return UInt64(info.resident_size)
}
