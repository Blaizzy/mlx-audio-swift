import AVFoundation
import Foundation
import MLX
import MLXNN
import Testing
@testable import MLXAudioCodecs

private func tinyMiMoConfig(codebooks: Int = 2) -> MiMoAudioTokenizerConfig {
    var c = MiMoAudioTokenizerConfig()
    c.dModel = 16
    c.nMels = 4
    c.nFFT = 32
    c.windowSize = 32
    c.hopLength = 8
    c.encoderLayers = 2
    c.encoderSkipLayerID = 1
    c.decoderLayers = 2
    c.encoderAttentionHeads = 2
    c.decoderAttentionHeads = 2
    c.encoderFFNDim = 32
    c.decoderFFNDim = 32
    c.vocoderDim = 8
    c.vocoderIntermediateDim = 16
    c.vocoderNumLayers = 1
    c.vocoderAttentionHeads = 2
    c.numQuantizers = codebooks
    c.codebookSize = Array(repeating: 4, count: codebooks)
    c.codebookSize[0] = 8
    return c
}

private func tinyMiMoWeights(_ model: MiMoAudioTokenizer) -> [String: MLXArray] {
    Dictionary(uniqueKeysWithValues: model.parameters().flattened().map { key, value in
        if key.hasSuffix("istft.window") { return (key, value) }
        let array = MLXArray((0..<value.size).map { Float(sin(Double($0 + key.count) * 0.17)) * 0.07 })
            .reshaped(value.shape)
        return (key, key.hasSuffix("norm.weight") ? array + 1 : array)
    })
}

private func originalMiMoWeights(_ prepared: [String: MLXArray]) -> [String: MLXArray] {
    var result: [String: MLXArray] = [:]
    for (name, tensor) in prepared {
        var key = name, value = tensor
        if name.hasPrefix("encoder.quantizer.codebooks.") {
            let index = name.split(separator: ".")[3]
            key = "encoder.quantizer.vq.layers.\(index)._codebook.embed"
        } else if name.hasSuffix("weight") && tensor.ndim == 3 {
            value = name.contains(".dconv") ? tensor.transposed(2, 0, 1) : tensor.transposed(0, 2, 1)
        }
        key = key.replacingOccurrences(of: "encoder.down_sample_layer.layers.0.",
                                       with: "encoder.down_sample_layer.0.")
        result[key] = value
    }
    result["encoder.quantizer.vq.layers.0._codebook.cluster_size"] = MLXArray.zeros([8])
    result["encoder.quantizer.vq.layers.0._codebook.embed_avg"] = MLXArray.zeros([8, 16])
    result["encoder.quantizer.vq.layers.0._codebook.inited"] = MLXArray([Float(1)])
    return result
}

@Suite("MiMo audio tokenizer", .serialized)
struct MiMoAudioTokenizerTests {
    @Test func configurationDefaultsRoundtripAndValidation() throws {
        let defaults = try JSONDecoder().decode(MiMoAudioTokenizerConfig.self, from: Data("{}".utf8))
        #expect(defaults.samplingRate == 24_000)
        #expect(defaults.downsampleRate == 960)
        #expect(defaults.numQuantizers == 20)
        #expect(defaults.codebookSize == [1024, 1024] + Array(repeating: 128, count: 18))
        let restored = try JSONDecoder().decode(MiMoAudioTokenizerConfig.self,
                                               from: JSONEncoder().encode(tinyMiMoConfig()))
        #expect(restored.dModel == 16)
        #expect(restored.encoderSkipLayerID == 1)
        let noSkip = try JSONDecoder().decode(MiMoAudioTokenizerConfig.self,
                                              from: Data("{\"encoder_skip_layer_id\":null}".utf8))
        #expect(noSkip.encoderSkipLayerID == nil)
        var invalid = tinyMiMoConfig()
        invalid.numQuantizers = 3
        #expect(throws: MiMoAudioTokenizerError.self) { try invalid.validate() }
        invalid = tinyMiMoConfig()
        invalid.decoderAttentionHeads = 3
        #expect(throws: MiMoAudioTokenizerError.self) { try invalid.validate() }
    }

    @Test func tinyWaveformRoundtripAndShortInput() throws {
        let model = try MiMoAudioTokenizer(config: tinyMiMoConfig())
        try model.update(parameters: ModuleParameters.unflattened(tinyMiMoWeights(model)), verify: .all)
        let waveform = MLXArray((0..<200).map { Float(sin(Double($0) / 7)) * 0.1 })
        let mel = mimoLogMelSpectrogram(waveform, config: model.config)
        #expect(mel.shape == [26, 4])
        let codes = try model.encode(waveform, sampleRate: 24_000)
        #expect(codes.shape == [2, 7])
        #expect(codes.dtype.isInteger)
        let audio = try model.decode(codes)
        #expect(audio.shape == [7 * 32])
        #expect(isFinite(audio).all().item(Bool.self))
        #expect(try model.decode(codes, length: 200).shape == [200])
        let short = try model.encode(MLXArray([Float(0)]), sampleRate: 24_000)
        #expect(short.dim(1) > 0)
    }

    @Test func allTwentyCodebooksAndPrefixAgreement() throws {
        let model = try MiMoAudioTokenizer(config: tinyMiMoConfig(codebooks: 20))
        try model.update(parameters: ModuleParameters.unflattened(tinyMiMoWeights(model)), verify: .all)
        let features = MLXArray((0..<(11 * 16)).map { Float(sin(Double($0) / 3)) }).reshaped([11, 16])
        let full = model.encoder.quantizer.encode(features, count: 20)
        let prefix = model.encoder.quantizer.encode(features, count: 8)
        #expect(full.shape == [20, 11])
        #expect((full[0..<8] .== prefix).all().item(Bool.self))
        #expect(try model.decode(full).shape == [11 * 32])
        #expect(try model.decode(prefix).shape == [11 * 32])
    }

    @Test func residualQuantizationMatchesKnownNearestCodes() throws {
        let quantizer = MiMoTokenizerRVQ(dim: 4, sizes: [2, 2])
        let first = MLXArray([[Float(0), 0, 0, 0], [1, 0, 0, 0]])
        let second = MLXArray([[Float(0), 0, 0, 0], [0, 1, 0, 0]])
        try quantizer.update(parameters: ModuleParameters.unflattened([
            "codebooks.0.weight": first, "codebooks.1.weight": second]), verify: .all)
        let features = MLXArray([[Float(1.2), 1.1, 0, 0], [0, 0, 0, 0]])
        let codes = quantizer.encode(features, count: 2)
        #expect(codes.asArray(Int32.self) == [1, 0, 1, 0])
        #expect(quantizer.decode(codes).asArray(Float.self) == [1, 1, 0, 0, 0, 0, 0, 0])
    }

    @Test func localAttentionWindowRestrictsValuesToCurrentFrame() throws {
        let attention = MiMoTokenizerAttention(dim: 4, heads: 1, config: tinyMiMoConfig(),
                                                causal: false, window: [0, 0])
        let identity = MLXArray([[Float(1), 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        let zero = MLXArray.zeros([4])
        try attention.update(parameters: ModuleParameters.unflattened([
            "q_proj.weight": MLXArray.zeros([4, 4]), "q_proj.bias": zero,
            "k_proj.weight": MLXArray.zeros([4, 4]), "v_proj.weight": identity,
            "v_proj.bias": zero, "out_proj.weight": identity, "out_proj.bias": zero]), verify: .all)
        let input = MLXArray((0..<12).map(Float.init), [1, 3, 4])
        #expect((attention(input) .== input).all().item(Bool.self))
    }

    @Test func segmentedLengthsAndEmptyDecode() throws {
        let model = try MiMoAudioTokenizer(config: tinyMiMoConfig())
        let mels = MLXArray.zeros([35, 4])
        let codes = try model.encodeMels(mels, segmentSize: 16)
        #expect(codes.shape == [2, 9])
        #expect(try model.decode(codes, segmentSize: 4).shape == [9 * 32])
        #expect(try model.decode(codes, length: 27).shape == [27])
        #expect(try model.decode(MLXArray.zeros([2, 0], dtype: .int32)).shape == [0])
    }

    @Test func monoMixdownAndResamplingContract() throws {
        let model = try MiMoAudioTokenizer(config: tinyMiMoConfig())
        let mono = MLXArray((0..<200).map { Float(sin(Double($0) / 7)) * 0.1 })
        let stereo = stacked([mono, mono], axis: -1)
        #expect((try model.encode(mono, sampleRate: 24_000)
                 .== model.encode(stereo, sampleRate: 24_000)).all().item(Bool.self))
        let upsampled = try model.encode(MLXArray.ones([120, 2]), sampleRate: 12_000)
        #expect(upsampled.shape == [2, 8])
    }

    @Test func stereoFileEncodingAveragesChannels() throws {
        let model = try MiMoAudioTokenizer(config: tinyMiMoConfig())
        let fileURL = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: fileURL) }
        let format = AVAudioFormat(standardFormatWithSampleRate: 24_000, channels: 2)!
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 200)!
        buffer.frameLength = 200
        let left = (0..<200).map { Float(sin(Double($0) / 7)) * 0.1 }
        let right = (0..<200).map { Float(cos(Double($0) / 5)) * 0.1 }
        for i in 0..<200 {
            buffer.floatChannelData![0][i] = left[i]
            buffer.floatChannelData![1][i] = right[i]
        }
        do {
            let file = try AVAudioFile(forWriting: fileURL, settings: format.settings)
            try file.write(from: buffer)
        }
        let expected = try model.encode(stacked([MLXArray(left), MLXArray(right)], axis: -1), sampleRate: 24_000)
        let actual = try model.encode(fileURL)
        #expect((expected .== actual).all().item(Bool.self))
    }

    @Test func invalidAudioAndCodeBoundsAreRecoverable() throws {
        let model = try MiMoAudioTokenizer(config: tinyMiMoConfig())
        for audio in [MLXArray.zeros([0]), MLXArray([Float.nan]), MLXArray.zeros([1, 1, 2])] {
            #expect(throws: MiMoAudioTokenizerError.self) { try model.encode(audio, sampleRate: 24_000) }
        }
        #expect(throws: MiMoAudioTokenizerError.self) { try model.encode(MLXArray.ones([20])) }
        for codes in [MLXArray([[8]]), MLXArray([[-1]]), MLXArray([[Float(0.5)]]),
                      MLXArray.zeros([3, 4], dtype: .int32), MLXArray([[0], [4]])] {
            #expect(throws: MiMoAudioTokenizerError.self) { try model.decode(codes) }
        }
        #expect(throws: MiMoAudioTokenizerError.self) { try model.encodeMels(MLXArray.zeros([4, 4]), numQuantizers: 0) }
        #expect(throws: MiMoAudioTokenizerError.self) { try model.decode(MLXArray([[0]]), segmentSize: 0) }
    }

    @Test func squaredWindowISTFTReconstructsKnownFrames() {
        let waveform = MLXArray((0..<80).map { Float(sin(Double($0) / 3)) })
        let window = mimoPeriodicHann(32)
        let frames = asStrided(waveform, [7, 32], strides: [8, 1])
        let spectrum = MLXFFT.rfft(frames * window, axis: -1).expandedDimensions(axis: 0)
        let reconstructed = mimoSameISTFT(spectrum, nFFT: 32, hopLength: 8, window: window)[0]
        #expect(reconstructed.shape == [56])
        #expect(max(abs(reconstructed - waveform[12..<68])).item(Float.self) < 1e-5)
        let silence = mimoSameISTFT(MLXArray.zeros([1, 7, 17], dtype: .complex64),
                                    nFFT: 32, hopLength: 8, window: window)
        #expect(max(abs(silence)).item(Float.self) == 0)
    }

    @Test func sourceAndPreparedLoadingAreStrictAndEquivalent() async throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let missingName = "mimo-missing-\(UUID().uuidString)"
        for path in [folder.appendingPathComponent(missingName).path, "./\(missingName)", "~/\(missingName)"] {
            do {
                _ = try await MiMoAudioTokenizer.fromPretrained(path)
                Issue.record("Missing local path loaded successfully: \(path)")
            } catch MiMoAudioTokenizerError.missingDirectory {
                // Explicit local paths fail locally instead of contacting the Hub.
            }
        }
        var config = tinyMiMoConfig()
        let native = try MiMoAudioTokenizer(config: config)
        let prepared = tinyMiMoWeights(native)
        try JSONEncoder().encode(config).write(to: folder.appendingPathComponent("config.json"))
        do {
            _ = try await MiMoAudioTokenizer.fromPretrained(folder.appendingPathComponent("config.json").path)
            Issue.record("A regular file loaded as a checkpoint directory")
        } catch MiMoAudioTokenizerError.missingDirectory {
        }
        try MLX.save(arrays: originalMiMoWeights(prepared), url: folder.appendingPathComponent("model.safetensors"))
        let source = try await MiMoAudioTokenizer.fromPretrained(folder.path, dtype: nil)
        config.weightsFormat = "mlx"
        try JSONEncoder().encode(config).write(to: folder.appendingPathComponent("config.json"))
        try MLX.save(arrays: prepared, url: folder.appendingPathComponent("model.safetensors"))
        let restored = try MiMoAudioTokenizer.fromModelDirectory(folder, dtype: nil)
        let shared = try await AudioCodecs.loadModel(modelRepo: folder.path)
        let reconstructed = try shared.reconstruct(MLXArray.zeros([1, 41, 1]))
        #expect(shared.sampleRate == 24_000)
        #expect(reconstructed.shape == [1, 41, 1])
        let left = Dictionary(uniqueKeysWithValues: source.parameters().flattened())
        let right = Dictionary(uniqueKeysWithValues: restored.parameters().flattened())
        #expect(left.keys == right.keys)
        for key in left.keys { #expect((left[key]! .== right[key]!).all().item(Bool.self)) }
        let cast = try MiMoAudioTokenizer.fromModelDirectory(folder)
        #expect(cast.encoder.conv1.weight.dtype == .bfloat16)
        #expect(cast.encoder.quantizer.codebooks[0].weight.dtype == .float32)
        #expect((cast.encoder.quantizer.codebooks[0].weight
                 .== prepared["encoder.quantizer.codebooks.0.weight"]!.asType(.bfloat16).asType(.float32))
            .all().item(Bool.self))
        var missing = prepared
        missing.removeValue(forKey: "decoder.layer_norm.weight")
        try MLX.save(arrays: missing, url: folder.appendingPathComponent("model.safetensors"))
        #expect(throws: (any Error).self) { try MiMoAudioTokenizer.fromModelDirectory(folder) }
        var extra = prepared
        extra["unused.weight"] = MLXArray.zeros([1])
        try MLX.save(arrays: extra, url: folder.appendingPathComponent("model.safetensors"))
        #expect(throws: (any Error).self) { try MiMoAudioTokenizer.fromModelDirectory(folder) }
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["MIMO_CODEC_CHECKPOINT"] != nil))
    func publishedCheckpointRoundtrip() throws {
        let directory = URL(fileURLWithPath: ProcessInfo.processInfo.environment["MIMO_CODEC_CHECKPOINT"]!)
        let model = try MiMoAudioTokenizer.fromModelDirectory(directory)
        let angularStep = 2.0 * Double.pi * 220.0 / 24_000.0
        let samples: [Float] = (0..<24_000).map { Float(Foundation.sin(Double($0) * angularStep)) * 0.05 }
        let waveform = MLXArray(samples)
        let codes = try model.encode(waveform, sampleRate: 24_000)
        #expect(codes.shape == [20, 26])
        let audio = try model.decode(codes, length: waveform.size)
        #expect(audio.shape == waveform.shape)
        #expect(isFinite(audio).all().item(Bool.self))
        #expect(mean(audio.square()).item(Float.self) > 0)
        #expect(try model.decode(codes[0..<8]).shape == [26 * 960])
    }
}
