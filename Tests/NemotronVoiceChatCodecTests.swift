import Foundation
import MLX
import MLXNN
import Testing
@testable import MLXAudioCodecs

@Suite("Nemotron VoiceChat Codec", .serialized)
struct NemotronVoiceChatCodecTests {
    private func tinyConfig() -> NemotronVoiceChatCodecConfig {
        .init(sampleRate: 160, baseChannels: 4, channelMultipliers: [1, 2, 4],
              downsampleRates: [2, 2, 2], blocksPerStage: 3, blockKernelSize: 3,
              latentDim: 8, nFFT: 4, hopLength: 2, numQuantizers: 2, codebookSize: 16)
    }

    private func deterministicModel() throws -> NemotronVoiceChatCodec {
        let model = NemotronVoiceChatCodec(config: tinyConfig())
        var weights: [String: MLXArray] = [:]
        for (key, value) in model.parameters().flattened() {
            if key.hasSuffix("norm.weight") || key.hasSuffix("variance") {
                weights[key] = MLXArray.ones(value.shape)
            } else {
                weights[key] = (sin(MLXArray(0..<value.size).asType(.float32) * 0.37) * 0.04)
                    .reshaped(value.shape)
            }
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        return model
    }

    @Test func defaultConfigurationAndAliases() throws {
        let config = NemotronVoiceChatCodecConfig()
        #expect(config.stftChannels == 18)
        #expect(config.waveformToTokenRatio == 1_764)
        #expect(config.frameRate == 12.5)
        let root: [String: Any] = ["model": ["speech_generation": ["model": ["codec_config": [
            "base_hidden_size": 4, "channel_mult": [1, 2, 4], "rates": [2, 2, 2],
            "num_blocks": 3, "kernel_size": 3, "latent_size": 8,
            "n_fft": 4, "hop_length": 2, "num_quantizers": 2, "codebook_size": 16,
        ]]]], "data": ["target_sample_rate": 160]]
        #expect(try NemotronVoiceChatCodec.codecConfiguration(root) == tinyConfig())
        #expect(try JSONDecoder().decode(NemotronVoiceChatCodecConfig.self, from: Data("{}".utf8)) == config)
        #expect(throws: (any Error).self) {
            try NemotronVoiceChatCodec.codecConfiguration(["unrelated": true])
        }
        #expect(throws: (any Error).self) {
            try JSONDecoder().decode(NemotronVoiceChatCodecConfig.self,
                                     from: Data("{\"downsample_rates\":[]}".utf8))
        }
    }

    @Test func cacheHistoryFlushAndReset() {
        let cache = NemotronVoiceChatCodecCache()
        let first = cache.update(MLXArray([Float(1), 2]).reshaped([1, 2, 1]),
                                 key: "layer", padding: 3, flush: false)
        #expect(first.asArray(Float.self) == [0, 0, 0, 1, 2])
        let second = cache.update(MLXArray([Float(3)]).reshaped([1, 1, 1]),
                                  key: "layer", padding: 3, flush: true)
        #expect(second.asArray(Float.self) == [0, 1, 2, 3])
        #expect(cache.isEmpty)
        _ = cache.update(MLXArray.ones([1, 1, 1]), key: "other", padding: 2, flush: false)
        cache.reset()
        #expect(cache.isEmpty)
    }

    @Test func residualQuantizationAndPrefixDecode() throws {
        let config = NemotronVoiceChatCodecConfig(baseChannels: 2, channelMultipliers: [1],
                                                  downsampleRates: [2], latentDim: 2,
                                                  numQuantizers: 2, codebookSize: 2)
        let quantizer = NemotronCodecPRVQ(config)
        try quantizer.update(parameters: ModuleParameters.unflattened([
            "mus_list.0": MLXArray([Float(0), 0, 1, 0]).reshaped([2, 2]),
            "mus_list.1": MLXArray([Float(0), 0, 0, 1]).reshaped([2, 2]),
        ]), verify: [.noUnusedKeys, .shapeMismatch])
        let codes = quantizer.encode(MLXArray([Float(0.9), 0.9]).reshaped([1, 1, 2]))
        #expect(codes.asArray(Int32.self) == [1, 1])
        #expect(quantizer.decode(codes).asArray(Float.self) == [1, 1])
        #expect(quantizer.decode(codes[0..., 0..<1, 0...]).asArray(Float.self) == [1, 0])
    }

    @Test func nativeRoundtripAndLengthPreservingProtocol() throws {
        let model = try deterministicModel()
        let waveform = (sin(MLXArray(0..<64).asType(.float32) * 0.3) * 0.1).reshaped([1, 1, 64])
        let codes = model.encode(waveform)
        let decoded = model.decode(codes)
        #expect(codes.shape == [1, 2, 4])
        #expect(decoded.shape == [1, 1, 64])
        #expect(MLX.all(isFinite(decoded)).item(Bool.self))
        for length in [1, 7, 17, 70] {
            let input = MLXArray.zeros([1, length, 1])
            let encoded = model.encodeAudio(input)
            #expect(encoded.originalLength == length)
            #expect(encoded.codes.dim(2) == (length + 15) / 16)
            #expect(model.decodeAudio(encoded).shape == [1, length, 1])
        }
        #expect(model.codecSampleRate == 160)
    }

    @Test func streamingActivationCachesLookaheadAndIndependentStreams() throws {
        let model = try deterministicModel()
        let codes = MLXArray([Int32(1), 2, 3, 4, 5, 6]).reshaped([1, 2, 3])
        let latents = model.prvq.decode(codes)
        let offlineFeatures = model.decoder(latents, cache: nil, flush: false)
        let featureCache = NemotronVoiceChatCodecCache()
        var featurePieces: [MLXArray] = []
        for index in 0..<3 {
            featurePieces.append(model.decoder(latents[0..., index..<(index + 1), 0...],
                                                cache: featureCache, flush: false))
        }
        #expect(allClose(concatenated(featurePieces, axis: 1), offlineFeatures,
                         rtol: 1e-5, atol: 1e-6).item(Bool.self))
        let cache = NemotronVoiceChatCodecCache()
        var pieces: [MLXArray] = []
        for index in 0..<3 {
            let piece = model.decodeStep(codes[0..., 0..., index..<(index + 1)], cache: cache)
            #expect(piece.shape == [1, 1, 16])
            pieces.append(piece)
        }
        let streamed = concatenated(pieces, axis: -1)
        #expect(model.streamingLookaheadSamples == 2)
        #expect(streamed[0, 0, 0..<2].asArray(Float.self) == [0, 0])
        // Nonzero-weight Python reference value exercises the delayed first
        // window and its normalization, independently of stored tensor fixtures.
        #expect(abs(streamed[0, 0, 2].item(Float.self) - 0.0013334155) < 1e-7)
        let other = NemotronVoiceChatCodecCache()
        let fresh = model.decodeStep(codes[0..., 0..., 0..<1], cache: other)
        cache.reset()
        let reset = model.decodeStep(codes[0..., 0..., 0..<1], cache: cache)
        #expect(allClose(fresh, reset, rtol: 0, atol: 0).item(Bool.self))
        let final = model.decodeStep(codes[0..., 0..., 1..<2], cache: cache, flush: true)
        // The reference flush appends one overlap hop for this tiny geometry.
        #expect(final.shape == [1, 1, 18])
        #expect(cache.isEmpty)
        let empty = model.decodeStep(MLXArray.zeros([1, 2, 0], dtype: .int32), cache: cache, flush: true)
        #expect(empty.shape == [1, 1, 0])
        #expect(cache.isEmpty)
    }

    @Test func rawAndPreparedLayoutsLoadStrictly() async throws {
        let model = try deterministicModel()
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let config = try JSONSerialization.jsonObject(with: JSONEncoder().encode(tinyConfig()))
        let runtime = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        var source: [String: MLXArray] = [:]
        for (key, original) in runtime {
            var value = original
            if key.hasSuffix(".weight"), value.ndim == 3 {
                let parts = key.split(separator: ".")
                let index = Int(parts[2]) ?? -1
                let transposed = key.hasPrefix("decoder.layers.") && [0, 4, 8].contains(index)
                    && !key.contains("dwconv")
                value = transposed ? value.transposed(2, 0, 1) : value.transposed(0, 2, 1)
            }
            let sourceKey = key.replacingOccurrences(of: "prvq.variance_list.", with: "prvq._variance_list.")
            source["tts_model.audio_codec.\(sourceKey)"] = value
        }
        try JSONSerialization.data(withJSONObject: ["codec_config": config, "model_type": "nemotron_voicechat"])
            .write(to: directory.appendingPathComponent("config.json"))
        try save(arrays: source, url: directory.appendingPathComponent("model.safetensors"))
        let loaded = try NemotronVoiceChatCodec.fromModelDirectory(directory)
        let restored = Dictionary(uniqueKeysWithValues: loaded.parameters().flattened())
        #expect(restored.keys == runtime.keys)
        for key in runtime.keys {
            #expect(arrayEqual(restored[key]!, runtime[key]!).item(Bool.self), "\(key)")
        }
        let shared = try await AudioCodecs.loadModel(modelRepo: directory.path)
        #expect(shared.sampleRate == 160 && shared.audioChannels == 1)
        #expect(try shared.reconstruct(MLXArray.zeros([1, 19, 1])).shape == [1, 19, 1])
        try JSONSerialization.data(withJSONObject: ["codec_config": config, "weights_format": "mlx"])
            .write(to: directory.appendingPathComponent("config.json"))
        try save(arrays: runtime, url: directory.appendingPathComponent("model.safetensors"))
        _ = try NemotronVoiceChatCodec.fromModelDirectory(directory)
        var missing = runtime
        missing.removeValue(forKey: "decoder.layers.0.weight")
        try save(arrays: missing, url: directory.appendingPathComponent("model.safetensors"))
        #expect(throws: (any Error).self) { try NemotronVoiceChatCodec.fromModelDirectory(directory) }
    }

    @Test func indexedParentLoadingSelectsOnlyCodecShard() throws {
        let model = try deterministicModel()
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let config = try JSONSerialization.jsonObject(with: JSONEncoder().encode(tinyConfig()))
        try JSONSerialization.data(withJSONObject: ["codec_config": config, "weights_format": "mlx"])
            .write(to: directory.appendingPathComponent("config.json"))
        let weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened().map {
            ("tts_model.audio_codec.\($0.0)", $0.1)
        })
        try save(arrays: weights, url: directory.appendingPathComponent("codec-shard.safetensors"))
        var map = weights.mapValues { _ in "codec-shard.safetensors" }
        map["stt_model.llm.unrelated.weight"] = "absent-language-shard.safetensors"
        try JSONSerialization.data(withJSONObject: ["weight_map": map])
            .write(to: directory.appendingPathComponent("model.safetensors.index.json"))
        _ = try NemotronVoiceChatCodec.fromModelDirectory(directory)
        map["tts_model.audio_codec.encoder.layers.0.weight"] = "../unsafe.safetensors"
        try JSONSerialization.data(withJSONObject: ["weight_map": map])
            .write(to: directory.appendingPathComponent("model.safetensors.index.json"))
        #expect(throws: (any Error).self) { try NemotronVoiceChatCodec.fromModelDirectory(directory) }
    }

    @Test func malformedTensorHeadersAndRangeResponsesAreRejected() throws {
        let url = URL(string: "https://huggingface.co/example/model/resolve/revision/weights.safetensors")!
        func parse(_ tensor: [String: Any], extra: [String: Any] = [:]) throws -> [NemotronCodecHub.Tensor] {
            var header = extra
            header["tts_model.audio_codec.encoder.layers.0.weight"] = tensor
            return try NemotronCodecHub.parseHeader(JSONSerialization.data(withJSONObject: header),
                                                    url: url, dataStart: 100, fileSize: 200)
        }
        let valid: [String: Any] = ["dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]]
        let selected = try parse(valid, extra: ["stt_model.llm.weight": ["unrelated": true]])
        #expect(selected.count == 1)
        #expect(selected[0].start == 100 && selected[0].end == 124)
        for invalid: [String: Any] in [
            ["dtype": "F32", "shape": [2, 3], "data_offsets": [0, 23]],
            ["dtype": "F32", "shape": [2, 3], "data_offsets": [-1, 23]],
            ["dtype": "F32", "shape": [2, 3], "data_offsets": [90, 114]],
            ["dtype": "F32", "shape": [-1, 3], "data_offsets": [0, 24]],
            ["dtype": "UNKNOWN", "shape": [2, 3], "data_offsets": [0, 24]],
        ] {
            #expect(throws: (any Error).self) { try parse(invalid) }
        }
        #expect(throws: (any Error).self) {
            try parse(valid, extra: ["tts_model.audio_codec.decoder.layers.0.weight": valid])
        }
        let validator = NemotronCodecRangeValidator(range: 100..<124)
        for (status, contentRange, expected) in [
            (206, "bytes 100-123/200", true),
            (200, "bytes 100-123/200", false),
            (206, "bytes 0-23/200", false),
            (416, "bytes */200", false),
        ] {
            let response = HTTPURLResponse(url: url, statusCode: status,
                                           httpVersion: "HTTP/1.1", headerFields: ["Content-Range": contentRange])!
            #expect(validator.accepts(response) == expected)
        }
    }

    @Test func optionalCheckpointRoundtrip() async throws {
        guard let path = ProcessInfo.processInfo.environment["NEMOTRON_CODEC_CHECKPOINT"] else { return }
        let model = try await NemotronVoiceChatCodec.fromPretrained(path)
        let length = model.waveformToTokenRatio * 3 + 29
        let input = (sin(MLXArray(0..<length).asType(.float32) * 0.07) * 0.1).reshaped([1, length, 1])
        let encoded = model.encodeAudio(input)
        let audio = model.decodeAudio(encoded)
        eval(encoded.codes, audio)
        #expect(encoded.codes.shape == [1, 31, 4])
        #expect(audio.shape == input.shape)
        #expect(MLX.all(isFinite(audio)).item(Bool.self))
    }
}
