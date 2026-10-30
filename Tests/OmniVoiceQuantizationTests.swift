import Foundation
import HuggingFace
import MLX
import MLXNN
import Testing
@testable import MLXAudioTTS

@Suite("OmniVoice row-wise int8 checkpoint packing", .serialized)
struct OmniVoiceQuantizationTests {
    @Test func packsSignedCodesAndAffineBiases() throws {
        let codes: [Int8] = [-127, -3, 0, 127, 1, -1, 42, -42]
        let scales = MLXArray([Float(0.01), Float(0.02)], [2, 1])
        let weights = [
            "layer.weight": MLXArray(codes, [2, 4]),
            "layer.weight.scales": scales
        ]

        let packed = try OmniVoiceModel.packRowwiseInt8ForMLX(weights: weights, groupSize: 4)

        #expect(packed["layer.weight"]?.shape == [2, 1])
        #expect(packed["layer.scales"]?.shape == [2, 1])
        #expect(packed["layer.biases"]?.shape == [2, 1])

        let packedCodes = try #require(packed["layer.weight"]).asArray(UInt32.self)
        let expectedFirst = UInt32(1) | (UInt32(125) << 8) | (UInt32(128) << 16) | (UInt32(255) << 24)
        let expectedSecond = UInt32(129) | (UInt32(127) << 8) | (UInt32(170) << 16) | (UInt32(86) << 24)
        #expect(packedCodes == [expectedFirst, expectedSecond])

        let packedScales = try #require(packed["layer.scales"]).asArray(Float.self)
        let packedBiases = try #require(packed["layer.biases"]).asArray(Float.self)
        #expect(packedScales == [0.01, 0.02])
        #expect(packedBiases[0].isApproximatelyEqual(to: -1.28, absoluteTolerance: 1e-5))
        #expect(packedBiases[1].isApproximatelyEqual(to: -2.56, absoluteTolerance: 1e-5))
    }

    @Test(arguments: [4, 8])
    func loadsNativeQuantizedCheckpoint(bits: Int) async throws {
        // Use a unique cached repository so the public loader runs without
        // network requests or changes to the process-wide cache configuration.
        let repoID = "omnivoice-tests/native-\(UUID().uuidString)"
        let directory = HubCache.default.cacheDirectory
            .appendingPathComponent("mlx-audio/\(repoID.replacingOccurrences(of: "/", with: "_"))")
        let audioDirectory = directory.appendingPathComponent("audio_tokenizer")
        try FileManager.default.createDirectory(at: audioDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let configJSON = """
        {
            "audio_vocab_size": 17, "audio_mask_id": 16, "num_audio_codebook": 2,
            "quantization": {"bits": \(bits), "group_size": 64},
            "llm_config": {
                "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 1,
                "num_attention_heads": 2, "num_key_value_heads": 1,
                "head_dim": 32, "vocab_size": 64
            }
        }
        """
        let configData = Data(configJSON.utf8)
        try configData.write(to: directory.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(OmniVoiceConfig.self, from: configData)
        let source = try OmniVoiceModel(config: config)
        quantize(model: source, groupSize: 64, bits: bits)
        let weights = Dictionary(uniqueKeysWithValues: source.parameters().flattened().map { key, value in
            let checkpointKey = key.hasPrefix("llm.model.")
                ? "llm." + String(key.dropFirst("llm.model.".count)) : key
            return (checkpointKey, value)
        })
        try MLX.save(arrays: weights, url: directory.appendingPathComponent("model.safetensors"))

        let codecJSON = """
        {
            "codebook_size": 16, "codebook_dim": 4, "n_codebooks": 2,
            "encoder_hidden_size": 8, "decoder_hidden_size": 64,
            "downsampling_ratios": [2, 2, 2], "upsampling_ratios": [2, 2, 2]
        }
        """
        let codecData = Data(codecJSON.utf8)
        try codecData.write(to: audioDirectory.appendingPathComponent("config.json"))
        let codecConfig = try JSONDecoder().decode(OmniVoiceAudioTokenizerConfig.self, from: codecData)
        let codec = OmniVoiceAudioTokenizer(config: codecConfig)
        try MLX.save(
            arrays: Dictionary(uniqueKeysWithValues: codec.parameters().flattened()),
            url: audioDirectory.appendingPathComponent("model.safetensors"))

        let tokenizer: [String: Any] = [
            "model": ["type": "BPE", "vocab": ["<unk>": 0], "merges": [], "unk_token": "<unk>"],
            "added_tokens": [["id": 0, "content": "<unk>", "special": true]],
        ]
        try JSONSerialization.data(withJSONObject: tokenizer)
            .write(to: directory.appendingPathComponent("tokenizer.json"))
        try Data(#"{"tokenizer_class":"PreTrainedTokenizerFast","unk_token":"<unk>"}"#.utf8)
            .write(to: directory.appendingPathComponent("tokenizer_config.json"))
        let filenames = [
            "config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
            "audio_tokenizer/config.json", "audio_tokenizer/model.safetensors",
        ]
        let manifest: [String: Any] = [
            "revision": "test-fixture", "files": filenames.map { ["rfilename": $0] },
        ]
        try JSONSerialization.data(withJSONObject: manifest)
            .write(to: directory.appendingPathComponent(".mlx-audio-snapshot.json"))

        let loaded = try await OmniVoiceModel.fromPretrained(repoID)
        let parameters = Dictionary(uniqueKeysWithValues: loaded.parameters().flattened())
        for key in ["audio_heads.0.weight", "audio_embeddings.0.weight", "llm.model.embed_tokens.weight"] {
            let weight = try #require(parameters[key])
            try #require(weight.dtype == .uint32)
        }

        let ids = MLXArray.zeros([1, 2, 6], dtype: .int32)
        let mask = MLXArray([false, false, true, true, true, true]).reshaped([1, 6])
        let expected = source.forward(inputIds: ids, audioMask: mask)
        let actual = loaded.forward(inputIds: ids, audioMask: mask)
        let difference = abs(expected - actual).max().item(Float.self)
        #expect(difference < 0.00001)
    }
}

private extension Float {
    func isApproximatelyEqual(to other: Float, absoluteTolerance: Float) -> Bool {
        abs(self - other) <= absoluteTolerance
    }
}
