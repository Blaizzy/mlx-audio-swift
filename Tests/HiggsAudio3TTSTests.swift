import Foundation
import MLX
import MLXAudioCore
import MLXNN
import Testing
import Tokenizers
@testable import MLXAudioTTS

private func higgsV3TinyConfig() -> HiggsAudio3TTSConfig {
    .init(textConfig: .init(vocabSize: 128, hiddenSize: 64, numHiddenLayers: 2,
        intermediateSize: 128, numAttentionHeads: 4, numKeyValueHeads: 2,
        headDim: 32), audioNumCodebooks: 3, audioCodebookSize: 10,
        audioBOCTokenID: 8, audioEOCTokenID: 9)
}

@Suite("Higgs Audio v3 TTS")
struct HiggsAudio3TTSTests {
    @Test func originalAndMLXConfigLayouts() throws {
        let text = "\"text_config\":{\"vocab_size\":128,\"hidden_size\":64,\"num_hidden_layers\":2,\"intermediate_size\":128,\"num_attention_heads\":4,\"num_key_value_heads\":2,\"head_dim\":32,\"rope_parameters\":{\"rope_theta\":1000000}}"
        for audio in ["\"audio_encoder_config\":{\"num_codebooks\":3,\"vocab_size\":10}", "\"audio_num_codebooks\":3,\"audio_codebook_size\":10"] {
            let c = try JSONDecoder().decode(HiggsAudio3TTSConfig.self, from: Data(("{" + text + "," + audio + "}").utf8))
            #expect(c.textConfig.headDim == 32)
            #expect(c.textConfig.ropeTheta == 1000000)
            #expect(c.audioNumCodebooks == 3)
            #expect(c.audioBOCTokenID == 8)
            #expect(c.audioEOCTokenID == 9)
        }
    }

    @Test func delayRoundTripAndSamplerFlush() {
        let raw = MLXArray((0..<12).map { Int32($0 % 8) }).reshaped([4, 3])
        let delayed = higgsV3Delay(raw, boc: 8, eoc: 9)
        #expect(delayed.shape == [6, 3])
        #expect(delayed[0].asArray(Int32.self) == [0, 8, 8])
        #expect(higgsV3Undelay(delayed).asArray(Int32.self) == raw.asArray(Int32.self))
        var state = HiggsV3SamplerState()
        #expect(state.step(MLXArray([1, 2, 3]), codebooks: 3, boc: 8, eoc: 9).asArray(Int32.self) == [1, 8, 8])
        _ = state.step(MLXArray([1, 2, 3]), codebooks: 3, boc: 8, eoc: 9)
        _ = state.step(MLXArray([1, 2, 3]), codebooks: 3, boc: 8, eoc: 9)
        _ = state.step(MLXArray([9, 2, 3]), codebooks: 3, boc: 8, eoc: 9)
        #expect(!state.done)
        _ = state.step(MLXArray([9, 9, 3]), codebooks: 3, boc: 8, eoc: 9)
        #expect(state.done)
    }

    @Test func fusedFeedbackHeadAndCachedAttention() throws {
        let config = higgsV3TinyConfig()
        let model = HiggsAudio3TTSModel(config)
        #expect(model.backbone.layers[0].selfAttention.qProj.weight.shape == [128, 64])
        let codes = MLXArray([1, 2, 3, 4, 5, 6]).reshaped([2, 3])
        let embedded = model.embedAudio(codes)
        let expected = model.multimodalEmbedding(MLXArray([1, 12, 23, 4, 15, 26]).reshaped([2, 3])).sum(axis: 1)
        #expect(abs(embedded - expected).max().item(Float.self) < 1e-6)
        #expect(model.audioLogits(embedded).shape == [2, 3, 10])
        let tokens = MLXArray([1, 2, 3, 4]).reshaped([1, 4])
        let full = try model.backbone(inputIDs: tokens)
        let cache = model.backbone.makeCache()
        _ = try model.backbone(inputIDs: tokens[0..., 0..<3], cache: cache)
        let last = try model.backbone(inputIDs: tokens[0..., 3..<4], cache: cache)
        // Metal uses different SDPA kernels for prefill and one-token decode.
        #expect(abs(full[0..., 3..<4, 0...] - last).max().item(Float.self) < 1e-2)
    }

    @Test func referencePromptAndInlineControls() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("higgs-v3-tokenizer-\(UUID())")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let specials = ["<|tts|>", "<|text|>", "<|audio|>", "<|ref_audio|>", "<|ref_text|>", "<|emotion:enthusiasm|>"]
        var vocabulary: [String: Int] = ["<unk>": 0, "a": 1]
        var added: [[String: Any]] = []
        for (index, token) in specials.enumerated() {
            vocabulary[token] = 100 + index
            added.append(["id": 100 + index, "content": token, "special": true])
        }
        try JSONSerialization.data(withJSONObject: [
            "model": ["type": "BPE", "vocab": vocabulary, "merges": [], "unk_token": "<unk>"],
            "added_tokens": added,
        ]).write(to: directory.appendingPathComponent("tokenizer.json"))
        try JSONSerialization.data(withJSONObject: ["tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "<unk>"])
            .write(to: directory.appendingPathComponent("tokenizer_config.json"))
        let model = HiggsAudio3TTSModel(higgsV3TinyConfig())
        model.tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        let codes = MLXArray([1, 2, 3, 4, 5, 6]).reshaped([2, 3])
        let (embeddings, ids) = try model.promptEmbeddings(text: "<|emotion:enthusiasm|>", references: [.init(codes: codes, text: "a")])
        #expect(ids == [100, 104, 1, 103, -100, -100, 101, 105, 102])
        #expect(embeddings.shape == [1, ids.count, 64])
        #expect(abs(embeddings[0, 4..<6, 0...] - model.embedAudio(codes)).max().item(Float.self) == 0)
    }

    @Test func quantizedBackboneStrictRoundTrip() throws {
        let source = HiggsAudio3TTSModel(higgsV3TinyConfig())
        quantize(model: source) { name, _ in name.hasPrefix("backbone.") ? (32, 4, .affine) : nil }
        let weights = Dictionary(uniqueKeysWithValues: source.parameters().flattened())
        #expect(weights["backbone.layers.0.mlp.gate_proj.scales"] != nil)
        let destination = HiggsAudio3TTSModel(higgsV3TinyConfig())
        quantize(model: destination) { name, _ in name.hasPrefix("backbone.") ? (32, 4, .affine) : nil }
        try destination.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        let ids = MLXArray([1, 2]).reshaped([1, 2])
        let a = try source.backbone(inputIDs: ids)
        let b = try destination.backbone(inputIDs: ids)
        #expect(abs(a - b).max().item(Float.self) == 0)
        #expect(destination.multimodalEmbedding is QuantizedEmbedding == false)
    }
}
