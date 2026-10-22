import Foundation
import MLX
import MLXNN

/// Metadata for the MLX 2-bit artifacts prepared by Python mlx-audio.
struct ParakeetReduxConfig: Decodable {
    struct Encoder: Decodable {
        let hiddenSize: Int
        let intermediateSize: Int
        let numHiddenLayers: Int
        let numAttentionHeads: Int
        let numKeyValueHeads: Int
        let numMelBins: Int
        let convKernelSize: Int
        let subsamplingFactor: Int
        let subsamplingConvChannels: Int
        let subsamplingConvKernelSize: Int
        let subsamplingConvStride: Int
        let maxPositionEmbeddings: Int
        let hiddenAct: String
        let attentionBias: Bool?
        let convolutionBias: Bool?
        let scaleInput: Bool?
    }

    let modelType: String
    let reduxWeightFormat: String?
    let encoderConfig: Encoder
    let decoderHiddenSize: Int
    let numDecoderLayers: Int
    let hiddenAct: String
    let blankTokenId: Int
    let vocabSize: Int
    let durations: [Int]
    let maxSymbolsPerStep: Int
    let ternaryGroupSize: Int
    let ternaryModules: [String]

    static func decode(_ data: Data) throws -> Self {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let config = try decoder.decode(Self.self, from: data)
        guard config.reduxWeightFormat == "mlx-2bit" else {
            throw STTError.invalidInput("Parakeet Redux requires a Python-prepared mlx-2bit checkpoint")
        }
        return config
    }

    func modelConfig(tokenizerAt url: URL) throws -> ParakeetRawConfig {
        let e = encoderConfig
        guard modelType == "parakeet_tdt", e.hiddenSize > 0, e.numHiddenLayers > 0,
            e.numAttentionHeads > 0, e.hiddenSize % e.numAttentionHeads == 0,
            e.numAttentionHeads == e.numKeyValueHeads,
            e.intermediateSize > 0, e.intermediateSize % e.hiddenSize == 0,
            e.hiddenAct == "silu", e.attentionBias != true, e.convolutionBias != true,
            [2, 4, 8].contains(e.subsamplingFactor), e.numMelBins > 0,
            e.numMelBins % e.subsamplingFactor == 0, e.subsamplingConvChannels > 0,
            e.subsamplingConvKernelSize == 3, e.subsamplingConvStride == 2,
            e.convKernelSize > 0, e.convKernelSize % 2 == 1, e.maxPositionEmbeddings > 0,
            hiddenAct == "relu", decoderHiddenSize > 0, numDecoderLayers > 0,
            blankTokenId > 0, blankTokenId == vocabSize - 1,
            durations.first == 0, zip(durations, durations.dropFirst()).allSatisfy({ $0 < $1 }),
            maxSymbolsPerStep > 0, ternaryGroupSize == 128,
            Set(ternaryModules).count == ternaryModules.count
        else { throw STTError.invalidInput("Unsupported Parakeet Redux configuration") }

        struct Tokenizer: Decodable {
            struct Model: Decodable { let vocab: [String: Int] }
            struct AddedToken: Decodable { let content: String; let id: Int }
            let model: Model
            let addedTokens: [AddedToken]
        }
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let tokenizer = try decoder.decode(Tokenizer.self, from: Data(contentsOf: url))
        let vocab = tokenizer.model.vocab
        guard vocab.count == blankTokenId, Set(vocab.values) == Set(0..<blankTokenId),
            tokenizer.addedTokens.filter({ $0.content == "<blank>" }).map(\.id) == [blankTokenId]
        else { throw STTError.invalidInput("Parakeet Redux tokenizer must have a contiguous vocabulary followed by blank") }
        let vocabulary = vocab.sorted { $0.value < $1.value }.map(\.key)
        let object: [String: Any] = [
            "target": "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel",
            "model_defaults": ["tdt_durations": durations],
            "preprocessor": ["sample_rate": 16000, "normalize": "per_feature", "window_size": 0.025,
                "window_stride": 0.01, "window": "hann", "features": e.numMelBins, "n_fft": 512,
                "dither": 0.0, "normalize_valid_frames": true],
            "encoder": ["feat_in": e.numMelBins, "n_layers": e.numHiddenLayers, "d_model": e.hiddenSize,
                "n_heads": e.numAttentionHeads, "ff_expansion_factor": e.intermediateSize / e.hiddenSize,
                "subsampling_factor": e.subsamplingFactor, "self_attention_model": "rel_pos",
                "subsampling": "dw_striding", "conv_kernel_size": e.convKernelSize,
                "subsampling_conv_channels": e.subsamplingConvChannels, "pos_emb_max_len": e.maxPositionEmbeddings,
                "use_bias": false, "xscaling": e.scaleInput ?? false, "mask_padding": true],
            "decoder": ["blank_as_pad": true, "vocab_size": vocabulary.count,
                "prednet": ["pred_hidden": decoderHiddenSize, "pred_rnn_layers": numDecoderLayers]],
            "joint": ["num_classes": vocabulary.count, "vocabulary": vocabulary, "num_extra_outputs": durations.count,
                "jointnet": ["joint_hidden": decoderHiddenSize, "activation": "relu",
                    "encoder_hidden": e.hiddenSize, "pred_hidden": decoderHiddenSize]],
            "decoding": ["model_type": "tdt", "durations": durations, "greedy": ["max_symbols": maxSymbolsPerStep]],
        ]
        return try JSONDecoder().decode(ParakeetRawConfig.self, from: JSONSerialization.data(withJSONObject: object))
    }

    /// Replaces only the encoder projections declared by the prepared checkpoint.
    func configure(_ model: ParakeetModel, weights: [String: MLXArray]) throws {
        let d = encoderConfig.hiddenSize, ff = encoderConfig.intermediateSize
        let shapes = [
            "feed_forward1.linear1": [ff, d], "feed_forward1.linear2": [d, ff],
            "feed_forward2.linear1": [ff, d], "feed_forward2.linear2": [d, ff],
            "self_attn.linear_q": [d, d], "self_attn.linear_k": [d, d],
            "self_attn.linear_v": [d, d], "self_attn.linear_out": [d, d], "self_attn.linear_pos": [d, d],
            "conv.pointwise_conv1": [2 * d, d], "conv.pointwise_conv2": [d, d],
        ]
        var expected: [String: [Int]] = [:]
        for i in 0..<encoderConfig.numHiddenLayers {
            for (suffix, shape) in shapes { expected["encoder.layers.\(i).\(suffix)"] = shape }
        }
        let names = ternaryModules.map { name in
            [("q_proj", "linear_q"), ("k_proj", "linear_k"), ("v_proj", "linear_v"),
             ("o_proj", "linear_out"), ("relative_k_proj", "linear_pos")].reduce(name) {
                $0.replacingOccurrences(of: "." + $1.0, with: "." + $1.1)
            }
        }
        guard Set(names) == Set(expected.keys), names.count == expected.count else {
            throw STTError.invalidInput("Redux ternary_modules must cover every encoder projection exactly once")
        }
        var replacements: [(String, Module)] = []
        for name in names {
            let shape = expected[name]!
            guard shape[1] % ternaryGroupSize == 0,
                let weight = weights[name + ".weight"], weight.dtype == .uint32,
                weight.shape == [shape[0], shape[1] / 16],
                let scales = weights[name + ".scales"], scales.dtype.isFloatingPoint,
                scales.shape == [shape[0], shape[1] / ternaryGroupSize],
                let biases = weights[name + ".biases"], biases.dtype == scales.dtype,
                biases.shape == scales.shape
            else { throw STTError.invalidInput("Invalid prepared Redux projection: \(name)") }
            replacements.append((name, QuantizedLinear(weight: weight, scales: scales, biases: biases,
                groupSize: ternaryGroupSize, bits: 2)))
        }
        try model.update(modules: ModuleChildren.unflattened(replacements), verify: .noUnusedKeys)
    }
}
