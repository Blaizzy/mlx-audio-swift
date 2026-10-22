import Foundation
import MLX
import MLXNN

/// Metadata and load-time adaptation for published ternary and prepared MLX Redux weights.
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

    var weightFormat: String { reduxWeightFormat ?? "thrush-ternary-v2" }

    static func decode(_ data: Data) throws -> Self {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let config = try decoder.decode(Self.self, from: data)
        guard ["mlx-2bit", "thrush-ternary-v2"].contains(config.weightFormat) else {
            throw STTError.invalidInput("Unsupported Parakeet Redux weight format: \(config.weightFormat)")
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

    private var projectionShapes: [String: [Int]] {
        let d = encoderConfig.hiddenSize, ff = encoderConfig.intermediateSize
        let shapes = [
            "feed_forward1.linear1": [ff, d], "feed_forward1.linear2": [d, ff],
            "feed_forward2.linear1": [ff, d], "feed_forward2.linear2": [d, ff],
            "self_attn.linear_q": [d, d], "self_attn.linear_k": [d, d],
            "self_attn.linear_v": [d, d], "self_attn.linear_out": [d, d], "self_attn.linear_pos": [d, d],
            "conv.pointwise_conv1": [2 * d, d], "conv.pointwise_conv2": [d, d],
        ]
        var result: [String: [Int]] = [:]
        for i in 0..<encoderConfig.numHiddenLayers {
            for (suffix, shape) in shapes { result["encoder.layers.\(i).\(suffix)"] = shape }
        }
        return result
    }

    /// Replaces only the encoder projections declared by the checkpoint.
    func configure(_ model: ParakeetModel, weights: [String: MLXArray]) throws {
        let expected = projectionShapes
        let names = ternaryModules.map(Self.mapWeightName)
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

    func validateManifest(in directory: URL) throws {
        guard weightFormat == "thrush-ternary-v2" else { return }
        struct Manifest: Decodable {
            struct Quant: Decodable { let mode: String; let groupSize: Int }
            struct Packing: Decodable { let base: Int; let elementsPerByte: Int; let codeOffset: Int }
            struct Entry: Decodable {
                let name: String
                let inFeatures: Int
                let outFeatures: Int
                let groupSize: Int
                let hasBias: Bool
                let asConv1D: Bool
            }
            let format: String
            let names: String
            let quant: Quant
            let packing: Packing
            let quantizedModules: [Entry]
        }
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let manifest = try decoder.decode(Manifest.self,
            from: Data(contentsOf: directory.appendingPathComponent("ternary.json")))
        guard manifest.format == weightFormat, manifest.names == "hf",
            manifest.quant.mode == "ternary", manifest.quant.groupSize == 128,
            manifest.packing.base == 3, manifest.packing.elementsPerByte == 5, manifest.packing.codeOffset == 1,
            manifest.quantizedModules.count == ternaryModules.count,
            Set(manifest.quantizedModules.map(\.name)) == Set(ternaryModules)
        else { throw STTError.invalidInput("Unsupported Redux ternary manifest") }
        let shapes = projectionShapes
        for entry in manifest.quantizedModules {
            guard let shape = shapes[Self.mapWeightName(entry.name)],
                shape == [entry.outFeatures, entry.inFeatures], entry.groupSize == 128, !entry.hasBias,
                entry.asConv1D == entry.name.contains(".conv.pointwise_conv")
            else { throw STTError.invalidInput("Redux manifest shape/type mismatch: \(entry.name)") }
        }
    }

    /// Losslessly expands base-3 bytes into the 2-bit words used by MLX's affine kernels.
    static func repackTernary(_ weight: MLXArray, inputDimensions: Int) throws -> MLXArray {
        guard weight.dtype == .uint8, weight.ndim == 2, inputDimensions > 0, inputDimensions % 16 == 0,
            weight.dim(1) == (inputDimensions + 4) / 5,
            !MLX.any(weight .> 242).item(Bool.self)
        else { throw STTError.invalidInput("Expected valid uint8 Redux ternary rows of ceil(in_features / 5) bytes") }
        let powers = MLXArray([UInt32(1), 3, 9, 27, 81])
        let digits = remainder(floorDivide(weight.asType(.uint32).expandedDimensions(axis: 2), powers), 3)
            .reshaped(weight.dim(0), -1)[0..., 0..<inputDimensions]
        let shifts = MLXArray((0..<16).map { UInt32(2 * $0) })
        let packed = (digits.reshaped(weight.dim(0), -1, 16) << shifts).sum(axis: -1).asType(.uint32)
        // Release each projection's expanded digits before adapting the next matrix.
        eval(packed)
        return packed
    }

    static func mapWeightName(_ name: String) -> String {
        let mappings = [
            ("encoder.subsampling.layers.", "encoder.pre_encode.conv."),
            ("encoder.subsampling.linear", "encoder.pre_encode.out"),
            (".conv.norm.", ".conv.batch_norm."),
            (".self_attn.q_proj", ".self_attn.linear_q"),
            (".self_attn.k_proj", ".self_attn.linear_k"),
            (".self_attn.v_proj", ".self_attn.linear_v"),
            (".self_attn.o_proj", ".self_attn.linear_out"),
            (".self_attn.relative_k_proj", ".self_attn.linear_pos"),
            (".self_attn.bias_u", ".self_attn.pos_bias_u"),
            (".self_attn.bias_v", ".self_attn.pos_bias_v"),
            ("decoder.embedding.", "decoder.prediction.embed."),
            ("decoder.decoder_projector.", "joint.pred."),
            ("encoder_projector.", "joint.enc."),
            ("joint.head.", "joint.joint_net.2."),
        ]
        var mapped = mappings.reduce(name) { $0.replacingOccurrences(of: $1.0, with: $1.1) }
        for (source, target) in [("ih", "Wx"), ("hh", "Wh")] {
            mapped = mapped.replacingOccurrences(of: "^decoder\\.lstm\\.weight_" + source + "_l(\\d+)$",
                with: "decoder.prediction.dec_rnn.lstm.$1." + target, options: .regularExpression)
        }
        return mapped
    }

    /// Adapts published weights in memory; prepared MLX tensors pass through unchanged.
    func sanitize(_ weights: [String: MLXArray]) throws -> [String: MLXArray] {
        guard weightFormat == "thrush-ternary-v2" else { return weights }
        let ignored = Set(["proj", "ctx", "out"].flatMap { layer in
            ["weight", "bias"].map { "vad_head.\(layer).\($0)" }
        } + (0..<encoderConfig.numHiddenLayers).map { "encoder.layers.\($0).conv.norm.num_batches_tracked" })
        let shapes = projectionShapes
        let names = Set(ternaryModules)
        var result: [String: MLXArray] = [:]
        for (name, original) in weights {
            if ignored.contains(name) || name.range(of: "^decoder\\.lstm\\.bias_(ih|hh)_l\\d+$", options: .regularExpression) != nil { continue }
            var target = Self.mapWeightName(name)
            var value = original
            if name.hasSuffix(".qweight") {
                let module = String(name.dropLast(".qweight".count))
                guard names.contains(module), let shape = shapes[Self.mapWeightName(module)], value.ndim == 2,
                    value.dim(0) == shape[0] else { throw STTError.invalidInput("Unexpected Redux ternary weight: \(name)") }
                value = try Self.repackTernary(value, inputDimensions: shape[1])
                target = Self.mapWeightName(module) + ".weight"
            } else if name.hasSuffix(".scales") {
                let module = String(name.dropLast(".scales".count))
                guard names.contains(module), value.dtype.isFloatingPoint,
                    MLX.all(isFinite(value)).item(Bool.self), MLX.all(value .>= 0).item(Bool.self)
                else { throw STTError.invalidInput("Invalid Redux ternary scales: \(name)") }
                guard result[Self.mapWeightName(module) + ".biases"] == nil else {
                    throw STTError.invalidInput("Duplicate Redux affine biases: \(name)")
                }
                result[Self.mapWeightName(module) + ".biases"] = -value
            } else if name.hasSuffix(".weight") {
                if value.ndim == 3 { value = value.transposed(0, 2, 1) }
                if value.ndim == 4 { value = value.transposed(0, 2, 3, 1) }
            }
            guard result[target] == nil else { throw STTError.invalidInput("Duplicate Redux weight: \(target)") }
            result[target] = value
        }
        for i in 0..<numDecoderLayers {
            guard let input = weights["decoder.lstm.bias_ih_l\(i)"], let hidden = weights["decoder.lstm.bias_hh_l\(i)"] else {
                throw STTError.invalidInput("Missing Redux LSTM biases for layer \(i)")
            }
            result["decoder.prediction.dec_rnn.lstm.\(i).bias"] = input + hidden
        }
        return result
    }
}
