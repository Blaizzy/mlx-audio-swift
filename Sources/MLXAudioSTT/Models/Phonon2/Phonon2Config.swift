import Foundation

/// Metadata for the prepared MLX Phonon-2 checkpoint, including its exact packed projections.
public struct Phonon2Config: Decodable, Sendable {
    public struct PackedModule: Codable, Sendable {
        public let name: String
        /// Output and input dimensions, including for kernel-1 convolutions.
        public let shape: [Int]
    }

    public let modelType: String
    public let modules: [PackedModule]
    public let parakeet: ParakeetTDTConfig

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case modules = "phonon2_modules"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decode(String.self, forKey: .modelType)
        modules = try c.decode([PackedModule].self, forKey: .modules)
        parakeet = try ParakeetConfigParser.parseTDT(ParakeetRawConfig(from: decoder))
        guard modelType == "phonon2", parakeet.decoding.modelType == "tdt",
            !parakeet.decoding.durations.isEmpty,
            parakeet.decoding.durations.allSatisfy({ $0 >= 0 }),
            parakeet.decoding.durations.count == parakeet.joint.numExtraOutputs,
            let maximum = parakeet.decoding.greedy?.maxSymbols, maximum > 0,
            parakeet.encoder.selfAttentionModel == "rel_pos",
            parakeet.encoder.nLayers > 0, parakeet.encoder.dModel > 0,
            parakeet.encoder.ffExpansionFactor > 0
        else {
            throw Phonon2Error.invalidConfiguration("Expected relative-position Parakeet TDT, nonnegative durations and positive max_symbols")
        }

        let d = parakeet.encoder.dModel
        let ff = d * parakeet.encoder.ffExpansionFactor
        let shapes = [
            "feed_forward1.linear1": [ff, d], "feed_forward1.linear2": [d, ff],
            "feed_forward2.linear1": [ff, d], "feed_forward2.linear2": [d, ff],
            "self_attn.linear_q": [d, d], "self_attn.linear_k": [d, d],
            "self_attn.linear_v": [d, d], "self_attn.linear_out": [d, d],
            "self_attn.linear_pos": [d, d],
            "conv.pointwise_conv1": [2 * d, d], "conv.pointwise_conv2": [d, d],
        ]
        var expected: [String: [Int]] = [:]
        for i in 0..<parakeet.encoder.nLayers {
            for (suffix, shape) in shapes { expected["encoder.layers.\(i).\(suffix)"] = shape }
        }
        guard modules.count == expected.count, Set(modules.map(\.name)).count == modules.count,
            modules.allSatisfy({ expected[$0.name] == $0.shape && $0.shape[1] > 0 && $0.shape[1] % 128 == 0 })
        else {
            throw Phonon2Error.invalidConfiguration("phonon2_modules must cover every encoder projection exactly once with matching shapes")
        }
    }
}

public enum Phonon2Error: Error, LocalizedError {
    case invalidConfiguration(String)
    case invalidPackedWeight(String)

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let message): "Invalid Phonon-2 configuration: \(message)"
        case .invalidPackedWeight(let message): "Invalid Phonon-2 packed weight: \(message)"
        }
    }
}
