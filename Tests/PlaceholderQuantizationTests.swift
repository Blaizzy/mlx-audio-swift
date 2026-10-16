//  Every quantized checkpoint loader builds the model with random init, swaps
//  in quantized layers, then overwrites their arrays from safetensors. With
//  `quantize(model:)` that left the unevaluated init graph of every quantized
//  layer alive for the lifetime of the process (ml-explore/mlx#4710). These
//  tests pin the placeholder replacement the loaders use instead: same shapes,
//  same output once loaded, and nothing left behind on a reload.

import MLX
import MLXNN
import Testing

@testable import MLXAudioCore

private final class TwoLayerModel: Module {
    @ModuleInfo var proj: Linear
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    override init() {
        _proj.wrappedValue = Linear(256, 64, bias: true)
        _embedTokens.wrappedValue = Embedding(embeddingCount: 32, dimensions: 256)
        super.init()
    }
}

/// Serialized: two of these read process-wide state (the global random key,
/// MLX's active memory) that a concurrently running test would move.
@Suite(.serialized)
struct PlaceholderQuantizationTests {
    private static let groupSize = 64
    private static let bits = 4

    /// A 4-bit checkpoint for `TwoLayerModel`, made the way a converter would.
    private static func makeCheckpoint() -> [String: MLXArray] {
        let proj = MLXRandom.uniform(low: Float(-1), high: Float(1), [64, 256])
        let embed = MLXRandom.normal([32, 256])
        let (projWeight, projScales, projBiases) = MLX.quantized(proj, groupSize: groupSize, bits: bits)
        let (embedWeight, embedScales, embedBiases) = MLX.quantized(
            embed, groupSize: groupSize, bits: bits)
        let checkpoint: [String: MLXArray] = [
            "proj.weight": projWeight,
            "proj.scales": projScales,
            "proj.biases": projBiases!,
            "proj.bias": MLXRandom.normal([64]),
            "embed_tokens.weight": embedWeight,
            "embed_tokens.scales": embedScales,
            "embed_tokens.biases": embedBiases!,
        ]
        eval(Array(checkpoint.values))
        return checkpoint
    }

    /// The loader recipe under test: construct, quantize, load, evaluate.
    private static func load(
        _ checkpoint: [String: MLXArray], quantizing: (TwoLayerModel) -> Void
    ) throws -> TwoLayerModel {
        let model = withThrowawayRandomState { TwoLayerModel() }
        quantizing(model)
        try model.update(parameters: ModuleParameters.unflattened(checkpoint), verify: .all)
        eval(model)
        return model
    }

    @Test func placeholdersHaveTheShapesQuantizeProduces() throws {
        let reference = TwoLayerModel()
        quantize(model: reference, groupSize: Self.groupSize, bits: Self.bits)
        let placeholder = TwoLayerModel()
        quantizeWithPlaceholderArrays(model: placeholder, groupSize: Self.groupSize, bits: Self.bits)

        let referenceParameters = reference.parameters().flattened()
        let placeholderParameters = Dictionary(
            uniqueKeysWithValues: placeholder.parameters().flattened())
        #expect(referenceParameters.count == placeholderParameters.count)
        for (key, expected) in referenceParameters {
            let actual = try #require(placeholderParameters[key], "\(key)")
            #expect(actual.shape == expected.shape, "\(key)")
        }

        let proj = try #require(placeholder.proj as? QuantizedLinear)
        #expect(proj.weight.dtype == .uint32)
        #expect(proj.groupSize == Self.groupSize)
        #expect(proj.bits == Self.bits)
        #expect(proj.mode == .affine)
        #expect(proj.shape.0 == 64)
        #expect(proj.shape.1 == 256)
        let embed = try #require(placeholder.embedTokens as? QuantizedEmbedding)
        #expect(embed.weight.dtype == .uint32)
        #expect(embed.shape.0 == 32)
        #expect(embed.shape.1 == 256)
    }

    @Test func loadedPlaceholdersComputeWhatTheLoadedLayersCompute() throws {
        let checkpoint = Self.makeCheckpoint()
        let reference = try Self.load(checkpoint) {
            quantize(model: $0, groupSize: Self.groupSize, bits: Self.bits)
        }
        let placeholder = try Self.load(checkpoint) {
            quantizeWithPlaceholderArrays(model: $0, groupSize: Self.groupSize, bits: Self.bits)
        }

        let hidden = MLXRandom.normal([3, 256])
        let tokens = MLXArray([Int32(1), 7, 31])
        #expect(arrayEqual(placeholder.proj(hidden), reference.proj(hidden)).item(Bool.self))
        #expect(
            arrayEqual(placeholder.embedTokens(tokens), reference.embedTokens(tokens)).item(Bool.self)
        )
    }

    @Test func constructingUnderAThrowawayKeyLeavesTheGlobalKeyAlone() {
        MLXRandom.seed(7)
        let expected = MLXRandom.uniform(0 ..< 1, [4])
        MLXRandom.seed(7)
        _ = withThrowawayRandomState { TwoLayerModel() }
        let actual = MLXRandom.uniform(0 ..< 1, [4])
        #expect(arrayEqual(actual, expected).item(Bool.self))
    }

    /// Regression: loading into layers made by `quantize(model:)` overwrote
    /// the lazy quantize outputs in place, which kept the init graph — and the
    /// scalar buffers feeding it — alive once per layer per load.
    ///
    /// Other suites may allocate concurrently, so a single before/after
    /// comparison could fail on noise. The leak is monotonic (a few KB per
    /// 100 reloads, never freed), so it is enough that *some* window of 100
    /// reloads leaves active memory exactly where it was; with the leak no
    /// window ever does.
    @Test func reloadingIntoPlaceholderLayersLeavesNoBuffersBehind() throws {
        let checkpoint = Self.makeCheckpoint()
        func reload(times: Int) throws {
            for _ in 0 ..< times {
                _ = try Self.load(checkpoint) {
                    quantizeWithPlaceholderArrays(model: $0, groupSize: Self.groupSize, bits: Self.bits)
                }
            }
        }
        try reload(times: 5)
        var growths: [Int] = []
        for _ in 0 ..< 5 {
            let activeMemoryBefore = Memory.activeMemory
            try reload(times: 100)
            let growth = Memory.activeMemory - activeMemoryBefore
            if growth == 0 { return }
            growths.append(growth)
        }
        Issue.record("active memory grew on every window of 100 reloads: \(growths) bytes")
    }
}
