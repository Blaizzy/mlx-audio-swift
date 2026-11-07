import MLX
import MLXNN
import Testing

@testable import MLXAudioTTS

struct VoxCPM2DtypeTests {
    private static func tinyConfig() -> VoxCPM2LMConfig {
        VoxCPM2LMConfig(
            hiddenSize: 32, numHiddenLayers: 1, numAttentionHeads: 4, numKeyValueHeads: 2,
            intermediateSize: 64, vocabSize: 16,
            ropeLongFactor: [1, 1, 1, 1], ropeShortFactor: [1, 1, 1, 1]
        )
    }

    private static func toBFloat16(_ module: Module) {
        let params = module.parameters().flattened().map {
            ($0.0, $0.1.dtype.isFloatingPoint ? $0.1.asType(.bfloat16) : $0.1)
        }
        module.update(parameters: ModuleParameters.unflattened(params))
    }

    @Test func lmKeepsWeightDtypeThroughRotaryEmbedding() {
        let lm = VoxMiniCPMModel(Self.tinyConfig())
        Self.toBFloat16(lm)

        let (hidden, cache) = lm(inputsEmbeds: MLXRandom.normal([1, 5, 32]).asType(.bfloat16))
        #expect(hidden.dtype == .bfloat16)
        #expect(cache[0].0.dtype == .bfloat16)
    }

    @Test func ditSamplesInConditioningDtype() {
        let config = Self.tinyConfig()
        let cfm = VoxUnifiedCFM(
            inChannels: 4, cfmParams: VoxCPM2CFMConfig(),
            estimator: VoxCPMLocDiT(config, inChannels: 4)
        )
        Self.toBFloat16(cfm)

        let mu = MLXRandom.normal([1, 64]).asType(.bfloat16)
        let cond = MLXRandom.normal([1, 4, 2])
        #expect(cond.dtype == .float32)
        let feat = cfm.sample(mu: mu, nTimesteps: 4, patchSize: 2, cond: cond, cfgValue: 2.0)
        #expect(feat.dtype == .bfloat16)
    }
}
