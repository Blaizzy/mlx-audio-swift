import MLXAudioCodecs

typealias OmniVoiceHubertModel = HiggsAudioHubertModel
typealias OmniVoiceSemanticEncoder = HiggsAudioSemanticEncoder

private extension OmniVoiceAudioTokenizerConfig {
    var higgsSemanticConfig: HiggsAudioSemanticConfig {
        var result = HiggsAudioSemanticConfig()
        result.hiddenSize = hiddenSize
        result.numHiddenLayers = numHiddenLayers
        result.numAttentionHeads = numAttentionHeads
        result.intermediateSize = intermediateSize
        result.convDim = convDim
        result.convKernel = convKernel
        result.convStride = convStride
        result.kernelSize = kernelSize
        return result
    }
}

extension HiggsAudioHubertModel {
    convenience init(config: OmniVoiceAudioTokenizerConfig) {
        self.init(config: config.higgsSemanticConfig)
    }
}
extension HiggsAudioSemanticEncoder {
    convenience init(config: OmniVoiceAudioTokenizerConfig) {
        self.init(config: config.higgsSemanticConfig)
    }
}

func omniVoiceSincResample(
    _ waveform: [Float], from originalRate: Int, to sampleRate: Int,
    lowpassFilterWidth: Int = 6, rolloff: Double = 0.99
) -> [Float] {
    higgsAudioSincResample(waveform, from: originalRate, to: sampleRate,
                         lowpassFilterWidth: lowpassFilterWidth, rolloff: rolloff)
}
