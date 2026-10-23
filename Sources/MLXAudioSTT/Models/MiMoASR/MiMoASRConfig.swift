import Foundation
import MLXAudioCore

public struct MiMoASRConfig: Codable {
    public var modelType = "mimo_v2_asr"
    public var languageModel = Qwen2Configuration(
        hiddenSize: 4096, hiddenLayers: 36,
        intermediateSize: 11008, attentionHeads: 32, rmsNormEps: 1e-6,
        vocabularySize: 151680, kvHeads: 8, ropeTheta: 640000)
    public var audioChannels = 8
    public var groupSize = 4
    public var delayPattern = Array(0 ..< 8)
    public var speechVocabSizes = [1025, 1025] + Array(repeating: 129, count: 6)
    public var speechEmptyIds = [1024, 1024] + Array(repeating: 128, count: 6)
    public var inputLocalLayers = 6
    public var inputLocalDim = 1024
    public var inputLocalHeads = 64
    public var inputLocalIntermediateSize = 4096
    public var inputFullAttention = true
    public var localLayers = 16
    public var localDim = 1024
    public var localHeads = 64
    public var localIntermediateSize = 4096
    public var localRopeTheta: Float = 640000
    public var endOfTurnId = 151672
    public var startOfSpeechId = 151665
    public var endOfSpeechId = 151666
    public var endOfTextSpeechId = 151671
    public var emptyId = 151667
    public var perLayerQuantization: BaseConfiguration.PerLayerQuantization?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case audioConfig = "audio_config"
        case audioChannels = "audio_channels"
        case groupSize = "group_size"
        case delayPattern = "delay_pattern"
        case speechVocabSizes = "speech_vocab_size"
        case speechEmptyIds = "speech_zeroemb_idx"
        case inputLocalLayers = "input_local_layers"
        case inputLocalDim = "input_local_dim"
        case inputLocalHeads = "input_local_attn_heads"
        case inputLocalIntermediateSize = "input_local_intermediate_size"
        case inputFullAttention = "input_full_attention"
        case localLayers = "local_layers"
        case localDim = "local_dim"
        case localHeads = "local_attn_heads"
        case localIntermediateSize = "local_ffn_dim"
        case localRopeTheta = "local_rotary_base"
        case endOfTurnId = "eot_idx"
        case startOfSpeechId = "sosp_idx"
        case endOfSpeechId = "eosp_idx"
        case endOfTextSpeechId = "eostm_idx"
        case emptyId = "empty_idx"
    }
    public init() {}
    public init(from decoder: Decoder) throws {
        let root = try decoder.container(keyedBy: CodingKeys.self)
        let nested = try root.nestedContainerIfPresent(
            keyedBy: CodingKeys.self, forKey: .audioConfig)
        func int(_ key: CodingKeys, _ fallback: Int) throws -> Int {
            try root.decodeIfPresent(Int.self, forKey: key) ?? nested?.decodeIfPresent(
                Int.self, forKey: key) ?? fallback
        }
        func ints(_ key: CodingKeys, _ fallback: [Int]) throws -> [Int] {
            let c = root.contains(key) ? root : (nested ?? root)
            if let array = try? c.decode([Int].self, forKey: key) { return array }
            if let string = try? c.decode(String.self, forKey: key) {
                return string.split(separator: "-").compactMap { Int($0) }
            }
            if let scalar = try c.decodeIfPresent(Int.self, forKey: key) {
                return Array(repeating: scalar, count: audioChannels)
            }
            return fallback
        }
        modelType = try root.decodeIfPresent(String.self, forKey: .modelType) ?? modelType
        if root.contains(.hiddenSize) { languageModel = try Qwen2Configuration(from: decoder) }
        audioChannels = try int(.audioChannels, audioChannels)
        groupSize = try int(.groupSize, groupSize)
        delayPattern = try ints(.delayPattern, Array(0 ..< audioChannels))
        speechVocabSizes = try ints(.speechVocabSizes, speechVocabSizes)
        speechEmptyIds = try ints(.speechEmptyIds, speechEmptyIds)
        inputLocalLayers = try int(.inputLocalLayers, inputLocalLayers)
        inputLocalDim = try int(.inputLocalDim, inputLocalDim)
        inputLocalHeads = try int(.inputLocalHeads, inputLocalHeads)
        inputLocalIntermediateSize = try int(
            .inputLocalIntermediateSize, inputLocalIntermediateSize)
        inputFullAttention =
            try root.decodeIfPresent(Bool.self, forKey: .inputFullAttention) ?? nested?
            .decodeIfPresent(Bool.self, forKey: .inputFullAttention) ?? true
        localLayers = try int(.localLayers, localLayers)
        localDim = try int(.localDim, localDim)
        localHeads = try int(.localHeads, localHeads)
        localIntermediateSize = try int(.localIntermediateSize, localIntermediateSize)
        localRopeTheta =
            try root.decodeIfPresent(Float.self, forKey: .localRopeTheta) ?? localRopeTheta
        endOfTurnId = try int(.endOfTurnId, endOfTurnId)
        startOfSpeechId = try int(.startOfSpeechId, startOfSpeechId)
        endOfSpeechId = try int(.endOfSpeechId, endOfSpeechId)
        endOfTextSpeechId = try int(.endOfTextSpeechId, endOfTextSpeechId)
        emptyId = try int(.emptyId, emptyId)
        perLayerQuantization = (try? BaseConfiguration(from: decoder))?.perLayerQuantization
    }
    public func encode(to encoder: Encoder) throws {
        try languageModel.encode(to: encoder)
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(modelType, forKey: .modelType)
        try c.encode(audioChannels, forKey: .audioChannels)
        try c.encode(groupSize, forKey: .groupSize)
        try c.encode(delayPattern, forKey: .delayPattern)
        try c.encode(
            speechVocabSizes.map(String.init).joined(separator: "-"), forKey: .speechVocabSizes)
        try c.encode(
            speechEmptyIds.map(String.init).joined(separator: "-"), forKey: .speechEmptyIds)
        try c.encode(inputLocalLayers, forKey: .inputLocalLayers)
        try c.encode(inputLocalDim, forKey: .inputLocalDim)
        try c.encode(inputLocalHeads, forKey: .inputLocalHeads)
        try c.encode(inputLocalIntermediateSize, forKey: .inputLocalIntermediateSize)
        try c.encode(inputFullAttention, forKey: .inputFullAttention)
        try c.encode(localLayers, forKey: .localLayers)
        try c.encode(localDim, forKey: .localDim)
        try c.encode(localHeads, forKey: .localHeads)
        try c.encode(localIntermediateSize, forKey: .localIntermediateSize)
        try c.encode(localRopeTheta, forKey: .localRopeTheta)
        try c.encode(endOfTurnId, forKey: .endOfTurnId)
        try c.encode(startOfSpeechId, forKey: .startOfSpeechId)
        try c.encode(endOfSpeechId, forKey: .endOfSpeechId)
        try c.encode(endOfTextSpeechId, forKey: .endOfTextSpeechId)
        try c.encode(emptyId, forKey: .emptyId)
    }
    func localConfiguration(input: Bool) -> Qwen2Configuration {
        .init(
            hiddenSize: input ? inputLocalDim : localDim,
            hiddenLayers: input ? inputLocalLayers : localLayers,
            intermediateSize: input ? inputLocalIntermediateSize : localIntermediateSize,
            attentionHeads: input ? inputLocalHeads : localHeads,
            rmsNormEps: languageModel.rmsNormEps,
            vocabularySize: languageModel.vocabularySize,
            kvHeads: input ? inputLocalHeads : localHeads,
            ropeTheta: localRopeTheta)
    }
    func validate() throws {
        let d = languageModel
        guard audioChannels > 0, groupSize > 0, speechVocabSizes.count == audioChannels,
            speechEmptyIds.count == audioChannels, delayPattern.count == audioChannels,
            zip(speechEmptyIds, speechVocabSizes).allSatisfy({ $0 >= 0 && $0 < $1 }),
            delayPattern.allSatisfy({ $0 >= 0 }), inputLocalLayers > 0, localLayers > 0,
            inputLocalDim > 0, localDim > 0, inputLocalHeads > 0, localHeads > 0,
            inputLocalDim % inputLocalHeads == 0, localDim % localHeads == 0,
            (inputLocalDim / inputLocalHeads) % 2 == 0, (localDim / localHeads) % 2 == 0,
            d.hiddenSize > 0, d.hiddenLayers > 0, d.attentionHeads > 0, d.kvHeads > 0,
            d.hiddenSize % d.attentionHeads == 0, d.attentionHeads % d.kvHeads == 0,
            [emptyId, endOfTurnId, startOfSpeechId, endOfSpeechId, endOfTextSpeechId].allSatisfy({
                (0 ..< d.vocabularySize).contains($0)
            })
        else { throw STTError.modelNotInitialized("Inconsistent MiMo ASR configuration") }
    }
}

extension KeyedDecodingContainer {
    fileprivate func nestedContainerIfPresent<NestedKey: CodingKey>(
        keyedBy type: NestedKey.Type, forKey key: Key
    ) throws -> KeyedDecodingContainer<NestedKey>? {
        contains(key) ? try nestedContainer(keyedBy: type, forKey: key) : nil
    }
}
