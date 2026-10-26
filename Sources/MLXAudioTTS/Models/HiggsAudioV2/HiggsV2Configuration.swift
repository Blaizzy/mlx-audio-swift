import Foundation
import MLXAudioCore

public struct HiggsV2Configuration: Decodable, Sendable {
    public var textConfig: LlamaTTSConfiguration
    public var audioNumCodebooks: Int
    public var audioCodebookSize: Int
    public var audioStreamBOS: Int
    public var audioStreamEOS: Int
    public var quantization: BaseConfiguration.Quantization?

    public init(textConfig: LlamaTTSConfiguration, audioNumCodebooks: Int = 8, audioCodebookSize: Int = 1024) {
        self.textConfig = textConfig
        self.audioNumCodebooks = audioNumCodebooks
        self.audioCodebookSize = audioCodebookSize
        audioStreamBOS = audioCodebookSize
        audioStreamEOS = audioCodebookSize + 1
    }
    enum CodingKeys: String, CodingKey {
        case textConfig = "text_config", audioNumCodebooks = "audio_num_codebooks"
        case audioCodebookSize = "audio_codebook_size", audioStreamBOS = "audio_stream_bos_id"
        case audioStreamEOS = "audio_stream_eos_id", quantization
        case audioDualFFNLayers = "audio_dual_ffn_layers", useAudioAttention = "use_audio_out_self_attention"
        case projectorLayers = "audio_decoder_proj_num_layers", delayPattern = "use_delay_pattern"
    }
    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        textConfig = try c.decode(LlamaTTSConfiguration.self, forKey: .textConfig)
        audioNumCodebooks = try c.decodeIfPresent(Int.self, forKey: .audioNumCodebooks) ?? 8
        audioCodebookSize = try c.decodeIfPresent(Int.self, forKey: .audioCodebookSize) ?? 1024
        audioStreamBOS = try c.decodeIfPresent(Int.self, forKey: .audioStreamBOS) ?? audioCodebookSize
        audioStreamEOS = try c.decodeIfPresent(Int.self, forKey: .audioStreamEOS) ?? audioCodebookSize + 1
        quantization = try c.decodeIfPresent(BaseConfiguration.Quantization.self, forKey: .quantization)
        let audioAttention = (try? c.decode(Bool.self, forKey: .useAudioAttention))
            ?? ((try? c.decode(Int.self, forKey: .useAudioAttention)) ?? 0 != 0)
        let dualLayers = try c.decodeIfPresent([Int].self, forKey: .audioDualFFNLayers)
            ?? Array(0..<textConfig.hiddenLayers)
        guard !audioAttention, (try c.decodeIfPresent(Int.self, forKey: .projectorLayers) ?? 0) == 0,
              (try c.decodeIfPresent(Bool.self, forKey: .delayPattern) ?? true),
              dualLayers == Array(0..<textConfig.hiddenLayers), audioNumCodebooks > 0,
              audioCodebookSize > 0, audioStreamBOS == audioCodebookSize,
              audioStreamEOS == audioCodebookSize + 1 else {
            throw DecodingError.dataCorrupted(.init(codingPath: decoder.codingPath,
                debugDescription: "Unsupported Higgs v2 architecture or delay-pattern configuration"))
        }
    }
}

public struct HiggsV2ConversationTurn: Sendable {
    public var role: String
    public var text: String
    /// Optional aligned audio codebooks [codebook][frame].
    public var audioCodes: [[Int32]]?
    public init(role: String, text: String = "", audioCodes: [[Int32]]? = nil) {
        self.role = role; self.text = text; self.audioCodes = audioCodes
    }
}

public struct HiggsV2GenerationOptions: Sendable {
    public var systemPrompt: String? = nil
    public var conversation: [HiggsV2ConversationTurn] = []
    public var repetitionWindow: Int = 7
    public var maximumRepeat: Int = 2
    public var samplingWarmupFrames: Int = 0
    public var fadeInMilliseconds: Float = 30
    public var fadeOutMilliseconds: Float = 15
    public var overlapMilliseconds: Float = 40
    public init() {}
}
