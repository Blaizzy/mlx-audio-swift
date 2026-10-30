import Foundation

public struct HiggsAudio3TTSConfig: Decodable, Sendable {
    public var textConfig: MossQwen3Config
    public var audioNumCodebooks: Int
    public var audioCodebookSize: Int
    public var audioBOCTokenID: Int
    public var audioEOCTokenID: Int
    public var useDelayPattern: Bool
    public var sampleRate: Int

    public init(
        textConfig: MossQwen3Config, audioNumCodebooks: Int = 8,
        audioCodebookSize: Int = 1026, audioBOCTokenID: Int = 1024,
        audioEOCTokenID: Int = 1025, useDelayPattern: Bool = true, sampleRate: Int = 24000
    ) {
        self.textConfig = textConfig
        self.audioNumCodebooks = audioNumCodebooks
        self.audioCodebookSize = audioCodebookSize
        self.audioBOCTokenID = audioBOCTokenID
        self.audioEOCTokenID = audioEOCTokenID
        self.useDelayPattern = useDelayPattern
        self.sampleRate = sampleRate
    }

    enum CodingKeys: String, CodingKey {
        case textConfig = "text_config", audioEncoderConfig = "audio_encoder_config"
        case audioNumCodebooks = "audio_num_codebooks", audioCodebookSize = "audio_codebook_size"
        case audioBOCTokenID = "audio_boc_token_id", audioEOCTokenID = "audio_eoc_token_id"
        case useDelayPattern = "use_delay_pattern", sampleRate = "sample_rate"
    }
    enum AudioKeys: String, CodingKey {
        case numCodebooks = "num_codebooks", vocabSize = "vocab_size", useDelayPattern = "use_delay_pattern"
    }
    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let a = try c.contains(.audioEncoderConfig) ? c.nestedContainer(keyedBy: AudioKeys.self, forKey: .audioEncoderConfig) : nil
        textConfig = try c.decode(MossQwen3Config.self, forKey: .textConfig)
        audioNumCodebooks = try c.decodeIfPresent(Int.self, forKey: .audioNumCodebooks)
            ?? a?.decodeIfPresent(Int.self, forKey: .numCodebooks) ?? 8
        audioCodebookSize = try c.decodeIfPresent(Int.self, forKey: .audioCodebookSize)
            ?? a?.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 1026
        audioBOCTokenID = try c.decodeIfPresent(Int.self, forKey: .audioBOCTokenID) ?? audioCodebookSize - 2
        audioEOCTokenID = try c.decodeIfPresent(Int.self, forKey: .audioEOCTokenID) ?? audioCodebookSize - 1
        useDelayPattern = try c.decodeIfPresent(Bool.self, forKey: .useDelayPattern)
            ?? a?.decodeIfPresent(Bool.self, forKey: .useDelayPattern) ?? true
        sampleRate = try c.decodeIfPresent(Int.self, forKey: .sampleRate) ?? 24000
    }
}
