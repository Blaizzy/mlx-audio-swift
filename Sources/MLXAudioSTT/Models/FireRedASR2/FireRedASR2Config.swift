import Foundation

public struct FireRedASR2EncoderConfig: Codable, Sendable {
    public let nLayers: Int
    public let nHead: Int
    public let dModel: Int
    public let kernelSize: Int
    public let peMaxlen: Int

    public init(
        nLayers: Int = 16,
        nHead: Int = 20,
        dModel: Int = 1280,
        kernelSize: Int = 33,
        peMaxlen: Int = 5000
    ) {
        self.nLayers = nLayers
        self.nHead = nHead
        self.dModel = dModel
        self.kernelSize = kernelSize
        self.peMaxlen = peMaxlen
    }

    enum CodingKeys: String, CodingKey {
        case nLayers = "n_layers"
        case nHead = "n_head"
        case dModel = "d_model"
        case kernelSize = "kernel_size"
        case peMaxlen = "pe_maxlen"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        nLayers = try container.decodeIfPresent(Int.self, forKey: .nLayers) ?? 16
        nHead = try container.decodeIfPresent(Int.self, forKey: .nHead) ?? 20
        dModel = try container.decodeIfPresent(Int.self, forKey: .dModel) ?? 1280
        kernelSize = try container.decodeIfPresent(Int.self, forKey: .kernelSize) ?? 33
        peMaxlen = try container.decodeIfPresent(Int.self, forKey: .peMaxlen) ?? 5000
    }
}

public struct FireRedASR2DecoderConfig: Codable, Sendable {
    public let nLayers: Int
    public let nHead: Int
    public let dModel: Int
    public let peMaxlen: Int

    public init(
        nLayers: Int = 16,
        nHead: Int = 20,
        dModel: Int = 1280,
        peMaxlen: Int = 5000
    ) {
        self.nLayers = nLayers
        self.nHead = nHead
        self.dModel = dModel
        self.peMaxlen = peMaxlen
    }

    enum CodingKeys: String, CodingKey {
        case nLayers = "n_layers"
        case nHead = "n_head"
        case dModel = "d_model"
        case peMaxlen = "pe_maxlen"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        nLayers = try container.decodeIfPresent(Int.self, forKey: .nLayers) ?? 16
        nHead = try container.decodeIfPresent(Int.self, forKey: .nHead) ?? 20
        dModel = try container.decodeIfPresent(Int.self, forKey: .dModel) ?? 1280
        peMaxlen = try container.decodeIfPresent(Int.self, forKey: .peMaxlen) ?? 5000
    }
}

public struct FireRedASR2Config: Codable, Sendable {
    public let modelType: String
    public let idim: Int
    public let odim: Int
    public let dModel: Int
    public let sosID: Int
    public let eosID: Int
    public let padID: Int
    public let blankID: Int
    public let encoder: FireRedASR2EncoderConfig
    public let decoder: FireRedASR2DecoderConfig

    public init(
        modelType: String = "fireredasr2",
        idim: Int = 80,
        odim: Int = 8667,
        dModel: Int = 1280,
        sosID: Int = 3,
        eosID: Int = 4,
        padID: Int = 2,
        blankID: Int = 0,
        encoder: FireRedASR2EncoderConfig = FireRedASR2EncoderConfig(),
        decoder: FireRedASR2DecoderConfig = FireRedASR2DecoderConfig()
    ) {
        self.modelType = modelType
        self.idim = idim
        self.odim = odim
        self.dModel = dModel
        self.sosID = sosID
        self.eosID = eosID
        self.padID = padID
        self.blankID = blankID
        self.encoder = encoder
        self.decoder = decoder
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case idim
        case odim
        case dModel = "d_model"
        case sosID = "sos_id"
        case eosID = "eos_id"
        case padID = "pad_id"
        case blankID = "blank_id"
        case encoder
        case decoder
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "fireredasr2"
        idim = try container.decodeIfPresent(Int.self, forKey: .idim) ?? 80
        odim = try container.decodeIfPresent(Int.self, forKey: .odim) ?? 8667
        dModel = try container.decodeIfPresent(Int.self, forKey: .dModel) ?? 1280
        sosID = try container.decodeIfPresent(Int.self, forKey: .sosID) ?? 3
        eosID = try container.decodeIfPresent(Int.self, forKey: .eosID) ?? 4
        padID = try container.decodeIfPresent(Int.self, forKey: .padID) ?? 2
        blankID = try container.decodeIfPresent(Int.self, forKey: .blankID) ?? 0
        encoder = try container.decodeIfPresent(FireRedASR2EncoderConfig.self, forKey: .encoder)
            ?? FireRedASR2EncoderConfig()
        self.decoder = try container.decodeIfPresent(FireRedASR2DecoderConfig.self, forKey: .decoder)
            ?? FireRedASR2DecoderConfig()
    }
}
