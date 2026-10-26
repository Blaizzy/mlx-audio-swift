import Foundation

public struct DescriptDACConfig: Codable, Sendable {
    public let encoderDim: Int
    public let encoderRates: [Int]
    public let latentDim: Int?
    public let decoderDim: Int
    public let decoderRates: [Int]
    public let nCodebooks: Int
    public let codebookSize: Int
    public let codebookDim: Int
    public let sampleRate: Int

    public init(
        encoderDim: Int = 64,
        encoderRates: [Int] = [2, 4, 5, 8],
        latentDim: Int? = nil,
        decoderDim: Int = 1536,
        decoderRates: [Int] = [8, 5, 4, 2],
        nCodebooks: Int = 12,
        codebookSize: Int = 1024,
        codebookDim: Int = 8,
        sampleRate: Int = 16_000
    ) {
        self.encoderDim = encoderDim
        self.encoderRates = encoderRates
        self.latentDim = latentDim
        self.decoderDim = decoderDim
        self.decoderRates = decoderRates
        self.nCodebooks = nCodebooks
        self.codebookSize = codebookSize
        self.codebookDim = codebookDim
        self.sampleRate = sampleRate
    }

    enum CodingKeys: String, CodingKey {
        case encoderDim = "encoder_dim"
        case encoderRates = "encoder_rates"
        case latentDim = "latent_dim"
        case decoderDim = "decoder_dim"
        case decoderRates = "decoder_rates"
        case nCodebooks = "n_codebooks"
        case codebookSize = "codebook_size"
        case codebookDim = "codebook_dim"
        case sampleRate = "sample_rate"
    }

    private enum PublishedKeys: String, CodingKey {
        case encoderDim = "encoder_hidden_size"
        case encoderRates = "downsampling_ratios"
        case latentDim = "hidden_size"
        case decoderDim = "decoder_hidden_size"
        case decoderRates = "upsampling_ratios"
        case sampleRate = "sampling_rate"
    }

    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let p = try decoder.container(keyedBy: PublishedKeys.self)
        encoderDim = try c.decodeIfPresent(Int.self, forKey: .encoderDim)
            ?? p.decode(Int.self, forKey: .encoderDim)
        encoderRates = try c.decodeIfPresent([Int].self, forKey: .encoderRates)
            ?? p.decode([Int].self, forKey: .encoderRates)
        latentDim = try c.decodeIfPresent(Int.self, forKey: .latentDim)
            ?? p.decodeIfPresent(Int.self, forKey: .latentDim)
        decoderDim = try c.decodeIfPresent(Int.self, forKey: .decoderDim)
            ?? p.decode(Int.self, forKey: .decoderDim)
        decoderRates = try c.decodeIfPresent([Int].self, forKey: .decoderRates)
            ?? p.decode([Int].self, forKey: .decoderRates)
        sampleRate = try c.decodeIfPresent(Int.self, forKey: .sampleRate)
            ?? p.decode(Int.self, forKey: .sampleRate)
        nCodebooks = try c.decode(Int.self, forKey: .nCodebooks)
        codebookSize = try c.decode(Int.self, forKey: .codebookSize)
        codebookDim = try c.decode(Int.self, forKey: .codebookDim)
    }
}
