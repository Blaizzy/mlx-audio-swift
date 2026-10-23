import Foundation

/// Configuration of the 22.05 kHz codec embedded in NemotronLabs VoiceChat.
public struct NemotronVoiceChatCodecConfig: Codable, Sendable, Equatable {
    public var sampleRate: Int
    public var baseChannels: Int
    public var channelMultipliers: [Int]
    public var downsampleRates: [Int]
    public var blocksPerStage: Int
    public var blockKernelSize: Int
    public var latentDim: Int
    public var nFFT: Int
    public var hopLength: Int
    public var numQuantizers: Int
    public var codebookSize: Int

    public init(
        sampleRate: Int = 22_050, baseChannels: Int = 384,
        channelMultipliers: [Int] = [1, 2, 4], downsampleRates: [Int] = [7, 7, 9],
        blocksPerStage: Int = 3, blockKernelSize: Int = 7, latentDim: Int = 512,
        nFFT: Int = 16, hopLength: Int = 4, numQuantizers: Int = 31,
        codebookSize: Int = 1_024
    ) {
        self.sampleRate = sampleRate
        self.baseChannels = baseChannels
        self.channelMultipliers = channelMultipliers
        self.downsampleRates = downsampleRates
        self.blocksPerStage = blocksPerStage
        self.blockKernelSize = blockKernelSize
        self.latentDim = latentDim
        self.nFFT = nFFT
        self.hopLength = hopLength
        self.numQuantizers = numQuantizers
        self.codebookSize = codebookSize
    }

    public var waveformToTokenRatio: Int { downsampleRates.reduce(hopLength, *) }
    public var frameRate: Double { Double(sampleRate) / Double(waveformToTokenRatio) }
    public var stftChannels: Int { 2 * (nFFT / 2 + 1) }

    enum CodingKeys: String, CodingKey {
        case sampleRate = "sample_rate", baseChannels = "base_channels"
        case channelMultipliers = "channel_multipliers", downsampleRates = "downsample_rates"
        case blocksPerStage = "blocks_per_stage", blockKernelSize = "block_kernel_size"
        case latentDim = "latent_dim", nFFT = "n_fft", hopLength = "hop_length"
        case numQuantizers = "num_quantizers", codebookSize = "codebook_size"
    }

    public init(from decoder: any Swift.Decoder) throws {
        self.init()
        let c = try decoder.container(keyedBy: CodingKeys.self)
        sampleRate = try c.decodeIfPresent(Int.self, forKey: .sampleRate) ?? sampleRate
        baseChannels = try c.decodeIfPresent(Int.self, forKey: .baseChannels) ?? baseChannels
        channelMultipliers = try c.decodeIfPresent([Int].self, forKey: .channelMultipliers) ?? channelMultipliers
        downsampleRates = try c.decodeIfPresent([Int].self, forKey: .downsampleRates) ?? downsampleRates
        blocksPerStage = try c.decodeIfPresent(Int.self, forKey: .blocksPerStage) ?? blocksPerStage
        blockKernelSize = try c.decodeIfPresent(Int.self, forKey: .blockKernelSize) ?? blockKernelSize
        latentDim = try c.decodeIfPresent(Int.self, forKey: .latentDim) ?? latentDim
        nFFT = try c.decodeIfPresent(Int.self, forKey: .nFFT) ?? nFFT
        hopLength = try c.decodeIfPresent(Int.self, forKey: .hopLength) ?? hopLength
        numQuantizers = try c.decodeIfPresent(Int.self, forKey: .numQuantizers) ?? numQuantizers
        codebookSize = try c.decodeIfPresent(Int.self, forKey: .codebookSize) ?? codebookSize
        try validate()
    }

    func validate() throws {
        guard sampleRate > 0, baseChannels > 0, latentDim > 0,
              !channelMultipliers.isEmpty, channelMultipliers.count == downsampleRates.count,
              channelMultipliers.allSatisfy({ $0 > 0 }), downsampleRates.allSatisfy({ $0 > 0 }),
              blocksPerStage >= 0, blockKernelSize > 0, nFFT >= 2, nFFT % 2 == 0,
              hopLength > 0, hopLength < nFFT, numQuantizers > 0, codebookSize > 0
        else { throw NemotronVoiceChatCodecError.invalidConfiguration }
        var ratio = hopLength
        for rate in downsampleRates {
            let product = ratio.multipliedReportingOverflow(by: rate)
            guard !product.overflow else { throw NemotronVoiceChatCodecError.invalidConfiguration }
            ratio = product.partialValue
        }
    }
}

public enum NemotronVoiceChatCodecError: LocalizedError {
    case invalidConfiguration
    case invalidArtifact(String)
    case invalidWeights(String)
    case invalidRangeResponse

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration: "Invalid Nemotron VoiceChat codec configuration."
        case .invalidArtifact(let message): "Invalid Nemotron VoiceChat codec artifact: \(message)"
        case .invalidWeights(let message): "Invalid Nemotron VoiceChat codec weights: \(message)"
        case .invalidRangeResponse: "The Hub did not return the requested codec tensor byte range."
        }
    }
}
