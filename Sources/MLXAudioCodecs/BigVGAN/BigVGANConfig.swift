import Foundation

public enum BigVGANResBlockType: String, Codable, Sendable {
    case one = "1"
    case two = "2"
}

public enum BigVGANActivationType: String, Codable, Sendable {
    case snake
    case snakebeta
}

public struct BigVGANConfig: Codable, Sendable {
    public let numMels: Int
    public let upsampleRates: [Int]
    public let upsampleKernelSizes: [Int]
    public let upsampleInitialChannel: Int
    public let resblock: BigVGANResBlockType
    public let resblockKernelSizes: [Int]
    public let resblockDilationSizes: [[Int]]
    public let activation: BigVGANActivationType
    public let snakeLogscale: Bool
    public let useBiasAtFinal: Bool
    public let useTanhAtFinal: Bool

    public init(
        numMels: Int,
        upsampleRates: [Int],
        upsampleKernelSizes: [Int],
        upsampleInitialChannel: Int,
        resblock: BigVGANResBlockType,
        resblockKernelSizes: [Int],
        resblockDilationSizes: [[Int]],
        activation: BigVGANActivationType,
        snakeLogscale: Bool,
        useBiasAtFinal: Bool = true,
        useTanhAtFinal: Bool = true
    ) {
        self.numMels = numMels
        self.upsampleRates = upsampleRates
        self.upsampleKernelSizes = upsampleKernelSizes
        self.upsampleInitialChannel = upsampleInitialChannel
        self.resblock = resblock
        self.resblockKernelSizes = resblockKernelSizes
        self.resblockDilationSizes = resblockDilationSizes
        self.activation = activation
        self.snakeLogscale = snakeLogscale
        self.useBiasAtFinal = useBiasAtFinal
        self.useTanhAtFinal = useTanhAtFinal
    }

    enum CodingKeys: String, CodingKey {
        case numMels = "num_mels"
        case upsampleRates = "upsample_rates"
        case upsampleKernelSizes = "upsample_kernel_sizes"
        case upsampleInitialChannel = "upsample_initial_channel"
        case resblock
        case resblockKernelSizes = "resblock_kernel_sizes"
        case resblockDilationSizes = "resblock_dilation_sizes"
        case activation
        case snakeLogscale = "snake_logscale"
        case useBiasAtFinal = "use_bias_at_final"
        case useTanhAtFinal = "use_tanh_at_final"
    }
}
