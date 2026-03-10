import Foundation

public struct EcapaTdnnConfig: Codable, Sendable {
    private static let defaultKernelSizes = [5, 3, 3, 3, 1]
    private static let defaultDilations = [1, 2, 3, 4, 1]

    public var inputSize: Int
    public var channels: Int
    public var embedDim: Int
    public var kernelSizes: [Int]
    public var dilations: [Int]
    public var attentionChannels: Int
    public var res2netScale: Int
    public var seChannels: Int
    public var globalContext: Bool

    public init(
        inputSize: Int = 60,
        channels: Int = 1024,
        embedDim: Int = 256,
        kernelSizes: [Int] = [5, 3, 3, 3, 1],
        dilations: [Int] = [1, 2, 3, 4, 1],
        attentionChannels: Int = 128,
        res2netScale: Int = 8,
        seChannels: Int = 128,
        globalContext: Bool = false
    ) {
        self.inputSize = inputSize
        self.channels = channels
        self.embedDim = embedDim
        self.kernelSizes = Self.normalized(kernelSizes, fallback: Self.defaultKernelSizes)
        self.dilations = Self.normalized(dilations, fallback: Self.defaultDilations)
        self.attentionChannels = attentionChannels
        self.res2netScale = res2netScale
        self.seChannels = seChannels
        self.globalContext = globalContext
    }

    enum CodingKeys: String, CodingKey {
        case inputSize
        case channels
        case embedDim
        case kernelSizes
        case dilations
        case attentionChannels
        case res2netScale
        case seChannels
        case globalContext
    }

    public init(from decoder: any Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            inputSize: container.decodeIfPresent(Int.self, forKey: .inputSize) ?? 60,
            channels: container.decodeIfPresent(Int.self, forKey: .channels) ?? 1024,
            embedDim: container.decodeIfPresent(Int.self, forKey: .embedDim) ?? 256,
            kernelSizes: container.decodeIfPresent([Int].self, forKey: .kernelSizes)
                ?? Self.defaultKernelSizes,
            dilations: container.decodeIfPresent([Int].self, forKey: .dilations)
                ?? Self.defaultDilations,
            attentionChannels: container.decodeIfPresent(Int.self, forKey: .attentionChannels) ?? 128,
            res2netScale: container.decodeIfPresent(Int.self, forKey: .res2netScale) ?? 8,
            seChannels: container.decodeIfPresent(Int.self, forKey: .seChannels) ?? 128,
            globalContext: container.decodeIfPresent(Bool.self, forKey: .globalContext) ?? false
        )
    }

    private static func normalized(_ values: [Int], fallback: [Int]) -> [Int] {
        var normalized = fallback
        for (index, value) in values.prefix(fallback.count).enumerated() {
            normalized[index] = value
        }
        return normalized
    }
}
