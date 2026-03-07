import Foundation

public struct EcapaTdnnConfig: Codable, Sendable {
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
        self.kernelSizes = kernelSizes
        self.dilations = dilations
        self.attentionChannels = attentionChannels
        self.res2netScale = res2netScale
        self.seChannels = seChannels
        self.globalContext = globalContext
    }
}
