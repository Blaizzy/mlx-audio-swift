import Foundation

public struct DeepFilterNetConfig: Codable, Sendable {
    public var sampleRate: Int = 48_000
    public var fftSize: Int = 960
    public var hopSize: Int = 480

    public var nbErb: Int = 32
    public var nbDf: Int = 96
    public var dfOrder: Int = 5
    public var dfLookahead: Int = 2
    public var convLookahead: Int = 2

    public var convCh: Int = 64
    public var embHiddenDim: Int = 256
    public var embNumLayers: Int = 3
    public var dfHiddenDim: Int = 256
    public var dfNumLayers: Int = 2
    public var linearGroups: Int = 16
    public var encLinearGroups: Int = 32
    public var encConcat: Bool = false
    public var dfGruSkip: String = "groupedlinear"

    public var convKernel: [Int] = [1, 3]
    public var convtKernel: [Int] = [1, 3]
    public var convKernelInp: [Int] = [3, 3]
    public var dfPathwayKernelSizeT: Int = 5

    public var lsnrMax: Int = 35
    public var lsnrMin: Int = -15
    public var modelVersion: String = "DeepFilterNet3"

    public var freqBins: Int { fftSize / 2 + 1 }

    public var modelType: String {
        switch modelVersion.lowercased() {
        case "deepfilternet":
            return "deepfilternet"
        case "deepfilternet2":
            return "deepfilternet2"
        default:
            return "deepfilternet3"
        }
    }

    enum CodingKeys: String, CodingKey {
        case sampleRate
        case fftSize
        case hopSize
        case nbErb
        case nbDf
        case dfOrder
        case dfLookahead
        case convLookahead
        case convCh
        case embHiddenDim
        case embNumLayers
        case dfHiddenDim
        case dfNumLayers
        case linearGroups
        case encLinearGroups
        case encConcat
        case dfGruSkip
        case convKernel
        case convtKernel
        case convKernelInp
        case dfPathwayKernelSizeT
        case lsnrMax
        case lsnrMin
        case modelVersion
    }

    public init() {}

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        sampleRate = try c.decodeIfPresent(Int.self, forKey: .sampleRate) ?? sampleRate
        fftSize = try c.decodeIfPresent(Int.self, forKey: .fftSize) ?? fftSize
        hopSize = try c.decodeIfPresent(Int.self, forKey: .hopSize) ?? hopSize
        nbErb = try c.decodeIfPresent(Int.self, forKey: .nbErb) ?? nbErb
        nbDf = try c.decodeIfPresent(Int.self, forKey: .nbDf) ?? nbDf
        dfOrder = try c.decodeIfPresent(Int.self, forKey: .dfOrder) ?? dfOrder
        dfLookahead = try c.decodeIfPresent(Int.self, forKey: .dfLookahead) ?? dfLookahead
        convLookahead = try c.decodeIfPresent(Int.self, forKey: .convLookahead) ?? convLookahead
        convCh = try c.decodeIfPresent(Int.self, forKey: .convCh) ?? convCh
        embHiddenDim = try c.decodeIfPresent(Int.self, forKey: .embHiddenDim) ?? embHiddenDim
        embNumLayers = try c.decodeIfPresent(Int.self, forKey: .embNumLayers) ?? embNumLayers
        dfHiddenDim = try c.decodeIfPresent(Int.self, forKey: .dfHiddenDim) ?? dfHiddenDim
        dfNumLayers = try c.decodeIfPresent(Int.self, forKey: .dfNumLayers) ?? dfNumLayers
        linearGroups = try c.decodeIfPresent(Int.self, forKey: .linearGroups) ?? linearGroups
        encLinearGroups = try c.decodeIfPresent(Int.self, forKey: .encLinearGroups) ?? encLinearGroups
        encConcat = try c.decodeIfPresent(Bool.self, forKey: .encConcat) ?? encConcat
        dfGruSkip = try c.decodeIfPresent(String.self, forKey: .dfGruSkip) ?? dfGruSkip
        convKernel = try c.decodeIfPresent([Int].self, forKey: .convKernel) ?? convKernel
        convtKernel = try c.decodeIfPresent([Int].self, forKey: .convtKernel) ?? convtKernel
        convKernelInp = try c.decodeIfPresent([Int].self, forKey: .convKernelInp) ?? convKernelInp
        dfPathwayKernelSizeT =
            try c.decodeIfPresent(Int.self, forKey: .dfPathwayKernelSizeT) ?? dfPathwayKernelSizeT
        lsnrMax = try c.decodeIfPresent(Int.self, forKey: .lsnrMax) ?? lsnrMax
        lsnrMin = try c.decodeIfPresent(Int.self, forKey: .lsnrMin) ?? lsnrMin
        modelVersion = try c.decodeIfPresent(String.self, forKey: .modelVersion) ?? modelVersion
    }
}
