import Foundation
import MLX

// MARK: - STSModel Protocol

public protocol STSModel: AnyObject {
    var sampleRate: Int { get }
}

// MARK: - Loaded Model Container

public enum LoadedSTSModel {
    case samAudio(SAMAudio)
    case lfmAudio(LFM2AudioModel)
    case mossFormer2SE(MossFormer2SEModel)

    public var model: any STSModel {
        switch self {
        case .samAudio(let m): return m
        case .lfmAudio(let m): return m
        case .mossFormer2SE(let m): return m
        }
    }

    public var sampleRate: Int { model.sampleRate }
}
