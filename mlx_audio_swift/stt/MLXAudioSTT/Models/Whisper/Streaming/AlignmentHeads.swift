import Foundation

public enum WhisperModel: String, CaseIterable, Sendable {
    case tiny
    case base
    case small
    case medium
    case largeV3
    case largeTurbo
}

public enum WhisperAlignmentHeads {
    public static func heads(for model: WhisperModel) -> [(layer: Int, head: Int)] {
        switch model {
        case .tiny:
            return [(2, 2), (3, 0), (3, 2), (3, 3), (3, 4), (3, 5)]
        case .base:
            return [(3, 1), (4, 2), (4, 3), (4, 7), (5, 1), (5, 2), (5, 4), (5, 6)]
        case .small:
            return [(5, 3), (5, 9), (8, 0), (8, 4), (8, 7), (8, 8), (9, 0), (9, 7), (9, 9), (10, 5)]
        case .medium:
            return [(13, 15), (15, 4), (15, 15), (16, 1), (20, 0), (23, 4)]
        case .largeV3:
            return [
                (10, 12), (13, 17), (16, 11), (17, 3), (18, 11), (19, 9),
                (20, 1), (20, 8), (21, 0), (21, 4), (21, 8), (22, 3),
                (22, 5), (22, 7), (22, 10), (22, 12), (22, 16), (23, 0),
                (23, 2), (23, 4), (23, 8), (23, 10), (23, 13)
            ]
        case .largeTurbo:
            // Provisional values for 4-layer decoder
            return [(1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)]
        }
    }
}
