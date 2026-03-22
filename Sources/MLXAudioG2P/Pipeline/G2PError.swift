public enum G2PError: Error, Sendable, Equatable {
    case emptyInput
    case unsupportedLocale(String)
    case phonemizationFailed(token: String, reason: String)
    case alignmentFailed(reason: String)
    case resourceLoadFailed(name: String, reason: String)
}
