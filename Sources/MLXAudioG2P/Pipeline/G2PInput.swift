public struct G2PInput: Sendable {
    public let text: String
    public let locale: String
    public let includeAlignment: Bool

    public init(text: String, locale: String = "en-US", includeAlignment: Bool = false) {
        self.text = text
        self.locale = locale
        self.includeAlignment = includeAlignment
    }
}
