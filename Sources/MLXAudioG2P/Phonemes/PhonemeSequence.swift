public struct PhonemeSequence: Sendable, Hashable {
    public let units: [PhonemeUnit]

    public init(units: [PhonemeUnit]) {
        self.units = units
    }

    public var isEmpty: Bool { units.isEmpty }

    public func render(separator: String = " ") -> String {
        units.map(\.symbol).joined(separator: separator)
    }
}
