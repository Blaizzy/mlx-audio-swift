import Foundation
import NaturalLanguage

extension NLTag {
    var isProperNoun: Bool {
        self == .personalName || self == .organizationName || self == .placeName
    }
}

extension Range where Bound: Comparable {
    func containsRange(_ other: Range<Bound>) -> Bool {
        lowerBound <= other.lowerBound && upperBound >= other.upperBound
    }
}

extension String {
    func replacingLastOccurrence(of target: Character, with replacement: Character) -> String {
        guard let lastIndex = self.lastIndex(of: target) else { return self }
        var result = self
        result.replaceSubrange(lastIndex...lastIndex, with: String(replacement))
        return result
    }
}
