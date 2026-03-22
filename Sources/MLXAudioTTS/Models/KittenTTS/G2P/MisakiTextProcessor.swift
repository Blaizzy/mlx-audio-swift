import Foundation

public final class MisakiTextProcessor: TextProcessor, @unchecked Sendable {
    private var usG2P: EnglishG2P?
    private var gbG2P: EnglishG2P?
    private let lock = NSLock()

    public init() {}

    public func process(text: String, language: String?) throws -> String {
        let british = language?.lowercased().contains("gb") == true
        let g2p = getG2P(british: british)
        let (phonemes, _) = g2p.phonemize(text: text)
        return phonemes
    }

    private func getG2P(british: Bool) -> EnglishG2P {
        lock.lock()
        defer { lock.unlock() }
        if british {
            if let cached = gbG2P { return cached }
            let g2p = EnglishG2P(british: true)
            gbG2P = g2p
            return g2p
        } else {
            if let cached = usG2P { return cached }
            let g2p = EnglishG2P(british: false)
            usG2P = g2p
            return g2p
        }
    }
}
