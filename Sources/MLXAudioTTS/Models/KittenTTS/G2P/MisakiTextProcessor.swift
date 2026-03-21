import Foundation

/// Built-in TextProcessor using MisakiSwift G2P for English phonemization.
/// Converts plain English text to IPA phonemes suitable for KittenTTS/Kokoro.
public final class MisakiTextProcessor: TextProcessor, @unchecked Sendable {
    private let g2p: EnglishG2P

    public init(british: Bool = false) {
        g2p = EnglishG2P(british: british)
    }

    public func process(text: String, language: String?) throws -> String {
        let (phonemes, _) = g2p.phonemize(text: text)
        return phonemes
    }
}
