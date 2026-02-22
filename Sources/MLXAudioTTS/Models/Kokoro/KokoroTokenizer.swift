import Foundation

/// Tokenizes phonemized text using the Kokoro vocabulary.
enum KokoroTokenizer {
    /// Tokenize phonemized text into integer IDs using the vocabulary.
    static func tokenize(phonemizedText text: String, vocab: [String: Int]) -> [Int] {
        text.compactMap { vocab[String($0)] }
    }
}
