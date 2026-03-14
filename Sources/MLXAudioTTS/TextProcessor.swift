import Foundation

/// Protocol for text preprocessing before speech synthesis.
///
/// Some TTS models (like Kokoro) require phonemized IPA input rather than raw text.
/// Implement this protocol to convert natural language text into the format your
/// target model expects.
///
/// Example: A Misaki G2P adapter for Kokoro:
/// ```swift
/// struct MisakiTextProcessor: TextProcessor {
///     let g2p: EnglishG2P
///     func process(text: String, language: String?) throws -> String {
///         let (phonemes, _) = g2p.phonemize(text: text)
///         return phonemes
///     }
/// }
/// ```
public protocol TextProcessor: Sendable {
    /// Convert input text into the format expected by the target model.
    ///
    /// For G2P processors, this converts natural language text (e.g., "Hello world")
    /// into phonemized output (e.g., IPA notation like "həlˈoʊ wˈɜɹld").
    ///
    /// - Parameters:
    ///   - text: The input text in natural language.
    ///   - language: Optional language code (e.g., "en-us", "en-gb").
    /// - Returns: Processed text string suitable for the target model.
    func process(text: String, language: String?) throws -> String
}
