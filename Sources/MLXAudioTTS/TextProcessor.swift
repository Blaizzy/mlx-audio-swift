import Foundation

/// Protocol for text preprocessing before speech synthesis.
///
/// Some TTS models (like Kokoro, KittenTTS) require phonemized IPA input rather than raw text.
/// Implement this protocol to convert natural language text into the format your
/// target model expects.
///
/// Example: A Misaki G2P adapter:
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
    /// Download or initialize any resources needed before processing.
    ///
    /// Call this before `process(text:language:)` to ensure the processor is ready.
    /// The default implementation is a no-op for processors that don't need preparation.
    func prepare() async throws

    /// Convert input text into the format expected by the target model.
    ///
    /// - Parameters:
    ///   - text: The input text in natural language.
    ///   - language: Optional language code (e.g., "en-us", "en-gb").
    /// - Returns: Processed text string suitable for the target model.
    func process(text: String, language: String?) throws -> String
}

extension TextProcessor {
    /// Default no-op implementation for processors that don't need preparation.
    public func prepare() async throws {}
}
