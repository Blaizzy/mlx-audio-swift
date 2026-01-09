import Foundation
import Tokenizers

/// Tokenizer wrapper for Whisper models
/// Provides encoding/decoding with Whisper-specific features like timestamps and language tokens
public final class WhisperTokenizer: @unchecked Sendable {

    // MARK: - Special Token Constants

    public static let eotToken: Int = 50257
    public static let sotToken: Int = 50258
    public static let translateToken: Int = 50358
    public static let transcribeToken: Int = 50359
    public static let sotPrevToken: Int = 50361
    public static let noSpeechToken: Int = 50362
    public static let noTimestampsToken: Int = 50363
    public static let timestampBegin: Int = 50364
    public static let languageTokenStart: Int = 50259
    public static let languageTokenEnd: Int = 50358

    /// Number of seconds per timestamp token (20ms resolution)
    public static let secondsPerTimestampToken: Double = 0.02

    // MARK: - Language Token Mapping

    /// ISO language codes to token offsets from languageTokenStart
    private static let languageOffsets: [String: Int] = [
        "en": 0,   // English
        "zh": 1,   // Chinese
        "de": 2,   // German
        "es": 3,   // Spanish
        "ru": 4,   // Russian
        "ko": 5,   // Korean
        "fr": 6,   // French
        "ja": 7,   // Japanese
        "pt": 8,   // Portuguese
        "tr": 9,   // Turkish
        "pl": 10,  // Polish
        "ca": 11,  // Catalan
        "nl": 12,  // Dutch
        "ar": 13,  // Arabic
        "sv": 14,  // Swedish
        "it": 15,  // Italian
        "id": 16,  // Indonesian
        "hi": 17,  // Hindi
        "fi": 18,  // Finnish
        "vi": 19,  // Vietnamese
        "he": 20,  // Hebrew
        "uk": 21,  // Ukrainian
        "el": 22,  // Greek
        "ms": 23,  // Malay
        "cs": 24,  // Czech
        "ro": 25,  // Romanian
        "da": 26,  // Danish
        "hu": 27,  // Hungarian
        "ta": 28,  // Tamil
        "no": 29,  // Norwegian
        "th": 30,  // Thai
        "ur": 31,  // Urdu
        "hr": 32,  // Croatian
        "bg": 33,  // Bulgarian
        "lt": 34,  // Lithuanian
        "la": 35,  // Latin
        "mi": 36,  // Maori
        "ml": 37,  // Malayalam
        "cy": 38,  // Welsh
        "sk": 39,  // Slovak
        "te": 40,  // Telugu
        "fa": 41,  // Persian
        "lv": 42,  // Latvian
        "bn": 43,  // Bengali
        "sr": 44,  // Serbian
        "az": 45,  // Azerbaijani
        "sl": 46,  // Slovenian
        "kn": 47,  // Kannada
        "et": 48,  // Estonian
        "mk": 49,  // Macedonian
        "br": 50,  // Breton
        "eu": 51,  // Basque
        "is": 52,  // Icelandic
        "hy": 53,  // Armenian
        "ne": 54,  // Nepali
        "mn": 55,  // Mongolian
        "bs": 56,  // Bosnian
        "kk": 57,  // Kazakh
        "sq": 58,  // Albanian
        "sw": 59,  // Swahili
        "gl": 60,  // Galician
        "mr": 61,  // Marathi
        "pa": 62,  // Punjabi
        "si": 63,  // Sinhala
        "km": 64,  // Khmer
        "sn": 65,  // Shona
        "yo": 66,  // Yoruba
        "so": 67,  // Somali
        "af": 68,  // Afrikaans
        "oc": 69,  // Occitan
        "ka": 70,  // Georgian
        "be": 71,  // Belarusian
        "tg": 72,  // Tajik
        "sd": 73,  // Sindhi
        "gu": 74,  // Gujarati
        "am": 75,  // Amharic
        "yi": 76,  // Yiddish
        "lo": 77,  // Lao
        "uz": 78,  // Uzbek
        "fo": 79,  // Faroese
        "ht": 80,  // Haitian Creole
        "ps": 81,  // Pashto
        "tk": 82,  // Turkmen
        "nn": 83,  // Nynorsk
        "mt": 84,  // Maltese
        "sa": 85,  // Sanskrit
        "lb": 86,  // Luxembourgish
        "my": 87,  // Myanmar
        "bo": 88,  // Tibetan
        "tl": 89,  // Tagalog
        "mg": 90,  // Malagasy
        "as": 91,  // Assamese
        "tt": 92,  // Tatar
        "haw": 93, // Hawaiian
        "ln": 94,  // Lingala
        "ha": 95,  // Hausa
        "ba": 96,  // Bashkir
        "jw": 97,  // Javanese
        "su": 98,  // Sundanese
        "yue": 99, // Cantonese
    ]

    // MARK: - Properties

    private let tokenizer: Tokenizer

    // MARK: - Initialization

    /// Initialize from a model folder containing tokenizer.json and tokenizer_config.json
    /// - Parameter modelFolder: URL to the model directory
    public init(modelFolder: URL) async throws {
        self.tokenizer = try await AutoTokenizer.from(modelFolder: modelFolder)
    }

    /// Initialize from a pretrained HuggingFace model
    /// - Parameter pretrained: The model identifier on HuggingFace (e.g., "openai/whisper-large-v3")
    public init(pretrained: String) async throws {
        self.tokenizer = try await AutoTokenizer.from(pretrained: pretrained)
    }

    /// Initialize with an existing tokenizer
    /// - Parameter tokenizer: A pre-loaded Tokenizer instance
    public init(tokenizer: Tokenizer) {
        self.tokenizer = tokenizer
    }

    // MARK: - Basic Encoding/Decoding

    /// Encode text to token IDs
    /// - Parameter text: The text to encode
    /// - Returns: Array of token IDs
    public func encode(_ text: String) -> [Int] {
        tokenizer.encode(text: text)
    }

    /// Decode token IDs to text
    /// - Parameters:
    ///   - tokens: Array of token IDs
    ///   - skipSpecialTokens: Whether to filter out special tokens (default: true)
    /// - Returns: Decoded text
    public func decode(_ tokens: [Int], skipSpecialTokens: Bool = true) -> String {
        if skipSpecialTokens {
            let filteredTokens = tokens.filter { !isSpecialToken($0) }
            return tokenizer.decode(tokens: filteredTokens)
        }
        return tokenizer.decode(tokens: tokens)
    }

    // MARK: - Special Token Helpers

    /// Check if a token ID is a special token
    /// - Parameter tokenId: The token ID to check
    /// - Returns: True if the token is a special token
    public func isSpecialToken(_ tokenId: Int) -> Bool {
        tokenId >= Self.eotToken
    }

    /// Check if a token ID is a timestamp token
    /// - Parameter tokenId: The token ID to check
    /// - Returns: True if the token is a timestamp token
    public func isTimestampToken(_ tokenId: Int) -> Bool {
        tokenId >= Self.timestampBegin
    }

    /// Check if a token ID is a language token
    /// - Parameter tokenId: The token ID to check
    /// - Returns: True if the token is a language token
    public func isLanguageToken(_ tokenId: Int) -> Bool {
        tokenId >= Self.languageTokenStart && tokenId < Self.languageTokenEnd
    }

    // MARK: - Timestamp Handling

    /// Convert a timestamp token to seconds
    /// - Parameter tokenId: A timestamp token ID
    /// - Returns: Time in seconds, or nil if not a timestamp token
    public func timestampToSeconds(_ tokenId: Int) -> Double? {
        guard isTimestampToken(tokenId) else { return nil }
        return Double(tokenId - Self.timestampBegin) * Self.secondsPerTimestampToken
    }

    /// Convert seconds to a timestamp token ID
    /// - Parameter seconds: Time in seconds
    /// - Returns: The corresponding timestamp token ID
    public func secondsToTimestampToken(_ seconds: Double) -> Int {
        Self.timestampBegin + Int(seconds / Self.secondsPerTimestampToken)
    }

    /// Decode tokens with timestamp extraction
    /// - Parameter tokens: Array of token IDs
    /// - Returns: Array of tuples containing text and time range
    public func decodeWithTimestamps(_ tokens: [Int]) -> [(text: String, start: Double, end: Double)] {
        var segments: [(text: String, start: Double, end: Double)] = []
        var currentStart: Double = 0.0
        var textTokens: [Int] = []

        for token in tokens {
            if isTimestampToken(token) {
                if let time = timestampToSeconds(token) {
                    if !textTokens.isEmpty {
                        let text = decode(textTokens, skipSpecialTokens: true)
                        if !text.trimmingCharacters(in: .whitespaces).isEmpty {
                            segments.append((text: text, start: currentStart, end: time))
                        }
                        textTokens.removeAll()
                    }
                    currentStart = time
                }
            } else if !isSpecialToken(token) {
                textTokens.append(token)
            }
        }

        // Handle any remaining tokens
        if !textTokens.isEmpty {
            let text = decode(textTokens, skipSpecialTokens: true)
            if !text.trimmingCharacters(in: .whitespaces).isEmpty {
                segments.append((text: text, start: currentStart, end: currentStart))
            }
        }

        return segments
    }

    // MARK: - Language Tokens

    /// Get the language token for an ISO language code
    /// - Parameter language: ISO 639-1 language code (e.g., "en", "ja", "zh")
    /// - Returns: The language token ID, or nil if the language is not supported
    public func languageToken(for language: String) -> Int? {
        guard let offset = Self.languageOffsets[language.lowercased()] else {
            return nil
        }
        return Self.languageTokenStart + offset
    }

    /// Get the ISO language code for a language token
    /// - Parameter tokenId: A language token ID
    /// - Returns: The ISO language code, or nil if not a language token
    public func languageCode(for tokenId: Int) -> String? {
        guard isLanguageToken(tokenId) else { return nil }
        let offset = tokenId - Self.languageTokenStart
        return Self.languageOffsets.first { $0.value == offset }?.key
    }

    /// List all supported language codes
    public static var supportedLanguages: [String] {
        Array(languageOffsets.keys).sorted()
    }

    // MARK: - Initial Decoder Tokens

    /// Create the initial tokens for the Whisper decoder
    /// - Parameters:
    ///   - language: Optional ISO language code for language-specific transcription
    ///   - task: The transcription task (transcribe or translate)
    ///   - includeTimestamps: Whether to include timestamp tokens (default: true)
    /// - Returns: Array of initial token IDs for the decoder
    public func initialTokens(
        language: String? = nil,
        task: TranscriptionOptions.TranscriptionTask = .transcribe,
        includeTimestamps: Bool = true
    ) -> [Int] {
        var tokens: [Int] = [Self.sotToken]

        // Add language token if specified
        if let language = language, let langToken = languageToken(for: language) {
            tokens.append(langToken)
        }

        // Add task token
        switch task {
        case .transcribe:
            tokens.append(Self.transcribeToken)
        case .translate:
            tokens.append(Self.translateToken)
        }

        // Add no-timestamps token if timestamps are disabled
        if !includeTimestamps {
            tokens.append(Self.noTimestampsToken)
        }

        return tokens
    }

    // MARK: - Vocabulary Access

    /// Convert a single token to its ID
    /// - Parameter token: The token string
    /// - Returns: The token ID, or nil if not found
    public func tokenToId(_ token: String) -> Int? {
        tokenizer.convertTokenToId(token)
    }

    /// Convert a single ID to its token
    /// - Parameter id: The token ID
    /// - Returns: The token string, or nil if not found
    public func idToToken(_ id: Int) -> String? {
        tokenizer.convertIdToToken(id)
    }
}
