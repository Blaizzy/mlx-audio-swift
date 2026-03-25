import Foundation
import HuggingFace
import MLXAudioCore
import MLXAudioG2P

/// Multilingual TextProcessor for Kokoro TTS.
///
/// - English: delegates to MisakiTextProcessor (CMUdict + rules)
/// - Other languages: uses IPA lexicon lookup from gruut espeak dictionaries
/// - Unknown words: passed through as-is
public final class KokoroMultilingualProcessor: TextProcessor, @unchecked Sendable {
    private let englishProcessor = MisakiTextProcessor()
    private let lexiconRepo: String
    private let neuralG2PRepo: String
    private var lexiconCache: [String: [String: String]] = [:]
    private var lexiconDirectory: URL?
    private var neuralG2P: G2P?
    private let lock = NSLock()

    /// Kokoro voice prefix → language code mapping.
    public static let voiceLanguageMap: [Character: String] = [
        "a": "en-us",
        "b": "en-gb",
        "e": "es",
        "f": "fr",
        "h": "hi",
        "i": "it",
        "j": "ja",
        "p": "pt",
        "z": "cmn",
    ]

    /// Language codes that use MisakiTextProcessor (English G2P).
    private static let englishCodes: Set<String> = ["en", "en-us", "en-gb"]

    private static let neuralLangMap: [String: String] = [
        "ja": "jpn",
        "hi": "hin",
        "cmn": "zho-s",
        "zh": "zho-s",
    ]

    private static let charSplitLangs: Set<String> = ["ja", "cmn", "zh"]

    /// Language code → TSV filename mapping.
    private static let langFileMap: [String: String] = [
        "es": "es_lexicon.tsv",
        "fr": "fr_lexicon.tsv",
        "it": "it_lexicon.tsv",
        "pt": "pt_lexicon.tsv",
        "pt-br": "pt_lexicon.tsv",
        "de": "de_lexicon.tsv",
        "ru": "ru_lexicon.tsv",
        "ar": "ar_lexicon.tsv",
        "cs": "cs_lexicon.tsv",
        "fa": "fa_lexicon.tsv",
        "nl": "nl_lexicon.tsv",
        "sv": "sv_lexicon.tsv",
        "sw": "sw_lexicon.tsv",
    ]

    public init(
        lexiconRepo: String = "beshkenadze/kokoro-ipa-lexicons",
        neuralG2PRepo: String = "beshkenadze/g2p-multilingual-byT5-tiny-mlx"
    ) {
        self.lexiconRepo = lexiconRepo
        self.neuralG2PRepo = neuralG2PRepo
    }

    private func getEnglishProcessor() -> MisakiTextProcessor {
        englishProcessor
    }

    /// Infer language code from Kokoro voice name prefix.
    public static func languageForVoice(_ voice: String) -> String? {
        guard let first = voice.first else { return nil }
        return voiceLanguageMap[first]
    }

    public func process(text: String, language: String?) throws -> String {
        let lang = language?.lowercased() ?? "en-us"

        if Self.englishCodes.contains(lang) || lang.hasPrefix("en") {
            return try getEnglishProcessor().process(text: text, language: language)
        }

        if let byT5Lang = Self.neuralLangMap[lang] {
            return try neuralPhonemize(text: text, lang: lang, byT5Lang: byT5Lang)
        }

        let lexicon = try loadLexicon(for: lang)
        return phonemize(text: text, lexicon: lexicon)
    }

    // MARK: - Lexicon Loading

    private func loadLexicon(for lang: String) throws -> [String: String] {
        lock.lock()
        if let cached = lexiconCache[lang] {
            lock.unlock()
            return cached
        }
        lock.unlock()

        guard let filename = Self.langFileMap[lang] else {
            throw LexiconError.unsupportedLanguage(lang)
        }

        let dir = lock.withLock { lexiconDirectory } ?? repoDirectory(for: lexiconRepo)
        let tsvURL = dir.appendingPathComponent(filename)

        guard FileManager.default.fileExists(atPath: tsvURL.path) else {
            throw LexiconError.lexiconNotFound(filename)
        }

        let content = try String(contentsOf: tsvURL, encoding: .utf8)
        var dict = [String: String]()
        dict.reserveCapacity(100_000)

        for line in content.split(separator: "\n") {
            guard let tabIdx = line.firstIndex(of: "\t") else { continue }
            let word = String(line[line.startIndex..<tabIdx]).lowercased()
            let phonemes = String(line[line.index(after: tabIdx)...])
            let ipa = phonemes.split(separator: " ").joined()
            dict[word] = ipa
        }

        lock.lock()
        lexiconCache[lang] = dict
        lock.unlock()

        return dict
    }

    private func repoDirectory(for repo: String) -> URL {
        let modelSubdir = repo.replacingOccurrences(of: "/", with: "_")
        return HubCache.default.cacheDirectory
            .appendingPathComponent("mlx-audio")
            .appendingPathComponent(modelSubdir)
    }

    public func prepare(for language: String) async throws {
        let lang = language.lowercased()

        if Self.englishCodes.contains(lang) || lang.hasPrefix("en") {
            try await englishProcessor.prepare()
            return
        }

        if Self.neuralLangMap[lang] != nil {
            try await ensureNeuralModelDownloaded()
            return
        }

        if let filename = Self.langFileMap[lang] {
            try await ensureLexiconDownloaded(filename: filename)
            _ = try loadLexicon(for: lang)
        }
    }

    private func ensureLexiconDownloaded(filename: String) async throws {
        let dir = repoDirectory(for: lexiconRepo)
        let fileURL = dir.appendingPathComponent(filename)

        if FileManager.default.fileExists(atPath: fileURL.path) {
            lock.withLock { lexiconDirectory = dir }
            return
        }

        guard let repoID = Repo.ID(rawValue: lexiconRepo) else {
            throw LexiconError.invalidRepo(lexiconRepo)
        }

        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        _ = try await HubClient().downloadSnapshot(
            of: repoID,
            kind: .model,
            to: dir,
            matching: [filename]
        )

        lock.withLock { lexiconDirectory = dir }
    }

    private func ensureNeuralModelDownloaded() async throws {
        let dir = repoDirectory(for: neuralG2PRepo)
        let hasSafetensors = (try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil))?
            .contains { $0.pathExtension == "safetensors" } ?? false

        if hasSafetensors { return }

        guard let repoID = Repo.ID(rawValue: neuralG2PRepo) else {
            throw LexiconError.invalidRepo(neuralG2PRepo)
        }

        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        _ = try await HubClient().downloadSnapshot(
            of: repoID,
            kind: .model,
            to: dir,
            matching: ["*.safetensors", "*.json"]
        )
    }

    // MARK: - Phonemization

    func phonemize(text: String, lexicon: [String: String]) -> String {
        let normalized = text.lowercased()
        var result = [String]()
        var currentWord = ""

        for ch in normalized {
            if ch.isLetter || ch == "'" || ch == "-" || ch == "\u{0301}" {
                currentWord.append(ch)
            } else {
                if !currentWord.isEmpty {
                    let ipa = lookupWord(currentWord, lexicon: lexicon)
                    result.append(ipa)
                    currentWord = ""
                }
                if ch == "," || ch == "." || ch == "!" || ch == "?" || ch == ":" || ch == ";" {
                    result.append(String(ch))
                }
            }
        }

        if !currentWord.isEmpty {
            let ipa = lookupWord(currentWord, lexicon: lexicon)
            result.append(ipa)
        }

        return result.joined(separator: " ")
    }

    // MARK: - Neural G2P (JA, HI, ZH)

    private func getOrLoadG2P() throws -> G2P {
        lock.lock()
        if let g2p = neuralG2P {
            lock.unlock()
            return g2p
        }
        lock.unlock()

        let modelDir = repoDirectory(for: neuralG2PRepo)

        guard FileManager.default.fileExists(atPath: modelDir.path) else {
            throw LexiconError.neuralModelNotDownloaded(neuralG2PRepo)
        }

        let g2p = try G2P(modelDirectory: modelDir)

        lock.lock()
        neuralG2P = g2p
        lock.unlock()

        return g2p
    }

    private func neuralPhonemize(text: String, lang: String, byT5Lang: String) throws -> String {
        let g2p = try getOrLoadG2P()
        let words = splitWords(text: text, lang: lang)
        var result = [String]()

        for token in words {
            if token.allSatisfy({ $0.isPunctuation || $0.isWhitespace }) {
                let punct = token.filter { ",.!?:;".contains($0) }
                if !punct.isEmpty {
                    result.append(punct)
                }
                continue
            }

            let ipa = g2p.convert(token, language: byT5Lang)
            if !ipa.isEmpty {
                result.append(ipa)
            }
        }

        return result.joined(separator: " ")
    }

    func splitWords(text: String, lang: String) -> [String] {
        if Self.charSplitLangs.contains(lang) {
            var tokens = [String]()
            for ch in text {
                if ch.isWhitespace { continue }
                tokens.append(String(ch))
            }
            return tokens
        }

        var tokens = [String]()
        var current = ""
        for ch in text {
            if ch.isLetter || ch == "'" || ch == "-" {
                current.append(ch)
            } else {
                if !current.isEmpty {
                    tokens.append(current)
                    current = ""
                }
                if ch.isPunctuation {
                    tokens.append(String(ch))
                }
            }
        }
        if !current.isEmpty {
            tokens.append(current)
        }
        return tokens
    }

    func lookupWord(_ word: String, lexicon: [String: String]) -> String {
        if let ipa = lexicon[word] {
            return ipa
        }

        let stripped = word.decomposedStringWithCanonicalMapping
            .unicodeScalars
            .filter { !("\u{0300}"..."\u{036F}").contains($0) }
            .map { String($0) }
            .joined()

        if stripped != word, let ipa = lexicon[stripped] {
            return ipa
        }

        return word
    }
}

// MARK: - Errors

public enum LexiconError: Error, LocalizedError {
    case unsupportedLanguage(String)
    case lexiconNotFound(String)
    case lexiconNotDownloaded(String)
    case neuralModelNotDownloaded(String)
    case invalidRepo(String)

    public var errorDescription: String? {
        switch self {
        case .unsupportedLanguage(let lang):
            return "Unsupported language: \(lang). Available: es, fr, it, pt, de, ru, ar, cs, fa, nl, sv, sw, ja, hi, cmn"
        case .lexiconNotFound(let file):
            return "Lexicon file not found: \(file). Call prepare(for:) before processing text."
        case .lexiconNotDownloaded(let repo):
            return "Lexicons not downloaded. Call prepare(for:) before processing text. Repo: \(repo)"
        case .neuralModelNotDownloaded(let repo):
            return "Neural G2P model not downloaded. Call prepare(for:) before processing text. Repo: \(repo)"
        case .invalidRepo(let repo):
            return "Invalid HuggingFace repo: \(repo)"
        }
    }
}
