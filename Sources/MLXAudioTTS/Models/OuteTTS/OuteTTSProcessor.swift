import Foundation
import MLXAudioCore
import Tokenizers

/// Acoustic feature bins used by OuteTTS speaker profiles.
public struct OuteTTSFeatures: Codable, Sendable, Equatable {
    public var energy: Int
    public var spectralCentroid: Int
    public var pitch: Int

    public init(energy: Int = 0, spectralCentroid: Int = 0, pitch: Int = 0) {
        self.energy = energy
        self.spectralCentroid = spectralCentroid
        self.pitch = pitch
    }

    enum CodingKeys: String, CodingKey {
        case energy, pitch
        case spectralCentroid = "spectral_centroid"
    }

    var tokens: String {
        "<|energy_\(energy)|><|spectral_centroid_\(spectralCentroid)|><|pitch_\(pitch)|>"
    }
}

/// A portable OuteTTS voice, compatible with the upstream speaker JSON format.
public struct OuteTTSSpeaker: Codable, Sendable, Equatable {
    public struct Word: Codable, Sendable, Equatable {
        public var word: String
        public var duration: Double
        public var c1: [Int]
        public var c2: [Int]
        public var features: OuteTTSFeatures

        public init(word: String, duration: Double, c1: [Int], c2: [Int], features: OuteTTSFeatures) {
            self.word = word
            self.duration = duration
            self.c1 = c1
            self.c2 = c2
            self.features = features
        }
    }

    public var text: String
    public var words: [Word]
    public var globalFeatures: OuteTTSFeatures?

    public init(text: String, words: [Word], globalFeatures: OuteTTSFeatures? = nil) {
        self.text = text
        self.words = words
        self.globalFeatures = globalFeatures
    }

    enum CodingKeys: String, CodingKey {
        case text, words
        case globalFeatures = "global_features"
    }

    public static func load(from url: URL) throws -> Self {
        let speaker = try JSONDecoder().decode(Self.self, from: Data(contentsOf: url))
        try speaker.validate()
        return speaker
    }

    public func save(to url: URL) throws {
        try validate()
        try JSONEncoder().encode(self).write(to: url, options: .atomic)
    }

    public func validate() throws {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, !words.isEmpty else {
            throw AudioGenerationError.invalidInput("OuteTTS speaker needs a transcript and aligned words")
        }
        for word in words {
            guard word.duration.isFinite, word.duration >= 0,
                  !word.word.isEmpty, !word.c1.isEmpty, word.c1.count == word.c2.count,
                  (word.c1 + word.c2).allSatisfy({ (0..<1024).contains($0) }),
                  [word.features.energy, word.features.spectralCentroid, word.features.pitch]
                    .allSatisfy({ (0...100).contains($0) }) else {
                throw AudioGenerationError.invalidInput("Invalid OuteTTS speaker word or codec frame")
            }
        }
    }
}

/// Word boundaries supplied by an optional aligner, in seconds at the reference sample rate.
public struct OuteTTSAlignedWord: Sendable {
    public let word: String
    public let start: Double
    public let end: Double

    public init(word: String, start: Double, end: Double) {
        self.word = word
        self.start = start
        self.end = end
    }
}

enum OuteTTSPrompt {
    static func normalize(_ text: String) -> String {
        var text = text.replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        for (from, to) in [("…", "..."), ("“", "\""), ("”", "\""), ("‘", "'"), ("’", "'"), ("–", "-"), ("—", "-")] {
            text = text.replacingOccurrences(of: from, with: to)
        }
        return text.replacingOccurrences(of: "[\\x00-\\x1F\\x7F-\\x9F]", with: "", options: .regularExpression)
    }

    static func completion(text: String, speaker: OuteTTSSpeaker?) throws -> String {
        let text = normalize(text)
        var merged = text
        var codes = ""
        if var speaker {
            try speaker.validate()
            let reference = speaker.text.trimmingCharacters(in: .whitespacesAndNewlines)
            let cjk = reference.unicodeScalars.contains {
                (0x3040...0x30FF).contains($0.value) || (0x4E00...0x9FFF).contains($0.value)
            }
            let endings = cjk ? "。？！?!" : ".?!"
            let separator = reference.last.map { endings.contains($0) } == true ? (cjk ? "" : " ") : (cjk ? "。" : ". ")
            merged = reference + separator + text
            speaker.words[speaker.words.count - 1].word += separator.trimmingCharacters(in: .whitespaces)
            codes = speaker.words.map { word in
                let duration = String(format: "%.2f", locale: Locale(identifier: "en_US_POSIX"), word.duration)
                let pairs = zip(word.c1, word.c2).map { "<|c1_\($0)|><|c2_\($1)|>" }.joined()
                return "<|word_start|>\(word.word)<|features|><|t_\(duration)|>\(word.features.tokens)<|code|>\(pairs)<|word_end|>"
            }.joined(separator: "\n") + "\n<|word_start|>"
        }
        return "<|im_start|>\n<|text_start|>\(merged)<|text_end|>\n<|audio_start|>\n" + codes
    }

    static func chunks(_ text: String, maxWords: Int = 30) -> [String] {
        let sentences = text.components(separatedBy: CharacterSet(charactersIn: ".!?。！？︕︖"))
        var result = [String]()
        var current = [String]()
        for sentence in sentences {
            let words = sentence.split(whereSeparator: { $0.isWhitespace }).map(String.init)
            guard !words.isEmpty else { continue }
            if !current.isEmpty && current.count + words.count > maxWords {
                result.append(current.joined(separator: " "))
                current.removeAll(keepingCapacity: true)
            }
            current += words
        }
        if !current.isEmpty { result.append(current.joined(separator: " ")) }
        return result
    }
}

struct OuteTTSTokenMap {
    let c1: [Int: Int]
    let c2: [Int: Int]
    let stop: Set<Int>
    let forbidden: Set<Int>

    init(tokenizer: any Tokenizer, vocabularySize: Int, codebookSize: Int = 1024) throws {
        func tokenID(_ token: String) -> Int? {
            guard let id = tokenizer.convertTokenToId(token), tokenizer.convertIdToToken(id) == token else { return nil }
            return id
        }
        var first = [Int: Int](), second = [Int: Int]()
        for code in 0..<codebookSize {
            guard let a = tokenID("<|c1_\(code)|>"),
                  let b = tokenID("<|c2_\(code)|>"),
                  (0..<vocabularySize).contains(a), (0..<vocabularySize).contains(b) else {
                throw AudioGenerationError.invalidInput("Tokenizer is missing OuteTTS codec tokens")
            }
            first[a] = code
            second[b] = code
        }
        c1 = first
        c2 = second
        guard first.count == codebookSize, second.count == codebookSize,
              Set(first.keys).isDisjoint(with: second.keys) else {
            throw AudioGenerationError.invalidInput("OuteTTS codec tokens must have distinct IDs")
        }
        stop = Set(["<|im_end|>", "<|audio_end|>"].compactMap(tokenID))
        guard !stop.isEmpty else {
            throw AudioGenerationError.invalidInput("Tokenizer is missing OuteTTS stop tokens")
        }
        // The checkpoint pads its vocabulary beyond the tokenizer. Never sample
        // padding, chat-control tokens or the out-of-range codebook sentinel.
        let allowedMarkers = ["<|word_start|>", "<|word_end|>", "<|features|>", "<|code|>"]
        var blocked = Set<Int>()
        for id in 0..<vocabularySize {
            guard let token = tokenizer.convertIdToToken(id) else { blocked.insert(id); continue }
            if token.hasPrefix("<|") && !stop.contains(id) && first[id] == nil && second[id] == nil
                && !allowedMarkers.contains(token)
                && !["<|t_", "<|energy_", "<|spectral_centroid_", "<|pitch_"].contains(where: token.hasPrefix) {
                blocked.insert(id)
            }
        }
        forbidden = blocked
    }

    init(c1: [Int: Int], c2: [Int: Int], stop: Set<Int>, forbidden: Set<Int> = []) {
        self.c1 = c1
        self.c2 = c2
        self.stop = stop
        self.forbidden = forbidden
    }

    func extract(_ tokens: [Int]) -> [[Int]] {
        let first = tokens.compactMap { c1[$0] }
        let second = tokens.compactMap { c2[$0] }
        let count = min(first.count, second.count)
        return [Array(first.prefix(count)), Array(second.prefix(count))]
    }
}
