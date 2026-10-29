// G2P GRU algorithm by Kyubyong Park and Jongseok Kim: https://github.com/Kyubyong/g2p
import Foundation
import MLX
import MLXNN
import MLXAudioCore
import MLXAudioG2P

struct MeloTextResult {
    let normalized: String
    let phones: [String]
    let phoneIDs: [Int32]
    let tones: [Int32]
    let tokenIDs: [Int32]
    let featureIndices: [Int32]
    let word2ph: [Int]
}

struct MeloEnglishFrontend {
    let vocabulary: [String: Int32]
    let pronunciations: [String: [String]]
    let homographs: [String: ([String], [String], String)]
    let tags: [String: String]
    let tagWeights: [String: [String: Double]]
    let tagClasses: [String]
    let gru: MeloLayers

    static func validateGRU(_ weights: [String: MLXArray]) throws {
        var shapes = ["enc_emb": [29, 256], "dec_emb": [74, 256], "fc_w": [74, 256], "fc_b": [74]]
        for p in ["enc", "dec"] {
            shapes[p + "_w_ih"] = [768, 256]; shapes[p + "_w_hh"] = [768, 256]
            shapes[p + "_b_ih"] = [768]; shapes[p + "_b_hh"] = [768]
        }
        try MeloTTSModel.validate(weights, shapes: shapes)
    }
    /// Melo calls g2p_en separately for each BERT word, so its POS context contains one word.
    func isolatedTag(_ word: String) -> String {
        if let tag = tags[word] { return tag }
        let features = ["bias", "i suffix \(word.suffix(3))", "i pref1 \(word.prefix(1))", "i-1 tag -START-",
            "i-2 tag -START2-", "i tag+i-2 tag -START- -START2-", "i word \(word)", "i-1 tag+i word -START- \(word)",
            "i-1 word -START2-", "i-1 suffix T2-", "i-2 word -START-", "i+1 word -END-", "i+1 suffix ND-", "i+2 word -END2-"]
        var scores: [String: Double] = [:]
        for feature in features { for (tag, weight) in tagWeights[feature] ?? [:] { scores[tag, default: 0] += weight } }
        return tagClasses.max { a, b in scores[a, default: 0] == scores[b, default: 0] ? a < b : scores[a, default: 0] < scores[b, default: 0] } ?? "NN"
    }
    static let phonemes = ["<pad>", "<unk>", "<s>", "</s>"] + "AA0 AA1 AA2 AE0 AE1 AE2 AH0 AH1 AH2 AO0 AO1 AO2 AW0 AW1 AW2 AY0 AY1 AY2 B CH D DH EH0 EH1 EH2 ER0 ER1 ER2 EY0 EY1 EY2 F G HH IH0 IH1 IH2 IY0 IY1 IY2 JH K L M N NG OW0 OW1 OW2 OY0 OY1 OY2 P R S SH T TH UH0 UH1 UH2 UW UW0 UW1 UW2 V W Y Z ZH".split(separator: " ").map(String.init)
    static func gruCell(_ x: MLXArray, h: MLXArray, prefix: String, weights: MeloLayers) -> MLXArray {
        let a = matmul(x, weights.w(prefix + "_w_ih").T) + weights.w(prefix + "_b_ih")
        let b = matmul(h, weights.w(prefix + "_w_hh").T) + weights.w(prefix + "_b_hh")
        let n = h.dim(-1)
        let gates = sigmoid(a[0..., ..<(2 * n)] + b[0..., ..<(2 * n)])
        let r = gates[0..., ..<n], z = gates[0..., n...]
        return (1 - z) * tanh(a[0..., (2 * n)...] + r * b[0..., (2 * n)...]) + z * h
    }
    func predict(_ word: String) -> [String] {
        var h = MLXArray.zeros([1, gru.w("enc_w_hh").dim(1)])
        for id in word.unicodeScalars.map({ s in s.value >= 97 && s.value <= 122 ? Int(s.value - 97 + 3) : 1 }) + [2] {
            h = Self.gruCell(gru.w("enc_emb")[id].reshaped(1, -1), h: h, prefix: "enc", weights: gru)
        }
        var id = 2, result: [String] = []
        for _ in 0..<20 {
            h = Self.gruCell(gru.w("dec_emb")[id].reshaped(1, -1), h: h, prefix: "dec", weights: gru)
            id = argMax(matmul(h, gru.w("fc_w").T) + gru.w("fc_b")).item(Int.self)
            if id == 3 { break }
            result.append(Self.phonemes[id])
        }
        return result
    }
    func pronunciation(_ word: String) -> [String] {
        // BERT basic tokenization already lowercases, strips accents and separates punctuation.
        let clean = String(word.unicodeScalars.filter { CharacterSet(charactersIn: "abcdefghijklmnopqrstuvwxyz'.,?!-").contains($0) })
        if clean.range(of: "[a-z]", options: .regularExpression) == nil { return clean.isEmpty ? [] : [clean] }
        if let (a, b, pos) = homographs[clean] { return isolatedTag(clean).hasPrefix(pos) ? a : b }
        return pronunciations[clean] ?? predict(clean)
    }
    func tokenize(_ text: String) -> [String] {
        let stripped = text.lowercased().decomposedStringWithCanonicalMapping.unicodeScalars.filter { !CharacterSet.nonBaseCharacters.contains($0) }
        var basic: [String] = [], word = ""
        func flush() { if !word.isEmpty { basic.append(word); word = "" } }
        for scalar in stripped {
            let n = scalar.value
            if CharacterSet.whitespacesAndNewlines.contains(scalar) { flush() }
            else if n == 0 || n == 0xfffd || CharacterSet.controlCharacters.contains(scalar) { continue }
            else if CharacterSet.punctuationCharacters.contains(scalar) || (33...47).contains(n) || (58...64).contains(n) || (91...96).contains(n) || (123...126).contains(n) || (0x4e00...0x9fff).contains(n) || (0x3400...0x4dbf).contains(n) || (0x20000...0x2fa1f).contains(n) {
                flush(); basic.append(String(scalar))
            } else { word.unicodeScalars.append(scalar) }
        }
        flush()
        return basic.flatMap { word -> [String] in
            let chars = Array(word), n = chars.count
            if n > 100 { return ["[UNK]"] }
            var pieces: [String] = [], start = 0
            while start < n {
                var end = n, found: String?
                while end > start {
                    let piece = (start == 0 ? "" : "##") + String(chars[start..<end])
                    if vocabulary[piece] != nil { found = piece; break }
                    end -= 1
                }
                guard let found else { return ["[UNK]"] }
                pieces.append(found); start = end
            }
            return pieces
        }
    }
    func process(_ text: String, symbols: [String], addBlank: Bool) throws -> MeloTextResult {
        let normalized = Self.normalize(text), tokens = tokenize(normalized)
        guard !tokens.isEmpty, tokens.count + 2 <= 512 else {
            throw AudioGenerationError.invalidInput("MeloTTS text must contain 1–510 BERT wordpieces")
        }
        var groups: [[String]] = []
        for token in tokens {
            if token.hasPrefix("##"), !groups.isEmpty { groups[groups.count - 1].append(String(token.dropFirst(2))) }
            else { groups.append([token]) }
        }
        let lookup = Dictionary(uniqueKeysWithValues: symbols.enumerated().map { ($1, Int32($0)) })
        guard let unknown = lookup["UNK"], lookup["_"] == 0 else { throw AudioGenerationError.invalidInput("Invalid MeloTTS symbols") }
        var phones = ["_"], tones: [Int32] = [0], alignment = [1]
        for group in groups {
            let pronunciation = pronunciation(group.joined())
            for phone in pronunciation {
                var p = phone, tone: Int32 = 0
                if let digit = p.last?.wholeNumberValue { tone = Int32(digit + 1); p.removeLast() }
                // The scoped reference's ARPAbet allowlist excludes Y, which maps to UNK.
                p = p == "Y" ? "UNK" : p.lowercased()
                if p == "v" { p = "V" }
                if lookup[p] == nil { p = "UNK" }
                phones.append(p); tones.append(tone)
            }
            alignment += (0..<group.count).map { pronunciation.count / group.count + ($0 < pronunciation.count % group.count ? 1 : 0) }
        }
        phones.append("_"); tones.append(0); alignment.append(1)
        var features: [Int32] = []
        for (i, n) in alignment.enumerated() { features += Array(repeating: Int32(i), count: addBlank ? n * 2 + (i == 0 ? 1 : 0) : n) }
        if addBlank {
            phones = ["_"] + phones.flatMap { [$0, "_"] }
            tones = [0] + tones.flatMap { [$0, 0] }
        }
        return MeloTextResult(normalized: normalized, phones: phones, phoneIDs: phones.map { lookup[$0] ?? unknown }, tones: tones.map { $0 + 7 }, tokenIDs: [101] + tokens.map { vocabulary[$0] ?? 100 } + [102], featureIndices: features, word2ph: alignment)
    }
    static let ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
    static func number(_ n: Int) -> String {
        if n == 0 { return "zero" }
        if n < 0 { return "minus " + number(-n) }
        if n >= 1000 { return number(n / 1000) + " thousand" + (n % 1000 == 0 ? "" : " " + number(n % 1000)) }
        if n >= 100 { return ones[n / 100] + " hundred" + (n % 100 == 0 ? "" : " " + number(n % 100)) }
        if n < 20 { return ones[n] }
        return ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"][n / 10] + (n % 10 == 0 ? "" : " " + ones[n % 10])
    }
    static func replacing(_ text: String, pattern: String, replacement: (String) -> String) -> String {
        let regex = try! NSRegularExpression(pattern: pattern)
        var result = text
        for match in regex.matches(in: text, range: NSRange(text.startIndex..., in: text)).reversed() {
            if let range = Range(match.range, in: result) { result.replaceSubrange(range, with: replacement(String(result[range]))) }
        }
        return result
    }
    static func normalize(_ text: String) -> String {
        var text = text.lowercased()
        for (from, to) in ["mrs":"missis", "mr":"mister", "dr":"doctor", "st":"saint", "co":"company", "jr":"junior", "maj":"major", "gen":"general", "drs":"doctors", "rev":"reverend", "lt":"lieutenant", "hon":"honorable", "sgt":"sergeant", "capt":"captain", "esq":"esquire", "ltd":"limited", "col":"colonel", "ft":"fort"] {
            text = replacing(text, pattern: "\\b" + from + "\\.") { _ in to }
        }
        text = replacing(text, pattern: #"\d{1,3}(,\d{3})+"#) { $0.replacingOccurrences(of: ",", with: "") }
        text = replacing(text, pattern: #"\d+\.\d+"#) { value in
            let p = value.split(separator: ".")
            guard let n = Int(p[0]) else { return value }
            return number(n) + " point " + p[1].map { ones[$0.wholeNumberValue ?? 0] }.joined(separator: " ")
        }
        return replacing(text, pattern: #"\d+"#) { Int($0).map(number) ?? $0 }
    }
}

extension MeloEnglishFrontend {
    init(directory: URL) throws {
        let vocabularyText = try String(contentsOf: directory.appendingPathComponent("vocab.txt"), encoding: .utf8)
        vocabulary = Dictionary(uniqueKeysWithValues: vocabularyText.split(separator: "\n").enumerated().map { (String($1), Int32($0)) })
        guard vocabulary["[CLS]"] == 101, vocabulary["[SEP]"] == 102, vocabulary.count == 30522 else {
            throw AudioGenerationError.invalidInput("MeloTTS requires the bert-base-uncased vocabulary")
        }
        let cmu = try ZipArchive.load(url: directory.appendingPathComponent("cmudict.zip"))
        guard let raw = cmu["cmudict/cmudict"], let dictionary = String(data: raw, encoding: .utf8) else {
            throw AudioGenerationError.invalidInput("MeloTTS CMU dictionary is missing")
        }
        var entries: [String: [String]] = [:]
        for line in dictionary.split(separator: "\n") {
            let parts = line.split(separator: " ")
            guard parts.count > 2, let entry = CMUDictParser.parseLine(String(parts[0]) + " " + parts.dropFirst(2).joined(separator: " ")) else { continue }
            if entries[entry.word] == nil { entries[entry.word] = entry.arpabet }
        }
        pronunciations = entries
        var homographs: [String: ([String], [String], String)] = [:]
        for line in try String(contentsOf: directory.appendingPathComponent("homographs.en"), encoding: .utf8).split(separator: "\n") where !line.hasPrefix("#") {
            let p = line.split(separator: "|").map(String.init)
            guard p.count == 4 else { throw AudioGenerationError.invalidInput("Invalid MeloTTS homograph entry") }
            homographs[p[0].lowercased()] = (p[1].split(separator: " ").map(String.init), p[2].split(separator: " ").map(String.init), p[3])
        }
        self.homographs = homographs
        let tagger = try ZipArchive.load(url: directory.appendingPathComponent("averaged_perceptron_tagger_eng.zip"))
        func resource(_ type: String) throws -> Data {
            guard let data = tagger["averaged_perceptron_tagger_eng/averaged_perceptron_tagger_eng.\(type).json"] else {
                throw AudioGenerationError.invalidInput("Missing MeloTTS English tagger \(type)")
            }
            return data
        }
        tags = try JSONDecoder().decode([String: String].self, from: resource("tagdict"))
        tagWeights = try JSONDecoder().decode([String: [String: Double]].self, from: resource("weights"))
        tagClasses = try JSONDecoder().decode([String].self, from: resource("classes"))
        let weights = try NumpyArchive.load(url: directory.appendingPathComponent("checkpoint20.npz"))
        try Self.validateGRU(weights)
        gru = MeloLayers(weights: weights)
    }
}
