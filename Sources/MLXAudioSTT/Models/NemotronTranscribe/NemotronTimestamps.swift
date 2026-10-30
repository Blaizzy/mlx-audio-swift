// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES and HuggingFace Inc.
// Adapted from NeMo Speech and Transformers timestamp alignment (Apache-2.0).

import Foundation
import MLX
import MLXAudioCore
import MLXNN
import Tokenizers

enum NemotronAlignmentTokenizer {
    case huggingFace(any Tokenizer)
    case sentencePiece(SentencePieceTokenizer)

    var unknownTokenId: Int? {
        switch self {
        case .huggingFace(let tokenizer): tokenizer.unknownTokenId
        case .sentencePiece(let tokenizer): tokenizer.tokenID(for: "<unk>")
        }
    }
    func convertTokenToId(_ token: String) -> Int? {
        switch self {
        case .huggingFace(let tokenizer): tokenizer.convertTokenToId(token)
        case .sentencePiece(let tokenizer): tokenizer.tokenID(for: token) ?? unknownTokenId
        }
    }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        switch self {
        case .huggingFace(let tokenizer):
            return tokenizer.encode(text: text, addSpecialTokens: addSpecialTokens)
        case .sentencePiece(let tokenizer):
            // The embedded tokenizer uses nmt_nfkc with whitespace preserved.
            var normalized = ""
            for scalar in text.unicodeScalars {
                switch scalar.value {
                case 0x0001 ... 0x0008, 0x000B, 0x000E ... 0x001F, 0x007F, 0x008F, 0x009F: continue
                case 0x0009, 0x000A, 0x000C, 0x000D, 0x1680, 0x200B ... 0x200F, 0x2028, 0x2029,
                    0x2581, 0xFEFF, 0xFFFD:
                    normalized.append(" ")
                default: normalized.unicodeScalars.append(scalar)
                }
            }
            // SentencePiece's nmt_nfkc retains the full-width tilde.
            normalized = normalized.components(separatedBy: "\u{FF5E}")
                .map(\.precomposedStringWithCompatibilityMapping).joined(separator: "\u{FF5E}")
            return tokenizer.encodeWithByteFallback(normalized)
        }
    }
}

public struct NemotronTimestampConfiguration: Decodable, Sendable {
    public var encoder: NemotronSpeechEncoderConfiguration
    public var classes: Int
    public var upsamplingFactor: Int
    public var residualScale: Float

    private enum Root: String, CodingKey { case decoder = "decoder_config" }
    private enum Key: String, CodingKey {
        case classes = "num_classes"
        case factor = "upsampling_factor"
        case scale = "residual_scale"
        case mode = "upsampling_mode"
        case transformer = "use_transformer"
        case residual
    }
    public init(from decoder: Swift.Decoder) throws {
        let root = try decoder.container(keyedBy: Root.self)
        let nested = try root.superDecoder(forKey: .decoder)
        let c = try nested.container(keyedBy: Key.self)
        encoder = try NemotronSpeechEncoderConfiguration(from: nested)
        encoder.width = encoder.features
        encoder.factor = 1
        classes = try c.decode(Int.self, forKey: .classes)
        upsamplingFactor = try c.decodeIfPresent(Int.self, forKey: .factor) ?? 1
        residualScale = try c.decodeIfPresent(Float.self, forKey: .scale) ?? 0.1
        guard classes > 0, upsamplingFactor > 0, residualScale.isFinite,
            (try c.decodeIfPresent(String.self, forKey: .mode) ?? "conv_transpose")
                == "conv_transpose",
            try c.decodeIfPresent(Bool.self, forKey: .transformer) ?? true,
            try c.decodeIfPresent(Bool.self, forKey: .residual) ?? true
        else {
            throw STTError.modelNotInitialized("Unsupported Nemotron CTC timestamp head")
        }
    }
}

final class NemotronTimestampDecoder: Module {
    let upsample: ConvTransposed1d?
    let transformer: NemotronSpeechEncoder
    @ParameterInfo(key: "residual_scale") var residualScale: MLXArray
    @ModuleInfo(key: "decoder_layers") var projections: [Linear]
    let factor: Int
    init(_ c: NemotronTimestampConfiguration) {
        factor = c.upsamplingFactor
        upsample =
            factor > 1
            ? ConvTransposed1d(
                inputChannels: c.encoder.width,
                outputChannels: c.encoder.width, kernelSize: factor, stride: factor) : nil
        transformer = NemotronSpeechEncoder(c.encoder, includeStacking: false)
        _residualScale.wrappedValue = MLXArray(c.residualScale)
        _projections.wrappedValue = [Linear(c.encoder.width, c.classes + 1)]
    }
    func callAsFunction(_ input: MLXArray) -> MLXArray {
        var states = input.asType(transformer.finalNorm.weight!.dtype)
        if let upsample { states = upsample(states) }
        let hidden =
            states + residualScale.asType(states.dtype) * transformer(states, length: states.dim(1))
        return projections[0](hidden)
    }
}

final class NemotronCTCTimestamp: Module {
    let decoder: NemotronTimestampDecoder
    init(_ c: NemotronTimestampConfiguration) { decoder = NemotronTimestampDecoder(c) }
    func compactLogProbabilities(
        states: MLXArray, columns: [Int], configuration: NemotronTranscribeConfiguration
    ) -> [Float] {
        var output: [Float] = []
        let factor = decoder.factor
        let ids = MLXArray(columns.map(Int32.init))
        for start in stride(from: 0, to: states.dim(1), by: configuration.windowLength) {
            let end = min(start + configuration.windowLength, states.dim(1))
            let first = max(0, start - configuration.leftContext)
            let last = min(states.dim(1), end + configuration.rightContext)
            let logits = decoder(states[0..., first ..< last, 0...])[
                0, ((start - first) * factor) ..< ((end - first) * factor), 0...
            ].asType(.float32)
            let compact = logits[0..., ids] - logSumExp(logits, axis: -1, keepDims: true)
            output.append(contentsOf: compact.reshaped(-1).asArray(Float.self))
        }
        return output
    }
}

enum NemotronWordAlignment {
    static func isKeptTag(_ word: String) -> Bool {
        word.range(of: #"^(?i:<spk:\d+>)$|^<(?:ev|sty):[^<>]*>$"#, options: .regularExpression)
            != nil
    }
    struct Word {
        var text: String
        var ids: [Int]
    }
    struct Stream {
        var speaker: Int?
        var words: [Word]
    }
    enum Failure: Error {
        case unsupportedLanguage, unrepresentableText, insufficientFrames, noPath, unmatchedSpeaker
    }

    static func tokenize(
        _ text: String, tokenizer: NemotronAlignmentTokenizer, blank: Int, language: String?
    )
        throws -> [Stream]
    {
        let pattern = try NSRegularExpression(pattern: "<spk:([0-9]+)>", options: .caseInsensitive)
        let text = text.replacingOccurrences(of: "—", with: " ") as NSString
        let matches = pattern.matches(
            in: text as String, range: NSRange(location: 0, length: text.length))
        var parts: [(String, Int?)] = []
        var first = 0
        var speaker: Int? = matches.isEmpty ? 0 : nil
        for match in matches {
            parts.append(
                (
                    text.substring(
                        with: NSRange(location: first, length: match.range.location - first)),
                    speaker
                ))
            speaker = Int(text.substring(with: match.range(at: 1)))
            first = NSMaxRange(match.range)
        }
        parts.append((text.substring(from: first), speaker))
        var lang = language?.lowercased()
        let aliases = ["chinese": "zh", "japanese": "ja", "thai": "th"]
        lang = lang.map { aliases[$0] ?? $0 }
        if lang == nil {
            let scalars = parts.flatMap { $0.0.split(whereSeparator: \.isWhitespace) }
                .filter { !isKeptTag(String($0)) }.flatMap { $0.unicodeScalars.map(\.value) }
            let scripts: [(String, (UInt32) -> Bool)] = [
                ("ko", { (0xAC00 ... 0xD7AF).contains($0) || (0x1100 ... 0x11FF).contains($0) }),
                ("ja", { (0x3040 ... 0x309F).contains($0) }),
                ("ja", { (0x30A0 ... 0x30FF).contains($0) }),
                (
                    "zh",
                    {
                        (0x3400 ... 0x9FFF).contains($0) || (0x20000 ... 0x323AF).contains($0)
                            || (0xF900 ... 0xFAFF).contains($0)
                    }
                ),
                ("th", { (0x0E00 ... 0x0E7F).contains($0) }),
            ]
            lang = scripts.first { _, contains in scalars.contains(where: contains) }?.0
        }
        if lang == "th" { throw Failure.unsupportedLanguage }
        var streams: [Stream] = []
        for (part, speaker) in parts {
            var words: [Word] = []
            for item in part.split(whereSeparator: \.isWhitespace) {
                let word = String(item)
                let kept = isKeptTag(word)
                if (lang == "zh" || lang == "ja") && !kept {
                    for scalar in word.precomposedStringWithCanonicalMapping.unicodeScalars {
                        let category = scalar.properties.generalCategory
                        guard
                            [
                                .uppercaseLetter, .lowercaseLetter, .titlecaseLetter,
                                .modifierLetter,
                                .otherLetter, .nonspacingMark, .spacingMark, .enclosingMark,
                                .decimalNumber, .letterNumber, .otherNumber,
                            ].contains(category)
                        else { continue }
                        let character = String(scalar)
                        var ids: [Int]
                        if let id = tokenizer.convertTokenToId(character),
                            id != tokenizer.unknownTokenId
                        {
                            ids = [id]
                        } else {
                            let boundary = tokenizer.convertTokenToId("▁")
                            ids = tokenizer.encode(text: " " + character, addSpecialTokens: false)
                                .filter { $0 != boundary }
                        }
                        if let unknown = tokenizer.unknownTokenId, ids.contains(unknown) {
                            throw Failure.unrepresentableText
                        }
                        words.append(Word(text: character, ids: ids))
                    }
                } else {
                    words.append(
                        Word(
                            text: word,
                            ids: tokenizer.encode(text: " " + word, addSpecialTokens: false)))
                }
            }
            guard
                words.allSatisfy({ !$0.ids.isEmpty && $0.ids.allSatisfy { $0 >= 0 && $0 < blank } })
            else {
                throw Failure.unrepresentableText
            }
            if !words.isEmpty {
                if let index = streams.firstIndex(where: { $0.speaker == speaker }) {
                    streams[index].words += words
                } else {
                    streams.append(Stream(speaker: speaker, words: words))
                }
            }
        }
        return streams
    }

    private struct Trigram: Hashable {
        let a: Int
        let b: Int
        let c: Int
    }
    static func bandStarts(greedy: [Int], target: [Int], states: Int, width: Int) -> [Int] {
        func trigrams(_ tokens: [Int]) -> [Trigram: [Int]] {
            var result: [Trigram: [Int]] = [:]
            if tokens.count >= 3 {
                for i in 0 ..< (tokens.count - 2) {
                    result[Trigram(a: tokens[i], b: tokens[i + 1], c: tokens[i + 2]), default: []]
                        .append(i)
                }
            }
            return result
        }
        var times: [Int] = []
        for i in greedy.indices where greedy[i] != 0 && (i == 0 || greedy[i] != greedy[i - 1]) {
            times.append(i)
        }
        let left = trigrams(times.map { greedy[$0] })
        let right = trigrams(target)
        let pairs: [(Int, Int)] = left.compactMap { key, value in
            guard value.count == 1, let position = right[key], position.count == 1 else {
                return nil
            }
            return (times[value[0] + 1], 2 * (position[0] + 1) + 1)
        }.sorted { $0.0 < $1.0 }
        var tails: [Int] = []
        var indices: [Int] = []
        var parents: [Int] = []
        for (i, pair) in pairs.enumerated() {
            let j = lowerBound(tails, pair.1)
            parents.append(j == 0 ? -1 : indices[j - 1])
            if j == tails.count {
                tails.append(pair.1)
                indices.append(i)
            } else {
                tails[j] = pair.1
                indices[j] = i
            }
        }
        var chosen: [(Int, Int)] = []
        var index = indices.last ?? -1
        while index >= 0 {
            chosen.append(pairs[index])
            index = parents[index]
        }
        let points =
            [(0, 0)] + chosen.reversed().filter { $0.0 > 0 && $0.0 < greedy.count - 1 } + [
                (greedy.count - 1, states - 1)
            ]
        var starts: [Int] = []
        var interval = 0
        var runningMinimum = Int.max
        for i in greedy.indices {
            while interval + 1 < points.count - 1 && points[interval + 1].0 < i { interval += 1 }
            let a = points[interval]
            let b = points[interval + 1]
            let fraction = b.0 == a.0 ? 0 : Double(i - a.0) / Double(b.0 - a.0)
            let center = Double(a.1) + fraction * Double(b.1 - a.1)
            let start = min(
                states - width, max(0, Int(center.rounded(.toNearestOrEven)) - width / 2))
            runningMinimum = min(runningMinimum, start - 2 * i)
            starts.append(
                max(runningMinimum + 2 * i, max(0, states - width - 2 * (greedy.count - 1 - i))))
        }
        return starts
    }

    static func lowerBound(_ values: [Int], _ value: Int, upper: Bool = false) -> Int {
        var low = 0
        var high = values.count
        while low < high {
            let middle = (low + high) / 2
            if values[middle] < value || (upper && values[middle] == value) {
                low = middle + 1
            } else {
                high = middle
            }
        }
        return low
    }

    static func ctcPath(logProbabilities: [Float], classes: Int, target: [Int], activity: [Float])
        throws -> (frames: [Int], path: [Int])
    {
        let frames = activity.indices.filter { activity[$0] >= 0.01 }
        let repeats = zip(target.dropFirst(), target).filter { $0 == $1 }.count
        guard !target.isEmpty, frames.count >= target.count + repeats else {
            throw Failure.insufficientFrames
        }
        var labels = Array(repeating: 0, count: 2 * target.count + 1)
        for (i, label) in target.enumerated() { labels[2 * i + 1] = label }
        let states = labels.count
        let width =
            frames.count < 1000
            ? states : min(states, 2 * Int(ceil(0.1 * Double(states) + 100)) + 1)
        let greedy = frames.map { frame in
            (0 ..< classes).max {
                logProbabilities[frame * classes + $0] < logProbabilities[frame * classes + $1]
            }!
        }
        let starts = bandStarts(greedy: greedy, target: target, states: states, width: width)
        var previous = Array(repeating: -Float.infinity, count: width)
        previous[0] = logProbabilities[frames[0] * classes]
        previous[1] =
            logProbabilities[frames[0] * classes + labels[1]] + log(max(activity[frames[0]], 1e-6))
        var pointers = Array(repeating: UInt8(0), count: frames.count * width)
        for t in 1 ..< frames.count {
            var next = Array(repeating: -Float.infinity, count: width)
            let start = starts[t]
            let delta = start - starts[t - 1]
            let prior = log(max(activity[frames[t]], 1e-6))
            for i in 0 ..< width {
                let state = start + i
                let offset = i + delta
                var best: Float = (0 ..< width).contains(offset) ? previous[offset] : -.infinity
                var direction: UInt8 = 0
                if (0 ..< width).contains(offset - 1), previous[offset - 1] > best {
                    best = previous[offset - 1]
                    direction = 1
                }
                if state >= 2, labels[state] != 0, labels[state] != labels[state - 2],
                    (0 ..< width).contains(offset - 2), previous[offset - 2] > best
                {
                    best = previous[offset - 2]
                    direction = 2
                }
                pointers[t * width + i] = direction
                next[i] =
                    best + logProbabilities[frames[t] * classes + labels[state]]
                    + (labels[state] == 0 ? 0 : prior)
            }
            previous = next
        }
        let end = states - 1 - starts.last!
        var state = previous[end] >= previous[end - 1] ? states - 1 : states - 2
        guard previous[state - starts.last!].isFinite else { throw Failure.noPath }
        var path = Array(repeating: 0, count: frames.count)
        for t in stride(from: frames.count - 1, through: 0, by: -1) {
            path[t] = state
            if t > 0 { state -= Int(pointers[t * width + state - starts[t]]) }
        }
        return (frames, path)
    }

    static func speakerMapping(
        logProbabilities: [Float], classes: Int, streams: [Stream], probabilities: [[Float]]
    ) throws -> [Int] {
        let speakers = probabilities[0].count
        guard streams.count <= speakers, speakers <= 16 else { throw Failure.unmatchedSpeaker }
        var counts = Array(
            repeating: Array(repeating: Double(0), count: classes), count: streams.count)
        for (row, stream) in streams.enumerated() {
            for id in stream.words.flatMap(\.ids) { counts[row][id] += 1 }
        }
        var spoken = Array(repeating: Array(repeating: Double(0), count: classes), count: speakers)
        var previous = -1
        for frame in probabilities.indices {
            let token = (0 ..< classes).max {
                logProbabilities[frame * classes + $0] < logProbabilities[frame * classes + $1]
            }!
            if token != 0 && token != previous {
                for speaker in 0 ..< speakers where probabilities[frame][speaker] >= 0.5 {
                    spoken[speaker][token] += 1
                }
            }
            previous = token
        }
        let idf = (0 ..< classes).map { column in
            log1p(Double(speakers) / Double(1 + spoken.filter { $0[column] > 0 }.count))
        }
        let a = counts.map { row in zip(row, idf).map { sqrt($0 * $1) } }
        let b = spoken.map { row in zip(row, idf).map { sqrt($0 * $1) } }
        let similarity = a.enumerated().map { row, vector in
            b.enumerated().map { column, other in
                let dot = zip(vector, other).reduce(0) { $0 + $1.0 * $1.1 }
                let norm = sqrt(
                    vector.reduce(0) { $0 + $1 * $1 } * other.reduce(0) { $0 + $1 * $1 })
                return dot / max(norm, 1e-8) - 0.1
                    * abs(log((counts[row].reduce(0, +) + 1) / (spoken[column].reduce(0, +) + 1)))
            }
        }
        // At most eight native speakers: exact assignment without a SciPy dependency.
        var assignments: [Int: (score: Double, columns: [Int])] = [0: (0, [])]
        for row in streams.indices {
            var next: [Int: (score: Double, columns: [Int])] = [:]
            for mask in assignments.keys.sorted() {
                let old = assignments[mask]!
                for column in 0 ..< speakers where mask & (1 << column) == 0 {
                    let key = mask | (1 << column)
                    let score = old.score + similarity[row][column]
                    if score > (next[key]?.score ?? -.infinity) {
                        next[key] = (score, old.columns + [column])
                    }
                }
            }
            assignments = next
        }
        return assignments.keys.sorted().max { assignments[$0]!.score < assignments[$1]!.score }.map
        { assignments[$0]!.columns }!
    }

    static func align(
        text: String, tokenizer: NemotronAlignmentTokenizer, head: NemotronCTCTimestamp,
        states: MLXArray, nativeProbabilities: MLXArray, duration: Double,
        configuration: NemotronTranscribeConfiguration, language: String?
    ) throws -> [[String: Any]] {
        let streams = try tokenize(
            text, tokenizer: tokenizer, blank: head.decoder.projections[0].weight.dim(0) - 1,
            language: language)
        guard !streams.isEmpty else { return [] }
        let blank = head.decoder.projections[0].weight.dim(0) - 1
        let columns = [blank] + Set(streams.flatMap { $0.words.flatMap(\.ids) }).sorted()
        let remap = Dictionary(uniqueKeysWithValues: columns.enumerated().map { ($1, $0) })
        let compactStreams = streams.map {
            Stream(
                speaker: $0.speaker,
                words: $0.words.map { Word(text: $0.text, ids: $0.ids.map { remap[$0]! }) })
        }
        let logProbabilities = head.compactLogProbabilities(
            states: states, columns: columns, configuration: configuration)
        return try alignCompact(
            streams: compactStreams, logProbabilities: logProbabilities, classes: columns.count,
            nativeProbabilities: nativeProbabilities, duration: duration,
            configuration: configuration, upsamplingFactor: head.decoder.factor)
    }

    static func alignCompact(
        streams: [Stream], logProbabilities: [Float], classes: Int,
        nativeProbabilities: MLXArray, duration: Double,
        configuration: NemotronTranscribeConfiguration, upsamplingFactor: Int
    ) throws -> [[String: Any]] {
        let frames = logProbabilities.count / classes
        let speakers = nativeProbabilities.dim(-1)
        let native = nativeProbabilities.reshaped(-1, speakers).asType(.float32)
        let nativeValues = native.reshaped(-1).asArray(Float.self)
        let pooled = native.reshaped(-1, configuration.speech.factor, speakers).mean(axis: 1)
        let values = pooled.reshaped(-1).asArray(Float.self)
        var probabilities: [[Float]] = []
        for frame in 0 ..< frames {
            let position =
                frames <= 1 ? 0 : Double(frame) * Double(pooled.dim(0) - 1) / Double(frames - 1)
            let low = Int(position)
            let high = min(low + 1, pooled.dim(0) - 1)
            let fraction = Float(position - Double(low))
            probabilities.append(
                (0 ..< speakers).map {
                    values[low * speakers + $0] * (1 - fraction) + values[high * speakers + $0]
                        * fraction
                })
        }
        let mapping = try speakerMapping(
            logProbabilities: logProbabilities, classes: classes, streams: streams,
            probabilities: probabilities)
        let step = duration / Double(frames)
        let nativeStep =
            Double(configuration.preprocessor.hopLength)
            / Double(configuration.preprocessor.sampleRate)
        let limit = 12 * Double(upsamplingFactor) * min(step, 0.08)
        var result: [[String: Any]] = []
        for (row, stream) in streams.enumerated() {
            let speaker = mapping[row]
            let aligned = try ctcPath(
                logProbabilities: logProbabilities, classes: classes,
                target: stream.words.flatMap(\.ids), activity: probabilities.map { $0[speaker] })
            var times: [(start: Double, end: Double)] = []
            var index = 0
            for word in stream.words {
                let first = lowerBound(aligned.path, 2 * index + 1)
                index += word.ids.count
                let last = lowerBound(aligned.path, 2 * index - 1, upper: true) - 1
                guard first < aligned.frames.count, last >= first else { throw Failure.noPath }
                let start = Double(aligned.frames[first]) * step
                times.append(
                    (start, min(Double(aligned.frames[last] + 1) * step, start + 1, duration)))
            }
            let valid = min(native.dim(0), max(1, Int(duration / nativeStep)))
            var segments: [(Double, Double)] = []
            var start: Int?
            for i in 0 ... valid {
                let active = i < valid && nativeValues[i * speakers + speaker] > 0.5
                if active && start == nil { start = i }
                if !active, let first = start {
                    segments.append(
                        (Double(first) * nativeStep, min(Double(i) * nativeStep, duration)))
                    start = nil
                }
            }
            let segmentIDs: [Int?] = times.map { word in
                segments.lastIndex { $0.0 <= word.start && word.start < $0.1 }
            }
            for i in times.indices {
                if let segment = segmentIDs[i], i == 0 || segmentIDs[i - 1] != segment,
                    times[i].start > segments[segment].0,
                    times[i].start - segments[segment].0 <= limit
                {
                    times[i].start = segments[segment].0
                }
            }
            for i in times.indices {
                let end: Double
                if let segment = segmentIDs[i], i + 1 == times.count || segmentIDs[i + 1] != segment
                {
                    end = segments[segment].1
                } else {
                    end = i + 1 < times.count ? times[i + 1].start : times[i].end
                }
                result.append([
                    "text": stream.words[i].text, "start": times[i].start,
                    "end": max(times[i].start, min(end, times[i].start + limit, duration)),
                    "speaker": stream.speaker.map { $0 as Any } ?? NSNull(),
                ])
            }
        }
        return result.sorted {
            let a = ($0["start"] as! Double, $0["end"] as! Double)
            let b = ($1["start"] as! Double, $1["end"] as! Double)
            return a < b
        }
    }
}
