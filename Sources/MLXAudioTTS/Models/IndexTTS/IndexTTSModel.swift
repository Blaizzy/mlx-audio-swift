import Foundation
import HuggingFace
@preconcurrency import MLX
import MLXAudioCore
import MLXLMCommon
import MLXNN

public enum IndexTTSError: Error, LocalizedError {
    case invalidRepositoryID(String)
    case missingWeights(URL)
    case missingTokenizer(URL)
    case invalidInput(String)
    case unsupportedFullPipeline(String)

    public var errorDescription: String? {
        switch self {
        case .invalidRepositoryID(let repo):
            "Invalid IndexTTS repository ID: \(repo)"
        case .missingWeights(let directory):
            "No IndexTTS safetensors weights found in \(directory.path)"
        case .missingTokenizer(let directory):
            "No IndexTTS tokenizer.model found in \(directory.path)"
        case .invalidInput(let message):
            "Invalid IndexTTS input: \(message)"
        case .unsupportedFullPipeline(let message):
            message
        }
    }
}

public enum IndexTTSTextNormalizer {
    public static func normalize(_ text: String) -> String {
        useChinese(text) ? normalizeChinese(text) : normalizeEnglish(text)
    }

    public static func tokenizeByCJKChar(_ text: String, uppercaseASCII: Bool = true) -> String {
        var pieces: [String] = []
        pieces.reserveCapacity(text.count)
        for scalar in text.unicodeScalars {
            let value = scalar.value
            let scalarText = String(scalar)
            if isCJK(value) {
                pieces.append(" \(scalarText) ")
            } else {
                pieces.append(uppercaseASCII ? scalarText.uppercased() : scalarText)
            }
        }
        return pieces.joined()
            .split(whereSeparator: \.isWhitespace)
            .joined(separator: " ")
    }

    private static let charMap: [(String, String)] = [
        ("\u{FF1A}", ","),
        ("\u{FF1B}", ","),
        (";", ","),
        ("\u{FF0C}", ","),
        ("\u{3002}", "."),
        ("\u{FF01}", "!"),
        ("\u{FF1F}", "?"),
        ("\n", " "),
        ("\u{00B7}", "-"),
        ("\u{3001}", ","),
        ("...", "\u{2026}"),
        (",,,", "\u{2026}"),
        ("\u{FF0C}\u{FF0C}\u{FF0C}", "\u{2026}"),
        ("\u{2026}\u{2026}", "\u{2026}"),
        ("\u{201C}", "'"),
        ("\u{201D}", "'"),
        ("\"", "'"),
        ("'", "'"),
        ("\u{FF08}", "'"),
        ("\u{FF09}", "'"),
        ("(", "'"),
        (")", "'"),
        ("\u{300A}", "'"),
        ("\u{300B}", "'"),
        ("\u{3010}", "'"),
        ("\u{3011}", "'"),
        ("[", "'"),
        ("]", "'"),
        ("\u{2014}", "-"),
        ("\u{FF5E}", "-"),
        ("~", "-"),
        ("\u{300C}", "'"),
        ("\u{300D}", "'"),
        (":", ","),
    ]

    private static let zhCharMap: [(String, String)] = [("$", ".")] + charMap

    private static func normalizeChinese(_ text: String) -> String {
        var result = expandContractions(trimTrailingWhitespace(text))
        result = replacingMatches(
            in: result,
            pattern: #"(?<![a-z])((?:[bpmfdtnlgkhjqxzcsryw]|[zcs]h)?(?:[aeiouüv]|[ae]i|u[aio]|ao|ou|i[aue]|[uüv]e|[uvü]ang?|uai|[aeiuv]n|[aeio]ng|ia[no]|i[ao]ng)|ng|er)([1-5])"#,
            options: [.caseInsensitive]
        ) { match, _ in
            correctPinyin(match)
        }
        return replaceChars(result, map: zhCharMap)
    }

    private static func normalizeEnglish(_ text: String) -> String {
        var result = expandContractions(text)
        result = replacingMatches(
            in: result,
            pattern: #"\$\s*[0-9,.\s]+"#
        ) { match, _ in
            let digits = extractDigits(match)
            guard let number = Int(digits), !digits.isEmpty else {
                return match
            }
            return "\(numberToWords(number)) dollar\(number == 1 ? "" : "s") "
        }
        result = trimTrailingWhitespace(result)

        result = replacingMatches(
            in: result,
            pattern: #"\b\d(\s+\d)+\b"#
        ) { match, _ in
            let parts = match.split(whereSeparator: \.isWhitespace)
            if parts.allSatisfy({ $0.count == 1 && $0.allSatisfy(\.isNumber) }) {
                return parts.compactMap { Int($0).map(numberToWords) }.joined(separator: " ")
            }
            let digits = extractDigits(match)
            return Int(digits).map(numberToWords) ?? match
        }

        result = replacingMatches(
            in: result,
            pattern: #"\b\d+(?:,\d+)*\b"#
        ) { match, _ in
            let digits = extractDigits(match)
            return Int(digits).map(numberToWords) ?? match
        }

        result = result.split(whereSeparator: \.isWhitespace).joined(separator: " ")
        return replaceChars(result, map: charMap)
    }

    private static func useChinese(_ text: String) -> Bool {
        hasChinese(text) || !hasAlpha(text) || isEmail(text) || hasPinyin(text)
    }

    private static func hasChinese(_ text: String) -> Bool {
        text.unicodeScalars.contains { (0x4E00...0x9FFF).contains($0.value) }
    }

    private static func hasAlpha(_ text: String) -> Bool {
        text.unicodeScalars.contains {
            (0x41...0x5A).contains($0.value) || (0x61...0x7A).contains($0.value)
        }
    }

    private static func isEmail(_ text: String) -> Bool {
        firstMatch(in: text, pattern: #"^[a-zA-Z0-9]+@[a-zA-Z0-9]+\.[a-zA-Z]+$"#) != nil
    }

    private static func hasPinyin(_ text: String) -> Bool {
        firstMatch(
            in: text,
            pattern: #"(?<![a-z])((?:[bpmfdtnlgkhjqxzcsryw]|[zcs]h)?(?:[aeiouüv]|[ae]i|u[aio]|ao|ou|i[aue]|[uüv]e|[uvü]ang?|uai|[aeiuv]n|[aeio]ng|ia[no]|i[ao]ng)|ng|er)([1-5])"#,
            options: [.caseInsensitive]
        ) != nil
    }

    private static func expandContractions(_ text: String) -> String {
        replacingMatches(
            in: text,
            pattern: #"(what|where|who|which|how|t?here|it|s?he|that|this)'s"#,
            options: [.caseInsensitive]
        ) { _, captures in
            guard let prefix = captures.first, !prefix.isEmpty else {
                return "is"
            }
            return "\(prefix) is"
        }
    }

    private static func correctPinyin(_ pinyin: String) -> String {
        guard let first = pinyin.unicodeScalars.first else {
            return pinyin
        }
        let lowerFirst = CharacterSet(charactersIn: "JQXjqx")
        guard lowerFirst.contains(first) else {
            return pinyin
        }
        var scalars = Array(pinyin.unicodeScalars)
        if scalars.count > 1 {
            let second = scalars[1]
            if second == "u" || second == "U" || second == "\u{00FC}" || second == "\u{00DC}" {
                scalars[1] = "v"
            }
        }
        return String(String.UnicodeScalarView(scalars)).uppercased()
    }

    private static func replaceChars(_ text: String, map: [(String, String)]) -> String {
        var output = ""
        var index = text.startIndex
        while index < text.endIndex {
            let suffix = text[index...]
            if let replacement = map.first(where: { suffix.hasPrefix($0.0) }) {
                output += replacement.1
                index = text.index(index, offsetBy: replacement.0.count)
            } else {
                output.append(text[index])
                index = text.index(after: index)
            }
        }
        return output
    }

    private static func numberToWords(_ number: Int) -> String {
        let ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        let teens = [
            "ten", "eleven", "twelve", "thirteen", "fourteen",
            "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
        ]
        let tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
        let thousands = ["", "thousand", "million", "billion", "trillion"]

        func convertHundreds(_ value: Int) -> String {
            if value == 0 {
                return ""
            }
            if value < 10 {
                return ones[value]
            }
            if value < 20 {
                return teens[value - 10]
            }
            if value < 100 {
                let suffix = value % 10 == 0 ? "" : " \(ones[value % 10])"
                return tens[value / 10] + suffix
            }
            let suffix = value % 100 == 0 ? "" : " \(convertHundreds(value % 100))"
            return "\(ones[value / 100]) hundred\(suffix)"
        }

        guard number != 0 else {
            return "zero"
        }

        var value = number
        var groupIndex = 0
        var groups: [String] = []
        while value > 0, groupIndex < thousands.count {
            let group = value % 1000
            if group != 0 {
                let label = thousands[groupIndex]
                let suffix = label.isEmpty ? "" : " \(label)"
                groups.append(convertHundreds(group) + suffix)
            }
            value /= 1000
            groupIndex += 1
        }
        return groups.reversed().joined(separator: " ")
    }

    private static func extractDigits(_ text: String) -> String {
        String(text.filter(\.isNumber))
    }

    private static func trimTrailingWhitespace(_ text: String) -> String {
        var result = text
        while result.last?.isWhitespace == true {
            result.removeLast()
        }
        return result
    }

    private static func firstMatch(
        in text: String,
        pattern: String,
        options: NSRegularExpression.Options = []
    ) -> NSTextCheckingResult? {
        guard let regex = try? NSRegularExpression(pattern: pattern, options: options) else {
            return nil
        }
        return regex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text))
    }

    private static func replacingMatches(
        in text: String,
        pattern: String,
        options: NSRegularExpression.Options = [],
        transform: (String, [String]) -> String
    ) -> String {
        guard let regex = try? NSRegularExpression(pattern: pattern, options: options) else {
            return text
        }
        let matches = regex.matches(in: text, range: NSRange(text.startIndex..., in: text))
        guard !matches.isEmpty else {
            return text
        }

        let source = text as NSString
        var output = text
        for match in matches.reversed() {
            guard let range = Range(match.range, in: output) else {
                continue
            }
            let captures = (1..<match.numberOfRanges).map { index -> String in
                let range = match.range(at: index)
                return range.location == NSNotFound ? "" : source.substring(with: range)
            }
            output.replaceSubrange(range, with: transform(source.substring(with: match.range), captures))
        }
        return output
    }

    private static func isCJK(_ value: UInt32) -> Bool {
        (0x1100...0x11FF).contains(value)
            || (0x2E80...0xA4CF).contains(value)
            || (0xA840...0xD7AF).contains(value)
            || (0xF900...0xFAFF).contains(value)
            || (0xFE30...0xFE4F).contains(value)
            || (0xFF65...0xFFDC).contains(value)
            || (0x20000...0x2FFFF).contains(value)
    }
}

public final class IndexTTSModel: SpeechGenerationModel, @unchecked Sendable {
    static let additionalDownloadPatterns = ["*.model"]

    public let config: IndexTTSConfig
    public let core: IndexTTSCore
    public let vocoder: IndexTTSBigVGANConditioning?
    public let modelDirectory: URL?
    public private(set) var tokenizer: SentencePieceTokenizer?

    public var sampleRate: Int { config.sampleRate }
    public var defaultGenerationParameters: GenerateParameters {
        GenerateParameters(maxTokens: min(5000, config.gpt.maxMelTokens), temperature: 0.8, topP: 1.0, topK: 30)
    }

    public init(
        config: IndexTTSConfig,
        core: IndexTTSCore? = nil,
        vocoder: IndexTTSBigVGANConditioning? = nil,
        tokenizer: SentencePieceTokenizer? = nil,
        modelDirectory: URL? = nil
    ) {
        self.config = config
        self.core = core ?? IndexTTSCore(config: config)
        self.vocoder = vocoder
        self.tokenizer = tokenizer
        self.modelDirectory = modelDirectory
    }

    public func encodeText(_ text: String) throws -> [Int] {
        guard let tokenizer else {
            throw IndexTTSError.missingTokenizer(modelDirectory ?? URL(fileURLWithPath: "."))
        }
        let normalized = IndexTTSTextNormalizer.tokenizeByCJKChar(
            IndexTTSTextNormalizer.normalize(text)
        )
        return tokenizer.encodeWithByteFallback(normalized)
    }

    public func prepareInputEmbedding(
        textTokenIDs: [Int],
        conditioningLatents: MLXArray
    ) throws -> IndexTTSPreparedEmbedding {
        try core.prepareInputEmbedding(textTokenIDs: textTokenIDs, conditioningLatents: conditioningLatents)
    }

    public func referenceFeatures(
        from audio: MLXArray,
        sampleRate: Int,
        nFft: Int = 1024,
        hopLength: Int = 256
    ) throws -> MLXArray {
        guard sampleRate > 0 else {
            throw IndexTTSError.invalidInput("sampleRate must be positive.")
        }
        var mono = try Self.monoReferenceAudio(audio)
        if sampleRate != config.sampleRate {
            mono = try resampleAudio(mono, from: sampleRate, to: config.sampleRate)
        }
        let features = Self.indexTTSLogMelSpectrogram(
            audio: mono.asType(.float32),
            sampleRate: config.sampleRate,
            nFft: nFft,
            hopLength: hopLength,
            nMels: config.gpt.conditionModule.inputSize
        ).asType(.float32)
        guard features.ndim == 2, features.dim(1) == config.gpt.conditionModule.inputSize else {
            throw IndexTTSError.invalidInput(
                "Reference features must have shape [frames, \(config.gpt.conditionModule.inputSize)]; got \(features.shape)."
            )
        }
        return features.expandedDimensions(axis: 0)
    }

    public func generateMelTokens(
        textTokenIDs: [Int],
        conditioningLatents: MLXArray,
        maxTokens: Int,
        temperature: Float = 0,
        topP: Float = 1.0,
        topK: Int = 0,
        minP: Float = 0
    ) throws -> IndexTTSMelGeneration {
        try core.generateMelTokens(
            textTokenIDs: textTokenIDs,
            conditioningLatents: conditioningLatents,
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP,
            topK: topK,
            minP: minP
        )
    }

    public func generateWaveform(
        textTokenIDs: [Int],
        referenceFeatures: MLXArray,
        maxTokens: Int,
        temperature: Float = 0,
        topP: Float = 1.0,
        topK: Int = 0,
        minP: Float = 0
    ) throws -> MLXArray {
        guard maxTokens > 0 else {
            throw IndexTTSError.invalidInput("maxTokens must be positive for IndexTTS waveform generation.")
        }
        let conditioning = try core.getConditioning(referenceFeatures: referenceFeatures)
        let melGeneration = try generateMelTokens(
            textTokenIDs: textTokenIDs,
            conditioningLatents: conditioning,
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP,
            topK: topK,
            minP: minP
        )
        return try decodeWaveform(
            latentStates: melGeneration.latentStates,
            referenceFeatures: referenceFeatures
        )
    }

    public func decodeWaveform(
        latentStates: MLXArray,
        speakerEmbedding: MLXArray
    ) throws -> MLXArray {
        guard let vocoder else {
            throw IndexTTSError.unsupportedFullPipeline(
                "IndexTTS BigVGAN conditioning weights are not loaded."
            )
        }
        return try vocoder(latentStates: latentStates, speakerEmbedding: speakerEmbedding)
    }

    public func decodeWaveform(
        latentStates: MLXArray,
        referenceFeatures: MLXArray
    ) throws -> MLXArray {
        guard let vocoder else {
            throw IndexTTSError.unsupportedFullPipeline(
                "IndexTTS BigVGAN conditioning weights are not loaded."
            )
        }
        return try vocoder(latentStates: latentStates, referenceFeatures: referenceFeatures)
    }

    public func generate(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray {
        _ = voice
        _ = refText
        _ = language
        guard let refAudio else {
            throw IndexTTSError.invalidInput(
                "IndexTTS generation requires reference audio for conditioning."
            )
        }
        let maxTokens = generationParameters.maxTokens ?? min(5000, config.gpt.maxMelTokens)
        let textTokenIDs = try encodeText(text)
        let features = try referenceFeatures(
            from: refAudio,
            sampleRate: config.sampleRate
        )
        return try generateWaveform(
            textTokenIDs: textTokenIDs,
            referenceFeatures: features,
            maxTokens: maxTokens,
            temperature: generationParameters.temperature,
            topP: generationParameters.topP,
            topK: generationParameters.topK,
            minP: generationParameters.minP
        )
    }

    public func generateStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        let (stream, continuation) = AsyncThrowingStream<AudioGeneration, Error>.makeStream()
        let task = Task { @Sendable [weak self] in
            guard let self else {
                continuation.finish(throwing: AudioGenerationError.modelNotInitialized("IndexTTS model deallocated"))
                return
            }
            do {
                let audio = try await self.generate(
                    text: text,
                    voice: voice,
                    refAudio: refAudio,
                    refText: refText,
                    language: language,
                    generationParameters: generationParameters
                )
                continuation.yield(.audio(audio))
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        continuation.onTermination = { @Sendable _ in task.cancel() }
        return stream
    }

    public func generateStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters,
        streamingInterval: Double
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        _ = streamingInterval
        return generateStream(
            text: text,
            voice: voice,
            refAudio: refAudio,
            refText: refText,
            language: language,
            generationParameters: generationParameters
        )
    }

    public static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let rawLayoutNeedsFixups = weights.keys.contains { $0.contains("num_batches_tracked") }
        let unsupportedPrefixes = [
            "bigvgan.",
            "ups.",
            "speaker_encoder.",
            "resblocks.",
            "conv_pre.",
            "conv_post.",
            "conds.",
            "cond_layer.",
            "activation_post.",
        ]
        var sanitized: [String: MLXArray] = [:]
        sanitized.reserveCapacity(weights.count)

        for (originalKey, originalValue) in weights {
            if originalKey.contains("num_batches_tracked") || originalKey.contains("pos_enc") {
                continue
            }
            if unsupportedPrefixes.contains(where: { originalKey.hasPrefix($0) }) {
                continue
            }

            var key = originalKey
            var value = originalValue

            if key.hasPrefix("model.") {
                key.removeFirst("model.".count)
            }
            if key.hasPrefix("indextts.") {
                key.removeFirst("indextts.".count)
            }

            if unsupportedPrefixes.contains(where: { key.hasPrefix($0) }) {
                continue
            }

            if key == "perceiver_encoder.norm.gamma" {
                key = "perceiver_encoder.norm.weight"
            } else if key == "perceiver_encoder.norm.beta" {
                continue
            }

            if rawLayoutNeedsFixups {
                if isGPT2Conv1DWeight(key), value.ndim == 2 {
                    value = value.transposed(1, 0)
                } else if key.contains("conv"), value.ndim == 3 {
                    value = value.transposed(0, 2, 1)
                } else if key.contains("conv"), value.ndim == 4 {
                    value = value.transposed(0, 2, 3, 1)
                }

                if let mappedKey = remapRawConformerKey(key) {
                    sanitized[mappedKey] = value
                    continue
                }
            }

            if let remapped = remapPerceiverLayerWeight(key: key, value: value) {
                for (mappedKey, mappedValue) in remapped {
                    sanitized[mappedKey] = mappedValue
                }
                continue
            }

            sanitized[key] = value
        }

        return sanitized
    }

    public static func fromPretrained(_ modelRepo: String, cache: HubCache = .default) async throws -> IndexTTSModel {
        guard let repoID = Repo.ID(rawValue: modelRepo) else {
            throw IndexTTSError.invalidRepositoryID(modelRepo)
        }
        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: ".safetensors",
            additionalMatchingPatterns: additionalDownloadPatterns,
            cache: cache
        )
        return try await fromModelDirectory(modelDir, cache: cache)
    }

    public static func fromModelDirectory(_ modelDir: URL, cache: HubCache = .default) async throws -> IndexTTSModel {
        let configData = try Data(contentsOf: modelDir.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(IndexTTSConfig.self, from: configData)
        try validateSupportedCore(config)
        let weights = try indexTTSLoadWeights(from: modelDir)
        let vocoder = try loadVocoderIfPresent(config: config.bigvgan, weights: weights)
        let model = IndexTTSModel(
            config: config,
            vocoder: vocoder,
            tokenizer: await loadTokenizer(from: modelDir, tokenizerName: config.tokenizerName, cache: cache),
            modelDirectory: modelDir
        )
        try model.core.update(
            parameters: ModuleParameters.unflattened(sanitize(weights: weights)),
            verify: .all
        )
        model.core.train(false)
        eval(model.core.parameters())
        return model
    }

    private static func validateSupportedCore(_ config: IndexTTSConfig) throws {
        guard config.gpt.useMelCodesAsInput else {
            throw IndexTTSError.unsupportedFullPipeline(
                "IndexTTS Swift currently supports use_mel_codes_as_input=true only."
            )
        }
        guard config.gpt.conditionType == "conformer_perceiver" else {
            throw IndexTTSError.unsupportedFullPipeline(
                "IndexTTS Swift currently supports condition_type=conformer_perceiver only."
            )
        }
    }

    private static func loadVocoderIfPresent(
        config: IndexTTSBigVGANConditioningConfig,
        weights: [String: MLXArray]
    ) throws -> IndexTTSBigVGANConditioning? {
        let vocoder = IndexTTSBigVGANConditioning(config: config)
        let sanitized = vocoder.sanitize(weights: weights)
        let requiredKeys = Set(vocoder.parameters().flattened().map(\.0))
        guard requiredKeys.isSubset(of: Set(sanitized.keys)) else {
            return nil
        }
        try vocoder.update(
            parameters: ModuleParameters.unflattened(sanitized),
            verify: .all
        )
        vocoder.train(false)
        eval(vocoder.parameters())
        return vocoder
    }

    static func resolveTokenizerModelURL(
        from modelDir: URL,
        tokenizerName: String,
        cache: HubCache = .default
    ) -> URL? {
        let fileManager = FileManager.default
        let localTokenizer = modelDir.appendingPathComponent("tokenizer.model")
        if fileManager.fileExists(atPath: localTokenizer.path) {
            return localTokenizer
        }

        guard !tokenizerName.isEmpty else { return nil }
        let tokenizerPath = (tokenizerName as NSString).expandingTildeInPath
        let tokenizerURL = URL(fileURLWithPath: tokenizerPath)
        if fileManager.fileExists(atPath: tokenizerURL.path), tokenizerURL.lastPathComponent == "tokenizer.model" {
            return tokenizerURL
        }

        let nestedTokenizerURL = tokenizerURL.appendingPathComponent("tokenizer.model")
        if fileManager.fileExists(atPath: nestedTokenizerURL.path) {
            return nestedTokenizerURL
        }

        guard let repoID = Repo.ID(rawValue: tokenizerName) else { return nil }
        let modelSubdir = repoID.description.replacingOccurrences(of: "/", with: "_")
        let customCacheTokenizer = cache.cacheDirectory
            .appendingPathComponent("mlx-audio")
            .appendingPathComponent(modelSubdir)
            .appendingPathComponent("tokenizer.model")
        if fileManager.fileExists(atPath: customCacheTokenizer.path) {
            return customCacheTokenizer
        }

        return cachedHubSnapshotTokenizer(repoID: repoID, cache: cache)
    }

    private static func loadTokenizer(
        from modelDir: URL,
        tokenizerName: String,
        cache: HubCache
    ) async -> SentencePieceTokenizer? {
        if let tokenizerURL = resolveTokenizerModelURL(from: modelDir, tokenizerName: tokenizerName, cache: cache) {
            return try? SentencePieceTokenizer.from(sentencePieceModelURL: tokenizerURL)
        }

        guard let repoID = Repo.ID(rawValue: tokenizerName) else { return nil }
        do {
            let tokenizerDir = try await ModelUtils.resolveOrDownloadModel(
                repoID: repoID,
                requiredExtension: ".model",
                additionalMatchingPatterns: ["*.model"],
                cache: cache
            )
            let tokenizerURL = tokenizerDir.appendingPathComponent("tokenizer.model")
            guard FileManager.default.fileExists(atPath: tokenizerURL.path) else { return nil }
            return try? SentencePieceTokenizer.from(sentencePieceModelURL: tokenizerURL)
        } catch {
            return nil
        }
    }

    private static func cachedHubSnapshotTokenizer(repoID: Repo.ID, cache: HubCache) -> URL? {
        let fileManager = FileManager.default
        let repoDir = cache.repoDirectory(repo: repoID, kind: .model)
        let snapshotsDir = repoDir.appendingPathComponent("snapshots")

        let mainRef = repoDir.appendingPathComponent("refs").appendingPathComponent("main")
        if let revision = try? String(contentsOf: mainRef, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !revision.isEmpty
        {
            let tokenizerURL = snapshotsDir
                .appendingPathComponent(revision)
                .appendingPathComponent("tokenizer.model")
            if fileManager.fileExists(atPath: tokenizerURL.path) {
                return tokenizerURL
            }
        }

        guard let snapshots = try? fileManager.contentsOfDirectory(
            at: snapshotsDir,
            includingPropertiesForKeys: nil
        ) else {
            return nil
        }

        for snapshot in snapshots.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            let tokenizerURL = snapshot.appendingPathComponent("tokenizer.model")
            if fileManager.fileExists(atPath: tokenizerURL.path) {
                return tokenizerURL
            }
        }
        return nil
    }

    private static func monoReferenceAudio(_ audio: MLXArray) throws -> MLXArray {
        let audio = audio.asType(.float32)
        if audio.ndim == 1 {
            return audio
        }
        if audio.ndim == 2 {
            if audio.dim(0) == 1 {
                return audio.squeezed(axis: 0)
            }
            if audio.dim(1) == 1 {
                return audio.squeezed(axis: 1)
            }
            if audio.dim(0) <= 8 && audio.dim(0) < audio.dim(1) {
                return MLX.mean(audio, axis: 0)
            }
            if audio.dim(1) <= 8 && audio.dim(1) < audio.dim(0) {
                return MLX.mean(audio, axis: 1)
            }
        }
        throw IndexTTSError.invalidInput(
            "Reference audio must be mono or stereo waveform data; got shape \(audio.shape)."
        )
    }

    private static func indexTTSLogMelSpectrogram(
        audio: MLXArray,
        sampleRate: Int,
        nFft: Int,
        hopLength: Int,
        nMels: Int
    ) -> MLXArray {
        let freqs = stft(
            audio: audio,
            window: hanningWindow(size: nFft),
            nFft: nFft,
            hopLength: hopLength
        )
        let magnitudes = MLX.abs(freqs)
        let filters = melFilters(
            sampleRate: sampleRate,
            nFft: nFft,
            nMels: nMels,
            norm: nil,
            melScale: .htk
        )
        let melSpec = MLX.matmul(magnitudes, filters)
        return MLX.log(MLX.maximum(melSpec, MLXArray(Float(1e-5))))
    }

    private static func isGPT2Conv1DWeight(_ key: String) -> Bool {
        guard key.hasPrefix("gpt.h."), key.hasSuffix(".weight") else {
            return false
        }
        return key.contains(".attn.c_attn.")
            || key.contains(".attn.c_proj.")
            || key.contains(".mlp.c_fc.")
            || key.contains(".mlp.c_proj.")
    }

    private static func remapRawConformerKey(_ key: String) -> String? {
        guard key.hasPrefix("conditioning_encoder.embed.conv.") else {
            return nil
        }
        let parts = key.split(separator: ".", omittingEmptySubsequences: false).map(String.init)
        guard parts.count >= 5, let rawIndex = Int(parts[3]), rawIndex > 0, rawIndex % 2 == 0 else {
            return nil
        }
        var rewritten = parts
        rewritten[3] = String(rawIndex / 2)
        return rewritten.joined(separator: ".")
    }

    private static func remapPerceiverLayerWeight(key: String, value: MLXArray) -> [String: MLXArray]? {
        guard key.hasPrefix("perceiver_encoder.layers.") else {
            return nil
        }
        if key.contains(".0.to_q.weight") {
            return [key.replacingOccurrences(of: ".0.to_q.weight", with: ".attention.linear_q.weight"): value]
        }
        if key.contains(".0.to_kv.weight") {
            let pieces = value.split(parts: 2, axis: 0)
            return [
                key.replacingOccurrences(of: ".0.to_kv.weight", with: ".attention.linear_k.weight"): pieces[0],
                key.replacingOccurrences(of: ".0.to_kv.weight", with: ".attention.linear_v.weight"): pieces[1],
            ]
        }
        if key.contains(".0.to_out.weight") {
            return [key.replacingOccurrences(of: ".0.to_out.weight", with: ".attention.linear_out.weight"): value]
        }
        if key.contains(".0.linear_q.") {
            return [key.replacingOccurrences(of: ".0.linear_q.", with: ".attention.linear_q."): value]
        }
        if key.contains(".0.linear_k.") {
            return [key.replacingOccurrences(of: ".0.linear_k.", with: ".attention.linear_k."): value]
        }
        if key.contains(".0.linear_v.") {
            return [key.replacingOccurrences(of: ".0.linear_v.", with: ".attention.linear_v."): value]
        }
        if key.contains(".0.linear_out.") {
            return [key.replacingOccurrences(of: ".0.linear_out.", with: ".attention.linear_out."): value]
        }
        if key.contains(".1.0.weight") {
            return [key.replacingOccurrences(of: ".1.0.weight", with: ".feed_forward.w_1.weight"): value]
        }
        if key.contains(".1.2.weight") {
            return [key.replacingOccurrences(of: ".1.2.weight", with: ".feed_forward.w_2.weight"): value]
        }
        if key.contains(".1.0.bias") {
            return [key.replacingOccurrences(of: ".1.0.bias", with: ".feed_forward.w_1.bias"): value]
        }
        if key.contains(".1.2.bias") {
            return [key.replacingOccurrences(of: ".1.2.bias", with: ".feed_forward.w_2.bias"): value]
        }
        if key.contains(".1.w_1.") {
            return [key.replacingOccurrences(of: ".1.w_1.", with: ".feed_forward.w_1."): value]
        }
        if key.contains(".1.w_2.") {
            return [key.replacingOccurrences(of: ".1.w_2.", with: ".feed_forward.w_2."): value]
        }
        return nil
    }
}

private func indexTTSLoadWeights(from directory: URL) throws -> [String: MLXArray] {
    let files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        .filter { $0.pathExtension == "safetensors" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
    guard !files.isEmpty else {
        throw IndexTTSError.missingWeights(directory)
    }

    var weights: [String: MLXArray] = [:]
    for file in files {
        for (key, value) in try MLX.loadArrays(url: file) {
            weights[key] = value
        }
    }
    return weights
}
