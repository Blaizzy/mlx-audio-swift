import Foundation
@preconcurrency import MLX
import MLXAudioCore
import MLXAudioSTT

enum AppError: Error, LocalizedError, CustomStringConvertible {
    case inputFileNotFound(String)
    case unsupportedModelRepo(String)
    case missingTextForForcedAlignment
    case streamUnsupportedForForcedAligner
    case invalidGenKwargs(String)

    var errorDescription: String? { description }

    var description: String {
        switch self {
        case .inputFileNotFound(let path):
            "Input audio file not found: \(path)"
        case .unsupportedModelRepo(let repo):
<<<<<<< Updated upstream
            "Unsupported STT model repo: \(repo). Expected GLMASR, Qwen3ASR, or Qwen3ForcedAligner."
=======
            "Unsupported STT model repo: \(repo). Expected GLMASR, Qwen3ASR, VoxtralRealtime, Parakeet, or Qwen3ForcedAligner."
>>>>>>> Stashed changes
        case .missingTextForForcedAlignment:
            "--text is required when using a forced aligner model."
        case .streamUnsupportedForForcedAligner:
            "--stream is not supported for forced aligner models."
        case .invalidGenKwargs(let value):
            "Invalid --gen-kwargs JSON: \(value)"
        }
    }
}

enum CLIError: Error, CustomStringConvertible {
    case missingValue(String)
    case unknownOption(String)
    case invalidValue(String, String)

    var description: String {
        switch self {
        case .missingValue(let key):
            "Missing value for \(key)"
        case .unknownOption(let key):
            "Unknown option \(key)"
        case .invalidValue(let key, let value):
            "Invalid value for \(key): \(value)"
        }
    }
}

enum OutputFormat: String {
    case txt
    case srt
    case vtt
    case json
}

private struct Segment {
    let text: String
    let start: Double
    let end: Double
}

private enum LoadedModel {
    case stt(any STTGenerationModel)
    case forcedAligner(Qwen3ForcedAlignerModel)
}

private struct Options {
    var model: String = "mlx-community/Qwen3-ASR-0.6B-4bit"
    var audio: String?
    var outputPath: String?
    var format: OutputFormat = .txt
    var verbose = false
    var maxTokens = 2048
    var language = "en"
    var chunkDuration: Float = 30.0
    var frameThreshold = 25
    var stream = false
    var context: String? = nil
    var prefillStepSize = 2048
    var genKwargsRaw: String? = nil
    var text = ""

    var temperature: Float? = nil
    var topP: Float? = nil
    var topK: Int? = nil
    var minChunkDuration: Float? = nil

    static func parse() throws -> Options {
        var options = Options()

        var it = CommandLine.arguments.dropFirst().makeIterator()
        while let arg = it.next() {
            switch arg {
            case "--model":
                guard let v = it.next() else { throw CLIError.missingValue(arg) }
                options.model = v
            case "--audio":
                guard let v = it.next() else { throw CLIError.missingValue(arg) }
                options.audio = v
            case "--output-path":
                guard let v = it.next() else { throw CLIError.missingValue(arg) }
                options.outputPath = v
            case "--format":
                guard let v = it.next() else { throw CLIError.missingValue(arg) }
                guard let format = OutputFormat(rawValue: v.lowercased()) else {
                    throw CLIError.invalidValue(arg, v)
                }
                options.format = format
            case "--verbose":
                options.verbose = true
            case "--max-tokens":
                guard let v = it.next() else { throw CLIError.missingValue(arg) }
                guard let value = Int(v) else { throw CLIError.invalidValue(arg, v) }
                options.maxTokens = value
            case "--language":
                guard let v = it.next() else { throw CLIError.missingValue(arg) }
                options.language = v
            case "--chunk-duration":
                guard let v = it.next() else { throw CLIError.missingValue(arg) }
                guard let value = Float(v) else { throw CLIError.invalidValue(arg, v) }
                options.chunkDuration = value
            case "--frame-threshold":
                guard let v = it.next() else { throw CLIError.missingValue(arg) }
                guard let value = Int(v) else { throw CLIError.invalidValue(arg, v) }
                options.frameThreshold = value
            case "--stream":
                options.stream = true
            case "--context":
                guard let v = it.next() else { throw CLIError.missingValue(arg) }
                options.context = v
            case "--prefill-step-size":
                guard let v = it.next() else { throw CLIError.missingValue(arg) }
                guard let value = Int(v) else { throw CLIError.invalidValue(arg, v) }
                options.prefillStepSize = value
            case "--gen-kwargs":
                guard let v = it.next() else { throw CLIError.missingValue(arg) }
                options.genKwargsRaw = v
            case "--text":
                guard let v = it.next() else { throw CLIError.missingValue(arg) }
                options.text = v
            case "--help", "-h":
                printUsage()
                exit(0)
            default:
                if options.audio == nil, !arg.hasPrefix("-") {
                    options.audio = arg
                } else {
                    throw CLIError.unknownOption(arg)
                }
            }
        }

        guard options.audio?.isEmpty == false else {
            throw CLIError.missingValue("--audio")
        }
        guard options.outputPath?.isEmpty == false else {
            throw CLIError.missingValue("--output-path")
        }

        try options.applyGenKwargs()
        return options
    }

    mutating func applyGenKwargs() throws {
        guard let genKwargsRaw, !genKwargsRaw.isEmpty else { return }
        guard let data = genKwargsRaw.data(using: .utf8) else {
            throw AppError.invalidGenKwargs(genKwargsRaw)
        }
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw AppError.invalidGenKwargs(genKwargsRaw)
        }

        for (key, value) in json {
            switch key {
            case "max_tokens":
                if let v = asInt(value) { maxTokens = v }
            case "language":
                if let v = value as? String { language = v }
            case "chunk_duration":
                if let v = asFloat(value) { chunkDuration = v }
            case "min_chunk_duration":
                if let v = asFloat(value) { minChunkDuration = v }
            case "temperature":
                if let v = asFloat(value) { temperature = v }
            case "top_p":
                if let v = asFloat(value) { topP = v }
            case "top_k":
                if let v = asInt(value) { topK = v }
            case "stream":
                if let v = value as? Bool { stream = v }
            case "text":
                if let v = value as? String { text = v }
            case "verbose":
                if let v = value as? Bool { verbose = v }
            case "context":
                if let v = value as? String { context = v }
            case "prefill_step_size":
                if let v = asInt(value) { prefillStepSize = v }
            case "frame_threshold":
                if let v = asInt(value) { frameThreshold = v }
            default:
                continue
            }
        }
    }

    static func printUsage() {
        let executable = (CommandLine.arguments.first as NSString?)?.lastPathComponent ?? "mlx-audio-swift-stt"
        print(
            """
            Usage:
              \(executable) --audio <path> --output-path <path> [options]

            Description:
              Generate STT transcriptions from audio using repo-id selected models.

            Options:
              --model <repo>                Model repo id.
                                            Default: mlx-community/Qwen3-ASR-0.6B-4bit
                                            Supported families: Qwen3-ASR, GLM-ASR, Voxtral, Parakeet, Qwen3-ForcedAligner
              --audio <path>                Input audio file (required if not passed as trailing arg)
              --output-path <path>          Output path stem (required). Extension is appended from --format.
              --format <txt|srt|vtt|json>   Output format. Default: txt
              --verbose                     Verbose logging
              --max-tokens <int>            Max generated tokens. Default: 2048
              --language <code|name>        Language hint. Default: en
              --chunk-duration <float>      Chunk duration seconds. Default: 30.0
              --frame-threshold <int>       Accepted for compatibility (currently unused). Default: 25
              --stream                      Stream token output while generating
              --context <text>              Accepted for compatibility (currently unused)
              --prefill-step-size <int>     Accepted for compatibility (currently unused). Default: 2048
              --gen-kwargs <json>           Additional kwargs JSON (e.g. '{"min_chunk_duration":1.0}')
              --text <text>                 Alignment text (required for forced aligner models)
              -h, --help                    Show this help
            """
        )
    }
}

@main
enum App {
    static func main() async {
        do {
            let options = try Options.parse()
            try await run(options: options)
        } catch {
            fputs("Error: \(error)\n", stderr)
            Options.printUsage()
            exit(1)
        }
    }

    private static func run(options: Options) async throws {
        let inputURL = resolveURL(path: options.audio!)
        guard FileManager.default.fileExists(atPath: inputURL.path) else {
            throw AppError.inputFileNotFound(inputURL.path)
        }

        let model = try await loadModel(repo: options.model)
        let (_, audio) = try loadAudioArray(from: inputURL)

        let startTime = CFAbsoluteTimeGetCurrent()

        if options.verbose {
            print("==========")
            print("Audio path: \(inputURL.path)")
            print("Output path: \(options.outputPath!).\(options.format.rawValue)")
            print("Format: \(options.format.rawValue)")
            if options.frameThreshold != 25 {
                print("Warning: --frame-threshold is currently ignored by this CLI.")
            }
            if options.prefillStepSize != 2048 {
                print("Warning: --prefill-step-size is currently ignored by this CLI.")
            }
            if options.context?.isEmpty == false {
                print("Warning: --context is currently ignored by this CLI.")
            }
        }

        let output: STTOutput
        switch model {
        case .stt(let sttModel):
            var params = sttModel.defaultGenerationParameters
            params = STTGenerateParameters(
                maxTokens: options.maxTokens,
                temperature: options.temperature ?? params.temperature,
                topP: options.topP ?? params.topP,
                topK: options.topK ?? params.topK,
                verbose: options.verbose,
                language: normalizeLanguage(options.language),
                chunkDuration: options.chunkDuration,
                minChunkDuration: options.minChunkDuration ?? params.minChunkDuration
            )

            if options.stream {
                output = try await runStreaming(model: sttModel, audio: audio, parameters: params)
            } else {
                output = sttModel.generate(audio: audio, generationParameters: params)
            }

        case .forcedAligner(let aligner):
            if options.stream {
                throw AppError.streamUnsupportedForForcedAligner
            }
            guard !options.text.isEmpty else {
                throw AppError.missingTextForForcedAlignment
            }
            let aligned = aligner.generate(audio: audio, text: options.text, language: normalizeLanguage(options.language))
            let promptTps = aligned.totalTime > 0 ? Double(aligned.promptTokens) / aligned.totalTime : 0
            output = STTOutput(
                text: aligned.text,
                segments: aligned.segments,
                language: options.language,
                promptTokens: aligned.promptTokens,
                generationTokens: 0,
                totalTokens: aligned.promptTokens,
                promptTps: promptTps,
                generationTps: 0,
                totalTime: aligned.totalTime,
                peakMemoryUsage: aligned.peakMemoryUsage
            )
        }

        try writeOutput(output, format: options.format, outputPathStem: options.outputPath!)

        if options.verbose {
            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            print("\n==========")
            print("Saved file to: \(options.outputPath!).\(options.format.rawValue)")
            print(String(format: "Processing time: %.2f seconds", elapsed))
            print(String(format: "Prompt: %d tokens, %.3f tokens-per-sec", output.promptTokens, output.promptTps))
            print(String(format: "Generation: %d tokens, %.3f tokens-per-sec", output.generationTokens, output.generationTps))
            print(String(format: "Peak memory: %.2f GB", output.peakMemoryUsage))
            if !output.text.isEmpty {
                print("Transcription:\n\(output.text.prefix(500))")
            }
        }
    }

    private static func runStreaming(
        model: any STTGenerationModel,
        audio: MLXArray,
        parameters: STTGenerateParameters
    ) async throws -> STTOutput {
        var finalOutput: STTOutput?
        var streamedText = ""
        var emittedToken = false

        for try await event in model.generateStream(audio: audio, generationParameters: parameters) {
            switch event {
            case .token(let token):
                emittedToken = true
                streamedText += token
                print(token, terminator: "")
                fflush(stdout)
            case .info:
                break
            case .result(let output):
                finalOutput = output
            }
        }

        if emittedToken {
            print()
        }

        if let finalOutput {
            return finalOutput
        }

        return STTOutput(text: streamedText.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    private static func loadModel(repo: String) async throws -> LoadedModel {
        let lower = repo.lowercased()

        if lower.contains("forcedalign") || lower.contains("forced-align") {
            return .forcedAligner(try await Qwen3ForcedAlignerModel.fromPretrained(repo))
        }
        if lower.contains("glmasr") || lower.contains("glm-asr") {
            return .stt(try await GLMASRModel.fromPretrained(repo))
        }
        if lower.contains("qwen3-asr") || lower.contains("qwen3_asr") {
            return .stt(try await Qwen3ASRModel.fromPretrained(repo))
        }
<<<<<<< Updated upstream
=======
        if lower.contains("voxtral") {
            return .stt(try await VoxtralRealtimeModel.fromPretrained(repo))
        }
        if lower.contains("parakeet") {
            return .stt(try await ParakeetModel.fromPretrained(repo))
        }
>>>>>>> Stashed changes

        throw AppError.unsupportedModelRepo(repo)
    }

    private static func normalizeLanguage(_ language: String) -> String {
        switch language.lowercased() {
        case "en", "english":
            return "English"
        default:
            return language
        }
    }

    private static func writeOutput(
        _ output: STTOutput,
        format: OutputFormat,
        outputPathStem: String
    ) throws {
        let segments = extractSegments(output)

        switch format {
        case .txt:
            let url = outputURL(stem: outputPathStem, ext: "txt")
            try output.text.write(to: url, atomically: true, encoding: .utf8)
        case .srt:
            guard let segments else {
                fputs("[WARNING] No segments found, saving as plain text.\n", stderr)
                let url = outputURL(stem: outputPathStem, ext: "txt")
                try output.text.write(to: url, atomically: true, encoding: .utf8)
                return
            }
            let url = outputURL(stem: outputPathStem, ext: "srt")
            try renderSRT(segments: segments).write(to: url, atomically: true, encoding: .utf8)
        case .vtt:
            guard let segments else {
                fputs("[WARNING] No segments found, saving as plain text.\n", stderr)
                let url = outputURL(stem: outputPathStem, ext: "txt")
                try output.text.write(to: url, atomically: true, encoding: .utf8)
                return
            }
            let url = outputURL(stem: outputPathStem, ext: "vtt")
            try renderVTT(segments: segments).write(to: url, atomically: true, encoding: .utf8)
        case .json:
            let url = outputURL(stem: outputPathStem, ext: "json")
            let jsonObject: [String: Any] = [
                "text": output.text,
                "segments": (segments ?? []).map {
                    [
                        "text": $0.text,
                        "start": $0.start,
                        "end": $0.end,
                        "duration": $0.end - $0.start,
                    ]
                },
                "language": output.language as Any,
                "prompt_tokens": output.promptTokens,
                "generation_tokens": output.generationTokens,
                "total_tokens": output.totalTokens,
                "prompt_tps": output.promptTps,
                "generation_tps": output.generationTps,
                "total_time": output.totalTime,
                "peak_memory_usage": output.peakMemoryUsage,
            ]
            let data = try JSONSerialization.data(withJSONObject: jsonObject, options: [.prettyPrinted])
            try data.write(to: url)
        }
    }

    private static func extractSegments(_ output: STTOutput) -> [Segment]? {
        guard let raw = output.segments, !raw.isEmpty else { return nil }
        var segments: [Segment] = []
        segments.reserveCapacity(raw.count)

        for item in raw {
            guard let text = item["text"] as? String,
                  let start = asDouble(item["start"]),
                  let end = asDouble(item["end"])
            else {
                continue
            }
            segments.append(Segment(text: text, start: start, end: end))
        }

        return segments.isEmpty ? nil : segments
    }

    private static func renderSRT(segments: [Segment]) -> String {
        var lines: [String] = []
        lines.reserveCapacity(segments.count * 4)

        for (idx, seg) in segments.enumerated() {
            lines.append("\(idx + 1)")
            lines.append("\(formatTimestamp(seg.start, forVTT: false)) --> \(formatTimestamp(seg.end, forVTT: false))")
            lines.append(seg.text)
            lines.append("")
        }

        return lines.joined(separator: "\n")
    }

    private static func renderVTT(segments: [Segment]) -> String {
        var lines: [String] = ["WEBVTT", ""]
        lines.reserveCapacity(2 + segments.count * 4)

        for (idx, seg) in segments.enumerated() {
            lines.append("\(idx + 1)")
            lines.append("\(formatTimestamp(seg.start, forVTT: true)) --> \(formatTimestamp(seg.end, forVTT: true))")
            lines.append(seg.text)
            lines.append("")
        }

        return lines.joined(separator: "\n")
    }

    private static func formatTimestamp(_ seconds: Double, forVTT: Bool) -> String {
        let hours = Int(seconds / 3600)
        let minutes = Int((seconds.truncatingRemainder(dividingBy: 3600)) / 60)
        let secs = seconds.truncatingRemainder(dividingBy: 60)
        let base = String(format: "%02d:%02d:%06.3f", hours, minutes, secs)
        return forVTT ? base : base.replacingOccurrences(of: ".", with: ",")
    }

    private static func outputURL(stem: String, ext: String) -> URL {
        let normalizedStem = (stem as NSString).expandingTildeInPath
        let stemExtension = URL(fileURLWithPath: normalizedStem).pathExtension.lowercased()
        let path = stemExtension == ext.lowercased() || !stemExtension.isEmpty
            ? normalizedStem
            : "\(normalizedStem).\(ext)"
        let url: URL
        if path.hasPrefix("/") {
            url = URL(fileURLWithPath: path)
        } else {
            url = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(path)
        }

        let dir = url.deletingLastPathComponent()
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return url
    }

    private static func resolveURL(path: String) -> URL {
        if path.hasPrefix("/") {
            return URL(fileURLWithPath: path)
        }
        return URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(path)
    }
}

private func asInt(_ value: Any?) -> Int? {
    switch value {
    case let v as Int:
        return v
    case let v as Double:
        return Int(v)
    case let v as Float:
        return Int(v)
    case let v as NSNumber:
        return v.intValue
    case let v as String:
        return Int(v)
    default:
        return nil
    }
}

private func asFloat(_ value: Any?) -> Float? {
    switch value {
    case let v as Float:
        return v
    case let v as Double:
        return Float(v)
    case let v as Int:
        return Float(v)
    case let v as NSNumber:
        return v.floatValue
    case let v as String:
        return Float(v)
    default:
        return nil
    }
}

private func asDouble(_ value: Any?) -> Double? {
    switch value {
    case let v as Double:
        return v
    case let v as Float:
        return Double(v)
    case let v as Int:
        return Double(v)
    case let v as NSNumber:
        return v.doubleValue
    case let v as String:
        return Double(v)
    default:
        return nil
    }
}
