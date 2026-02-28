import AVFoundation
import Foundation
@preconcurrency import MLX
import MLXAudioCore
import MLXAudioSTT
import MLXAudioTTS
import MLXLMCommon

enum AppError: Error, LocalizedError, CustomStringConvertible {
    case invalidRepositoryID(String)
    case unsupportedModelType(String?)
    case failedToCreateAudioBuffer
    case failedToAccessAudioBufferData

    var errorDescription: String? {
        description
    }

    var description: String {
        switch self {
        case .invalidRepositoryID(let model):
            "Invalid repository ID: \(model)"
        case .unsupportedModelType(let modelType):
            "Unsupported model type: \(String(describing: modelType))"
        case .failedToCreateAudioBuffer:
            "Failed to create audio buffer"
        case .failedToAccessAudioBufferData:
            "Failed to access audio buffer data"
        }
    }
}

@main
enum App {
    private static let forcedAlignerRepo = "mlx-community/Qwen3-ForcedAligner-0.6B-4bit"

    static func main() async {
        do {
            let args = try CLI.parse()
            if args.benchmark {
                try await benchmark(
                    model: args.model,
                    text: args.text,
                    voice: args.voice,
                    refAudioPath: args.refAudioPath,
                    refText: args.refText,
                    maxTokens: args.maxTokens,
                    temperature: args.temperature,
                    topP: args.topP
                )
            } else {
                try await run(
                    model: args.model,
                    text: args.text,
                    voice: args.voice,
                    outputPath: args.outputPath,
                    refAudioPath: args.refAudioPath,
                    refText: args.refText,
                    maxTokens: args.maxTokens,
                    temperature: args.temperature,
                    topP: args.topP,
                    timestamps: args.timestamps
                )
            }
        } catch {
            fputs("Error: \(error)\n", stderr)
            CLI.printUsage()
            exit(1)
        }
    }

    private static func run(
        model: String,
        text: String,
        voice: String?,
        outputPath: String?,
        refAudioPath: String?,
        refText: String?,
        maxTokens: Int?,
        temperature: Float?,
        topP: Float?,
        timestamps: Bool,
        hfToken: String? = nil
    ) async throws {
        Memory.cacheLimit = 100 * 1024 * 1024

        print("Loading model (\(model))")

        // Check for HF token in environment (macOS) or Info.plist (iOS) as a fallback
        let hfToken: String? = hfToken ?? ProcessInfo.processInfo.environment["HF_TOKEN"] ?? Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String

        let loadedModel: SpeechGenerationModel
        do {
            loadedModel = try await TTS.loadModel(modelRepo: model, hfToken: hfToken)
        } catch let error as TTSModelError {
            switch error {
            case .invalidRepositoryID(let modelRepo):
                throw AppError.invalidRepositoryID(modelRepo)
            case .unsupportedModelType(let modelType):
                throw AppError.unsupportedModelType(modelType)
            }
        }

        print("Generating")
        let started = CFAbsoluteTimeGetCurrent()

        let refAudio: MLXArray?
        if let refAudioPath, !refAudioPath.isEmpty {
            let refAudioURL = resovleURL(path: refAudioPath)
            (_, refAudio) = try loadAudioArray(from: refAudioURL)
        } else {
            refAudio = nil
        }

        var generationParameters = loadedModel.defaultGenerationParameters
        if let maxTokens {
            generationParameters.maxTokens = maxTokens
        }
        if let temperature {
            generationParameters.temperature = temperature
        }
        if let topP {
            generationParameters.topP = topP
        }

        let audioData = try await loadedModel.generate(
            text: text,
            voice: voice,
            refAudio: refAudio,
            refText: refText,
            language: nil,
            generationParameters: generationParameters
        ).asArray(Float.self)

        let outputURL = makeOutputURL(outputPath: outputPath)
        let sampleRate = Double(loadedModel.sampleRate)
        try writeWavFile(samples: audioData, sampleRate: sampleRate, outputURL: outputURL)
        print("Wrote WAV to \(outputURL.path)")

        if timestamps {
            print("Loading forced aligner (\(forcedAlignerRepo))")
            let forcedAligner = try await Qwen3ForcedAlignerModel.fromPretrained(forcedAlignerRepo)
            let alignmentAudio = try resampleAudio(
                MLXArray(audioData),
                from: Int(loadedModel.sampleRate),
                to: 16000
            )
            let aligned = forcedAligner.generate(audio: alignmentAudio, text: text, language: "English")

            print("Timestamps:")
            for item in aligned.items {
                print(
                    String(
                        format: "  [%.3fs - %.3fs] %@",
                        item.startTime,
                        item.endTime,
                        item.text
                    )
                )
            }
        }

        print(String(format: "Finished generation in %0.2fs", CFAbsoluteTimeGetCurrent() - started))
        print("Memory usage:\n\(Memory.snapshot())")

        let elapsed = CFAbsoluteTimeGetCurrent() - started
        print(String(format: "Done. Elapsed: %.2fs", elapsed))
    }

    // MARK: - Streaming TTFB Benchmark

    private static func benchmark(
        model: String,
        text: String,
        voice: String?,
        refAudioPath: String?,
        refText: String?,
        maxTokens: Int?,
        temperature: Float?,
        topP: Float?,
        hfToken: String? = nil
    ) async throws {
        Memory.cacheLimit = 100 * 1024 * 1024

        let hfToken: String? = hfToken ?? ProcessInfo.processInfo.environment["HF_TOKEN"]
            ?? Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String

        // Phase 1: Load model
        print("Phase 1: Loading model (\(model))")
        let loadStart = CFAbsoluteTimeGetCurrent()
        let loadedModel: SpeechGenerationModel
        do {
            loadedModel = try await TTS.loadModel(modelRepo: model, hfToken: hfToken)
        } catch let error as TTSModelError {
            switch error {
            case .invalidRepositoryID(let r): throw AppError.invalidRepositoryID(r)
            case .unsupportedModelType(let t): throw AppError.unsupportedModelType(t)
            }
        }
        let loadTime = CFAbsoluteTimeGetCurrent() - loadStart
        print(String(format: "Model loaded in %.3fs", loadTime))

        let refAudio: MLXArray?
        if let refAudioPath, !refAudioPath.isEmpty {
            let refAudioURL = resovleURL(path: refAudioPath)
            (_, refAudio) = try loadAudioArray(from: refAudioURL)
        } else {
            refAudio = nil
        }

        var generationParameters = loadedModel.defaultGenerationParameters
        if let maxTokens { generationParameters.maxTokens = maxTokens }
        if let temperature { generationParameters.temperature = temperature }
        if let topP { generationParameters.topP = topP }

        // Phase 2: Warmup (JIT/shader compilation)
        print("Phase 2: Warmup run...")
        var warmup = loadedModel.defaultGenerationParameters
        warmup.maxTokens = 3
        let _ = try await loadedModel.generate(
            text: "Hi",
            voice: voice,
            refAudio: refAudio,
            refText: refText,
            language: nil,
            generationParameters: warmup
        )
        Memory.clearCache()
        print("Warmup complete")

        // Phase 3: Streaming TTFB measurement
        print(String(format: "Phase 3: Streaming TTFB for: \"%@\"", text))
        let genStart = CFAbsoluteTimeGetCurrent()
        var timeToFirstToken: Double?
        var timeToFirstAudio: Double?
        var tokenCount = 0
        var audioChunkCount = 0
        var totalAudioSamples = 0
        var genInfo: AudioGenerationInfo?

        let stream = loadedModel.generateStream(
            text: text,
            voice: voice,
            refAudio: refAudio,
            refText: refText,
            language: nil,
            generationParameters: generationParameters
        )

        for try await event in stream {
            switch event {
            case .token(_):
                tokenCount += 1
                if timeToFirstToken == nil {
                    timeToFirstToken = CFAbsoluteTimeGetCurrent() - genStart
                }
            case .audio(let audio):
                audioChunkCount += 1
                let samples = audio.dim(0)
                totalAudioSamples += samples
                if timeToFirstAudio == nil {
                    timeToFirstAudio = CFAbsoluteTimeGetCurrent() - genStart
                }
                let chunkMs = Double(samples) / Double(loadedModel.sampleRate) * 1000
                print(String(format: "  Chunk %d: %d samples (%.0fms audio) at t=%.3fs",
                             audioChunkCount, samples, chunkMs,
                             CFAbsoluteTimeGetCurrent() - genStart))
            case .info(let info):
                genInfo = info
            }
        }

        let totalTime = CFAbsoluteTimeGetCurrent() - genStart
        let audioDuration = Double(totalAudioSamples) / Double(loadedModel.sampleRate)

        // Results
        print("\n========== TTFB Benchmark Results ==========")
        print(String(format: "  Model load time:       %.3fs", loadTime))
        if let ttft = timeToFirstToken {
            print(String(format: "  Time to first token:   %.3fs (%.1fms)", ttft, ttft * 1000))
        }
        if let ttfa = timeToFirstAudio {
            print(String(format: "  Time to first audio:   %.3fs (%.1fms) << TTFB", ttfa, ttfa * 1000))
        }
        print(String(format: "  Total generation time: %.3fs", totalTime))
        print(String(format: "  Tokens generated:      %d", tokenCount))
        print(String(format: "  Audio chunks:          %d", audioChunkCount))
        print(String(format: "  Total audio duration:  %.3fs", audioDuration))
        if let info = genInfo {
            print(String(format: "  Tokens/sec:            %.1f", info.tokensPerSecond))
            print(String(format: "  Peak memory:           %.2f GB", info.peakMemoryUsage))
        }
        print("=============================================")

        print("\nStreaming config:")
        print("  First chunk:       1 token (~80ms audio) — emit ASAP")
        print("  Subsequent chunks: 8 tokens (~640ms audio)")
        print("  Context overlap:   10 tokens")
        print("\nExpected TTFB estimates (6-bit, M-series):")
        print("  Prefill:           15-30ms")
        print("  First token gen:   ~73ms (at ~13.7 tok/s)")
        print("  Speech decode:     ~5-15ms (direct decode, 1 token)")
        print("  Expected TTFB:     ~100-120ms")
        print("  Qwen CUDA claim:   97ms (12Hz-0.6B), 101ms (12Hz-1.7B)")

        if let ttfa = timeToFirstAudio {
            let ms = ttfa * 1000
            if ms < 101 {
                print(String(format: "\n  RESULT: %.1fms — beats Qwen's 101ms CUDA latency!", ms))
            } else if ms < 150 {
                print(String(format: "\n  RESULT: %.1fms — competitive with Qwen's 101ms CUDA", ms))
            } else {
                print(String(format: "\n  RESULT: %.1fms — above target", ms))
            }
        }
    }

    private static func makeOutputURL(outputPath: String?) -> URL {
        let outputName = outputPath?.isEmpty == false ? outputPath! : "output.wav"
        if outputName.hasPrefix("/") {
            return URL(fileURLWithPath: outputName)
        }
        return URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(outputName)
    }

    private static func resovleURL(path: String) -> URL {
        if path.hasPrefix("/") {
            return URL(fileURLWithPath: path)
        }
        return URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(path)
    }

    private static func writeWavFile(samples: [Float], sampleRate: Double, outputURL: URL) throws {
        let frameCount = AVAudioFrameCount(samples.count)
        guard let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1),
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            throw AppError.failedToCreateAudioBuffer
        }
        buffer.frameLength = frameCount
        guard let channelData = buffer.floatChannelData else {
            throw AppError.failedToAccessAudioBufferData
        }
        for i in 0 ..< samples.count {
            channelData[0][i] = samples[i]
        }
        let audioFile = try AVAudioFile(forWriting: outputURL, settings: format.settings)
        try audioFile.write(from: buffer)
    }

}

// MARK: -

enum CLIError: Error, CustomStringConvertible {
    case missingValue(String)
    case unknownOption(String)
    case invalidValue(String, String)

    var description: String {
        switch self {
        case .missingValue(let k): "Missing value for \(k)"
        case .unknownOption(let k): "Unknown option \(k)"
        case .invalidValue(let k, let v): "Invalid value for \(k): \(v)"
        }
    }
}

struct CLI {
    let model: String
    let text: String
    let voice: String?
    let outputPath: String?
    let refAudioPath: String?
    let refText: String?
    let maxTokens: Int?
    let temperature: Float?
    let topP: Float?
    let timestamps: Bool
    let benchmark: Bool

    static func parse() throws -> CLI {
        var text: String?
        var voice: String? = nil
        var outputPath: String? = nil
        var model = "Marvis-AI/marvis-tts-250m-v0.2-MLX-8bit"
        var refAudioPath: String? = nil
        var refText: String? = nil
        var maxTokens: Int? = nil
        var temperature: Float? = nil
        var topP: Float? = nil
        var timestamps = false
        var benchmark = false

        var it = CommandLine.arguments.dropFirst().makeIterator()
        while let arg = it.next() {
            switch arg {
            case "--text", "-t":
                guard let v = it.next() else { throw CLIError.missingValue(arg) }
                text = v
            case "--voice", "-v":
                guard let v = it.next() else { throw CLIError.missingValue(arg) }
                voice = v
            case "--model":
                guard let v = it.next() else { throw CLIError.missingValue(arg) }
                model = v
            case "--output", "-o":
                guard let v = it.next() else { throw CLIError.missingValue(arg) }
                outputPath = v
            case "--ref_audio":
                guard let v = it.next() else { throw CLIError.missingValue(arg) }
                refAudioPath = v
            case "--ref_text":
                guard let v = it.next() else { throw CLIError.missingValue(arg) }
                refText = v
            case "--max_tokens":
                guard let v = it.next() else { throw CLIError.missingValue(arg) }
                guard let value = Int(v) else { throw CLIError.invalidValue(arg, v) }
                maxTokens = value
            case "--temperature":
                guard let v = it.next() else { throw CLIError.missingValue(arg) }
                guard let value = Float(v) else { throw CLIError.invalidValue(arg, v) }
                temperature = value
            case "--top_p":
                guard let v = it.next() else { throw CLIError.missingValue(arg) }
                guard let value = Float(v) else { throw CLIError.invalidValue(arg, v) }
                topP = value
            case "--timestamps":
                timestamps = true
            case "--benchmark":
                benchmark = true
            case "--help", "-h":
                printUsage()
                exit(0)
            default:
                if text == nil, !arg.hasPrefix("-") {
                    text = arg
                } else {
                    throw CLIError.unknownOption(arg)
                }
            }
        }

        guard let finalText = text, !finalText.isEmpty else {
            throw CLIError.missingValue("--text")
        }

        return CLI(
            model: model,
            text: finalText,
            voice: voice,
            outputPath: outputPath,
            refAudioPath: refAudioPath,
            refText: refText,
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP,
            timestamps: timestamps,
            benchmark: benchmark
        )
    }

    static func printUsage() {
        let exe = (CommandLine.arguments.first as NSString?)?.lastPathComponent ?? "marvis-tts-cli"
        print("""
        Usage:
          \(exe) --text "Hello world" [--voice conversational_b] [--model <hf-repo>] [--output <path>] [--ref_audio <path>] [--ref_text <string>] [--max_tokens <int>] [--temperature <float>] [--top_p <float>] [--timestamps]

        Options:
          -t, --text <string>           Text to synthesize (required if not passed as trailing arg)
          -v, --voice <name>            Voice id
              --model <repo>            HF repo id. Default: Marvis-AI/marvis-tts-250m-v0.2-MLX-8bit
          -o, --output <path>           Output WAV path. Default: ./output.wav
              --ref_audio <path>       Path to reference audio
              --ref_text <string>      Caption for reference audio
              --max_tokens <int>       Maximum number of tokens to generate (overrides model default)
              --temperature <float>    Sampling temperature (overrides model default)
              --top_p <float>          Top-p sampling (overrides model default)
              --timestamps             Emit word timestamps using mlx-community/Qwen3-ForcedAligner-0.6B-4bit
          -h, --help                    Show this help
        """)
    }
}
