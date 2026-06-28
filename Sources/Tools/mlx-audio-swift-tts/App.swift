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
    case invalidGeneratedAudioShape([Int])
    case mismatchedAudioChannelCount(Int, Int)
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
        case .invalidGeneratedAudioShape(let shape):
            "Invalid generated audio shape: \(shape)"
        case .mismatchedAudioChannelCount(let expected, let actual):
            "Mismatched generated audio channel count: expected \(expected), got \(actual)"
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
                timestamps: args.timestamps,
                benchmark: args.benchmark,
                rawIPA: args.rawIPA,
                language: args.language
            )
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
        benchmark: Bool,
        rawIPA: Bool = false,
        language: String? = nil,
        hfToken: String? = nil
    ) async throws {
        Memory.cacheLimit = 256 * 1024 * 1024

        print("Loading model (\(model))")

        // Check for HF token in environment (macOS) or Info.plist (iOS) as a fallback
        let hfToken: String? = hfToken ?? ProcessInfo.processInfo.environment["HF_TOKEN"] ?? Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String

        let loadedModel: SpeechGenerationModel
        do {
            let processor: TextProcessor? = rawIPA ? PassthroughProcessor() : nil
            loadedModel = try await TTS.loadModel(modelRepo: model, textProcessor: processor, hfToken: hfToken)
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
            (_, refAudio) = try loadAudioArray(from: refAudioURL, sampleRate: loadedModel.sampleRate)
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

        let audioFrames: PCMFrames
        var benchmarkMetrics: BenchmarkMetrics?
        if benchmark {
            let sampleRate = Double(loadedModel.sampleRate)
            let stream = loadedModel.generateStream(
                text: text,
                voice: voice,
                refAudio: refAudio,
                refText: refText,
                language: language,
                generationParameters: generationParameters,
                streamingInterval: 0.32
            )

            var collectedAudio = PCMFrames()
            var totalFrames = 0
            var firstChunkLatency: TimeInterval?
            var generationInfo: AudioGenerationInfo?

            for try await event in stream {
                switch event {
                case .token:
                    break
                case .info(let info):
                    generationInfo = info
                case .audio(let chunk):
                    let chunkFrames = try PCMFrames(audio: chunk)
                    guard chunkFrames.frameCount > 0 else { continue }

                    let now = CFAbsoluteTimeGetCurrent()
                    let chunkLatency = now - started
                    if firstChunkLatency == nil {
                        firstChunkLatency = chunkLatency
                    }

                    try collectedAudio.append(chunkFrames)
                    totalFrames += chunkFrames.frameCount
                }
            }

            audioFrames = collectedAudio
            let elapsed = CFAbsoluteTimeGetCurrent() - started
            let audioDuration = sampleRate > 0 ? Double(totalFrames) / sampleRate : 0
            benchmarkMetrics = BenchmarkMetrics(
                elapsed: elapsed,
                audioDuration: audioDuration,
                firstChunkLatency: firstChunkLatency,
                sampleRate: sampleRate,
                generationInfo: generationInfo
            )
        } else {
            let audio = try await loadedModel.generate(
                text: text,
                voice: voice,
                refAudio: refAudio,
                refText: refText,
                language: language,
                generationParameters: generationParameters
            )
            audioFrames = try PCMFrames(audio: audio)
        }

        print(String(format: "Finished generation in %0.2fs", CFAbsoluteTimeGetCurrent() - started))

        let outputURL = makeOutputURL(outputPath: outputPath)
        let sampleRate = Double(loadedModel.sampleRate)
        try writeWavFile(audioFrames: audioFrames, sampleRate: sampleRate, outputURL: outputURL)
        print("Wrote WAV to \(outputURL.path)")

        if let benchmarkMetrics {
            print("Benchmark:")
            print(String(format: "  Audio duration: %.2fs", benchmarkMetrics.audioDuration))
            if let firstChunkLatency = benchmarkMetrics.firstChunkLatency {
                print(String(format: "  TTFB: %.3fs", firstChunkLatency))
            } else {
                print("  TTFB: n/a")
            }
            if benchmarkMetrics.audioDuration > 0 {
                let rtf = benchmarkMetrics.audioDuration / benchmarkMetrics.elapsed
                print(String(format: "  RTFx: %.3f", rtf))
            } else {
                print("  RTFx: n/a")
            }
            if let info = benchmarkMetrics.generationInfo {
                print(String(format: "  Tokens/s: %.2f", info.tokensPerSecond))
            }
        }

        if timestamps {
            print("Loading forced aligner (\(forcedAlignerRepo))")
            let forcedAligner = try await Qwen3ForcedAlignerModel.fromPretrained(forcedAlignerRepo)
            let alignmentAudio = try resampleAudio(
                MLXArray(audioFrames.monoSamples()),
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

        print("Memory usage:\n\(Memory.snapshot())")

        let elapsed = CFAbsoluteTimeGetCurrent() - started
        print(String(format: "Done. Elapsed: %.2fs", elapsed))
    }

    private struct BenchmarkMetrics {
        let elapsed: TimeInterval
        let audioDuration: TimeInterval
        let firstChunkLatency: TimeInterval?
        let sampleRate: Double
        let generationInfo: AudioGenerationInfo?
    }

    private struct PCMFrames {
        var channels: [[Float]]

        init(channels: [[Float]] = []) {
            self.channels = channels
        }

        init(audio: MLXArray) throws {
            var audio = audio.asType(.float32)
            if audio.ndim == 3 {
                guard audio.dim(0) == 1 else {
                    throw AppError.invalidGeneratedAudioShape(audio.shape)
                }
                audio = audio[0]
            }

            switch audio.ndim {
            case 1:
                self.channels = [audio.asArray(Float.self)]
            case 2:
                let first = audio.dim(0)
                let second = audio.dim(1)
                let values = audio.asArray(Float.self)
                if second <= 8 {
                    self.channels = Self.channelsFromSampleMajor(values: values, frameCount: first, channelCount: second)
                } else if first <= 8 {
                    self.channels = (0 ..< first).map { channel in
                        let start = channel * second
                        return Array(values[start ..< (start + second)])
                    }
                } else {
                    throw AppError.invalidGeneratedAudioShape(audio.shape)
                }
            default:
                throw AppError.invalidGeneratedAudioShape(audio.shape)
            }
        }

        var channelCount: Int {
            channels.count
        }

        var frameCount: Int {
            channels.first?.count ?? 0
        }

        mutating func append(_ other: PCMFrames) throws {
            guard !channels.isEmpty else {
                channels = other.channels
                return
            }
            guard other.channelCount == channelCount else {
                throw AppError.mismatchedAudioChannelCount(channelCount, other.channelCount)
            }
            for index in channels.indices {
                channels[index].append(contentsOf: other.channels[index])
            }
        }

        func monoSamples() -> [Float] {
            guard channelCount > 1 else {
                return channels.first ?? []
            }
            var samples = Array(repeating: Float(0), count: frameCount)
            for frame in 0 ..< frameCount {
                var sum: Float = 0
                for channel in 0 ..< channelCount {
                    sum += channels[channel][frame]
                }
                samples[frame] = sum / Float(channelCount)
            }
            return samples
        }

        private static func channelsFromSampleMajor(
            values: [Float],
            frameCount: Int,
            channelCount: Int
        ) -> [[Float]] {
            var channels = Array(
                repeating: Array(repeating: Float(0), count: frameCount),
                count: channelCount
            )
            for frame in 0 ..< frameCount {
                for channel in 0 ..< channelCount {
                    channels[channel][frame] = values[frame * channelCount + channel]
                }
            }
            return channels
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

    private static func writeWavFile(audioFrames: PCMFrames, sampleRate: Double, outputURL: URL) throws {
        let channelCount = audioFrames.channelCount
        guard channelCount > 0 else {
            throw AppError.invalidGeneratedAudioShape([audioFrames.frameCount, channelCount])
        }
        let frameCount = AVAudioFrameCount(audioFrames.frameCount)
        guard let bufferFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: AVAudioChannelCount(channelCount),
            interleaved: false
        ), let fileFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: AVAudioChannelCount(channelCount),
            interleaved: true
        ), let buffer = AVAudioPCMBuffer(pcmFormat: bufferFormat, frameCapacity: frameCount) else {
            throw AppError.failedToCreateAudioBuffer
        }
        buffer.frameLength = frameCount
        guard let channelData = buffer.floatChannelData else {
            throw AppError.failedToAccessAudioBufferData
        }
        for channel in 0 ..< channelCount {
            for frame in 0 ..< audioFrames.frameCount {
                channelData[channel][frame] = audioFrames.channels[channel][frame]
            }
        }
        let audioFile = try AVAudioFile(
            forWriting: outputURL,
            settings: fileFormat.settings,
            commonFormat: bufferFormat.commonFormat,
            interleaved: bufferFormat.isInterleaved
        )
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

struct PassthroughProcessor: TextProcessor {
    func process(text: String, language: String?) throws -> String { text }
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
    let rawIPA: Bool
    let language: String?

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
        var rawIPA = false
        var language: String? = nil

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
            case "--raw-ipa":
                rawIPA = true
            case "--language", "-l":
                guard let v = it.next() else { throw CLIError.missingValue(arg) }
                language = v
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
            benchmark: benchmark,
            rawIPA: rawIPA,
            language: language
        )
    }

    static func printUsage() {
        let exe = (CommandLine.arguments.first as NSString?)?.lastPathComponent ?? "mlx-audio-swift-tts"
        print("""
        Usage:
          \(exe) --text "Hello world" [--voice conversational_b] [--model <hf-repo>] [--output <path>] [--ref_audio <path>] [--ref_text <string>] [--max_tokens <int>] [--temperature <float>] [--top_p <float>] [--timestamps] [--benchmark]

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
              --benchmark              Run streaming benchmark and log TTFB/RTF metrics
              --raw-ipa                Skip text processing, pass IPA phonemes directly
          -l, --language <code>       Language code (e.g., es, fr, it, pt). Auto-detected from voice prefix if omitted
          -h, --help                    Show this help
        """)
    }
}
