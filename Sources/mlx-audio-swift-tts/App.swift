import AVFoundation
import Foundation
@preconcurrency import MLX
import MLXAudioCore
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
    static func main() async {
        do {
            let args = try CLI.parse()
            try await run(
                model: args.model,
                text: args.text,
                voice: args.voice,
                outputPath: args.outputPath,
                instruct: args.instruct,
                refAudioPath: args.refAudioPath,
                refText: args.refText,
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
        instruct: String? = nil,
        refAudioPath: String? = nil,
        refText: String? = nil,
        language: String? = nil,
        hfToken: String? = nil
    ) async throws {
        Memory.cacheLimit = 100 * 1024 * 1024

        print("Loading model (\(model))")

        // Check for HF token in environment (macOS) or Info.plist (iOS) as a fallback
        let hfToken: String? = hfToken ?? ProcessInfo.processInfo.environment["HF_TOKEN"] ?? Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String

        let loadedModel: SpeechGenerationModel
        do {
            loadedModel = try await TTSModelUtils.loadModel(modelRepo: model, hfToken: hfToken)
        } catch let error as TTSModelUtilsError {
            switch error {
            case .invalidRepositoryID(let modelRepo):
                throw AppError.invalidRepositoryID(modelRepo)
            case .unsupportedModelType(let modelType):
                throw AppError.unsupportedModelType(modelType)
            }
        }

        // Load reference audio for voice cloning if provided
        var refAudio: MLXArray? = nil
        if let path = refAudioPath {
            let url = URL(fileURLWithPath: path)
            let (_, audioArray) = try loadAudioArray(from: url)
            refAudio = audioArray
            print("Loaded reference audio from \(path)")
        }

        print("Generating")
        let started = CFAbsoluteTimeGetCurrent()

        let audioData = try await loadedModel.generate(
            text: text,
            voice: voice,
            refAudio: refAudio,
            refText: refText,
            language: language,
            instruct: instruct,
            generationParameters: GenerateParameters()
        ).asArray(Float.self)

        let outputURL = makeOutputURL(outputPath: outputPath)
        let sampleRate = Double(loadedModel.sampleRate)
        try writeWavFile(samples: audioData, sampleRate: sampleRate, outputURL: outputURL)
        print("Wrote WAV to \(outputURL.path)")

        print(String(format: "Finished generation in %0.2fs", CFAbsoluteTimeGetCurrent() - started))
        print("Memory usage:\n\(Memory.snapshot())")

        let elapsed = CFAbsoluteTimeGetCurrent() - started
        print(String(format: "Done. Elapsed: %.2fs", elapsed))
    }

    private static func makeOutputURL(outputPath: String?) -> URL {
        let outputName = outputPath?.isEmpty == false ? outputPath! : "output.wav"
        if outputName.hasPrefix("/") {
            return URL(fileURLWithPath: outputName)
        }
        return URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(outputName)
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

    var description: String {
        switch self {
        case .missingValue(let k): "Missing value for \(k)"
        case .unknownOption(let k): "Unknown option \(k)"
        }
    }
}

struct CLI {
    let model: String
    let text: String
    let voice: String?
    let outputPath: String?
    let instruct: String?
    let refAudioPath: String?
    let refText: String?
    let language: String?

    static func parse() throws -> CLI {
        var text: String?
        var voice: String? = nil
        var outputPath: String? = nil
        var model = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit"
        var instruct: String? = nil
        var refAudioPath: String? = nil
        var refText: String? = nil
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
            case "--instruct":
                guard let v = it.next() else { throw CLIError.missingValue(arg) }
                instruct = v
            case "--ref-audio":
                guard let v = it.next() else { throw CLIError.missingValue(arg) }
                refAudioPath = v
            case "--ref-text":
                guard let v = it.next() else { throw CLIError.missingValue(arg) }
                refText = v
            case "--language":
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
            model: model, text: finalText, voice: voice, outputPath: outputPath,
            instruct: instruct, refAudioPath: refAudioPath, refText: refText, language: language
        )
    }

    static func printUsage() {
        let exe = (CommandLine.arguments.first as NSString?)?.lastPathComponent ?? "marvis-tts-cli"
        print("""
        Usage:
          \(exe) --text "Hello world" [--voice conversational_b] [--model <hf-repo>] [--output <path>]

        Options:
          -t, --text <string>           Text to synthesize (required if not passed as trailing arg)
          -v, --voice <name>            Voice id (CustomVoice models)
              --model <repo>            HF repo id. Default: mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit
          -o, --output <path>           Output WAV path. Default: ./output.wav
              --instruct <string>       Voice description / style instruction (VoiceDesign/CustomVoice)
              --ref-audio <path>        Reference audio WAV for voice cloning (Base models)
              --ref-text <string>       Transcription of the reference audio
              --language <code>         Language code (e.g. en, zh, ja)
          -h, --help                    Show this help
        """)
    }
}
