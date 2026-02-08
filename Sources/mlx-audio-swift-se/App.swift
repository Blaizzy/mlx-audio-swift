import AVFoundation
import Foundation
@preconcurrency import MLX
import MLXAudioCore
import MLXAudioEnhancement

@main
enum App {
    static func main() async {
        do {
            let args = try CLI.parse()
            try await run(
                model: args.model,
                inputPath: args.inputPath,
                outputPath: args.outputPath
            )
        } catch {
            fputs("Error: \(error)\n", stderr)
            CLI.printUsage()
            exit(1)
        }
    }

    private static func run(
        model: String,
        inputPath: String,
        outputPath: String?
    ) async throws {
        Memory.cacheLimit = 100 * 1024 * 1024

        print("Loading model (\(model))...")
        let started = CFAbsoluteTimeGetCurrent()

        let seModel = try await MossFormer2SEModel.fromPretrained(model)
        let loadTime = CFAbsoluteTimeGetCurrent() - started
        print(String(format: "Model loaded in %.2fs", loadTime))

        let inputURL = resolveURL(path: inputPath)
        let (sampleRate, audioData) = try loadAudioArray(from: inputURL)
        print("Input audio: \(audioData.shape[0]) samples @ \(sampleRate) Hz (%.2fs)"
            .replacingOccurrences(of: "%.2fs", with: String(format: "%.2fs", Float(audioData.shape[0]) / Float(sampleRate))))

        if sampleRate != seModel.config.sampleRate {
            print("Warning: Input sample rate \(sampleRate) Hz != model expected \(seModel.config.sampleRate) Hz")
            print("For best results, resample your audio to \(seModel.config.sampleRate) Hz first.")
        }

        print("Enhancing...")
        let enhanceStart = CFAbsoluteTimeGetCurrent()
        let enhanced = try seModel.enhance(audioData)
        eval(enhanced)
        let enhanceTime = CFAbsoluteTimeGetCurrent() - enhanceStart
        print(String(format: "Enhancement done in %.2fs", enhanceTime))
        print("Output shape: \(enhanced.shape)")

        let outputURL = makeOutputURL(inputPath: inputPath, outputPath: outputPath)
        try writeWavFile(
            samples: enhanced.asArray(Float.self),
            sampleRate: Double(seModel.config.sampleRate),
            outputURL: outputURL
        )
        print("Wrote enhanced audio to \(outputURL.path)")
        print("Memory usage:\n\(Memory.snapshot())")
    }

    private static func makeOutputURL(inputPath: String, outputPath: String?) -> URL {
        if let outputPath, !outputPath.isEmpty {
            return resolveURL(path: outputPath)
        }
        let inputURL = resolveURL(path: inputPath)
        let stem = inputURL.deletingPathExtension().lastPathComponent
        return inputURL.deletingLastPathComponent().appendingPathComponent("\(stem)_enhanced.wav")
    }

    private static func resolveURL(path: String) -> URL {
        if path.hasPrefix("/") || path.hasPrefix("~") {
            return URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        }
        return URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(path)
    }

    private static func writeWavFile(samples: [Float], sampleRate: Double, outputURL: URL) throws {
        let frameCount = AVAudioFrameCount(samples.count)
        guard let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1),
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            fatalError("Failed to create audio buffer")
        }
        buffer.frameLength = frameCount
        guard let channelData = buffer.floatChannelData else {
            fatalError("Failed to access audio buffer data")
        }
        for i in 0..<samples.count {
            channelData[0][i] = samples[i]
        }
        let audioFile = try AVAudioFile(forWriting: outputURL, settings: format.settings)
        try audioFile.write(from: buffer)
    }
}

// MARK: - CLI

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
    let inputPath: String
    let outputPath: String?

    static func parse() throws -> CLI {
        var inputPath: String?
        var outputPath: String?
        var model = MossFormer2SEModel.defaultRepo

        var it = CommandLine.arguments.dropFirst().makeIterator()
        while let arg = it.next() {
            switch arg {
            case "--input", "-i":
                guard let v = it.next() else { throw CLIError.missingValue(arg) }
                inputPath = v
            case "--output", "-o":
                guard let v = it.next() else { throw CLIError.missingValue(arg) }
                outputPath = v
            case "--model", "-m":
                guard let v = it.next() else { throw CLIError.missingValue(arg) }
                model = v
            case "--help", "-h":
                printUsage()
                exit(0)
            default:
                if inputPath == nil, !arg.hasPrefix("-") {
                    inputPath = arg
                } else {
                    throw CLIError.unknownOption(arg)
                }
            }
        }

        guard let finalInput = inputPath, !finalInput.isEmpty else {
            throw CLIError.missingValue("--input")
        }

        return CLI(model: model, inputPath: finalInput, outputPath: outputPath)
    }

    static func printUsage() {
        let exe = (CommandLine.arguments.first as NSString?)?.lastPathComponent ?? "mlx-audio-swift-se"
        print("""
        Usage:
          \(exe) --input <audio.wav> [--output <enhanced.wav>] [--model <hf-repo>]

        Options:
          -i, --input <path>    Input WAV file to enhance (required)
          -o, --output <path>   Output WAV path. Default: <input>_enhanced.wav
          -m, --model <repo>    HF repo id. Default: \(MossFormer2SEModel.defaultRepo)
          -h, --help            Show this help
        """)
    }
}
