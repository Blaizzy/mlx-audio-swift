import AVFoundation
import Foundation
@preconcurrency import MLX
import MLXAudioCodecs
import MLXAudioCore

enum AppError: Error, LocalizedError, CustomStringConvertible {
    case inputFileNotFound(String)
    case failedToCreateAudioBuffer
    case failedToAccessAudioBufferData

    var errorDescription: String? { description }

    var description: String {
        switch self {
        case .inputFileNotFound(let path):
            "Input audio file not found: \(path)"
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
                modelRepo: args.model,
                audioPath: args.audioPath,
                outputPath: args.outputPath,
                chunkSize: args.chunkSize
            )
        } catch {
            fputs("Error: \(error)\n", stderr)
            CLI.printUsage()
            exit(1)
        }
    }

    private static func run(
        modelRepo: String,
        audioPath: String,
        outputPath: String?,
        chunkSize: Int?
    ) async throws {
        let inputURL = resolveURL(path: audioPath)
        guard FileManager.default.fileExists(atPath: inputURL.path) else {
            throw AppError.inputFileNotFound(inputURL.path)
        }

        print("Loading DACVAE model (\(modelRepo))")
        let model = try await DACVAE.fromPretrained(modelRepo)

        print("Loading audio (\(inputURL.path))")
        let (inputSampleRate, audio) = try loadAudioArray(from: inputURL)
        if inputSampleRate != model.sampleRate {
            print("Warning: input sample rate \(inputSampleRate) != model sample rate \(model.sampleRate). No resampling is applied.")
        }

        let waveform = audio.expandedDimensions(axis: 0).expandedDimensions(axis: -1)  // (B, T, 1)

        print("Running encode -> decode")
        let encoded = model.encode(waveform)
        let decoded = model.decode(encoded, chunkSize: chunkSize)
        let reconstructed = decoded.squeezed().asArray(Float.self)

        let outputURL = makeOutputURL(outputPath: outputPath, inputURL: inputURL)
        try writeWavFile(samples: reconstructed, sampleRate: Double(inputSampleRate), outputURL: outputURL)
        print("Wrote reconstructed WAV to \(outputURL.path)")
    }

    private static func makeOutputURL(outputPath: String?, inputURL: URL) -> URL {
        if let outputPath, !outputPath.isEmpty {
            if outputPath.hasPrefix("/") {
                return URL(fileURLWithPath: outputPath)
            }
            return URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent(outputPath)
        }

        let stem = inputURL.deletingPathExtension().lastPathComponent
        return inputURL.deletingLastPathComponent()
            .appendingPathComponent("\(stem).reconstructed.wav")
    }

    private static func resolveURL(path: String) -> URL {
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
        for i in 0..<samples.count {
            channelData[0][i] = samples[i]
        }
        let audioFile = try AVAudioFile(forWriting: outputURL, settings: format.settings)
        try audioFile.write(from: buffer)
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

struct CLI {
    let audioPath: String
    let model: String
    let outputPath: String?
    let chunkSize: Int?

    static func parse() throws -> CLI {
        var audioPath: String?
        var model = "mlx-community/dacvae-watermarked"
        var outputPath: String? = nil
        var chunkSize: Int? = nil

        var iterator = CommandLine.arguments.dropFirst().makeIterator()
        while let arg = iterator.next() {
            switch arg {
            case "--audio", "-i":
                guard let value = iterator.next() else { throw CLIError.missingValue(arg) }
                audioPath = value
            case "--model":
                guard let value = iterator.next() else { throw CLIError.missingValue(arg) }
                model = value
            case "--output", "-o":
                guard let value = iterator.next() else { throw CLIError.missingValue(arg) }
                outputPath = value
            case "--chunk_size":
                guard let value = iterator.next() else { throw CLIError.missingValue(arg) }
                guard let parsed = Int(value) else { throw CLIError.invalidValue(arg, value) }
                chunkSize = parsed
            case "--help", "-h":
                printUsage()
                exit(0)
            default:
                if audioPath == nil, !arg.hasPrefix("-") {
                    audioPath = arg
                } else {
                    throw CLIError.unknownOption(arg)
                }
            }
        }

        guard let finalAudioPath = audioPath, !finalAudioPath.isEmpty else {
            throw CLIError.missingValue("--audio")
        }

        return CLI(
            audioPath: finalAudioPath,
            model: model,
            outputPath: outputPath,
            chunkSize: chunkSize
        )
    }

    static func printUsage() {
        let executable = (CommandLine.arguments.first as NSString?)?.lastPathComponent ?? "mlx-audio-swift-codec"
        print(
            """
            Usage:
              \(executable) --audio <path> [--model <hf-repo>] [--output <path>] [--chunk_size <int>]

            Description:
              Loads a DACVAE codec, runs encode() -> decode() on the input audio, and writes reconstructed WAV output.

            Options:
              -i, --audio <path>         Input audio file path (required if not passed as trailing arg)
                  --model <repo>         HF model repo id. Default: mlx-community/dacvae-watermarked
              -o, --output <path>        Output WAV path. Default: <input_stem>.reconstructed.wav
                  --chunk_size <int>     Optional decode chunk size
              -h, --help                 Show this help
            """
        )
    }
}
