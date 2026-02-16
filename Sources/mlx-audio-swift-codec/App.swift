import AVFoundation
import Foundation
@preconcurrency import MLX
import MLXAudioCodecs
import MLXAudioCore

enum AppError: Error, LocalizedError, CustomStringConvertible {
    case inputFileNotFound(String)
    case unsupportedModelRepo(String)
    case failedToCreateAudioBuffer
    case failedToAccessAudioBufferData

    var errorDescription: String? { description }

    var description: String {
        switch self {
        case .inputFileNotFound(let path):
            "Input audio file not found: \(path)"
        case .unsupportedModelRepo(let repo):
            """
            Unsupported codec repo: \(repo)
            Expected repo id containing one of: dacvae, encodec, snac, mimi.
            """
        case .failedToCreateAudioBuffer:
            "Failed to create audio buffer"
        case .failedToAccessAudioBufferData:
            "Failed to access audio buffer data"
        }
    }
}

private struct AnyAudioCodecModel {
    let codecSampleRate: Double?
    private let reconstructImpl: (MLXArray) -> MLXArray

    init<M: AudioCodecModel>(_ model: M) {
        codecSampleRate = model.codecSampleRate
        reconstructImpl = { waveform in
            model.reconstruct(waveform)
        }
    }

    func reconstruct(_ waveform: MLXArray) -> MLXArray {
        reconstructImpl(waveform)
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
                outputPath: args.outputPath
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
        outputPath: String?
    ) async throws {
        let inputURL = resolveURL(path: audioPath)
        guard FileManager.default.fileExists(atPath: inputURL.path) else {
            throw AppError.inputFileNotFound(inputURL.path)
        }

        print("Loading codec model (\(modelRepo))")
        let model = try await loadCodecModel(from: modelRepo)

        print("Loading audio (\(inputURL.path))")
        let (inputSampleRate, audio) = try loadAudioArray(from: inputURL)
        if let codecSampleRate = model.codecSampleRate {
            let roundedCodecRate = Int(codecSampleRate.rounded())
            if inputSampleRate != roundedCodecRate {
                print("Warning: input sample rate \(inputSampleRate) != model sample rate \(roundedCodecRate). No resampling is applied.")
            }
        }
        let waveform = audio.expandedDimensions(axis: 0).expandedDimensions(axis: -1)  // (B, T, 1)

        print("Running reconstruct (encodeAudio -> decodeAudio)")
        let reconstructed = model.reconstruct(waveform).squeezed().asArray(Float.self)

        let outputURL = makeOutputURL(outputPath: outputPath, inputURL: inputURL)
        try writeWavFile(samples: reconstructed, sampleRate: Double(inputSampleRate), outputURL: outputURL)
        print("Wrote reconstructed WAV to \(outputURL.path)")
    }

    private static func loadCodecModel(from modelRepo: String) async throws -> AnyAudioCodecModel {
        let repo = modelRepo.lowercased()

        if repo.contains("dacvae") {
            return AnyAudioCodecModel(try await DACVAE.fromPretrained(modelRepo))
        }
        if repo.contains("encodec") {
            return AnyAudioCodecModel(try await Encodec.fromPretrained(modelRepo))
        }
        if repo.contains("snac") {
            return AnyAudioCodecModel(try await SNAC.fromPretrained(modelRepo))
        }
        if repo.contains("mimi") || repo.contains("moshiko") || modelRepo.hasPrefix("kyutai/") {
            return AnyAudioCodecModel(
                try await Mimi.fromPretrained(
                    repoId: modelRepo,
                    progressHandler: { _ in }
                )
            )
        }

        throw AppError.unsupportedModelRepo(modelRepo)
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

    var description: String {
        switch self {
        case .missingValue(let key):
            "Missing value for \(key)"
        case .unknownOption(let key):
            "Unknown option \(key)"
        }
    }
}

struct CLI {
    let audioPath: String
    let model: String
    let outputPath: String?

    static func parse() throws -> CLI {
        var audioPath: String?
        var model = "mlx-community/dacvae-watermarked"
        var outputPath: String? = nil

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
            outputPath: outputPath
        )
    }

    static func printUsage() {
        let executable = (CommandLine.arguments.first as NSString?)?.lastPathComponent ?? "mlx-audio-swift-codec"
        print(
            """
            Usage:
              \(executable) --audio <path> [--model <hf-repo>] [--output <path>]

            Description:
              Loads a codec model, runs protocol-based reconstruct() on input audio, and writes reconstructed WAV output.

            Options:
              -i, --audio <path>         Input audio file path (required if not passed as trailing arg)
                  --model <repo>         HF model repo id. Supported patterns: dacvae, encodec, snac, mimi.
                                         Default: mlx-community/dacvae-watermarked
              -o, --output <path>        Output WAV path. Default: <input_stem>.reconstructed.wav
              -h, --help                 Show this help
            """
        )
    }
}
