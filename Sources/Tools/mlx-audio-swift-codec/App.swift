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
        let model = try await AudioCodecs.loadModel(modelRepo: modelRepo)

        print("Loading audio (\(inputURL.path))")
        let waveform = try loadWaveform(from: inputURL, sampleRate: model.sampleRate, channels: model.audioChannels)

        print("Running codec reconstruction")
        let reconstructed = try model.reconstruct(waveform)

        let outputURL = makeOutputURL(outputPath: outputPath, inputURL: inputURL)
        try writeWavFile(waveform: reconstructed, sampleRate: Double(model.sampleRate), outputURL: outputURL)
        print("Wrote reconstructed WAV to \(outputURL.path)")
    }

    private static func loadWaveform(from url: URL, sampleRate: Int, channels: Int) throws -> MLXArray {
        let file = try AVAudioFile(forReading: url)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length)) else {
            throw AppError.failedToCreateAudioBuffer
        }
        try file.read(into: buffer)
        guard let data = buffer.floatChannelData else { throw AppError.failedToAccessAudioBufferData }
        let frames = Int(buffer.frameLength)
        let inputChannels = Int(buffer.format.channelCount)
        guard frames > 0, inputChannels > 0 else {
            throw AudioCodecError.invalidWaveform("Input audio must contain samples and channels")
        }
        var signals = (0..<inputChannels).map { Array(UnsafeBufferPointer(start: data[$0], count: frames)) }
        if channels == 1 && inputChannels > 1 {
            var mono = [Float](repeating: 0, count: frames)
            for signal in signals {
                for frame in 0..<frames { mono[frame] += signal[frame] / Float(inputChannels) }
            }
            signals = [mono]
        } else if inputChannels == 1 && channels > 1 {
            signals = Array(repeating: signals[0], count: channels)
        } else if inputChannels != channels {
            throw AudioCodecError.invalidWaveform("Codec requires \(channels) channels, input has \(inputChannels)")
        }
        let inputRate = Int(buffer.format.sampleRate.rounded())
        if inputRate != sampleRate {
            signals = try signals.map { try resampleAudio($0, from: inputRate, to: sampleRate) }
        }
        let outputFrames = signals[0].count
        var interleaved = [Float](repeating: 0, count: outputFrames * channels)
        for channel in 0..<channels {
            for frame in 0..<outputFrames { interleaved[frame * channels + channel] = signals[channel][frame] }
        }
        return MLXArray(interleaved, [1, outputFrames, channels])
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
        URL(fileURLWithPath: NSString(string: path).expandingTildeInPath)
    }

    private static func writeWavFile(waveform: MLXArray, sampleRate: Double, outputURL: URL) throws {
        let frames = waveform.dim(1)
        let channels = waveform.dim(2)
        let samples = waveform.asType(.float32).asArray(Float.self)
        guard let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: AVAudioChannelCount(channels)),
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(frames)) else {
            throw AppError.failedToCreateAudioBuffer
        }
        buffer.frameLength = AVAudioFrameCount(frames)
        guard let data = buffer.floatChannelData else { throw AppError.failedToAccessAudioBufferData }
        for channel in 0..<channels {
            for frame in 0..<frames { data[channel][frame] = samples[frame * channels + channel] }
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
              \(executable) --audio <path> [--model <repo-or-path>] [--output <path>]

            Description:
              Loads a codec, resamples input audio to its sample rate, and writes reconstructed WAV output.

            Options:
              -i, --audio <path>         Input audio file path (required if not passed as trailing arg)
                  --model <repo-or-path> Codec repository ID or local checkpoint directory.
                                         Default: mlx-community/dacvae-watermarked
              -o, --output <path>        Output WAV path. Default: <input_stem>.reconstructed.wav
              -h, --help                 Show this help
            """
        )
    }
}
