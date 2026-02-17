import AVFoundation
import Foundation
@preconcurrency import MLX
import MLXAudioCore
import MLXAudioSTS

enum AppError: Error, LocalizedError, CustomStringConvertible {
    case inputFileNotFound(String)
    case anchorsUnsupportedForMode(SeparationMode)
    case failedToCreateAudioBuffer
    case failedToAccessAudioBufferData

    var errorDescription: String? { description }

    var description: String {
        switch self {
        case .inputFileNotFound(let path):
            "Input audio file not found: \(path)"
        case .anchorsUnsupportedForMode(let mode):
            "Anchors are only supported with --mode short. Received --mode \(mode.rawValue)."
        case .failedToCreateAudioBuffer:
            "Failed to create audio buffer"
        case .failedToAccessAudioBufferData:
            "Failed to access audio buffer data"
        }
    }
}

enum SeparationMode: String {
    case short
    case long
    case stream
}

@main
enum App {
    static func main() async {
        do {
            let args = try CLI.parse()
            try await run(
                modelRepo: args.model,
                audioPath: args.audioPath,
                description: args.description,
                mode: args.mode,
                outputTargetPath: args.outputTargetPath,
                outputResidualPath: args.outputResidualPath,
                writeResidual: args.writeResidual,
                chunkSeconds: args.chunkSeconds,
                overlapSeconds: args.overlapSeconds,
                odeMethod: args.odeMethod,
                stepSize: args.stepSize,
                odeDecodeChunkSize: args.odeDecodeChunkSize,
                anchors: args.anchors,
                strict: args.strict,
                hfToken: args.hfToken
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
        description: String,
        mode: SeparationMode,
        outputTargetPath: String?,
        outputResidualPath: String?,
        writeResidual: Bool,
        chunkSeconds: Float,
        overlapSeconds: Float,
        odeMethod: SAMAudioODEMethod,
        stepSize: Float,
        odeDecodeChunkSize: Int?,
        anchors: [SAMAudioAnchor],
        strict: Bool,
        hfToken: String?
    ) async throws {
        let inputURL = resolveURL(path: audioPath)
        guard FileManager.default.fileExists(atPath: inputURL.path) else {
            throw AppError.inputFileNotFound(inputURL.path)
        }

        if !anchors.isEmpty, mode != .short {
            throw AppError.anchorsUnsupportedForMode(mode)
        }

        let resolvedHFToken = hfToken
            ?? ProcessInfo.processInfo.environment["HF_TOKEN"]
            ?? Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String

        print("Loading SAM Audio model (\(modelRepo))")
        let model = try await SAMAudio.fromPretrained(
            modelRepo,
            hfToken: resolvedHFToken,
            strict: strict
        )

        let targetOutputURL = makeOutputURL(
            outputPath: outputTargetPath,
            inputURL: inputURL,
            defaultSuffix: "target.wav"
        )

        let residualOutputURL = makeOutputURL(
            outputPath: outputResidualPath,
            inputURL: inputURL,
            defaultSuffix: "residual.wav"
        )

        let ode = SAMAudioODEOptions(method: odeMethod, stepSize: stepSize)

        print("Running SAM Audio (mode=\(mode.rawValue), description=\"\(description)\")")
        let started = CFAbsoluteTimeGetCurrent()

        switch mode {
        case .short:
            let result = try await model.separate(
                audioPaths: [inputURL.path],
                descriptions: [description],
                anchors: anchors.isEmpty ? nil : [anchors],
                noise: nil,
                ode: ode,
                odeDecodeChunkSize: odeDecodeChunkSize
            )

            try writeWavArray(
                result.target[0],
                sampleRate: Double(model.sampleRate),
                outputURL: targetOutputURL
            )
            print("Wrote target WAV to \(targetOutputURL.path)")

            if writeResidual {
                try writeWavArray(
                    result.residual[0],
                    sampleRate: Double(model.sampleRate),
                    outputURL: residualOutputURL
                )
                print("Wrote residual WAV to \(residualOutputURL.path)")
            }

        case .long:
            let result = try await model.separateLong(
                audioPaths: [inputURL.path],
                descriptions: [description],
                chunkSeconds: chunkSeconds,
                overlapSeconds: overlapSeconds,
                ode: ode,
                odeDecodeChunkSize: odeDecodeChunkSize
            )

            try writeWavArray(
                result.target[0],
                sampleRate: Double(model.sampleRate),
                outputURL: targetOutputURL
            )
            print("Wrote target WAV to \(targetOutputURL.path)")

            if writeResidual {
                try writeWavArray(
                    result.residual[0],
                    sampleRate: Double(model.sampleRate),
                    outputURL: residualOutputURL
                )
                print("Wrote residual WAV to \(residualOutputURL.path)")
            }

        case .stream:
            try await separateStreamingToDisk(
                model: model,
                audioPath: inputURL.path,
                description: description,
                chunkSeconds: chunkSeconds,
                overlapSeconds: overlapSeconds,
                ode: ode,
                targetOutputURL: targetOutputURL,
                residualOutputURL: residualOutputURL,
                writeResidual: writeResidual
            )
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - started
        print(String(format: "Done. Elapsed: %.2fs", elapsed))
    }

    private static func separateStreamingToDisk(
        model: SAMAudio,
        audioPath: String,
        description: String,
        chunkSeconds: Float,
        overlapSeconds: Float,
        ode: SAMAudioODEOptions,
        targetOutputURL: URL,
        residualOutputURL: URL,
        writeResidual: Bool
    ) async throws {
        let targetWriter = try StreamingWAVWriter(
            url: targetOutputURL,
            sampleRate: Double(model.sampleRate)
        )

        let residualWriter: StreamingWAVWriter? = writeResidual
            ? try StreamingWAVWriter(url: residualOutputURL, sampleRate: Double(model.sampleRate))
            : nil

        let stream = try model.separateStreaming(
            audioPaths: [audioPath],
            descriptions: [description],
            chunkSeconds: chunkSeconds,
            overlapSeconds: overlapSeconds,
            ode: ode
        )

        var chunks = 0
        for try await chunk in stream {
            try targetWriter.writeChunk(chunk.target.squeezed().asArray(Float.self))
            if let residualWriter {
                try residualWriter.writeChunk(chunk.residual.squeezed().asArray(Float.self))
            }
            chunks += 1
        }

        _ = targetWriter.finalize()
        _ = residualWriter?.finalize()

        print("Wrote target WAV to \(targetOutputURL.path)")
        if writeResidual {
            print("Wrote residual WAV to \(residualOutputURL.path)")
        }
        print("Streamed \(chunks) chunk(s)")
    }

    private static func resolveURL(path: String) -> URL {
        if path.hasPrefix("/") {
            return URL(fileURLWithPath: path)
        }
        return URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(path)
    }

    private static func makeOutputURL(outputPath: String?, inputURL: URL, defaultSuffix: String) -> URL {
        if let outputPath, !outputPath.isEmpty {
            if outputPath.hasPrefix("/") {
                return URL(fileURLWithPath: outputPath)
            }
            return URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent(outputPath)
        }

        let stem = inputURL.deletingPathExtension().lastPathComponent
        return inputURL.deletingLastPathComponent()
            .appendingPathComponent("\(stem).\(defaultSuffix)")
    }

    private static func writeWavArray(_ audio: MLXArray, sampleRate: Double, outputURL: URL) throws {
        try writeWavFile(samples: audio.squeezed().asArray(Float.self), sampleRate: sampleRate, outputURL: outputURL)
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
            return "Missing value for \(key)"
        case .unknownOption(let key):
            return "Unknown option \(key)"
        case .invalidValue(let key, let value):
            return "Invalid value for \(key): \(value)"
        }
    }
}

struct CLI {
    let audioPath: String
    let model: String
    let description: String
    let mode: SeparationMode
    let outputTargetPath: String?
    let outputResidualPath: String?
    let writeResidual: Bool
    let chunkSeconds: Float
    let overlapSeconds: Float
    let odeMethod: SAMAudioODEMethod
    let stepSize: Float
    let odeDecodeChunkSize: Int?
    let anchors: [SAMAudioAnchor]
    let strict: Bool
    let hfToken: String?

    static func parse() throws -> CLI {
        var audioPath: String?
        var model = SAMAudio.defaultRepo
        var description = "speech"
        var mode: SeparationMode = .short
        var outputTargetPath: String?
        var outputResidualPath: String?
        var writeResidual = true
        var chunkSeconds: Float = 10.0
        var overlapSeconds: Float = 3.0
        var odeMethod: SAMAudioODEMethod = .midpoint
        var stepSize: Float = 2.0 / 32.0
        var odeDecodeChunkSize: Int?
        var anchors: [SAMAudioAnchor] = []
        var strict = false
        var hfToken: String?

        var iterator = CommandLine.arguments.dropFirst().makeIterator()
        while let arg = iterator.next() {
            switch arg {
            case "--audio", "-i":
                guard let value = iterator.next() else { throw CLIError.missingValue(arg) }
                audioPath = value
            case "--model":
                guard let value = iterator.next() else { throw CLIError.missingValue(arg) }
                model = value
            case "--description", "--prompt", "-d":
                guard let value = iterator.next() else { throw CLIError.missingValue(arg) }
                description = value
            case "--mode":
                guard let value = iterator.next() else { throw CLIError.missingValue(arg) }
                guard let parsed = SeparationMode(rawValue: value.lowercased()) else {
                    throw CLIError.invalidValue(arg, value)
                }
                mode = parsed
            case "--output-target", "-o":
                guard let value = iterator.next() else { throw CLIError.missingValue(arg) }
                outputTargetPath = value
            case "--output-residual":
                guard let value = iterator.next() else { throw CLIError.missingValue(arg) }
                outputResidualPath = value
            case "--no-residual":
                writeResidual = false
            case "--chunk-seconds":
                guard let value = iterator.next() else { throw CLIError.missingValue(arg) }
                guard let parsed = Float(value) else { throw CLIError.invalidValue(arg, value) }
                chunkSeconds = parsed
            case "--overlap-seconds":
                guard let value = iterator.next() else { throw CLIError.missingValue(arg) }
                guard let parsed = Float(value) else { throw CLIError.invalidValue(arg, value) }
                overlapSeconds = parsed
            case "--ode-method":
                guard let value = iterator.next() else { throw CLIError.missingValue(arg) }
                guard let parsed = SAMAudioODEMethod(rawValue: value.lowercased()) else {
                    throw CLIError.invalidValue(arg, value)
                }
                odeMethod = parsed
            case "--step-size":
                guard let value = iterator.next() else { throw CLIError.missingValue(arg) }
                guard let parsed = Float(value) else { throw CLIError.invalidValue(arg, value) }
                stepSize = parsed
            case "--decode-chunk-size":
                guard let value = iterator.next() else { throw CLIError.missingValue(arg) }
                guard let parsed = Int(value), parsed > 0 else { throw CLIError.invalidValue(arg, value) }
                odeDecodeChunkSize = parsed
            case "--anchor":
                guard let value = iterator.next() else { throw CLIError.missingValue(arg) }
                anchors.append(try parseAnchor(value, key: arg))
            case "--strict":
                strict = true
            case "--hf-token":
                guard let value = iterator.next() else { throw CLIError.missingValue(arg) }
                hfToken = value
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
            description: description,
            mode: mode,
            outputTargetPath: outputTargetPath,
            outputResidualPath: outputResidualPath,
            writeResidual: writeResidual,
            chunkSeconds: chunkSeconds,
            overlapSeconds: overlapSeconds,
            odeMethod: odeMethod,
            stepSize: stepSize,
            odeDecodeChunkSize: odeDecodeChunkSize,
            anchors: anchors,
            strict: strict,
            hfToken: hfToken
        )
    }

    private static func parseAnchor(_ raw: String, key: String) throws -> SAMAudioAnchor {
        let parts = raw.split(separator: ":", omittingEmptySubsequences: false)
        guard parts.count == 3 else {
            throw CLIError.invalidValue(key, raw)
        }

        let token = String(parts[0])
        guard token == "+" || token == "-" else {
            throw CLIError.invalidValue(key, raw)
        }

        guard let startTime = Float(parts[1]), let endTime = Float(parts[2]), endTime > startTime, startTime >= 0 else {
            throw CLIError.invalidValue(key, raw)
        }

        return (token: token, startTime: startTime, endTime: endTime)
    }

    static func printUsage() {
        let executable = (CommandLine.arguments.first as NSString?)?.lastPathComponent ?? "mlx-audio-swift-sts"
        print(
            """
            Usage:
              \(executable) --audio <path> [--description <text>] [--mode short|long|stream] [options]

            Description:
              Runs SAM Audio source separation and writes target/residual WAV output.

            Options:
              -i, --audio <path>           Input audio file path (required if not passed as trailing arg)
                  --model <repo-or-path>   SAM Audio model repo or local folder.
                                           Default: \(SAMAudio.defaultRepo)
              -d, --description <text>     Text prompt describing target sound.
                                           Default: speech
                  --mode <mode>            Separation mode: short, long, stream.
                                           Default: short
              -o, --output-target <path>   Target WAV output path.
                                           Default: <input_stem>.target.wav
                  --output-residual <path> Residual WAV output path.
                                           Default: <input_stem>.residual.wav
                  --no-residual            Skip writing residual output file.

                  --chunk-seconds <float>  Chunk duration for long/stream modes.
                                           Default: 10.0
                  --overlap-seconds <float> Overlap duration for long/stream modes.
                                           Default: 3.0
                  --ode-method <method>    ODE method: midpoint or euler.
                                           Default: midpoint
                  --step-size <float>      ODE step size (0 < value < 1).
                                           Default: 0.0625
                  --decode-chunk-size <n>  Optional decoder chunk size.

                  --anchor <tok:start:end> Temporal anchor (repeatable). tok is + or -.
                                           Example: --anchor +:1.5:3.0
                                           Note: anchors are only supported in --mode short.

                  --strict                 Enable strict model weight loading.
                  --hf-token <token>       Hugging Face token (or set HF_TOKEN env var)
              -h, --help                   Show this help
            """
        )
    }
}
