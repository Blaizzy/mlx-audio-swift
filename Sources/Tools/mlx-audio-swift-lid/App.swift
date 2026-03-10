import Foundation
@preconcurrency import MLX
import MLXAudioCore
import MLXAudioLID

enum AppError: Error, LocalizedError, CustomStringConvertible {
    case inputFileNotFound(String)
    case unsupportedModelRepo(String)
    case mlxRuntimeNotConfigured(String)

    var errorDescription: String? { description }

    var description: String {
        switch self {
        case .inputFileNotFound(let path):
            "Input audio file not found: \(path)"
        case .unsupportedModelRepo(let repo):
            "Unsupported LID model repo: \(repo). Expected a repo containing mms, wav2vec2, ecapa, or voxlingua."
        case .mlxRuntimeNotConfigured(let detail):
            "MLX command-line runtime is not configured: \(detail)"
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

struct CLI {
    let audioPath: String
    let model: String
    let outputPath: String?
    let topK: Int

    static func parse() throws -> CLI {
        try parse(Array(CommandLine.arguments.dropFirst()))
    }

    static func parse(_ arguments: [String]) throws -> CLI {
        var audioPath: String?
        var model = "beshkenadze/lang-id-voxlingua107-ecapa-mlx"
        var outputPath: String?
        var topK = 5

        var iterator = arguments.makeIterator()
        while let arg = iterator.next() {
            switch arg {
            case "--audio", "-i":
                guard let value = iterator.next() else { throw CLIError.missingValue(arg) }
                audioPath = value
            case "--model":
                guard let value = iterator.next() else { throw CLIError.missingValue(arg) }
                model = value
            case "--output-path", "-o":
                guard let value = iterator.next() else { throw CLIError.missingValue(arg) }
                outputPath = value
            case "--top-k":
                guard let value = iterator.next() else { throw CLIError.missingValue(arg) }
                guard let parsed = Int(value), parsed > 0 else {
                    throw CLIError.invalidValue(arg, value)
                }
                topK = parsed
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
            topK: topK
        )
    }

    static func printUsage() {
        let executable = (CommandLine.arguments.first as NSString?)?.lastPathComponent ?? "mlx-audio-swift-lid"
        print(
            """
            Usage:
              \(executable) --audio <path> [options]

            Description:
              Run spoken language identification from an audio file using MLXAudioLID.

            Options:
              --audio, -i <path>           Input audio file (required)
              --model <repo>               Hugging Face repo id.
                                           Default: beshkenadze/lang-id-voxlingua107-ecapa-mlx
              --top-k <int>                Number of top predictions to print. Default: 5
              --output-path, -o <path>     Optional path to save JSON results
              --help, -h                   Show this help

            Examples:
              \(executable) --audio Tests/media/intention.wav
              \(executable) --audio Tests/media/intention.wav --model facebook/mms-lid-256 --top-k 3
              \(executable) --audio Tests/media/intention.wav --output-path lid-output.json
            """
        )
    }
}

private struct JSONPrediction: Encodable {
    let language: String
    let confidence: Float
}

private struct JSONOutput: Encodable {
    let model: String
    let language: String
    let confidence: Float
    let topLanguages: [JSONPrediction]
}

private enum AnyLIDModel {
    case wav2vec2(Wav2Vec2ForSequenceClassification)
    case ecapa(EcapaTdnn)

    func predict(waveform: MLXArray, topK: Int) -> LIDOutput {
        switch self {
        case .wav2vec2(let model):
            model.predict(waveform: waveform, topK: topK)
        case .ecapa(let model):
            model.predict(waveform: waveform, topK: topK)
        }
    }

    static func load(from modelRepo: String) async throws -> AnyLIDModel {
        let repo = modelRepo.lowercased()
        if repo.contains("mms") || repo.contains("wav2vec2") {
            return .wav2vec2(try await Wav2Vec2ForSequenceClassification.fromPretrained(modelRepo))
        }
        if repo.contains("ecapa") || repo.contains("voxlingua") {
            return .ecapa(try await EcapaTdnn.fromPretrained(modelRepo))
        }
        throw AppError.unsupportedModelRepo(modelRepo)
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
                topK: args.topK
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
        topK: Int
    ) async throws {
        let inputURL = resolveURL(path: audioPath)
        guard FileManager.default.fileExists(atPath: inputURL.path) else {
            throw AppError.inputFileNotFound(inputURL.path)
        }

        try ensureMLXRuntimeReadyForShell()

        print("Loading LID model (\(modelRepo))")
        let model = try await AnyLIDModel.load(from: modelRepo)

        print("Loading audio (\(inputURL.path))")
        let (_, waveform) = try loadAudioArray(from: inputURL)

        print("Running language identification")
        let output = model.predict(waveform: waveform, topK: topK)

        print("Top prediction: \(output.language) (\(String(format: "%.2f", output.confidence * 100))%)")
        for prediction in output.topLanguages {
            print("  \(prediction.language): \(String(format: "%.2f", prediction.confidence * 100))%")
        }

        if let outputPath, !outputPath.isEmpty {
            let jsonOutput = JSONOutput(
                model: modelRepo,
                language: output.language,
                confidence: output.confidence,
                topLanguages: output.topLanguages.map {
                    JSONPrediction(language: $0.language, confidence: $0.confidence)
                }
            )
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(jsonOutput)
            let outputURL = resolveURL(path: outputPath)
            try data.write(to: outputURL)
            print("Wrote JSON output to \(outputURL.path)")
        }
    }

    private static func resolveURL(path: String) -> URL {
        if path.hasPrefix("/") {
            return URL(fileURLWithPath: path)
        }
        return URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(path)
    }

    static func ensureMLXRuntimeReadyForShell(
        executableURL: URL? = CommandLine.arguments.first.map { URL(fileURLWithPath: $0) },
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws {
        let searchRoots = runtimeSearchRoots(executableURL: executableURL, environment: environment)
        let candidatePaths = metallibCandidates(searchRoots: searchRoots)

        if candidatePaths.contains(where: { FileManager.default.fileExists(atPath: $0.path) }) {
            return
        }

        let searchedPaths = candidatePaths.map(\.path).joined(separator: ", ")
        throw AppError.mlxRuntimeNotConfigured(
            """
            could not find MLX metal resources near the executable or on DYLD_FRAMEWORK_PATH. \
            Searched: \(searchedPaths). Run the tool from Xcode, or export DYLD_FRAMEWORK_PATH \
            to the SwiftPM build directory before invoking the CLI.
            """
        )
    }

    private static func runtimeSearchRoots(
        executableURL: URL?,
        environment: [String: String]
    ) -> [URL] {
        var roots: [URL] = []

        if let executableURL {
            roots.append(executableURL.deletingLastPathComponent())
        }

        if let frameworkPath = environment["DYLD_FRAMEWORK_PATH"] {
            for rawPath in frameworkPath.split(separator: ":") where !rawPath.isEmpty {
                roots.append(URL(fileURLWithPath: String(rawPath)))
            }
        }

        return roots
    }

    private static func metallibCandidates(searchRoots: [URL]) -> [URL] {
        let suffixes = [
            "default.metallib",
            "mlx.metallib",
            "Resources/default.metallib",
            "Resources/mlx.metallib",
            "mlx-swift_Cmlx.bundle/default.metallib",
            "mlx-swift_Cmlx.bundle/mlx.metallib",
            "mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib",
            "mlx-swift_Cmlx.bundle/Contents/Resources/mlx.metallib",
        ]

        return searchRoots.flatMap { root in
            suffixes.map { root.appendingPathComponent($0) }
        }
    }
}
