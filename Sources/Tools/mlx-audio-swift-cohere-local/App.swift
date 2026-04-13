import Foundation
import MLX
import MLXNN
import MLXAudioCore
import MLXAudioSTT

enum CohereLocalError: Error, LocalizedError {
    case missingValue(String)
    case invalidValue(String, String)
    case missingRequired(String)
    case missingFile(String)

    var errorDescription: String? {
        switch self {
        case .missingValue(let option):
            return "Missing value for \(option)"
        case .invalidValue(let option, let value):
            return "Invalid value for \(option): \(value)"
        case .missingRequired(let option):
            return "Missing required option \(option)"
        case .missingFile(let path):
            return "Required file not found: \(path)"
        }
    }
}

private struct Options {
    var sourceDir: String?
    var outputDir: String?
    var audio: String?
    var bits: Int?
    var groupSize: Int = 64
    var language: String = "en"
    var maxTokens: Int = 256

    static func parse() throws -> Options {
        var options = Options()
        var iterator = CommandLine.arguments.dropFirst().makeIterator()

        while let argument = iterator.next() {
            switch argument {
            case "--source-dir":
                guard let value = iterator.next() else { throw CohereLocalError.missingValue(argument) }
                options.sourceDir = value
            case "--output-dir":
                guard let value = iterator.next() else { throw CohereLocalError.missingValue(argument) }
                options.outputDir = value
            case "--audio":
                guard let value = iterator.next() else { throw CohereLocalError.missingValue(argument) }
                options.audio = value
            case "--bits":
                guard let value = iterator.next() else { throw CohereLocalError.missingValue(argument) }
                guard let parsed = Int(value), [4, 8].contains(parsed) else {
                    throw CohereLocalError.invalidValue(argument, value)
                }
                options.bits = parsed
            case "--group-size":
                guard let value = iterator.next() else { throw CohereLocalError.missingValue(argument) }
                guard let parsed = Int(value), parsed > 0 else {
                    throw CohereLocalError.invalidValue(argument, value)
                }
                options.groupSize = parsed
            case "--language":
                guard let value = iterator.next() else { throw CohereLocalError.missingValue(argument) }
                options.language = value
            case "--max-tokens":
                guard let value = iterator.next() else { throw CohereLocalError.missingValue(argument) }
                guard let parsed = Int(value), parsed > 0 else {
                    throw CohereLocalError.invalidValue(argument, value)
                }
                options.maxTokens = parsed
            case "--help", "-h":
                Self.printUsage()
                exit(0)
            default:
                throw CohereLocalError.invalidValue("argument", argument)
            }
        }

        guard options.sourceDir != nil else {
            throw CohereLocalError.missingRequired("--source-dir")
        }

        if options.outputDir != nil, options.bits == nil {
            throw CohereLocalError.missingRequired("--bits when using --output-dir")
        }

        return options
    }

    static func printUsage() {
        print(
            """
            Usage:
              mlx-audio-swift-cohere-local --source-dir <dir> [--output-dir <dir> --bits <4|8>] [--audio <wav>]

            Options:
              --source-dir <dir>   Local fp16 or quantized Cohere model directory
              --output-dir <dir>   Directory to export a quantized local model
              --bits <4|8>         Quantization bit-width for export
              --group-size <int>   Quantization group size (default: 64)
              --audio <wav>        Optional audio file to transcribe after loading/export
              --language <code>    Language hint for generation (default: en)
              --max-tokens <int>   Maximum generated tokens (default: 256)
              -h, --help           Show this help
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
            fputs("Error: \(error.localizedDescription)\n", stderr)
            Options.printUsage()
            exit(1)
        }
    }

    private static func run(options: Options) async throws {
        let sourceURL = URL(fileURLWithPath: options.sourceDir!, isDirectory: true)
        let workingModelURL: URL

        if let outputDir = options.outputDir {
            let outputURL = URL(fileURLWithPath: outputDir, isDirectory: true)
            try exportQuantizedModel(
                from: sourceURL,
                to: outputURL,
                bits: options.bits!,
                groupSize: options.groupSize
            )
            workingModelURL = outputURL
        } else {
            workingModelURL = sourceURL
        }

        if let audioPath = options.audio {
            let audioURL = URL(fileURLWithPath: audioPath)
            let result = try transcribe(
                modelDir: workingModelURL,
                audioURL: audioURL,
                language: options.language,
                maxTokens: options.maxTokens
            )
            print(result)
        }
    }

    private static func exportQuantizedModel(
        from sourceURL: URL,
        to outputURL: URL,
        bits: Int,
        groupSize: Int
    ) throws {
        let fileManager = FileManager.default
        try fileManager.createDirectory(at: outputURL, withIntermediateDirectories: true)

        let model = try CohereTranscribeModel.fromDirectory(sourceURL)
        quantize(model: model, groupSize: groupSize, bits: bits)
        eval(model)

        let flattened = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        let modelURL = outputURL.appendingPathComponent("model.safetensors")
        try MLX.save(arrays: flattened, url: modelURL)

        try copySidecarFiles(from: sourceURL, to: outputURL)
        try writeQuantizedConfig(from: sourceURL, to: outputURL, bits: bits, groupSize: groupSize)

        print("Exported quantized model to \(outputURL.path)")
    }

    private static func transcribe(
        modelDir: URL,
        audioURL: URL,
        language: String,
        maxTokens: Int
    ) throws -> String {
        let model = try CohereTranscribeModel.fromDirectory(modelDir)
        let (_, audio) = try loadAudioArray(from: audioURL, sampleRate: 16_000)
        let output = model.generate(
            audio: audio,
            generationParameters: STTGenerateParameters(
                maxTokens: maxTokens,
                temperature: 0.0,
                topP: 1.0,
                topK: 0,
                verbose: false,
                language: language
            )
        )

        let payload: [String: Any] = [
            "text": output.text,
            "prompt_tokens": output.promptTokens,
            "generation_tokens": output.generationTokens,
            "total_tokens": output.totalTokens,
            "peak_memory_gb": output.peakMemoryUsage,
        ]

        let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
        return String(decoding: data, as: UTF8.self)
    }

    private static func copySidecarFiles(from sourceURL: URL, to outputURL: URL) throws {
        let fileManager = FileManager.default
        let filenames = [
            "tokenizer.model",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "preprocessor_config.json",
        ]

        for filename in filenames {
            let sourceFile = sourceURL.appendingPathComponent(filename)
            guard fileManager.fileExists(atPath: sourceFile.path) else {
                throw CohereLocalError.missingFile(sourceFile.path)
            }

            let destinationFile = outputURL.appendingPathComponent(filename)
            if fileManager.fileExists(atPath: destinationFile.path) {
                try fileManager.removeItem(at: destinationFile)
            }
            try fileManager.copyItem(at: sourceFile, to: destinationFile)
        }
    }

    private static func writeQuantizedConfig(
        from sourceURL: URL,
        to outputURL: URL,
        bits: Int,
        groupSize: Int
    ) throws {
        let configURL = sourceURL.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        guard var object = try JSONSerialization.jsonObject(with: configData) as? [String: Any] else {
            throw CohereLocalError.invalidValue("config.json", configURL.path)
        }

        object["quantization_config"] = [
            "group_size": groupSize,
            "bits": bits,
        ]
        object.removeValue(forKey: "quantization")

        let serialized = try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
        try serialized.write(to: outputURL.appendingPathComponent("config.json"))
    }
}
