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
    case unsupportedModelRepo(String)
    case lfmRequiresText
    case lfmRequiresAudioForMode(LFMMode)

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
        case .unsupportedModelRepo(let repo):
            "Unsupported STS model repo: \(repo). Expected SAMAudio or LFM model."
        case .lfmRequiresText:
            "--text is required for LFM text-to-text and text-to-speech modes."
        case .lfmRequiresAudioForMode(let mode):
            "--audio is required for LFM \(mode.rawValue) mode."
        }
    }
}

enum SeparationMode: String {
    case short
    case long
    case stream
}

enum LFMMode: String {
    case t2t
    case tts
    case stt
    case sts
}

@main
enum App {
    static func main() async {
        do {
            let args = try CLI.parse()

            if isLFMModel(args.model) {
                try await runLFM(args: args)
            } else {
                try await runSAMAudio(args: args)
            }
        } catch {
            fputs("Error: \(error)\n", stderr)
            CLI.printUsage()
            exit(1)
        }
    }

    private static func isLFMModel(_ model: String) -> Bool {
        let lower = model.lowercased()
        return lower.contains("lfm") || lower.contains("lfm2")
    }

    // MARK: - LFM2.5-Audio

    private static func runLFM(args: CLI) async throws {
        let lfmMode = args.lfmMode ?? .sts

        // Validate inputs for mode
        switch lfmMode {
        case .t2t, .tts:
            guard let text = args.text, !text.isEmpty else {
                throw AppError.lfmRequiresText
            }
        case .stt, .sts:
            guard args.audioPath != nil else {
                throw AppError.lfmRequiresAudioForMode(lfmMode)
            }
        }

        print("Loading LFM2.5-Audio model (\(args.model))")
        let model = try await LFM2AudioModel.fromPretrained(args.model)
        let processor = model.processor!

        let chat = ChatState(processor: processor)

        // Default system prompts per mode (user can override with --system)
        let defaultSystemPrompts: [LFMMode: String] = [
            .t2t: "You are a helpful assistant.",
            .tts: "Perform TTS. Use a UK male voice.",
            .stt: "You are a helpful assistant that transcribes audio.",
            .sts: "Respond to the user with interleaved text and speech audio. Use a UK male voice.",
        ]
        let systemPrompt = args.systemPrompt ?? defaultSystemPrompts[lfmMode]!

        // Build prompt based on mode
        switch lfmMode {
        case .t2t:
            chat.newTurn(role: "system")
            chat.addText(systemPrompt)
            chat.endTurn()
            chat.newTurn(role: "user")
            chat.addText(args.text!)
            chat.endTurn()
            chat.newTurn(role: "assistant")

        case .tts:
            chat.newTurn(role: "system")
            chat.addText(systemPrompt)
            chat.endTurn()
            chat.newTurn(role: "user")
            chat.addText(args.text!)
            chat.endTurn()
            chat.newTurn(role: "assistant")
            chat.addAudioStartToken()

        case .stt:
            let inputURL = resolveURL(path: args.audioPath!)
            guard FileManager.default.fileExists(atPath: inputURL.path) else {
                throw AppError.inputFileNotFound(inputURL.path)
            }
            let (sampleRate, audioData) = try loadAudioArray(from: inputURL)
            chat.newTurn(role: "system")
            chat.addText(systemPrompt)
            chat.endTurn()
            chat.newTurn(role: "user")
            chat.addAudio(audioData, sampleRate: sampleRate)
            chat.addText(args.text ?? "Transcribe the audio.")
            chat.endTurn()
            chat.newTurn(role: "assistant")

        case .sts:
            let inputURL = resolveURL(path: args.audioPath!)
            guard FileManager.default.fileExists(atPath: inputURL.path) else {
                throw AppError.inputFileNotFound(inputURL.path)
            }
            let (sampleRate, audioData) = try loadAudioArray(from: inputURL)
            chat.newTurn(role: "system")
            chat.addText(systemPrompt)
            chat.endTurn()
            chat.newTurn(role: "user")
            chat.addAudio(audioData, sampleRate: sampleRate)
            if let text = args.text {
                chat.addText(text)
            }
            chat.endTurn()
            chat.newTurn(role: "assistant")
        }

        let genConfig = LFMGenerationConfig(
            maxNewTokens: args.maxNewTokens,
            temperature: args.temperature,
            topK: args.topK,
            audioTemperature: args.audioTemperature,
            audioTopK: args.audioTopK
        )

        let started = CFAbsoluteTimeGetCurrent()
        print("Generating (mode=\(lfmMode.rawValue))")

        var textTokens: [Int] = []
        var audioCodes: [MLXArray] = []

        let useSequential = (lfmMode == .tts)
        let collectText = (lfmMode == .t2t || lfmMode == .stt || lfmMode == .sts)
        let collectAudio = (lfmMode == .tts || lfmMode == .sts)

        let stream: AsyncThrowingStream<(MLXArray, LFMModality), Error>

        if useSequential {
            stream = model.generateSequential(
                textTokens: chat.getTextTokens(),
                audioFeatures: chat.getAudioFeatures(),
                modalities: chat.getModalities(),
                config: genConfig
            )
        } else {
            stream = model.generateInterleaved(
                textTokens: chat.getTextTokens(),
                audioFeatures: chat.getAudioFeatures(),
                modalities: chat.getModalities(),
                config: genConfig
            )
        }

        for try await (token, modality) in stream {
            eval(token)
            if modality == .text && collectText {
                let tokenId = token.item(Int.self)
                textTokens.append(tokenId)
                if args.stream {
                    print(processor.decodeText([tokenId]), terminator: "")
                    fflush(stdout)
                }
            } else if modality == .audioOut && collectAudio {
                if useSequential {
                    // TTS: break on EOS, don't append it (matches smoke test)
                    if token[0].item(Int.self) == lfmAudioEOSToken { break }
                    audioCodes.append(token)
                } else {
                    // STS: collect audio frames, skip EOS frames (all 2048) to avoid decode artifacts
                    if token[0].item(Int.self) != lfmAudioEOSToken {
                        audioCodes.append(token)
                    }
                }
            }
        }

        if args.stream && !textTokens.isEmpty { print() }

        let elapsed = CFAbsoluteTimeGetCurrent() - started

        if collectText && !textTokens.isEmpty {
            let decodedText = processor.decodeText(textTokens)
            if !args.stream {
                print("Text: \(decodedText)")
            }
            print("Generated \(textTokens.count) text tokens")

            if let outputPath = args.outputTextPath {
                let url = resolveURL(path: outputPath)
                try decodedText.write(to: url, atomically: true, encoding: .utf8)
                print("Wrote text to \(url.path)")
            }
        }

        if collectAudio && !audioCodes.isEmpty {
            print("Generated \(audioCodes.count) audio frames")
            let stacked = MLX.stacked(audioCodes, axis: 0)
            let codesInput = stacked.transposed(1, 0).expandedDimensions(axis: 0)
            eval(codesInput)

            let detokenizer = try LFM2AudioDetokenizer.fromPretrained(modelPath: model.modelDirectory!)
            let waveform = detokenizer(codesInput)
            eval(waveform)
            let samples = waveform[0].asArray(Float.self)

            let duration = Double(samples.count) / 24000.0
            print(String(format: "Decoded %d audio samples (%.1fs at 24kHz)", samples.count, duration))

            let outputURL: URL
            if let path = args.outputTargetPath {
                outputURL = resolveURL(path: path)
            } else {
                outputURL = resolveURL(path: "lfm_output.wav")
            }

            try AudioUtils.writeWavFile(samples: samples, sampleRate: 24000, fileURL: outputURL)
            print("Wrote WAV to \(outputURL.path)")
        }

        print(String(format: "Done. Elapsed: %.2fs", elapsed))
    }

    // MARK: - SAM Audio

    private static func runSAMAudio(args: CLI) async throws {
        let mode = args.mode

        guard let audioPath = args.audioPath else {
            throw AppError.inputFileNotFound("(none)")
        }

        let inputURL = resolveURL(path: audioPath)
        guard FileManager.default.fileExists(atPath: inputURL.path) else {
            throw AppError.inputFileNotFound(inputURL.path)
        }

        if !args.anchors.isEmpty, mode != .short {
            throw AppError.anchorsUnsupportedForMode(mode)
        }

        let resolvedHFToken = args.hfToken
            ?? ProcessInfo.processInfo.environment["HF_TOKEN"]
            ?? Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String

        print("Loading SAM Audio model (\(args.model))")
        let model = try await SAMAudio.fromPretrained(
            args.model,
            hfToken: resolvedHFToken,
            strict: args.strict
        )

        let targetOutputURL = makeOutputURL(
            outputPath: args.outputTargetPath,
            inputURL: inputURL,
            defaultSuffix: "target.wav"
        )

        let residualOutputURL = makeOutputURL(
            outputPath: args.outputResidualPath,
            inputURL: inputURL,
            defaultSuffix: "residual.wav"
        )

        let ode = SAMAudioODEOptions(method: args.odeMethod, stepSize: args.stepSize)

        print("Running SAM Audio (mode=\(mode.rawValue), description=\"\(args.description)\")")
        let started = CFAbsoluteTimeGetCurrent()

        switch mode {
        case .short:
            let result = try await model.separate(
                audioPaths: [inputURL.path],
                descriptions: [args.description],
                anchors: args.anchors.isEmpty ? nil : [args.anchors],
                noise: nil,
                ode: ode,
                odeDecodeChunkSize: args.odeDecodeChunkSize
            )

            try writeWavArray(
                result.target[0],
                sampleRate: Double(model.sampleRate),
                outputURL: targetOutputURL
            )
            print("Wrote target WAV to \(targetOutputURL.path)")

            if args.writeResidual {
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
                descriptions: [args.description],
                chunkSeconds: args.chunkSeconds,
                overlapSeconds: args.overlapSeconds,
                ode: ode,
                odeDecodeChunkSize: args.odeDecodeChunkSize
            )

            try writeWavArray(
                result.target[0],
                sampleRate: Double(model.sampleRate),
                outputURL: targetOutputURL
            )
            print("Wrote target WAV to \(targetOutputURL.path)")

            if args.writeResidual {
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
                description: args.description,
                chunkSeconds: args.chunkSeconds,
                overlapSeconds: args.overlapSeconds,
                ode: ode,
                targetOutputURL: targetOutputURL,
                residualOutputURL: residualOutputURL,
                writeResidual: args.writeResidual
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

    // MARK: - Helpers

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

// MARK: - CLI

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
    // Common
    let model: String
    let audioPath: String?
    let text: String?
    let outputTargetPath: String?
    let outputTextPath: String?
    let hfToken: String?
    let stream: Bool

    // SAMAudio-specific
    let description: String
    let mode: SeparationMode
    let outputResidualPath: String?
    let writeResidual: Bool
    let chunkSeconds: Float
    let overlapSeconds: Float
    let odeMethod: SAMAudioODEMethod
    let stepSize: Float
    let odeDecodeChunkSize: Int?
    let anchors: [SAMAudioAnchor]
    let strict: Bool

    // LFM-specific
    let lfmMode: LFMMode?
    let systemPrompt: String?
    let maxNewTokens: Int
    let temperature: Float
    let topK: Int
    let audioTemperature: Float
    let audioTopK: Int

    static func parse() throws -> CLI {
        var audioPath: String?
        var model = SAMAudio.defaultRepo
        var text: String?
        var description = "speech"
        var mode: SeparationMode = .short
        var outputTargetPath: String?
        var outputResidualPath: String?
        var outputTextPath: String?
        var writeResidual = true
        var chunkSeconds: Float = 10.0
        var overlapSeconds: Float = 3.0
        var odeMethod: SAMAudioODEMethod = .midpoint
        var stepSize: Float = 2.0 / 32.0
        var odeDecodeChunkSize: Int?
        var anchors: [SAMAudioAnchor] = []
        var strict = false
        var hfToken: String?
        var stream = false

        // LFM-specific
        var lfmMode: LFMMode?
        var systemPrompt: String?
        var maxNewTokens = 512
        var temperature: Float = 0.8
        var topK = 50
        var audioTemperature: Float = 0.7
        var audioTopK = 30

        var iterator = CommandLine.arguments.dropFirst().makeIterator()
        while let arg = iterator.next() {
            switch arg {
            case "--audio", "-i":
                guard let value = iterator.next() else { throw CLIError.missingValue(arg) }
                audioPath = value
            case "--model":
                guard let value = iterator.next() else { throw CLIError.missingValue(arg) }
                model = value
            case "--text", "-t":
                guard let value = iterator.next() else { throw CLIError.missingValue(arg) }
                text = value
            case "--description", "--prompt", "-d":
                guard let value = iterator.next() else { throw CLIError.missingValue(arg) }
                description = value
            case "--mode":
                guard let value = iterator.next() else { throw CLIError.missingValue(arg) }
                // Try LFM mode first, then SAMAudio separation mode
                if let parsed = LFMMode(rawValue: value.lowercased()) {
                    lfmMode = parsed
                } else if let parsed = SeparationMode(rawValue: value.lowercased()) {
                    mode = parsed
                } else {
                    throw CLIError.invalidValue(arg, value)
                }
            case "--system":
                guard let value = iterator.next() else { throw CLIError.missingValue(arg) }
                systemPrompt = value
            case "--max-new-tokens":
                guard let value = iterator.next() else { throw CLIError.missingValue(arg) }
                guard let parsed = Int(value) else { throw CLIError.invalidValue(arg, value) }
                maxNewTokens = parsed
            case "--temperature":
                guard let value = iterator.next() else { throw CLIError.missingValue(arg) }
                guard let parsed = Float(value) else { throw CLIError.invalidValue(arg, value) }
                temperature = parsed
            case "--top-k":
                guard let value = iterator.next() else { throw CLIError.missingValue(arg) }
                guard let parsed = Int(value) else { throw CLIError.invalidValue(arg, value) }
                topK = parsed
            case "--audio-temperature":
                guard let value = iterator.next() else { throw CLIError.missingValue(arg) }
                guard let parsed = Float(value) else { throw CLIError.invalidValue(arg, value) }
                audioTemperature = parsed
            case "--audio-top-k":
                guard let value = iterator.next() else { throw CLIError.missingValue(arg) }
                guard let parsed = Int(value) else { throw CLIError.invalidValue(arg, value) }
                audioTopK = parsed
            case "--stream":
                stream = true
            case "--output-target", "-o":
                guard let value = iterator.next() else { throw CLIError.missingValue(arg) }
                outputTargetPath = value
            case "--output-text":
                guard let value = iterator.next() else { throw CLIError.missingValue(arg) }
                outputTextPath = value
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

        return CLI(
            model: model,
            audioPath: audioPath,
            text: text,
            outputTargetPath: outputTargetPath,
            outputTextPath: outputTextPath,
            hfToken: hfToken,
            stream: stream,
            description: description,
            mode: mode,
            outputResidualPath: outputResidualPath,
            writeResidual: writeResidual,
            chunkSeconds: chunkSeconds,
            overlapSeconds: overlapSeconds,
            odeMethod: odeMethod,
            stepSize: stepSize,
            odeDecodeChunkSize: odeDecodeChunkSize,
            anchors: anchors,
            strict: strict,
            lfmMode: lfmMode,
            systemPrompt: systemPrompt,
            maxNewTokens: maxNewTokens,
            temperature: temperature,
            topK: topK,
            audioTemperature: audioTemperature,
            audioTopK: audioTopK
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
              \(executable) [--model <repo>] [--mode <mode>] [options]

            Description:
              Runs STS (Speech-to-Speech) models. Supports SAM Audio source separation
              and LFM2.5-Audio multimodal generation (text-to-text, text-to-speech,
              speech-to-text, speech-to-speech).

            Model Selection:
              --model <repo>               Model repo or local path.
                                           SAM Audio default: \(SAMAudio.defaultRepo)
                                           LFM example: mlx-community/LFM2.5-Audio-1.5B-6bit

            LFM2.5-Audio Options:
              --mode <t2t|tts|stt|sts>     LFM generation mode.
                                             t2t: text-to-text
                                             tts: text-to-speech
                                             stt: speech-to-text
                                             sts: speech-to-speech (default)
              -t, --text <string>          Input text (required for t2t/tts, optional prompt for stt)
              -i, --audio <path>           Input audio file (required for stt/sts)
              --system <string>            System prompt (overrides per-mode default)
              --max-new-tokens <int>       Max tokens to generate. Default: 512
              --temperature <float>        Text sampling temperature. Default: 0.8
              --top-k <int>                Text top-K. Default: 50
              --audio-temperature <float>  Audio sampling temperature. Default: 0.7
              --audio-top-k <int>          Audio top-K. Default: 30
              --stream                     Stream text output to stdout
              -o, --output-target <path>   Audio WAV output path. Default: lfm_output.wav
              --output-text <path>         Text output path (optional)

            SAM Audio Options:
              --mode <short|long|stream>   Separation mode. Default: short
              -i, --audio <path>           Input audio file (required)
              -d, --description <text>     Target description. Default: speech
              -o, --output-target <path>   Target WAV output. Default: <input>.target.wav
              --output-residual <path>     Residual WAV output. Default: <input>.residual.wav
              --no-residual                Skip residual write
              --chunk-seconds <float>      Chunk duration for long/stream. Default: 10.0
              --overlap-seconds <float>    Overlap for long/stream. Default: 3.0
              --ode-method <method>        midpoint or euler. Default: midpoint
              --step-size <float>          ODE step size. Default: 0.0625
              --decode-chunk-size <n>      Optional decoder chunk size
              --anchor <tok:start:end>     Anchor (short mode only, repeatable)
              --strict                     Strict weight loading

            Common:
              --hf-token <token>           Hugging Face token (or set HF_TOKEN env var)
              -h, --help                   Show this help
            """
        )
    }
}
