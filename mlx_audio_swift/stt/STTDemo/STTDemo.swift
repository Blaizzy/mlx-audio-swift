import AVFoundation
import Foundation
import MLX
import MLXAudioSTT

@main
struct STTDemo {
    enum ProcessingMode: String, CaseIterable {
        case short
        case long
        case auto

        var description: String {
            switch self {
            case .short: return "Use WhisperSession directly (for audio <30s)"
            case .long: return "Use LongAudioProcessor with chunking"
            case .auto: return "Auto-select based on audio duration"
            }
        }
    }

    enum ChunkingStrategy: String, CaseIterable {
        case sequential
        case vad
        case sliding
        case auto

        var description: String {
            switch self {
            case .sequential: return "Sequential fixed-size chunks"
            case .vad: return "Voice Activity Detection based chunking"
            case .sliding: return "Sliding window with overlap"
            case .auto: return "Auto-select best strategy"
            }
        }

        func toLongAudioStrategy() -> LongAudioProcessor.StrategyType {
            switch self {
            case .sequential:
                return .sequential()
            case .vad:
                return .vad()
            case .sliding:
                return .slidingWindow()
            case .auto:
                return .auto
            }
        }
    }

    struct Config {
        var audioPath: String
        var modelName: String = "largeTurbo"
        var mode: ProcessingMode = .auto
        var strategy: ChunkingStrategy = .auto
        var language: String?
        var fast: Bool = false
        var verbose: Bool = false

        static let autoThresholdSeconds: Double = 30.0
    }

    static func main() async {
        let args = CommandLine.arguments

        guard args.count >= 2, args[1] != "--help", args[1] != "-h" else {
            printUsage()
            return
        }

        guard let config = parseArguments(args) else {
            return
        }

        guard let model = parseModel(config.modelName) else {
            print("Error: Unknown model '\(config.modelName)'")
            print("Available models: tiny, base, small, medium, largeV3, largeTurbo")
            return
        }

        do {
            let audio = try loadAudio(from: config.audioPath)
            let audioDuration = Double(audio.shape[0]) / Double(AudioConstants.sampleRate)
            print("Loaded audio: \(audio.shape[0]) samples (\(String(format: "%.1f", audioDuration))s)")

            let effectiveMode = resolveMode(config.mode, audioDuration: audioDuration)
            print("Processing mode: \(effectiveMode.rawValue)")

            if effectiveMode == .long {
                print("Chunking strategy: \(config.strategy.rawValue)")
            }

            let loadingMode = config.fast ? "fast (int4)" : "default (float16)"
            print("\nLoading \(config.modelName) model (\(loadingMode))...")

            switch effectiveMode {
            case .short:
                try await transcribeShort(audio: audio, model: model, language: config.language, fast: config.fast)
            case .long, .auto:
                try await transcribeLong(
                    audio: audio,
                    model: model,
                    strategy: config.strategy,
                    language: config.language,
                    fast: config.fast,
                    verbose: config.verbose
                )
            }

        } catch {
            print("Error: \(error)")
        }
    }

    static func resolveMode(_ mode: ProcessingMode, audioDuration: Double) -> ProcessingMode {
        switch mode {
        case .auto:
            return audioDuration > Config.autoThresholdSeconds ? .long : .short
        case .short, .long:
            return mode
        }
    }

    static func transcribeShort(audio: MLXArray, model: WhisperModel, language: String?, fast: Bool) async throws {
        let loadingOptions = fast ? ModelLoadingOptions.fast : ModelLoadingOptions.default
        let session = try await WhisperSession.fromPretrained(
            model: model,
            options: loadingOptions,
            progressHandler: createProgressHandler()
        )

        if fast {
            // Wait for background loading to complete
            _ = try await session.waitUntilReady()
        }

        var transcriptionOptions = TranscriptionOptions.default
        transcriptionOptions.language = language

        print("\nTranscribing...")
        print("─────────────────────────────────────────")

        for try await result in session.transcribe(audio, sampleRate: AudioConstants.sampleRate, options: transcriptionOptions) {
            if result.isFinal {
                print("\r\(result.text)")
            } else {
                print("\r\(result.text)...", terminator: "")
                fflush(stdout)
            }
        }

        print("─────────────────────────────────────────")
        print("Done!")
    }

    static func transcribeLong(
        audio: MLXArray,
        model: WhisperModel,
        strategy: ChunkingStrategy,
        language: String?,
        fast: Bool,
        verbose: Bool
    ) async throws {
        let loadingOptions = fast ? ModelLoadingOptions.fast : ModelLoadingOptions.default
        let processor = try await LongAudioProcessor.create(
            model: model,
            loadingOptions: loadingOptions,
            strategy: strategy.toLongAudioStrategy(),
            progressHandler: createProgressHandler()
        )

        if fast {
            // Wait for background loading to complete
            _ = try await processor.waitUntilReady()
        }

        var transcriptionOptions = TranscriptionOptions.default
        transcriptionOptions.language = language

        print("\nTranscribing long audio...")
        print("─────────────────────────────────────────")

        let stream: AsyncThrowingStream<TranscriptionProgress, Error> = processor.transcribe(
            audio,
            sampleRate: AudioConstants.sampleRate,
            options: transcriptionOptions
        )

        for try await progress in stream {
            let percent = Int(progress.progress * 100)
            let chunkInfo = "[\(progress.chunkIndex + 1)/\(progress.totalChunks)]"
            let timeInfo = String(format: "%.1f/%.1fs", progress.processedDuration, progress.audioDuration)

            if progress.isFinal {
                if verbose {
                    print("")  // Newline after streaming output
                }
                print("\r\u{1B}[K") // Clear line
                print("Progress: 100% \(chunkInfo) \(timeInfo)")
                print("─────────────────────────────────────────")
                if !verbose {
                    print(progress.text)
                }
            } else if verbose {
                if progress.isPartial {
                    // AlignAtt streaming: show word-by-word updates within a chunk
                    print("\r\u{1B}[K\u{1B}[90m[\(progress.chunkIndex + 1)]\u{1B}[0m \(progress.chunkText)", terminator: "")
                    fflush(stdout)
                } else {
                    // Chunk completed
                    print("\r\u{1B}[K\u{1B}[90m[\(progress.chunkIndex + 1)]\u{1B}[0m \(progress.chunkText)")
                }
            } else {
                print("\rProgress: \(percent)% \(chunkInfo) \(timeInfo)", terminator: "")
                fflush(stdout)
            }
        }

        print("─────────────────────────────────────────")
        print("Done!")
    }

    static func createProgressHandler() -> (WhisperProgress) -> Void {
        return { progress in
            switch progress {
            case .downloading(let fraction):
                print("\r  Downloading: \(Int(fraction * 100))%", terminator: "")
                fflush(stdout)
            case .loading(let fraction):
                if fraction >= 1.0 {
                    print("\r  Loading: done        ")
                }
            case .encoding:
                print("  Encoding audio...")
            case .decoding:
                break
            }
        }
    }

    static func parseArguments(_ args: [String]) -> Config? {
        var config = Config(audioPath: "")
        var i = 1

        while i < args.count {
            let arg = args[i]

            switch arg {
            case "--audio", "-a":
                guard i + 1 < args.count else {
                    print("Error: --audio requires a file path")
                    return nil
                }
                i += 1
                config.audioPath = args[i]

            case "--model", "-m":
                guard i + 1 < args.count else {
                    print("Error: --model requires a model name")
                    return nil
                }
                i += 1
                config.modelName = args[i]

            case "--mode":
                guard i + 1 < args.count else {
                    print("Error: --mode requires a value (short|long|auto)")
                    return nil
                }
                i += 1
                guard let mode = ProcessingMode(rawValue: args[i]) else {
                    print("Error: Invalid mode '\(args[i])'. Options: short, long, auto")
                    return nil
                }
                config.mode = mode

            case "--strategy":
                guard i + 1 < args.count else {
                    print("Error: --strategy requires a value (sequential|vad|sliding|auto)")
                    return nil
                }
                i += 1
                guard let strategy = ChunkingStrategy(rawValue: args[i]) else {
                    print("Error: Invalid strategy '\(args[i])'. Options: sequential, vad, sliding, auto")
                    return nil
                }
                config.strategy = strategy

            case "--language", "-l":
                guard i + 1 < args.count else {
                    print("Error: --language requires a language code")
                    return nil
                }
                i += 1
                config.language = args[i]

            case "--fast", "-f":
                config.fast = true

            default:
                // Positional argument: treat first as audio path, second as model
                if config.audioPath.isEmpty {
                    config.audioPath = arg
                } else if config.modelName == "largeTurbo" {
                    config.modelName = arg
                } else {
                    print("Error: Unknown argument '\(arg)'")
                    return nil
                }
            }

            i += 1
        }

        if config.audioPath.isEmpty {
            print("Error: Audio file path is required")
            printUsage()
            return nil
        }

        return config
    }

    static func printUsage() {
        print("""
        STT Demo - Speech to Text using Whisper

        Usage: stt-demo <audio-file> [model] [options]
               stt-demo --audio <file> [options]

        Arguments:
          audio-file              Path to audio file (wav, mp3, m4a, etc.)
          model                   Model size (default: largeTurbo)

        Options:
          --audio, -a <file>      Path to audio file
          --model, -m <model>     Model size: tiny, base, small, medium, largeV3, largeTurbo
          --fast, -f              Use fast loading (int4 quantization + background init)
          --mode <mode>           Processing mode:
                                    short  - Use WhisperSession directly (for audio <30s)
                                    long   - Use LongAudioProcessor with chunking
                                    auto   - Auto-select based on audio duration (default)
          --strategy <strategy>   Chunking strategy for long audio mode:
                                    sequential - Sequential fixed-size chunks
                                    vad        - Voice Activity Detection based chunking
                                    sliding    - Sliding window with overlap
                                    auto       - Auto-select best strategy (default)
          --language, -l <code>   Language code (e.g., en, ja, zh)
          --help, -h              Show this help message

        Examples:
          # Short audio (uses WhisperSession directly)
          stt-demo test.wav

          # Fast loading with int4 quantization (smaller model, faster load)
          stt-demo test.wav --fast

          # Long audio with sliding window strategy
          stt-demo --audio long.wav --mode long --strategy sliding

          # Long audio with VAD chunking
          stt-demo --audio long.wav --mode long --strategy vad

          # Auto mode - choose based on audio duration
          stt-demo --audio test.wav --mode auto

          # Specify language
          stt-demo speech.mp3 --language en
        """)
    }

    static func parseModel(_ name: String) -> WhisperModel? {
        switch name.lowercased() {
        case "tiny": return .tiny
        case "base": return .base
        case "small": return .small
        case "medium": return .medium
        case "largev3", "large-v3", "large_v3": return .largeV3
        case "largeturbo", "large-turbo", "large_turbo", "turbo": return .largeTurbo
        default: return nil
        }
    }

    static func loadAudio(from path: String) throws -> MLXArray {
        let url = URL(fileURLWithPath: path)

        guard FileManager.default.fileExists(atPath: path) else {
            throw AudioError.fileNotFound(path)
        }

        let file = try AVAudioFile(forReading: url)
        let format = file.processingFormat
        let frameCount = AVAudioFrameCount(file.length)

        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            throw AudioError.bufferCreationFailed
        }

        try file.read(into: buffer)

        guard let floatData = buffer.floatChannelData else {
            throw AudioError.noFloatData
        }

        let channelCount = Int(format.channelCount)
        let samples = Array(UnsafeBufferPointer(start: floatData[0], count: Int(frameCount)))

        var audio = MLXArray(samples)

        // Convert stereo to mono if needed
        if channelCount > 1 {
            let leftChannel = Array(UnsafeBufferPointer(start: floatData[0], count: Int(frameCount)))
            let rightChannel = Array(UnsafeBufferPointer(start: floatData[1], count: Int(frameCount)))
            let mono = zip(leftChannel, rightChannel).map { ($0 + $1) / 2.0 }
            audio = MLXArray(mono)
        }

        // Resample to 16kHz if needed
        let sourceSampleRate = Int(format.sampleRate)
        if sourceSampleRate != AudioConstants.sampleRate {
            audio = resample(audio, from: sourceSampleRate, to: AudioConstants.sampleRate)
        }

        return audio
    }

    static func resample(_ audio: MLXArray, from sourceSampleRate: Int, to targetSampleRate: Int) -> MLXArray {
        let ratio = Double(targetSampleRate) / Double(sourceSampleRate)
        let sourceLength = audio.shape[0]
        let targetLength = Int(Double(sourceLength) * ratio)

        // Simple linear interpolation resampling
        var resampled = [Float](repeating: 0, count: targetLength)
        let sourceData: [Float] = audio.asArray(Float.self)

        for i in 0..<targetLength {
            let sourceIndex = Double(i) / ratio
            let lower = Int(sourceIndex)
            let upper = min(lower + 1, sourceLength - 1)
            let fraction = Float(sourceIndex - Double(lower))

            resampled[i] = sourceData[lower] * (1 - fraction) + sourceData[upper] * fraction
        }

        return MLXArray(resampled)
    }
}

enum AudioError: Error, CustomStringConvertible {
    case fileNotFound(String)
    case bufferCreationFailed
    case noFloatData

    var description: String {
        switch self {
        case .fileNotFound(let path):
            return "Audio file not found: \(path)"
        case .bufferCreationFailed:
            return "Failed to create audio buffer"
        case .noFloatData:
            return "Audio file contains no float data"
        }
    }
}
