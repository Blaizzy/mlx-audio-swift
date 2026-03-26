@preconcurrency import AVFoundation
import Foundation
@preconcurrency import MLX
import MLXAudioCore
import MLXAudioSTS

// MARK: - Real-Time DeepFilterNet Audio Demo
//
// A real-time audio processing pipeline that applies DeepFilterNet noise reduction
// inline as audio "arrives" — simulating a phone call where incoming audio is
// processed before reaching your ears.
//
// The audio source (currently a looping file, but designed to be swapped for a
// microphone or live call input) feeds into a processing thread that runs at
// real-time pace via backpressure from the audio output. The processing thread
// can only get a few hops ahead of what the speaker is actually playing.
//
// Controls (single keypress, no Enter needed):
//   Space     Play / Pause
//   e         Toggle DeepFilterNet effect on/off
//   q         Quit
//
// Usage:
//   mlx-audio-swift-demo --audio <path.wav> [--model <path-or-repo>]

@main
enum App {
    static func main() {
        do {
            let args = try parseArgs()
            try run(args)
        } catch {
            fputs("Error: \(error)\n", stderr)
            printUsage()
            exit(1)
        }
    }

    // MARK: - Args

    struct Args {
        var audioPath: String
        var modelPath: String?
    }

    static func parseArgs() throws -> Args {
        let argv = CommandLine.arguments
        var audioPath: String?
        var modelPath: String?
        var i = 1
        while i < argv.count {
            switch argv[i] {
            case "--audio", "-i":
                i += 1
                guard i < argv.count else { throw DemoError.missingArgValue("--audio") }
                audioPath = argv[i]
            case "--model", "-m":
                i += 1
                guard i < argv.count else { throw DemoError.missingArgValue("--model") }
                modelPath = argv[i]
            case "--help", "-h":
                printUsage()
                exit(0)
            default:
                if audioPath == nil {
                    audioPath = argv[i]
                } else {
                    fputs("Unknown argument: \(argv[i])\n", stderr)
                }
            }
            i += 1
        }
        guard let audioPath else { throw DemoError.missingAudio }
        return Args(audioPath: audioPath, modelPath: modelPath)
    }

    static func printUsage() {
        print("""
        Usage: mlx-audio-swift-demo --audio <path.wav> [--model <path-or-repo>]

        Real-time audio processing demo. Applies DeepFilterNet noise reduction
        inline as audio plays, simulating a live phone call.

        Controls (single keypress):
          Space     Play / Pause
          e         Toggle effect on/off
          q         Quit

        Options:
          --audio, -i   Input audio file (WAV, loops continuously)
          --model, -m   DeepFilterNet model path (local directory)
        """)
    }

    // MARK: - Main

    static func run(_ args: Args) throws {
        // --- Load model ---
        let model = try loadModel(args.modelPath)
        print("Model: \(model.modelVersion) (sr=\(model.sampleRate), hop=\(model.config.hopSize))")

        // --- Load audio ---
        let inputURL = URL(fileURLWithPath: args.audioPath).standardizedFileURL
        guard FileManager.default.fileExists(atPath: inputURL.path) else {
            throw DemoError.fileNotFound(args.audioPath)
        }
        let (inputSR, rawAudio) = try loadAudioArray(from: inputURL, sampleRate: model.sampleRate)
        let samples = rawAudio.asArray(Float.self)
        let duration = Double(samples.count) / Double(model.sampleRate)
        print("Audio: \(samples.count) samples (\(String(format: "%.1f", duration))s at \(inputSR)Hz)")

        // --- Create streamer ---
        let streamer = model.createStreamer(
            config: DeepFilterNetStreamingConfig(
                padEndFrames: 0,
                compensateDelay: true,
                enableProfiling: true,
                profilingForceEvalPerStage: false,
                materializeEveryHops: 1
            )
        )

        // --- Setup AVAudioEngine ---
        let sampleRate = Double(model.sampleRate)
        let hopSize = model.config.hopSize
        let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)!

        let engine = AVAudioEngine()
        let playerNode = AVAudioPlayerNode()
        engine.attach(playerNode)
        engine.connect(playerNode, to: engine.mainMixerNode, format: format)
        engine.prepare()
        try engine.start()
        playerNode.play()

        // --- Shared state ---
        let state = PlaybackState()

        // --- Backpressure ---
        // Limits how far ahead the processing thread can get.
        // The player consumes buffers at real-time rate (10ms each).
        // With maxAhead=4, processing can be at most ~40ms ahead of playback.
        // When all slots are used, the processing thread blocks until the
        // player finishes one — this IS the real-time pacing mechanism.
        let maxAhead = 4
        let bufferSlots = DispatchSemaphore(value: maxAhead)

        print("")
        print("Playing with DeepFilterNet effect ON (looping)")
        print("   Space = play/pause | e = toggle effect | q = quit")
        print("")

        // --- Processing thread ---
        // Reads audio hop-by-hop (simulating incoming audio from a call),
        // processes through DeepFilterNet, and schedules output to the player.
        // Backpressure from bufferSlots keeps it at real-time pace.
        let processingThread = Thread {
            // Prime the streamer pipeline with silence so it produces output
            // immediately when real audio arrives.
            let delayHops = (model.config.fftSize - hopSize) / hopSize + model.config.convLookahead
            for _ in 0..<delayHops {
                let _ = try? streamer.processChunk([Float](repeating: 0.0, count: hopSize))
            }

            var fileOffset = 0
            var loopCount = 0
            var hopCount = 0
            var processedHopTimes = [Double]()
            var bypassHopCount = 0

            while !state.quit {
                // --- Pause gate ---
                // When paused, spin-wait so we stop feeding audio.
                // The player node is also paused so existing buffers hold.
                while state.paused && !state.quit {
                    Thread.sleep(forTimeInterval: 0.05)
                }
                if state.quit { break }

                // --- Read next hop from source (simulated incoming audio) ---
                let end = min(fileOffset + hopSize, samples.count)
                let inputChunk = Array(samples[fileOffset..<end])
                fileOffset = end

                // Loop: when we reach the end, wrap around
                if fileOffset >= samples.count {
                    fileOffset = 0
                    loopCount += 1
                    // Reset streamer state for a clean loop transition
                    streamer.reset()
                    // Re-prime the pipeline
                    for _ in 0..<delayHops {
                        let _ = try? streamer.processChunk([Float](repeating: 0.0, count: hopSize))
                    }
                }

                // --- Process ---
                let t0 = CFAbsoluteTimeGetCurrent()

                // Always feed the streamer to keep GRU state warm
                let processedChunk: [Float]
                do {
                    processedChunk = try streamer.processChunk(inputChunk)
                } catch {
                    fputs("\nStreamer error: \(error)\n", stderr)
                    break
                }

                let outputChunk: [Float]
                if state.effectEnabled {
                    outputChunk = processedChunk.isEmpty
                        ? [Float](repeating: 0.0, count: inputChunk.count)
                        : processedChunk
                    let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0
                    processedHopTimes.append(elapsed)
                } else {
                    outputChunk = inputChunk
                    bypassHopCount += 1
                }

                // --- Backpressure: wait for a free playback slot ---
                // This blocks until the player has consumed a buffer,
                // keeping the processing thread paced to real-time.
                bufferSlots.wait()

                // --- Schedule for playback ---
                if !outputChunk.isEmpty {
                    let buffer = AVAudioPCMBuffer(
                        pcmFormat: format,
                        frameCapacity: AVAudioFrameCount(outputChunk.count)
                    )!
                    buffer.frameLength = AVAudioFrameCount(outputChunk.count)
                    outputChunk.withUnsafeBufferPointer { src in
                        buffer.floatChannelData![0].update(
                            from: src.baseAddress!, count: outputChunk.count
                        )
                    }
                    playerNode.scheduleBuffer(
                        buffer,
                        completionCallbackType: .dataConsumed
                    ) { _ in
                        // Buffer consumed by player — free up a slot
                        bufferSlots.signal()
                    }
                } else {
                    // Nothing to play, give back the slot
                    bufferSlots.signal()
                }

                hopCount += 1

                // Print progress at ~30fps (every ~33ms = 3 hops at 10ms/hop)
                if hopCount % 3 == 0 {
                    let progressSec = Double(fileOffset) / sampleRate
                    let effect = state.effectEnabled ? "ON " : "OFF"
                    let latencyStr: String
                    if !processedHopTimes.isEmpty {
                        let recent = processedHopTimes.suffix(100)
                        let avg = recent.reduce(0, +) / Double(recent.count)
                        latencyStr = String(format: "%.1fms/hop", avg)
                    } else {
                        latencyStr = "---"
                    }
                    fputs(String(
                        format: "\r  [%5.1fs / %5.1fs] loop=%d  effect=%@  latency=%@   ",
                        progressSec, duration, loopCount + 1, effect, latencyStr
                    ), stdout)
                    fflush(stdout)
                }
            }

            // --- Print final stats ---
            fputs("\r\n", stdout)
            if !processedHopTimes.isEmpty {
                let sorted = processedHopTimes.sorted()
                let count = sorted.count
                let avg = sorted.reduce(0, +) / Double(count)
                let median = sorted[count / 2]
                let p95 = sorted[Int(Double(count) * 0.95)]
                let p99 = sorted[min(Int(Double(count) * 0.99), count - 1)]
                let maxVal = sorted.last!
                let budget = Double(hopSize) / sampleRate * 1000.0
                let over = sorted.filter { $0 > budget }.count
                print(String(format: """
                Effect-ON stats (%d hops, budget=%.1fms):
                  avg=%.2fms  median=%.2fms  p95=%.2fms  p99=%.2fms  max=%.2fms
                  over budget: %d/%d (%.1f%%)
                """, count, budget, avg, median, p95, p99, maxVal,
                             over, count, Double(over) / Double(count) * 100.0))
            }
            if bypassHopCount > 0 {
                print("Effect-OFF hops: \(bypassHopCount)")
            }
            if let summary = streamer.profilingSummary() {
                print(summary)
            }

            state.done = true
        }

        processingThread.qualityOfService = .userInteractive
        processingThread.start()

        // --- Main thread: raw keyboard input ---
        let originalTermios = enableRawTerminalMode()

        while !state.quit {
            var byte: UInt8 = 0
            let n = read(STDIN_FILENO, &byte, 1)
            guard n == 1 else { continue }

            switch byte {
            case 0x20: // Space
                state.paused.toggle()
                if state.paused {
                    playerNode.pause()
                    fputs("\n   PAUSED\n", stdout)
                } else {
                    playerNode.play()
                    fputs("\n   PLAYING\n", stdout)
                }
                fflush(stdout)

            case 0x65, 0x45: // 'e' or 'E'
                state.effectEnabled.toggle()
                let mode = state.effectEnabled ? "ON" : "OFF"
                fputs("\n   Effect: \(mode)\n", stdout)
                fflush(stdout)

            case 0x71, 0x51: // 'q' or 'Q'
                state.quit = true
                fputs("\n   Quitting...\n", stdout)
                fflush(stdout)

            default:
                break
            }
        }

        restoreTerminalMode(originalTermios)

        // Wait for processing thread to finish
        while !state.done {
            Thread.sleep(forTimeInterval: 0.05)
        }

        Thread.sleep(forTimeInterval: 0.2)
        playerNode.stop()
        engine.stop()
    }

    // MARK: - Model Loading

    static func loadModel(_ pathOrRepo: String?) throws -> DeepFilterNetModel {
        if let pathOrRepo {
            let url = URL(fileURLWithPath: pathOrRepo).standardizedFileURL
            if FileManager.default.fileExists(atPath: url.path) {
                print("Loading model from: \(pathOrRepo)")
                return try DeepFilterNetModel.fromLocal(url)
            }
            throw DemoError.fileNotFound(pathOrRepo)
        }

        let home = NSHomeDirectory()
        let localPaths = [
            "\(home)/.cache/huggingface/hub/deepfilternet-mlx/iky1e_DeepFilterNet3-MLX",
            "\(home)/.cache/huggingface/hub/mlx-audio/iky1e_DeepFilterNet3-MLX",
            "\(home)/Library/Caches/huggingface/hub/mlx-audio/iky1e_DeepFilterNet3-MLX",
        ]
        for path in localPaths {
            if FileManager.default.fileExists(atPath: "\(path)/config.json") {
                print("Loading model from cache: \(path)")
                return try DeepFilterNetModel.fromLocal(URL(fileURLWithPath: path))
            }
        }

        fputs("""
        No cached DeepFilterNet model found. Provide a local model path:
          mlx-audio-swift-demo --model /path/to/DeepFilterNet3 --audio input.wav
        \n
        """, stderr)
        throw DemoError.modelNotFound
    }

    // MARK: - Raw Terminal Mode

    static func enableRawTerminalMode() -> termios {
        var original = termios()
        tcgetattr(STDIN_FILENO, &original)
        var raw = original
        // Disable canonical mode (line buffering) and echo
        raw.c_lflag &= ~UInt(ECHO | ICANON)
        // Read returns after 1 byte, no timeout
        raw.c_cc.4 = 1   // VMIN
        raw.c_cc.5 = 0   // VTIME
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw)
        return original
    }

    static func restoreTerminalMode(_ original: termios) {
        var orig = original
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig)
    }
}

// MARK: - Shared State

final class PlaybackState: @unchecked Sendable {
    var effectEnabled = true
    var paused = false
    var quit = false
    var done = false
}

// MARK: - Errors

enum DemoError: Error, LocalizedError, CustomStringConvertible {
    case missingAudio
    case missingArgValue(String)
    case fileNotFound(String)
    case modelNotFound

    var errorDescription: String? { description }

    var description: String {
        switch self {
        case .missingAudio:
            return "Missing required --audio argument."
        case .missingArgValue(let flag):
            return "Missing value for \(flag)."
        case .fileNotFound(let path):
            return "File not found: \(path)"
        case .modelNotFound:
            return "No DeepFilterNet model found. Provide --model <path>."
        }
    }
}
