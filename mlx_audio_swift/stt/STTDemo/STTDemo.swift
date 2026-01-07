import AVFoundation
import Foundation
import MLX
import MLXAudioSTT

@main
struct STTDemo {
    static func main() async {
        let args = CommandLine.arguments

        guard args.count >= 2 else {
            printUsage()
            return
        }

        let audioPath = args[1]
        let modelName = args.count > 2 ? args[2] : "largeTurbo"

        guard let model = parseModel(modelName) else {
            print("Error: Unknown model '\(modelName)'")
            print("Available models: tiny, base, small, medium, largeV3, largeTurbo")
            return
        }

        do {
            let audio = try loadAudio(from: audioPath)
            print("Loaded audio: \(audio.shape[0]) samples (\(String(format: "%.1f", Double(audio.shape[0]) / 16000.0))s)")

            print("\nLoading \(modelName) model...")
            let session = try await WhisperSession.fromPretrained(
                model: model,
                progressHandler: { progress in
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
            )

            print("\nTranscribing...")
            print("─────────────────────────────────────────")

            for try await result in session.transcribe(audio, sampleRate: AudioConstants.sampleRate) {
                if result.isFinal {
                    print("\r\(result.text)")
                } else {
                    print("\r\(result.text)...", terminator: "")
                    fflush(stdout)
                }
            }

            print("─────────────────────────────────────────")
            print("Done!")

        } catch {
            print("Error: \(error)")
        }
    }

    static func printUsage() {
        print("""
        STT Demo - Speech to Text using Whisper

        Usage: stt-demo <audio-file> [model]

        Arguments:
          audio-file    Path to audio file (wav, mp3, m4a, etc.)
          model         Model size (default: largeTurbo)
                        Options: tiny, base, small, medium, largeV3, largeTurbo

        Examples:
          stt-demo speech.wav
          stt-demo speech.mp3 tiny
          stt-demo meeting.m4a medium
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
