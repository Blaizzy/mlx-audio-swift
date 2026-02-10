import Foundation
import MLX
import MLXAudioTTS

@main
enum Qwen3TTSTest {
    static func main() async {
        // Disable stdout buffering for immediate output
        setbuf(stdout, nil)

        do {
            try await run()
        } catch {
            fputs("Error: \(error)\n", stderr)
            exit(1)
        }
    }

    static func run() async throws {
        let model = CommandLine.arguments.count > 1
            ? CommandLine.arguments[1]
            : "smdesai/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit"

        let text = CommandLine.arguments.count > 2
            ? CommandLine.arguments[2]
            : "Hello, this is a test of the Qwen3 text to speech system."

        let voice = CommandLine.arguments.count > 3
            ? CommandLine.arguments[3]
            : "serena"

        let output = CommandLine.arguments.count > 4
            ? CommandLine.arguments[4]
            : "/tmp/qwen3tts_test.wav"

        print("Loading model: \(model)")
        let ttsModel = try await Qwen3TTSModel.load(from: model)
        print("Model loaded successfully")

        print("Generating audio for: \"\(text)\"")
        print("Voice: \(voice)")

        let startTime = CFAbsoluteTimeGetCurrent()

        let audio = try ttsModel.generate(
            text: text,
            temperature: 1.0,
            topK: 50,
            topP: 0.95,
            maxTokens: 2000,
            speaker: voice
        )

        let elapsed = CFAbsoluteTimeGetCurrent() - startTime

        let samples = audio.shape[0]
        let duration = Float(samples) / Float(ttsModel.sampleRate)

        print(String(format: "Generation time: %.2fs", elapsed))
        print(String(format: "Audio duration: %.2fs (%d samples)", duration, samples))
        print(String(format: "RTF: %.2fx", Double(duration) / elapsed))

        let outputURL = URL(fileURLWithPath: output)
        try Qwen3TTSModel.saveWAV(audio: audio, to: outputURL)
        print("Saved to: \(outputURL.path)")

        print("\nPlay with: afplay \(outputURL.path)")
    }
}
