import Foundation
import MLX
import MLXAudioSTT

struct Transcription: Identifiable {
    let id = UUID()
    let text: String
    let timestamp: Date
    let tokensPerSecond: Double
    let duration: TimeInterval
}

enum AppState: Equatable {
    case loading
    case ready
    case listening
    case transcribing
}

@MainActor
@Observable
final class TranscriptionViewModel {
    var state: AppState = .loading
    var transcriptions: [Transcription] = []
    var currentText: String = ""
    var tokensPerSecond: Double = 0
    var peakMemory: Double = 0
    var errorMessage: String?

    var energyThreshold: Float = 0.01 {
        didSet { bridge.vad.energyThreshold = energyThreshold }
    }

    var hangTime: Double = 1.5 {
        didSet { bridge.vad.hangTime = hangTime }
    }

    private var model: VoxtralRealtimeModel?
    private let audioCapture = AudioCapture()
    private let bridge = DelegateBridge()

    init() {
        bridge.viewModel = self
        audioCapture.delegate = bridge
        bridge.vad.delegate = bridge
    }

    func loadModel() async {
        state = .loading
        errorMessage = nil
        do {
            model = try await VoxtralRealtimeModel.fromPretrained(
                "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"
            )
            state = .ready
        } catch {
            errorMessage = "Failed to load model: \(error.localizedDescription)"
            state = .ready
        }
    }

    func startListening() {
        guard model != nil else { return }
        bridge.vad.reset()
        audioCapture.start()
        state = .listening
        errorMessage = nil
    }

    func stopListening() {
        audioCapture.stop()
        bridge.vad.reset()
        state = .ready
        currentText = ""
    }

    func transcribe(audio: [Float]) async {
        guard let model else { return }

        state = .transcribing
        currentText = ""
        let startTime = Date()

        do {
            let audioArray = MLXArray(audio).reshaped([1, audio.count])
            for try await event in model.generateStream(audio: audioArray) {
                switch event {
                case .token(let text):
                    currentText += text
                case .info(let info):
                    tokensPerSecond = info.tokensPerSecond
                    peakMemory = info.peakMemoryUsage
                case .result(let output):
                    let duration = Date().timeIntervalSince(startTime)
                    let transcription = Transcription(
                        text: output.text.trimmingCharacters(in: .whitespacesAndNewlines),
                        timestamp: startTime,
                        tokensPerSecond: output.generationTps,
                        duration: duration
                    )
                    if !transcription.text.isEmpty {
                        transcriptions.insert(transcription, at: 0)
                    }
                    currentText = ""
                    tokensPerSecond = output.generationTps
                    peakMemory = output.peakMemoryUsage
                }
            }
        } catch {
            errorMessage = "Transcription failed: \(error.localizedDescription)"
        }

        state = .listening
    }

    // MARK: - Delegate Bridge

    /// Bridges non-MainActor delegate callbacks to the @MainActor ViewModel.
    /// Owns the EnergyVAD so audio processing stays on the audio thread.
    final class DelegateBridge: AudioCaptureDelegate, EnergyVADDelegate {
        weak var viewModel: TranscriptionViewModel?
        let vad = EnergyVAD()

        func audioCapture(_ capture: AudioCapture, didReceiveBuffer buffer: [Float]) {
            vad.processBuffer(buffer)
        }

        func vadDidDetectSpeechStart() {
            // Speech detection is reflected by the VAD state;
            // the UI observes the ViewModel state which transitions on transcribe.
        }

        func vadDidDetectSpeechEnd(audio: [Float]) {
            let vm = viewModel
            Task { @MainActor in
                await vm?.transcribe(audio: audio)
            }
        }
    }
}
