import AVFoundation
import MLX
import MLXAudioCore
import MLXAudioTTS
import MLXLMCommon

protocol SpeechControllerDelegate: AnyObject {
    func speechController(_ controller: SpeechController, didFinish transcription: String)
}

@Observable
final class SpeechController {
    @ObservationIgnored
    weak var delegate: SpeechControllerDelegate?

    private(set) var isActive: Bool = false
    private(set) var isDetectingSpeech = false
    private(set) var canSpeak: Bool = false
    private(set) var isSpeaking: Bool = false

    var isMicrophoneMuted: Bool {
        audioEngine.isMicrophoneMuted
    }

    @ObservationIgnored
    private let audioEngine: AudioEngine
    @ObservationIgnored
    private var configuredAudioEngine = false
    @ObservationIgnored
    private let vad: SimpleVAD
    @ObservationIgnored
    private var model: SpeechGenerationModel?

    init(ttsRepoId: String = "mlx-community/pocket-tts") {
        self.audioEngine = AudioEngine(inputBufferSize: 1024)
        self.vad = SimpleVAD()
        audioEngine.delegate = self
        vad.delegate = self

        Task {
            do {
                print("Loading TTS model: \(ttsRepoId)")
                self.model = try await TTSModelUtils.loadModel(modelRepo: ttsRepoId)
                print("Loaded TTS model.")
            } catch {
                print("Error loading model: \(error)")
            }
            self.canSpeak = model != nil
        }
    }

    func start() async throws {
        let session = AVAudioSession.sharedInstance()
        try session.setActive(false)
        try session.setCategory(.playAndRecord, mode: .voiceChat, policy: .default, options: [.defaultToSpeaker])
        try session.setPreferredIOBufferDuration(0.02)
        try session.setActive(true)

        try await ensureEngineStarted()
        isActive = true
    }

    func stop() async throws {
        audioEngine.endSpeaking()
        audioEngine.stop()
        isDetectingSpeech = false
        vad.reset()
        try AVAudioSession.sharedInstance().setActive(false)
        isActive = false
    }

    func toggleInputMute(toMuted: Bool?) async {
        let currentMuted = audioEngine.isMicrophoneMuted
        let newMuted = toMuted ?? !currentMuted
        audioEngine.isMicrophoneMuted = newMuted

        if newMuted, isDetectingSpeech {
            vad.reset()
            isDetectingSpeech = false
        }
    }

    func stopSpeaking() async {
        audioEngine.endSpeaking()
    }

    func speak(text: String) async throws {
        guard let model else {
            print("Error: TTS model not yet loaded.")
            return
        }

        let audioStream = model.generateStream(
            text: text,
            voice: "cosette",
            refAudio: nil,
            refText: nil,
            language: "en"
        )
        try await ensureEngineStarted()

        audioEngine.speak(samplesStream: proxyAudioStream(audioStream, extract: {
            switch $0 {
            case .audio(let samples): samples.asArray(Float.self)
            default: fatalError("Unsupported sample type.")
            }
        }))
    }

    private func ensureEngineStarted() async throws {
        if !configuredAudioEngine {
            try audioEngine.setup()
            configuredAudioEngine = true
            print("Configured audio engine.")
        }
        try audioEngine.start()
        audioEngine.isMicrophoneMuted = false
        print("Started audio engine.")
    }

    private func proxyAudioStream<T>(_ upstream: AsyncThrowingStream<T, any Error>, extract: @escaping (T) -> [Float]) -> AsyncThrowingStream<[Float], any Error> {
        AsyncThrowingStream<[Float], any Error> { continuation in
            let task = Task {
                do {
                    for try await value in upstream {
                        continuation.yield(extract(value))
                    }
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish(throwing: CancellationError())
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }
}

// MARK: - AudioEngineDelegate

extension SpeechController: AudioEngineDelegate {
    func audioCaptureEngine(_ engine: AudioEngine, didReceive buffer: AVAudioPCMBuffer) {
        guard !audioEngine.isSpeaking else { return }

        Task {
            vad.process(buffer: buffer)
        }
    }

    func audioCaptureEngine(_ engine: AudioEngine, isSpeakingDidChange speaking: Bool) {
        isSpeaking = speaking
    }
}

// MARK: - SimpleVADDelegate

extension SpeechController: SimpleVADDelegate {
    func didStartSpeaking() {
        isDetectingSpeech = true
    }

    func didStopSpeaking(transcription: String?) {
        if let transcription {
            delegate?.speechController(self, didFinish: transcription)
        }
        vad.reset()
        isDetectingSpeech = false
    }
}
