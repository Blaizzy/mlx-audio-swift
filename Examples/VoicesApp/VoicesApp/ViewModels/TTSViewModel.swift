import Foundation
import SwiftUI
import MLXAudioTTS
import MLXAudioCore
import MLX
import AVFoundation
import Combine

@MainActor
@Observable
class TTSViewModel {
    var isLoading = false
    var isGenerating = false
    var generationProgress: String = ""
    var errorMessage: String?
    var audioURL: URL?
    var tokensPerSecond: Double = 0

    // Generation parameters
    var maxTokens: Int = 1200
    var temperature: Float = 0.6
    var topP: Float = 0.8

    // Model configuration
    var modelId: String = "mlx-community/VyvoTTS-EN-Beta-4bit"
    private var loadedModelId: String?

    // Audio player state (manually synced from AudioPlayerManager)
    var isPlaying: Bool = false
    var currentTime: TimeInterval = 0
    var duration: TimeInterval = 0

    private var model: Qwen3Model?
    private let audioPlayer = AudioPlayerManager()
    private var cancellables = Set<AnyCancellable>()

    var isModelLoaded: Bool {
        model != nil
    }

    init() {
        setupAudioPlayerObservers()
    }

    private func setupAudioPlayerObservers() {
        audioPlayer.$isPlaying
            .receive(on: DispatchQueue.main)
            .sink { [weak self] value in
                self?.isPlaying = value
            }
            .store(in: &cancellables)

        audioPlayer.$currentTime
            .receive(on: DispatchQueue.main)
            .sink { [weak self] value in
                self?.currentTime = value
            }
            .store(in: &cancellables)

        audioPlayer.$duration
            .receive(on: DispatchQueue.main)
            .sink { [weak self] value in
                self?.duration = value
            }
            .store(in: &cancellables)
    }

    func loadModel() async {
        // Skip if same model already loaded
        guard model == nil || loadedModelId != modelId else { return }

        isLoading = true
        errorMessage = nil
        generationProgress = "Downloading model..."

        do {
            model = try await Qwen3Model.fromPretrained(modelId)
            loadedModelId = modelId
            generationProgress = ""  // Clear progress on success
        } catch {
            errorMessage = "Failed to load model: \(error.localizedDescription)"
            generationProgress = ""
        }

        isLoading = false
    }

    func reloadModel() async {
        // Unload current model and clear GPU memory
        model = nil
        loadedModelId = nil
        Memory.clearCache()

        await loadModel()
    }

    func synthesize(text: String, voice: Voice? = nil) async {
        guard let model = model else {
            errorMessage = "Model not loaded"
            return
        }

        guard !text.isEmpty else {
            errorMessage = "Please enter text to synthesize"
            return
        }

        isGenerating = true
        errorMessage = nil
        generationProgress = "Starting generation..."
        tokensPerSecond = 0

        do {
            var tokenCount = 0
            var audio: MLXArray?

            for try await event in model.generateStream(
                text: text,
                voice: voice?.name,
                parameters: .init(
                    maxTokens: maxTokens,
                    temperature: temperature,
                    topP: topP,
                    repetitionPenalty: 1.3,
                    repetitionContextSize: 20
                )
            ) {
                switch event {
                case .token:
                    tokenCount += 1
                    if tokenCount % 10 == 0 {
                        generationProgress = "Generated \(tokenCount) tokens..."
                    }
                case .info(let info):
                    tokensPerSecond = info.tokensPerSecond
                    generationProgress = "Processing audio..."
                case .audio(let audioData):
                    audio = audioData
                }
            }

            guard let audioData = audio else {
                throw NSError(
                    domain: "TTSViewModel",
                    code: 1,
                    userInfo: [NSLocalizedDescriptionKey: "No audio generated"]
                )
            }

            // Save audio to temp file
            let tempURL = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString)
                .appendingPathExtension("wav")

            try saveAudioArray(audioData, sampleRate: Double(model.sampleRate), to: tempURL)

            audioURL = tempURL
            generationProgress = ""  // Clear progress

            // Load audio for playback
            audioPlayer.loadAudio(from: tempURL)

        } catch {
            errorMessage = "Generation failed: \(error.localizedDescription)"
            generationProgress = ""
        }
        
        Memory.clearCache()

        isGenerating = false
    }

    func play() {
        audioPlayer.play()
    }

    func pause() {
        audioPlayer.pause()
    }

    func togglePlayPause() {
        audioPlayer.togglePlayPause()
    }

    func stop() {
        audioPlayer.stop()
    }

    func seek(to time: TimeInterval) {
        audioPlayer.seek(to: time)
    }
}
