import AVFoundation
import Combine
import Foundation
import MLX
import MLXAudioCore
import MLXAudioTTS
import MLXLMCommon
import SwiftUI
#if os(macOS)
import AppKit
#endif

@MainActor
@Observable
class TTSViewModel {
    private static let defaultModelId = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit"
    private static let modelIdStorageKey = "VoicesApp.TTSViewModel.modelId"
    private static let maxTokensStorageKey = "VoicesApp.TTSViewModel.maxTokens"
    private static let temperatureStorageKey = "VoicesApp.TTSViewModel.temperature"
    private static let topPStorageKey = "VoicesApp.TTSViewModel.topP"
    private static let repetitionPenaltyStorageKey = "VoicesApp.TTSViewModel.repetitionPenalty"
    private static let voiceDescriptionStorageKey = "VoicesApp.TTSViewModel.voiceDescription"
    private static let useVoiceDesignStorageKey = "VoicesApp.TTSViewModel.useVoiceDesign"
    private static let enableChunkingStorageKey = "VoicesApp.TTSViewModel.enableChunking"
    private static let maxChunkLengthStorageKey = "VoicesApp.TTSViewModel.maxChunkLength"
    private static let splitPatternStorageKey = "VoicesApp.TTSViewModel.splitPattern"
    private static let streamingPlaybackStorageKey = "VoicesApp.TTSViewModel.streamingPlayback"

    var isLoading = false
    var isGenerating = false
    var generationProgress: String = ""
    var errorMessage: String?
    var audioURL: URL?
    var tokensPerSecond: Double = 0

    // Generation parameters: model defaults + optional UI overrides
    private var defaultGenerationParameters = GenerateParameters(
        maxTokens: 1200,
        temperature: 0.6,
        topP: 0.8,
        repetitionPenalty: 1.3,
        repetitionContextSize: 20
    )
    private var maxTokensOverride: Int? = UserDefaults.standard.object(forKey: TTSViewModel.maxTokensStorageKey).map { _ in
        UserDefaults.standard.integer(forKey: TTSViewModel.maxTokensStorageKey)
    }
    private var temperatureOverride: Float? = UserDefaults.standard.object(forKey: TTSViewModel.temperatureStorageKey).map { _ in
        UserDefaults.standard.float(forKey: TTSViewModel.temperatureStorageKey)
    }
    private var topPOverride: Float? = UserDefaults.standard.object(forKey: TTSViewModel.topPStorageKey).map { _ in
        UserDefaults.standard.float(forKey: TTSViewModel.topPStorageKey)
    }
    private var repetitionPenaltyOverride: Float? = UserDefaults.standard.object(forKey: TTSViewModel.repetitionPenaltyStorageKey).map { _ in
        UserDefaults.standard.float(forKey: TTSViewModel.repetitionPenaltyStorageKey)
    }

    var maxTokens: Int {
        get { maxTokensOverride ?? defaultMaxTokens }
        set {
            maxTokensOverride = (newValue == defaultMaxTokens) ? nil : newValue
            if let maxTokensOverride {
                UserDefaults.standard.set(maxTokensOverride, forKey: TTSViewModel.maxTokensStorageKey)
            } else {
                UserDefaults.standard.removeObject(forKey: TTSViewModel.maxTokensStorageKey)
            }
        }
    }

    var temperature: Float {
        get { temperatureOverride ?? defaultGenerationParameters.temperature }
        set {
            let defaultValue = defaultGenerationParameters.temperature
            temperatureOverride = abs(newValue - defaultValue) < 0.0001 ? nil : newValue
            if let temperatureOverride {
                UserDefaults.standard.set(temperatureOverride, forKey: TTSViewModel.temperatureStorageKey)
            } else {
                UserDefaults.standard.removeObject(forKey: TTSViewModel.temperatureStorageKey)
            }
        }
    }

    var topP: Float {
        get { topPOverride ?? defaultGenerationParameters.topP }
        set {
            let defaultValue = defaultGenerationParameters.topP
            topPOverride = abs(newValue - defaultValue) < 0.0001 ? nil : newValue
            if let topPOverride {
                UserDefaults.standard.set(topPOverride, forKey: TTSViewModel.topPStorageKey)
            } else {
                UserDefaults.standard.removeObject(forKey: TTSViewModel.topPStorageKey)
            }
        }
    }
    
    var repetitionPenalty: Float {
        get { repetitionPenaltyOverride ?? defaultGenerationParameters.repetitionPenalty ?? 1.3 }
        set {
            let defaultValue = defaultGenerationParameters.repetitionPenalty ?? 1.3
            repetitionPenaltyOverride = abs(newValue - defaultValue) < 0.0001 ? nil : newValue
            if let repetitionPenaltyOverride {
                UserDefaults.standard.set(repetitionPenaltyOverride, forKey: TTSViewModel.repetitionPenaltyStorageKey)
            } else {
                UserDefaults.standard.removeObject(forKey: TTSViewModel.repetitionPenaltyStorageKey)
            }
        }
    }
    
    // Voice Design (for Qwen3-TTS VoiceDesign models)
    var voiceDescription: String = UserDefaults.standard.string(forKey: TTSViewModel.voiceDescriptionStorageKey) ?? "" {
        didSet {
            UserDefaults.standard.set(voiceDescription, forKey: TTSViewModel.voiceDescriptionStorageKey)
        }
    }
    var useVoiceDesign: Bool = UserDefaults.standard.bool(forKey: TTSViewModel.useVoiceDesignStorageKey) {
        didSet {
            UserDefaults.standard.set(useVoiceDesign, forKey: TTSViewModel.useVoiceDesignStorageKey)
        }
    }

    // Text chunking
    var enableChunking: Bool = UserDefaults.standard.object(forKey: TTSViewModel.enableChunkingStorageKey).map { _ in
        UserDefaults.standard.bool(forKey: TTSViewModel.enableChunkingStorageKey)
    } ?? true {
        didSet {
            UserDefaults.standard.set(enableChunking, forKey: TTSViewModel.enableChunkingStorageKey)
        }
    }
    var maxChunkLength: Int = UserDefaults.standard.object(forKey: TTSViewModel.maxChunkLengthStorageKey).map { _ in
        UserDefaults.standard.integer(forKey: TTSViewModel.maxChunkLengthStorageKey)
    } ?? 200 {
        didSet {
            UserDefaults.standard.set(maxChunkLength, forKey: TTSViewModel.maxChunkLengthStorageKey)
        }
    }
    var splitPattern: String = UserDefaults.standard.string(forKey: TTSViewModel.splitPatternStorageKey) ?? "\n" {
        didSet {
            UserDefaults.standard.set(splitPattern, forKey: TTSViewModel.splitPatternStorageKey)
        }
    } // Can be regex like "\\n" or "[.!?]\\s+"

    // Streaming playback
    var streamingPlayback: Bool = UserDefaults.standard.object(forKey: TTSViewModel.streamingPlaybackStorageKey).map { _ in
        UserDefaults.standard.bool(forKey: TTSViewModel.streamingPlaybackStorageKey)
    } ?? true {
        didSet {
            UserDefaults.standard.set(streamingPlayback, forKey: TTSViewModel.streamingPlaybackStorageKey)
        }
    } // Play audio as chunks are generated

    // Model configuration
    var modelId: String = UserDefaults.standard.string(forKey: TTSViewModel.modelIdStorageKey) ?? TTSViewModel.defaultModelId {
        didSet {
            UserDefaults.standard.set(modelId, forKey: TTSViewModel.modelIdStorageKey)
        }
    }
    private var loadedModelId: String?

    // Audio player state (manually synced from AudioPlayerManager)
    var isPlaying: Bool = false
    var currentTime: TimeInterval = 0
    var duration: TimeInterval = 0

    private var model: SpeechGenerationModel?
    private let audioPlayer = AudioPlayer()
    private var cancellables = Set<AnyCancellable>()
    private var generationTask: Task<Void, Never>?

    private var defaultMaxTokens: Int {
        defaultGenerationParameters.maxTokens ?? 1200
    }

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
            defaultGenerationParameters = model?.defaultGenerationParameters ?? defaultGenerationParameters
            model = try await TTS.loadModel(modelRepo: modelId)
            loadedModelId = modelId
            generationProgress = "" // Clear progress on success
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

    func resetGenerationParameterOverrides() {
        maxTokensOverride = nil
        temperatureOverride = nil
        topPOverride = nil
        repetitionPenaltyOverride = nil
        UserDefaults.standard.removeObject(forKey: TTSViewModel.maxTokensStorageKey)
        UserDefaults.standard.removeObject(forKey: TTSViewModel.temperatureStorageKey)
        UserDefaults.standard.removeObject(forKey: TTSViewModel.topPStorageKey)
        UserDefaults.standard.removeObject(forKey: TTSViewModel.repetitionPenaltyStorageKey)
    }

    func resetSettingsToDefaults() {
        modelId = TTSViewModel.defaultModelId
        resetGenerationParameterOverrides()
        useVoiceDesign = false
        voiceDescription = ""
        enableChunking = true
        maxChunkLength = 200
        splitPattern = "\n"
        streamingPlayback = true

        UserDefaults.standard.removeObject(forKey: TTSViewModel.modelIdStorageKey)
        UserDefaults.standard.removeObject(forKey: TTSViewModel.voiceDescriptionStorageKey)
        UserDefaults.standard.removeObject(forKey: TTSViewModel.useVoiceDesignStorageKey)
        UserDefaults.standard.removeObject(forKey: TTSViewModel.enableChunkingStorageKey)
        UserDefaults.standard.removeObject(forKey: TTSViewModel.maxChunkLengthStorageKey)
        UserDefaults.standard.removeObject(forKey: TTSViewModel.splitPatternStorageKey)
        UserDefaults.standard.removeObject(forKey: TTSViewModel.streamingPlaybackStorageKey)
    }

    private func effectiveGenerationParameters() -> GenerateParameters {
        var parameters = defaultGenerationParameters
        if let maxTokensOverride {
            parameters.maxTokens = maxTokensOverride
        }
        if let temperatureOverride {
            parameters.temperature = temperatureOverride
        }
        if let topPOverride {
            parameters.topP = topPOverride
        }
        if let repetitionPenaltyOverride {
            parameters.repetitionPenalty = repetitionPenaltyOverride
        }
        return parameters
    }

    /// Split text into chunks based on pattern and max length
    private func chunkText(_ text: String) -> [String] {
        guard enableChunking, text.count > maxChunkLength else {
            return [text]
        }

        // First split by pattern (supports regex)
        var segments: [String]
        if let regex = try? NSRegularExpression(pattern: splitPattern, options: []) {
            let range = NSRange(text.startIndex..., in: text)
            segments = regex.stringByReplacingMatches(in: text, range: range, withTemplate: "\u{0000}")
                .components(separatedBy: "\u{0000}")
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
        } else {
            // Fallback to simple string split
            segments = text.components(separatedBy: splitPattern)
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
        }

        // Group segments into chunks respecting max length
        var chunks: [String] = []
        var currentChunk = ""

        for segment in segments {
            if currentChunk.isEmpty {
                currentChunk = segment
            } else if currentChunk.count + segment.count + 1 <= maxChunkLength {
                currentChunk += " " + segment
            } else {
                chunks.append(currentChunk)
                currentChunk = segment
            }
        }

        if !currentChunk.isEmpty {
            chunks.append(currentChunk)
        }

        // Handle case where a single segment is too long - split by sentence boundaries
        var finalChunks: [String] = []
        for chunk in chunks {
            if chunk.count > maxChunkLength {
                // Try splitting by sentence boundaries
                let sentencePattern = "[.!?]+\\s*"
                if let sentenceRegex = try? NSRegularExpression(pattern: sentencePattern, options: []) {
                    let range = NSRange(chunk.startIndex..., in: chunk)
                    let sentences = sentenceRegex.stringByReplacingMatches(in: chunk, range: range, withTemplate: "$0\u{0000}")
                        .components(separatedBy: "\u{0000}")
                        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                        .filter { !$0.isEmpty }

                    var subChunk = ""
                    for sentence in sentences {
                        if subChunk.isEmpty {
                            subChunk = sentence
                        } else if subChunk.count + sentence.count + 1 <= maxChunkLength {
                            subChunk += " " + sentence
                        } else {
                            finalChunks.append(subChunk)
                            subChunk = sentence
                        }
                    }
                    if !subChunk.isEmpty {
                        finalChunks.append(subChunk)
                    }
                } else {
                    finalChunks.append(chunk)
                }
            } else {
                finalChunks.append(chunk)
            }
        }

        return finalChunks.isEmpty ? [text] : finalChunks
    }

    /// Start synthesis in a cancellable task
    func startSynthesis(text: String, voice: Voice? = nil) {
        generationTask = Task {
            await synthesize(text: text, voice: voice)
        }
    }

    func synthesize(text: String, voice: Voice? = nil) async {
        guard let model else {
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
            // Load reference audio if this is a cloned voice
            var refAudio: MLXArray?
            var refText: String?

            if let voice, voice.isClonedVoice,
               let audioURL = voice.audioFileURL,
               let transcription = voice.transcription {
                generationProgress = "Loading reference audio..."
                let (_, audioData) = try loadAudioArray(from: audioURL)
                refAudio = audioData
                refText = transcription
            }

            // Split text into chunks
            let chunks = chunkText(text)
            let sampleRate = Double(model.sampleRate)

            // Create streaming WAV writer - writes directly to file
            let tempURL = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString)
                .appendingPathExtension("wav")
            let wavWriter = try StreamingWAVWriter(url: tempURL, sampleRate: sampleRate)

            var totalTokenCount = 0

            // Streaming playback — model yields audio chunks progressively
            let useStreaming = streamingPlayback
            if useStreaming {
                audioPlayer.startStreaming(sampleRate: sampleRate)
            }

            for (index, chunk) in chunks.enumerated() {
                // Check for cancellation between chunks
                try Task.checkCancellation()

                if chunks.count > 1 {
                    generationProgress = "Processing chunk \(index + 1)/\(chunks.count)..."
                }

                var chunkTokenCount = 0

                // Set cache limit for this chunk
                Memory.cacheLimit = 512 * 1024 * 1024 // 512MB cache limit
                let generationParameters = effectiveGenerationParameters()

                // Determine voice parameter (VoiceDesign description or voice name)
                let voiceParam: String? = useVoiceDesign && !voiceDescription.isEmpty
                    ? voiceDescription
                    : voice?.name

                // Each chunk needs a fresh generation
                // Audio chunks arrive progressively during streaming
                for try await event in model.generateStream(
                    text: chunk,
                    voice: voiceParam,
                    refAudio: refAudio,
                    refText: refText,
                    language: nil,
                    generationParameters: generationParameters
                ) {
                    // Throw if cancelled - this will exit the loop and be caught below
                    try Task.checkCancellation()

                    switch event {
                    case .token:
                        chunkTokenCount += 1
                        totalTokenCount += 1
                        if chunkTokenCount % 50 == 0 {
                            if chunks.count > 1 {
                                generationProgress = "Chunk \(index + 1)/\(chunks.count): \(chunkTokenCount) tokens..."
                            } else {
                                generationProgress = "Generated \(chunkTokenCount) tokens..."
                            }
                        }
                    case .info(let info):
                        tokensPerSecond = info.tokensPerSecond
                    case .audio(let audioData):
                        autoreleasepool {
                            let samples = audioData.asArray(Float.self)

                            if useStreaming {
                                audioPlayer.scheduleAudioChunk(samples, withCrossfade: true)
                            }

                            try? wavWriter.writeChunk(samples)
                        }
                    }
                }

                // Clear GPU cache after each chunk
                Memory.clearCache()
            }

            // Finalize the WAV file
            let finalURL = wavWriter.finalize()

            guard wavWriter.framesWritten > 0 else {
                throw NSError(
                    domain: "TTSViewModel",
                    code: 1,
                    userInfo: [NSLocalizedDescriptionKey: "No audio generated"]
                )
            }

            Memory.clearCache()

            audioURL = finalURL
            generationProgress = "" // Clear progress

            // Finalize playback state once generation completes.
            if useStreaming {
                audioPlayer.finishStreamingInput()
            } else {
                audioPlayer.loadAudio(from: finalURL)
                audioPlayer.play()
            }

        } catch is CancellationError {
            // User cancelled - clean up silently
            audioPlayer.stop()
            Memory.clearCache()
            generationProgress = ""
        } catch {
            errorMessage = "Generation failed: \(error.localizedDescription)"
            generationProgress = ""
        }

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
        // Cancel any ongoing generation
        generationTask?.cancel()
        generationTask = nil

        // Stop audio playback
        audioPlayer.stop()

        // Reset state
        if isGenerating {
            isGenerating = false
            generationProgress = ""
        }
    }

    func seek(to time: TimeInterval) {
        audioPlayer.seek(to: time)
    }

    func saveAudioFile() {
        guard let audioURL = audioURL else { return }

        #if os(macOS)
        let savePanel = NSSavePanel()
        savePanel.allowedContentTypes = [.wav]
        savePanel.canCreateDirectories = true
        savePanel.isExtensionHidden = false
        savePanel.title = "Save Audio File"
        savePanel.nameFieldStringValue = "generated_audio.wav"

        savePanel.begin { response in
            if response == .OK, let destinationURL = savePanel.url {
                do {
                    // If file exists, remove it first
                    if FileManager.default.fileExists(atPath: destinationURL.path) {
                        try FileManager.default.removeItem(at: destinationURL)
                    }
                    // Copy the audio file to the selected location
                    try FileManager.default.copyItem(at: audioURL, to: destinationURL)
                } catch {
                    DispatchQueue.main.async {
                        self.errorMessage = "Failed to save file: \(error.localizedDescription)"
                    }
                }
            }
        }
        #else
        // iOS implementation would use UIDocumentPickerViewController
        // For now, just show error message
        errorMessage = "Save functionality is available on macOS"
        #endif
    }
}
