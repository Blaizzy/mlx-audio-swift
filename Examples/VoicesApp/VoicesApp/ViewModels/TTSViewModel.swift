import Foundation
import SwiftUI
import MLXAudioTTS
import MLXAudioCore
import MLX
import AVFoundation
import Combine

/// What voice controls to show based on current model
enum VoiceMode {
    case voiceList       // VyvoTTS - browse voice list
    case presetVoices    // Qwen3 CustomVoice - pick from preset names
    case voiceCloning    // Qwen3 Base - reference audio
    case voiceDesign     // Qwen3 VoiceDesign - describe the voice
}

/// Available TTS model types
enum TTSModelType: String, CaseIterable, Identifiable {
    case vyvoTTS = "VyvoTTS"
    case qwen3TTS = "Qwen3-TTS"

    var id: String { rawValue }

    var defaultModelId: String {
        switch self {
        case .vyvoTTS:
            return "mlx-community/VyvoTTS-EN-Beta-4bit"
        case .qwen3TTS:
            return "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit"
        }
    }

    var availableModels: [(id: String, name: String)] {
        switch self {
        case .vyvoTTS:
            return [
                ("mlx-community/VyvoTTS-EN-Beta-4bit", "VyvoTTS EN Beta (4-bit)")
            ]
        case .qwen3TTS:
            return [
                ("mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit", "0.6B CustomVoice (4-bit)"),
                ("mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit", "0.6B Base (4-bit)"),
                ("mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit", "1.7B VoiceDesign (4-bit)")
            ]
        }
    }
}

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
    var maxTokens: Int = 2000
    var temperature: Float = 0.6
    var topP: Float = 1.0

    // Text chunking
    var enableChunking: Bool = false
    var maxChunkLength: Int = 200
    var splitPattern: String = "\n"  // Can be regex like "\\n" or "[.!?]\\s+"

    // Streaming playback
    var streamingPlayback: Bool = true  // Play audio as chunks are generated

    // Model configuration
    var modelType: TTSModelType = .qwen3TTS
    var modelId: String = "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit"
    private(set) var loadedModelId: String?

    // Qwen3 voice selection
    var selectedQwen3Voice: String = "serena"
    var voiceInstruct: String = ""

    // Qwen3 voice cloning (Base model)
    var referenceAudioURL: URL?
    var referenceTranscription: String = ""

    static let qwen3Voices: [(id: String, name: String)] = [
        ("serena", "Serena"), ("vivian", "Vivian"), ("ryan", "Ryan"),
        ("aiden", "Aiden"), ("eric", "Eric"), ("dylan", "Dylan"),
        ("uncle_fu", "Uncle Fu"), ("ono_anna", "Ono Anna"), ("sohee", "Sohee")
    ]

    /// What voice controls to show based on current model
    var voiceMode: VoiceMode {
        switch modelType {
        case .vyvoTTS: return .voiceList
        case .qwen3TTS:
            if modelId.contains("Base") { return .voiceCloning }
            if modelId.contains("VoiceDesign") { return .voiceDesign }
            return .presetVoices
        }
    }

    // Audio player state (manually synced from AudioPlayerManager)
    var isPlaying: Bool = false
    var currentTime: TimeInterval = 0
    var duration: TimeInterval = 0

    private var model: SpeechGenerationModel?
    private let audioPlayer = AudioPlayerManager()
    private var cancellables = Set<AnyCancellable>()
    private var generationTask: Task<Void, Never>?

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
        generationProgress = "Loading model..."

        do {
            // Use TTSModelUtils which auto-routes based on model_type in config
            model = try await TTSModelUtils.loadModel(modelRepo: modelId)
            loadedModelId = modelId
            generationProgress = ""  // Clear progress on success
        } catch {
            errorMessage = "Failed to load model: \(error.localizedDescription)"
            generationProgress = ""
        }

        isLoading = false
    }

    /// Switch model type and update to default model for that type
    func switchModelType(to type: TTSModelType) {
        modelType = type
        modelId = type.defaultModelId
    }

    func reloadModel() async {
        // Unload current model and clear GPU memory
        model = nil
        loadedModelId = nil
        Memory.clearCache()

        await loadModel()
    }

    /// Split text into chunks based on pattern and max length
    private func chunkText(_ text: String) -> [String] {
        guard enableChunking && text.count > maxChunkLength else {
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
            let effectiveVoice: Voice?
            switch voiceMode {
            case .presetVoices:
                // Qwen3 CustomVoice: use preset voice name
                effectiveVoice = Voice(name: selectedQwen3Voice)
            case .voiceCloning:
                // Qwen3 Base: use reference audio for voice cloning
                if let refURL = referenceAudioURL {
                    effectiveVoice = Voice(
                        name: refURL.deletingPathExtension().lastPathComponent,
                        audioFileURL: refURL,
                        transcription: referenceTranscription.isEmpty ? nil : referenceTranscription
                    )
                } else {
                    effectiveVoice = nil
                }
            case .voiceDesign:
                // Qwen3 VoiceDesign: no voice object needed (instruct not yet in protocol)
                effectiveVoice = nil
            case .voiceList:
                // VyvoTTS: use the selected voice from VoicesView
                effectiveVoice = voice
            }
            await synthesize(text: text, voice: effectiveVoice)
        }
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
            // Load reference audio if this is a cloned voice
            var refAudio: MLXArray? = nil
            var refText: String? = nil

            if let voice = voice, voice.isClonedVoice,
               let audioURL = voice.audioFileURL,
               let transcription = voice.transcription {
                generationProgress = "Loading reference audio..."
                let (_, audioData) = try loadAudioArray(from: audioURL)
                refAudio = audioData
                refText = transcription
            }

            // Pass voice instruct for VoiceDesign/CustomVoice models
            let instruct: String? = voiceInstruct.isEmpty ? nil : voiceInstruct

            // Split text into chunks
            let chunks = chunkText(text)
            let sampleRate = Double(model.sampleRate)

            // Create streaming WAV writer - writes directly to file
            let tempURL = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString)
                .appendingPathExtension("wav")
            let wavWriter = try StreamingWAVWriter(url: tempURL, sampleRate: sampleRate)

            // Start streaming playback if enabled and we have multiple chunks
            let useStreaming = streamingPlayback && chunks.count > 1
            if useStreaming {
                audioPlayer.startStreaming(sampleRate: sampleRate)
            }

            for (index, chunk) in chunks.enumerated() {
                // Check for cancellation between chunks
                try Task.checkCancellation()

                if chunks.count > 1 {
                    generationProgress = "Processing chunk \(index + 1)/\(chunks.count)..."
                }

                var audio: MLXArray?

                // Set cache limit for this chunk
                Memory.cacheLimit = 100 * 1024 * 1024  // 100MB cache limit (match CLI)

                generationProgress = "Generating..."

                // Use the same non-streaming code path as the CLI
                let startTime = Date()
                let audioData = try await model.generate(
                    text: chunk,
                    voice: voice?.name,
                    refAudio: refAudio,
                    refText: refText,
                    language: nil,
                    instruct: instruct,
                    generationParameters: .init(
                        maxTokens: maxTokens,
                        temperature: temperature,
                        topP: topP,
                        repetitionPenalty: 1.0,
                        repetitionContextSize: 20
                    )
                )
                audio = audioData
                let elapsed = Date().timeIntervalSince(startTime)

                // Convert to CPU samples and write directly to file
                if let audioData = audio {
                    autoreleasepool {
                        let samples = audioData.asArray(Float.self)

                        // Stream playback immediately as chunks are ready
                        if useStreaming {
                            audioPlayer.scheduleAudioChunk(samples, withCrossfade: true)
                        }

                        // Write directly to file - no memory accumulation
                        try? wavWriter.writeChunk(samples)
                    }
                }
                audio = nil

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
            generationProgress = ""  // Clear progress

            // For single chunk, load normally for playback
            if !useStreaming {
                audioPlayer.loadAudio(from: finalURL)
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
}
