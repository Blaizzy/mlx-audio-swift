//
//  Qwen3TTSViewModel.swift
//  Qwen3TTS
//
//  ViewModel for Qwen3-TTS model loading and audio generation.
//

import Foundation
import AVFoundation
import Combine
import MLX
import MLXNN
import MLXAudioTTS

#if canImport(UIKit)
import UIKit
#endif

/// Available TTS models.
enum Qwen3Model: String, CaseIterable, Identifiable {
    case base4bit = "smdesai/Qwen3-TTS-12Hz-0.6B-Base-4bit"
    case customVoice4bit = "smdesai/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit"
    case customVoice8bit = "smdesai/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit"
    case voiceDesignBf16 = "smdesai/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .base4bit: return "0.6B Base (4-bit)"
        case .customVoice4bit: return "0.6B CustomVoice (4-bit)"
        case .customVoice8bit: return "0.6B CustomVoice (8-bit)"
        case .voiceDesignBf16: return "1.7B VoiceDesign (bf16)"
        }
    }

    var description: String {
        switch self {
        case .base4bit: return "Base model with voice cloning support."
        case .customVoice4bit: return "Fast, smaller model with preset voices."
        case .customVoice8bit: return "Higher precision model with preset voices."
        case .voiceDesignBf16: return "Larger model with voice design via instructions."
        }
    }

    /// Whether this model uses voice names (CustomVoice) or instruct prompts (VoiceDesign)
    var usesVoiceNames: Bool {
        switch self {
        case .base4bit: return false
        case .customVoice4bit: return true
        case .customVoice8bit: return true
        case .voiceDesignBf16: return false
        }
    }

    /// Whether this model supports voice cloning
    var supportsVoiceCloning: Bool {
        switch self {
        case .base4bit: return true
        default: return false
        }
    }
}

/// Available voices for CustomVoice model.
enum Qwen3Voice: String, CaseIterable, Identifiable {
    case serena = "serena"
    case vivian = "vivian"
    case ryan = "ryan"
    case aiden = "aiden"
    case eric = "eric"
    case dylan = "dylan"
    case uncleFu = "uncle_fu"
    case onoAnna = "ono_anna"
    case sohee = "sohee"

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .uncleFu: return "Uncle Fu"
        case .onoAnna: return "Ono Anna"
        default: return rawValue.capitalized
        }
    }
}

/// Generation state.
enum Qwen3GenerationState: Equatable {
    case idle
    case loading
    case generating(step: Int, codes: Int)
    case decoding
    case playing
    case error(String)

    var isActive: Bool {
        switch self {
        case .idle, .error: return false
        default: return true
        }
    }

    var isGenerating: Bool {
        switch self {
        case .generating, .decoding: return true
        default: return false
        }
    }
}

/// ViewModel for TTS operations.
@MainActor
class Qwen3TTSViewModel: ObservableObject {
    // MARK: - Published Properties

    @Published var state: Qwen3GenerationState = .idle
    @Published var selectedModel: Qwen3Model = .customVoice4bit
    @Published var selectedVoice: Qwen3Voice = .serena
    @Published var voiceInstruct: String = "A calm, clear female voice with a professional tone."
    @Published var downloadProgress: Double = 0
    @Published var generationTime: TimeInterval = 0
    @Published var audioDuration: TimeInterval = 0
    @Published var lastAudioURL: URL?

    // MARK: - Private Properties

    private var model: Qwen3TTSModel?
    private var loadedModelId: Qwen3Model?

    // Audio playback
    private var audioPlayer: AVAudioPlayer?
    private var generationTask: Task<Void, Never>?

    // Generation parameters (adjustable)
    @Published var temperature: Float = 0.3
    @Published var topK: Int = 50
    @Published var topP: Float = 0.95
    @Published var maxTokens: Int = 2000
    @Published var repetitionPenalty: Float = 1.05

    // Audio sample rate
    private let sampleRate: Double = 24000

    // MARK: - Public Methods

    /// Load the TTS model from HuggingFace.
    func loadModel(forceReload: Bool = false) async {
        // Skip if same model already loaded
        if !forceReload && model != nil && loadedModelId == selectedModel {
            return
        }

        // Clear existing model if switching
        if loadedModelId != selectedModel {
            model = nil
            loadedModelId = nil
        }

        state = .loading
        downloadProgress = 0

        // Set memory cache limit based on model size
        let cacheLimit: Int
        switch selectedModel {
        case .base4bit, .customVoice4bit:
            cacheLimit = 300 * 1024 * 1024  // 300MB for 4-bit
        case .customVoice8bit:
            cacheLimit = 400 * 1024 * 1024  // 400MB for 8-bit
        case .voiceDesignBf16:
            cacheLimit = 800 * 1024 * 1024  // 800MB for bf16
        }
        GPU.set(cacheLimit: cacheLimit)

        // Setup audio session
        setupAudioSession()

        let modelRepo = selectedModel.rawValue

        do {
            print("Loading model from: \(modelRepo)")

            model = try await Qwen3TTSModel.load(from: modelRepo) { [weak self] progress in
                // Update progress on main actor
                // Note: Hub library only calls this at start (0%) and end (100%)
                // For per-file progress, the Progress object updates internally
                let fraction = progress.fractionCompleted
                Task { @MainActor in
                    self?.downloadProgress = fraction
                    print("Download progress: \(Int(fraction * 100))%")
                }
            }

            loadedModelId = selectedModel
            print("Model loaded successfully!")
            state = .idle
        } catch {
            print("Failed to load model: \(error)")
            state = .error("Failed to load model: \(error.localizedDescription)")
        }
    }

    /// Generate speech from text (does not auto-play).
    func generate(text: String) async {
        guard let model = model else {
            state = .error("Model not loaded")
            return
        }

        let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedText.isEmpty else {
            state = .error("Please enter some text")
            return
        }

        #if canImport(UIKit)
        // Check app is in foreground on iOS
        guard UIApplication.shared.applicationState == .active else {
            state = .error("Cannot generate while app is in background")
            return
        }
        #endif

        state = .generating(step: 0, codes: 0)
        generationTime = 0
        audioDuration = 0
        lastAudioURL = nil

        let startTime = Date()

        // Determine voice/instruct based on model type
        let voice: String? = selectedModel.usesVoiceNames ? selectedVoice.rawValue : nil
        let instruct: String? = selectedModel.usesVoiceNames ? nil : voiceInstruct

        do {
            // Generate audio
            let audio = try model.generate(
                text: trimmedText,
                temperature: temperature,
                topK: topK,
                topP: topP,
                maxTokens: maxTokens,
                repetitionPenalty: repetitionPenalty,
                speaker: voice,
                instruct: instruct
            ) { [weak self] step, codes in
                Task { @MainActor in
                    // Only update if still in generating state (avoid race with completion)
                    if case .generating = self?.state {
                        self?.state = .generating(step: step, codes: codes)
                    }
                }
            }

            // Check for cancellation
            if Task.isCancelled {
                state = .idle
                return
            }

            state = .decoding
            let samples = audio.asArray(Float.self)
            generationTime = Date().timeIntervalSince(startTime)
            audioDuration = Double(samples.count) / sampleRate

            print("Generated \(samples.count) samples, duration: \(audioDuration)s")

            // Save audio to file (but don't play)
            await saveAudioSamples(samples)

            state = .idle

        } catch {
            print("Generation failed: \(error)")
            if !Task.isCancelled {
                state = .error("Generation failed: \(error.localizedDescription)")
            }
        }
    }

    /// Start generation in a cancellable task.
    func startGeneration(text: String) {
        generationTask = Task {
            await generate(text: text)
        }
    }

    /// Stop ongoing generation.
    func stopGeneration() {
        generationTask?.cancel()
        generationTask = nil
        state = .idle
    }

    /// Play the last generated audio.
    func playAudio() async {
        guard let audioURL = lastAudioURL else { return }

        state = .playing

        do {
            audioPlayer = try AVAudioPlayer(contentsOf: audioURL)
            audioPlayer?.prepareToPlay()
            audioPlayer?.play()

            // Wait for playback to complete
            while audioPlayer?.isPlaying == true {
                try await Task.sleep(nanoseconds: 100_000_000) // 100ms
            }

            state = .idle
        } catch {
            print("Failed to play audio: \(error)")
            state = .error("Playback failed: \(error.localizedDescription)")
        }
    }

    /// Stop audio playback.
    func stopPlayback() {
        audioPlayer?.stop()
        audioPlayer = nil
        if case .playing = state {
            state = .idle
        }
    }

    /// Check if model is loaded.
    var isModelLoaded: Bool {
        model != nil
    }

    /// Check if audio has been generated and is ready to play.
    var hasGeneratedAudio: Bool {
        lastAudioURL != nil
    }

    // MARK: - Private Methods

    private func setupAudioSession() {
        #if canImport(UIKit)
        do {
            let audioSession = AVAudioSession.sharedInstance()
            try audioSession.setCategory(.playback, mode: .default, options: [])
            try audioSession.setActive(true)
            print("Audio session configured")
        } catch {
            print("Failed to configure audio session: \(error)")
        }
        #endif
    }

    private func saveAudioSamples(_ samples: [Float]) async {
        // Create WAV data
        let wavData = createWAVData(from: samples)

        // Save to documents directory
        let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let audioURL = documentsURL.appendingPathComponent("qwen3_tts_output.wav")

        do {
            try? FileManager.default.removeItem(at: audioURL)
            try wavData.write(to: audioURL)
            print("Saved audio to: \(audioURL.path)")

            lastAudioURL = audioURL
        } catch {
            print("Failed to save audio: \(error)")
        }
    }

    private func createWAVData(from samples: [Float]) -> Data {
        var data = Data()

        // Convert to 16-bit PCM
        let int16Samples = samples.map { sample -> Int16 in
            let clamped = max(-1.0, min(1.0, sample))
            return Int16(clamped * Float(Int16.max))
        }

        let numSamples = UInt32(int16Samples.count)
        let numChannels: UInt16 = 1
        let bitsPerSample: UInt16 = 16
        let sampleRateInt = UInt32(sampleRate)
        let byteRate = sampleRateInt * UInt32(numChannels) * UInt32(bitsPerSample) / 8
        let blockAlign = numChannels * bitsPerSample / 8
        let dataSize = numSamples * UInt32(blockAlign)
        let fileSize = 36 + dataSize

        // RIFF header
        data.append(contentsOf: "RIFF".utf8)
        data.append(contentsOf: withUnsafeBytes(of: fileSize.littleEndian) { Array($0) })
        data.append(contentsOf: "WAVE".utf8)

        // fmt chunk
        data.append(contentsOf: "fmt ".utf8)
        data.append(contentsOf: withUnsafeBytes(of: UInt32(16).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt16(1).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: numChannels.littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: sampleRateInt.littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: byteRate.littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: blockAlign.littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: bitsPerSample.littleEndian) { Array($0) })

        // data chunk
        data.append(contentsOf: "data".utf8)
        data.append(contentsOf: withUnsafeBytes(of: dataSize.littleEndian) { Array($0) })

        // Audio samples
        for sample in int16Samples {
            data.append(contentsOf: withUnsafeBytes(of: sample.littleEndian) { Array($0) })
        }

        return data
    }
}
