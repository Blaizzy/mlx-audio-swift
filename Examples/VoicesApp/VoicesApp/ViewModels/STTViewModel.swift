import Foundation
import SwiftUI
import MLXAudioSTT
import MLXAudioCore
import MLX
@preconcurrency import AVFoundation
import Combine

@MainActor
@Observable
class STTViewModel {
    private static let defaultModelId = "mlx-community/Qwen3-ASR-0.6B-4bit"
    private static let defaultMaxTokens = 1024
    private static let defaultTemperature: Float = 0.0
    private static let defaultLanguage = "English"
    private static let defaultChunkDuration: Float = 30.0
    private static let defaultStreamingDelayMs = 480
    private static let modelIdStorageKey = "VoicesApp.STTViewModel.modelId"
    private static let maxTokensStorageKey = "VoicesApp.STTViewModel.maxTokens"
    private static let temperatureStorageKey = "VoicesApp.STTViewModel.temperature"
    private static let languageStorageKey = "VoicesApp.STTViewModel.language"
    private static let chunkDurationStorageKey = "VoicesApp.STTViewModel.chunkDuration"
    private static let streamingDelayMsStorageKey = "VoicesApp.STTViewModel.streamingDelayMs"

    var isLoading = false
    var isGenerating = false
    var generationProgress: String = ""
    var errorMessage: String?
    var transcriptionText: String = ""
    var tokensPerSecond: Double = 0
    var peakMemory: Double = 0

    // Generation parameters
    var maxTokens: Int = UserDefaults.standard.object(forKey: STTViewModel.maxTokensStorageKey).map { _ in
        UserDefaults.standard.integer(forKey: STTViewModel.maxTokensStorageKey)
    } ?? STTViewModel.defaultMaxTokens {
        didSet {
            UserDefaults.standard.set(maxTokens, forKey: STTViewModel.maxTokensStorageKey)
        }
    }
    var temperature: Float = UserDefaults.standard.object(forKey: STTViewModel.temperatureStorageKey).map { _ in
        UserDefaults.standard.float(forKey: STTViewModel.temperatureStorageKey)
    } ?? STTViewModel.defaultTemperature {
        didSet {
            UserDefaults.standard.set(temperature, forKey: STTViewModel.temperatureStorageKey)
        }
    }
    var language: String = UserDefaults.standard.string(forKey: STTViewModel.languageStorageKey) ?? STTViewModel.defaultLanguage {
        didSet {
            UserDefaults.standard.set(language, forKey: STTViewModel.languageStorageKey)
        }
    }
    var chunkDuration: Float = UserDefaults.standard.object(forKey: STTViewModel.chunkDurationStorageKey).map { _ in
        UserDefaults.standard.float(forKey: STTViewModel.chunkDurationStorageKey)
    } ?? STTViewModel.defaultChunkDuration {
        didSet {
            UserDefaults.standard.set(chunkDuration, forKey: STTViewModel.chunkDurationStorageKey)
        }
    }

    // Streaming parameters
    var streamingDelayMs: Int = UserDefaults.standard.object(forKey: STTViewModel.streamingDelayMsStorageKey).map { _ in
        UserDefaults.standard.integer(forKey: STTViewModel.streamingDelayMsStorageKey)
    } ?? STTViewModel.defaultStreamingDelayMs {
        didSet {
            UserDefaults.standard.set(streamingDelayMs, forKey: STTViewModel.streamingDelayMsStorageKey)
        }
    } // .agent default

    // Model configuration
    var modelId: String = UserDefaults.standard.string(forKey: STTViewModel.modelIdStorageKey) ?? STTViewModel.defaultModelId {
        didSet {
            UserDefaults.standard.set(modelId, forKey: STTViewModel.modelIdStorageKey)
        }
    }
    private var loadedModelId: String?

    private var languageHint: String? {
        let trimmed = language.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    // Audio file
    var selectedAudioURL: URL?
    var audioFileName: String?

    // Audio player state
    var isPlaying: Bool = false
    var currentTime: TimeInterval = 0
    var duration: TimeInterval = 0

    // Recording state
    var isRecording: Bool { recorder.isRecording }
    var recordingDuration: TimeInterval { recorder.recordingDuration }
    var audioLevel: Float { recorder.audioLevel }

    private var model: (any STTGenerationModel)?
    private let audioPlayer = AudioPlayer()
    private let recorder = AudioRecorderManager()
    private var cancellables = Set<AnyCancellable>()
    private var generationTask: Task<Void, Never>?

    var isModelLoaded: Bool {
        model != nil
    }

    var canRecord: Bool {
        isRecording || (supportsRealtimeRecording && !isLoading && !isGenerating)
    }

    var supportsRealtimeRecording: Bool {
        model is Qwen3ASRModel || model is CohereTranscribeModel || model is MossTranscribeDiarizeModel
    }

    var usesRealtimeRecording: Bool {
        isRecording && streamingSession != nil
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
        guard model == nil || loadedModelId != modelId else { return }

        isLoading = true
        errorMessage = nil
        generationProgress = "Downloading model..."

        do {
            let loadedModel = try await loadSTTModel(modelId)
            model = loadedModel
            loadedModelId = modelId
            generationProgress = ""
        } catch {
            errorMessage = "Failed to load model: \(error.localizedDescription)"
            generationProgress = ""
        }

        isLoading = false
    }

    func reloadModel() async {
        model = nil
        loadedModelId = nil
        Memory.clearCache()
        await loadModel()
    }

    func resetSettingsToDefaults() {
        modelId = STTViewModel.defaultModelId
        maxTokens = STTViewModel.defaultMaxTokens
        temperature = STTViewModel.defaultTemperature
        language = STTViewModel.defaultLanguage
        chunkDuration = STTViewModel.defaultChunkDuration
        streamingDelayMs = STTViewModel.defaultStreamingDelayMs

        UserDefaults.standard.removeObject(forKey: STTViewModel.modelIdStorageKey)
        UserDefaults.standard.removeObject(forKey: STTViewModel.maxTokensStorageKey)
        UserDefaults.standard.removeObject(forKey: STTViewModel.temperatureStorageKey)
        UserDefaults.standard.removeObject(forKey: STTViewModel.languageStorageKey)
        UserDefaults.standard.removeObject(forKey: STTViewModel.chunkDurationStorageKey)
        UserDefaults.standard.removeObject(forKey: STTViewModel.streamingDelayMsStorageKey)
    }

    func selectAudioFile(_ url: URL) {
        selectedAudioURL = url
        audioFileName = url.lastPathComponent
        audioPlayer.loadAudio(from: url)
    }

    func removeAudioFile() {
        guard !isGenerating && !isRecording else { return }

        audioPlayer.unloadAudio()
        selectedAudioURL = nil
        audioFileName = nil
        currentTime = 0
        duration = 0
        isPlaying = false
    }

    func startTranscription() {
        guard let audioURL = selectedAudioURL else {
            errorMessage = "No audio file selected"
            return
        }

        generationTask = Task {
            await transcribe(audioURL: audioURL)
        }
    }

    func transcribe(audioURL: URL) async {
        guard let model = model else {
            errorMessage = "Model not loaded"
            return
        }

        isGenerating = true
        errorMessage = nil
        transcriptionText = ""
        generationProgress = "Loading audio..."
        resetGenerationStats()

        do {
            let (sampleRate, audioData) = try loadAudioArray(from: audioURL, sampleRate: 16_000)
            if sampleRate != 16_000 {
                generationProgress = "Resampling \(sampleRate)Hz → 16000Hz..."
            }

            try await transcribe(audioData: audioData, with: model)

            generationProgress = ""
        } catch is CancellationError {
            Memory.clearCache()
            generationProgress = ""
        } catch {
            errorMessage = "Transcription failed: \(error.localizedDescription)"
            generationProgress = ""
        }

        isGenerating = false
    }

    private func transcribeRecording(audioData: MLXArray, clearExistingText: Bool = true) async {
        guard let model = model else {
            errorMessage = "Model not loaded"
            return
        }

        isGenerating = true
        errorMessage = nil
        if clearExistingText {
            transcriptionText = ""
        }
        generationProgress = "Transcribing recording..."
        resetGenerationStats()

        do {
            try await transcribe(audioData: audioData, with: model)
            generationProgress = ""
        } catch is CancellationError {
            Memory.clearCache()
            generationProgress = ""
        } catch {
            errorMessage = "Transcription failed: \(error.localizedDescription)"
            generationProgress = ""
        }

        isGenerating = false
    }

    private func transcribe(audioData: MLXArray, with model: any STTGenerationModel) async throws {
        generationProgress = "Transcribing..."

        let parameters = generationParameters(for: model)

        var tokenCount = 0
        for try await event in model.generateStream(audio: audioData, generationParameters: parameters) {
            try Task.checkCancellation()

            switch event {
            case .token(let token):
                transcriptionText += token
                tokenCount += 1
                generationProgress = "Transcribing... \(tokenCount) tokens"
            case .info(let info):
                tokensPerSecond = info.tokensPerSecond
                peakMemory = info.peakMemoryUsage
            case .result(let output):
                if transcriptionText.isEmpty {
                    transcriptionText = output.text
                }
                tokensPerSecond = output.generationTps
                peakMemory = output.peakMemoryUsage
                generationProgress = ""
            }
        }
    }

    private func generationParameters(for model: any STTGenerationModel) -> STTGenerateParameters {
        let defaultParameters = model.defaultGenerationParameters
        return STTGenerateParameters(
            maxTokens: maxTokens,
            temperature: temperature,
            topP: defaultParameters.topP,
            topK: defaultParameters.topK,
            verbose: false,
            language: languageHint,
            chunkDuration: chunkDuration,
            minChunkDuration: defaultParameters.minChunkDuration,
            repetitionPenalty: defaultParameters.repetitionPenalty,
            repetitionContextSize: defaultParameters.repetitionContextSize
        )
    }

    // MARK: - Live Recording & Streaming Transcription

    private var liveTask: Task<Void, Never>?
    private var eventTask: Task<Void, Never>?
    private var streamingSession: StreamingInferenceSession?
    private var activeRecordingID: UUID?
    private var lastReadPos: Int = 0

    func startRecording() async {
        guard let model else {
            errorMessage = "Model not loaded"
            return
        }
        guard supportsRealtimeRecording else {
            errorMessage = "Realtime recording is available for Qwen3-ASR, Cohere Transcribe, and MOSS Transcribe models"
            return
        }

        let recordingID = UUID()
        activeRecordingID = recordingID
        errorMessage = nil
        transcriptionText = ""
        resetGenerationStats()
        lastReadPos = 0

        do {
            try await recorder.startRecording()
        } catch {
            errorMessage = error.localizedDescription
            return
        }

        // Create streaming session
        let config = StreamingConfig(
            decodeIntervalSeconds: 1.0,
            maxCachedWindows: 60,
            delayPreset: .custom(ms: streamingDelayMs),
            language: languageHint,
            temperature: temperature,
            maxTokensPerPass: maxTokens
        )
        let session = StreamingInferenceSession(model: model, config: config)
        streamingSession = session

        // Listen to events from the session
        eventTask = Task {
            for await event in session.events {
                guard activeRecordingID == recordingID else { continue }
                switch event {
                case .displayUpdate(let confirmed, let provisional):
                    transcriptionText = confirmed + provisional
                case .confirmed:
                    break  // displayUpdate handles the UI
                case .provisional:
                    break
                case .stats(let stats):
                    tokensPerSecond = stats.tokensPerSecond
                    peakMemory = stats.peakMemoryGB
                case .ended(let fullText):
                    transcriptionText = fullText
                }
            }
            // Stream ended naturally — clean up
            if activeRecordingID == recordingID {
                activeRecordingID = nil
                streamingSession = nil
                eventTask = nil
            }
        }

        // Audio feed loop: read new samples every 100ms and feed to session
        liveTask = Task {
            while !Task.isCancelled && recorder.isRecording {
                if let (audio, endPos) = recorder.getAudio(from: lastReadPos) {
                    lastReadPos = endPos
                    let samples = audio.asArray(Float.self)
                    session.feedAudio(samples: samples)
                }
                try? await Task.sleep(for: .milliseconds(100))
            }
        }
    }

    func stopRecording() {
        liveTask?.cancel()
        liveTask = nil

        // Feed any remaining audio, then stop session.
        if let session = streamingSession {
            if let (audio, endPos) = recorder.getAudio(from: lastReadPos) {
                lastReadPos = endPos
                let samples = audio.asArray(Float.self)
                session.feedAudio(samples: samples)
            }

            _ = recorder.stopRecording()

            // Stop promotes all provisional tokens and emits .ended
            // The eventTask will process .ended and clean up naturally
            session.stop()
            return
        }

        _ = recorder.stopRecording()
        activeRecordingID = nil
    }

    func cancelRecording() {
        liveTask?.cancel()
        liveTask = nil
        streamingSession?.cancel()
        streamingSession = nil
        eventTask?.cancel()
        eventTask = nil
        recorder.cancelRecording()
        activeRecordingID = nil
        lastReadPos = 0
    }

    func stop() {
        liveTask?.cancel()
        liveTask = nil
        streamingSession?.cancel()
        streamingSession = nil
        eventTask?.cancel()
        eventTask = nil
        generationTask?.cancel()
        generationTask = nil
        activeRecordingID = nil

        if isRecording {
            recorder.cancelRecording()
            lastReadPos = 0
        }

        if isGenerating {
            isGenerating = false
            generationProgress = ""
        }
    }

    private func loadSTTModel(_ repo: String) async throws -> any STTGenerationModel {
        try await STT.loadModel(modelRepo: repo)
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

    func seek(to time: TimeInterval) {
        audioPlayer.seek(to: time)
    }

    func copyTranscription() {
        #if os(iOS)
        UIPasteboard.general.string = transcriptionText
        #else
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(transcriptionText, forType: .string)
        #endif
    }

    func clearTranscription() {
        guard !isGenerating && !isRecording else { return }

        transcriptionText = ""
        errorMessage = nil
        generationProgress = ""
        tokensPerSecond = 0
        peakMemory = 0
    }

    private func resetGenerationStats() {
        tokensPerSecond = 0
        peakMemory = 0
        Memory.clearCache()
        Memory.peakMemory = 0
    }

    private func resampleAudio(_ audio: MLXArray, from sourceSR: Int, to targetSR: Int) throws -> MLXArray {
        let samples = audio.asArray(Float.self)

        guard let inputFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: Double(sourceSR), channels: 1, interleaved: false
        ), let outputFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: Double(targetSR), channels: 1, interleaved: false
        ) else {
            throw NSError(domain: "STT", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to create audio formats"])
        }

        guard let converter = AVAudioConverter(from: inputFormat, to: outputFormat) else {
            throw NSError(domain: "STT", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to create audio converter"])
        }

        let inputFrameCount = AVAudioFrameCount(samples.count)
        guard let inputBuffer = AVAudioPCMBuffer(pcmFormat: inputFormat, frameCapacity: inputFrameCount) else {
            throw NSError(domain: "STT", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed to create input buffer"])
        }
        inputBuffer.frameLength = inputFrameCount
        memcpy(inputBuffer.floatChannelData![0], samples, samples.count * MemoryLayout<Float>.size)

        let ratio = Double(targetSR) / Double(sourceSR)
        let outputFrameCount = AVAudioFrameCount(Double(samples.count) * ratio)
        guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: outputFrameCount) else {
            throw NSError(domain: "STT", code: 4, userInfo: [NSLocalizedDescriptionKey: "Failed to create output buffer"])
        }

        var error: NSError?
        converter.convert(to: outputBuffer, error: &error) { _, outStatus in
            outStatus.pointee = .haveData
            return inputBuffer
        }

        if let error { throw error }

        let outputSamples = Array(UnsafeBufferPointer(
            start: outputBuffer.floatChannelData![0], count: Int(outputBuffer.frameLength)
        ))
        return MLXArray(outputSamples)
    }
}
