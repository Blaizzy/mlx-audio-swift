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
    var isLoading = false
    var isGenerating = false
    var generationProgress: String = ""
    var errorMessage: String?
    var transcriptionText: String = ""
    var tokensPerSecond: Double = 0
    var peakMemory: Double = 0

    // Generation parameters
    var maxTokens: Int = 1024
    var temperature: Float = 0.0
    var language: String = "English"
    var chunkDuration: Float = 30.0

    // Streaming parameters
    var streamingDelayMs: Int = 480  // .agent default

    // Model configuration
    var modelId: String = "mlx-community/Qwen3-ASR-0.6B-4bit"
    private var loadedModelId: String?

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
    private var modelSampleRate: Int = 16000
    private var qwen3Model: Qwen3ASRModel? { model as? Qwen3ASRModel }
    private let audioPlayer = AudioPlayer()
    private let recorder = AudioRecorderManager()
    private var cancellables = Set<AnyCancellable>()
    private var generationTask: Task<Void, Never>?
    private var previousMemoryCacheLimit: Int?

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
        guard model == nil || loadedModelId != modelId else { return }

        isLoading = true
        errorMessage = nil
        generationProgress = "Downloading model..."

        do {
            let lowerId = modelId.lowercased()
            if lowerId.contains("parakeet") {
                let parakeet = try await ParakeetModel.fromPretrained(modelId)
                model = parakeet
                modelSampleRate = parakeet.preprocessConfig.sampleRate
            } else {
                let qwen3 = try await Qwen3ASRModel.fromPretrained(modelId)
                model = qwen3
                modelSampleRate = qwen3.sampleRate
            }
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

    func selectAudioFile(_ url: URL) {
        selectedAudioURL = url
        audioFileName = url.lastPathComponent
        audioPlayer.loadAudio(from: url)
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
        tokensPerSecond = 0
        peakMemory = 0

        do {
            let (sampleRate, audioData) = try loadAudioArray(from: audioURL)
            let targetRate = modelSampleRate

            let resampled: MLXArray
            if sampleRate != targetRate {
                generationProgress = "Resampling \(sampleRate)Hz → \(targetRate)Hz..."
                resampled = try resampleAudio(audioData, from: sampleRate, to: targetRate)
            } else {
                resampled = audioData
            }

            generationProgress = "Transcribing..."

            let params = STTGenerateParameters(
                maxTokens: maxTokens,
                temperature: temperature,
                language: language,
                chunkDuration: chunkDuration
            )

            var tokenCount = 0
            for try await event in model.generateStream(
                audio: resampled,
                generationParameters: params
            ) {
                try Task.checkCancellation()

                switch event {
                case .token(let token):
                    transcriptionText += token
                    tokenCount += 1
                    generationProgress = "Transcribing... \(tokenCount) tokens"
                case .info(let info):
                    tokensPerSecond = info.tokensPerSecond
                    peakMemory = info.peakMemoryUsage
                case .result:
                    generationProgress = ""
                }
            }

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

    // MARK: - Live Recording & Streaming Transcription

    private var liveTask: Task<Void, Never>?
    private var eventTask: Task<Void, Never>?
    private var streamingSession: (any StreamingSession)?
    private var lastReadPos: Int = 0
    private let liveStreamingCacheLimitBytes = 256 * 1024 * 1024

    private func applyLiveStreamingMemoryBudget() {
        if previousMemoryCacheLimit == nil {
            previousMemoryCacheLimit = Memory.cacheLimit
        }
        let target = min(previousMemoryCacheLimit ?? liveStreamingCacheLimitBytes, liveStreamingCacheLimitBytes)
        if Memory.cacheLimit != target {
            Memory.cacheLimit = target
        }
        Memory.clearCache()
    }

    private func restoreMemoryBudgetIfNeeded() {
        guard let previous = previousMemoryCacheLimit else { return }
        Memory.cacheLimit = previous
        previousMemoryCacheLimit = nil
        Memory.clearCache()
    }

    func startRecording() async {
        guard model != nil else {
            errorMessage = "Model not loaded"
            return
        }

        errorMessage = nil
        transcriptionText = ""
        tokensPerSecond = 0
        peakMemory = 0
        lastReadPos = 0

        do {
            try await recorder.startRecording()
        } catch {
            errorMessage = error.localizedDescription
            return
        }

        // Base streaming config; model-specific tweaks are applied below.
        var config = StreamingConfig(
            decodeIntervalSeconds: 1.0,
            maxCachedWindows: 60,
            delayPreset: .custom(ms: streamingDelayMs),
            language: language,
            temperature: temperature,
            maxTokensPerPass: maxTokens
        )

        let session: any StreamingSession
        if let qwen3 = qwen3Model {
            session = StreamingInferenceSession(model: qwen3, config: config)
        } else if let parakeet = model as? ParakeetModel {
            // Faster decode interval for Parakeet
            config.decodeIntervalSeconds = 0.25
            session = StreamingInferenceSession(model: parakeet, config: config)
        } else {
            errorMessage = "Loaded model does not support live streaming"
            recorder.cancelRecording()
            return
        }
        streamingSession = session
        applyLiveStreamingMemoryBudget()

        // Listen to events from the session
        eventTask = Task {
            for await event in session.events {
                switch event {
                case .displayUpdate(let confirmed, let provisional):
                    if confirmed.isEmpty {
                        transcriptionText = provisional
                    } else if provisional.isEmpty {
                        transcriptionText = confirmed
                    } else {
                        transcriptionText = confirmed + provisional
                    }
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
            streamingSession = nil
            eventTask = nil
            restoreMemoryBudgetIfNeeded()
        }

        // Audio feed loop: read new samples every 100ms and feed to session
        liveTask = Task {
            while !Task.isCancelled && recorder.isRecording {
                if let (samples, endPos) = recorder.getSamples(from: lastReadPos) {
                    lastReadPos = endPos
                    session.feedAudio(samples: samples)
                    // Keep live capture memory bounded by discarding consumed samples.
                    recorder.discardAudio(before: lastReadPos)
                }
                try? await Task.sleep(for: .milliseconds(100))
            }
        }
    }

    func stopRecording() {
        liveTask?.cancel()
        liveTask = nil

        // Feed any remaining audio, then stop session
        if let session = streamingSession {
            if let (samples, endPos) = recorder.getSamples(from: lastReadPos) {
                lastReadPos = endPos
                session.feedAudio(samples: samples)
            }

            // Stop capture after flushing pending samples from the recorder buffer.
            _ = recorder.stopRecording()

            // Stop promotes all provisional tokens and emits .ended
            // The eventTask will process .ended and clean up naturally
            session.stop()
        } else {
            _ = recorder.stopRecording()
            restoreMemoryBudgetIfNeeded()
        }
    }

    func cancelRecording() {
        liveTask?.cancel()
        liveTask = nil
        streamingSession?.cancel()
        streamingSession = nil
        eventTask?.cancel()
        eventTask = nil
        recorder.cancelRecording()
        lastReadPos = 0
        restoreMemoryBudgetIfNeeded()
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

        if isRecording {
            recorder.cancelRecording()
            lastReadPos = 0
        }

        if isGenerating {
            isGenerating = false
            generationProgress = ""
        }
        restoreMemoryBudgetIfNeeded()
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
