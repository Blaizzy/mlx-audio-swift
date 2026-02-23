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

    private var model: Qwen3ASRModel?
    private let audioPlayer = AudioPlayer()
    private let recorder = AudioRecorderManager()
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
        guard model == nil || loadedModelId != modelId else { return }

        isLoading = true
        errorMessage = nil
        generationProgress = "Downloading model..."

        do {
            model = try await Qwen3ASRModel.fromPretrained(modelId)
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
            let targetRate = model.sampleRate

            let resampled: MLXArray
            if sampleRate != targetRate {
                generationProgress = "Resampling \(sampleRate)Hz → \(targetRate)Hz..."
                resampled = try resampleAudio(audioData, from: sampleRate, to: targetRate)
            } else {
                resampled = audioData
            }

            generationProgress = "Transcribing..."

            var tokenCount = 0
            for try await event in model.generateStream(
                audio: resampled,
                maxTokens: maxTokens,
                temperature: temperature,
                language: language,
                chunkDuration: chunkDuration
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
    private var streamingSession: StreamingInferenceSession?
    private var lastReadPos: Int = 0

    func startRecording() async {
        guard let model = model else {
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

        // Create streaming session
        let config = StreamingConfig(
            decodeIntervalSeconds: 1.0,
            maxCachedWindows: 60,
            delayPreset: .custom(ms: streamingDelayMs),
            language: language,
            temperature: temperature,
            maxTokensPerPass: maxTokens
        )
        let session = StreamingInferenceSession(model: model, config: config)
        streamingSession = session

        // Listen to events from the session
        eventTask = Task {
            for await event in session.events {
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
            streamingSession = nil
            eventTask = nil
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

        _ = recorder.stopRecording()

        // Feed any remaining audio, then stop session
        if let session = streamingSession {
            if let (audio, endPos) = recorder.getAudio(from: lastReadPos) {
                lastReadPos = endPos
                let samples = audio.asArray(Float.self)
                session.feedAudio(samples: samples)
            }

            // Stop promotes all provisional tokens and emits .ended
            // The eventTask will process .ended and clean up naturally
            session.stop()
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
