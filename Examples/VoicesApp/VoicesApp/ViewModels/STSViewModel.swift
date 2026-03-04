//
//  STSViewModel.swift
//  VoicesApp
//
//  Created by Alpay Calalli on 25.02.26.
//

import AVFoundation
import Combine
import Foundation
import MLX
import MLXAudioCore
import MLXAudioSTS
import SwiftUI
#if os(macOS)
import AppKit
#endif

@MainActor
@Observable
class STSViewModel {
    var isLoading = false
    var isGenerating = false
    var generationProgress: String = ""
    var errorMessage: String?
    var audioURL: URL?

    // What to isolate. "speech", "music", "drums", etc.
    var separationDescription: String = "speech"

    // Whether to play the residual (everything *except* the target) instead
    var useResidual: Bool = false

    // Model config
    var modelId: String = "mlx-community/sam-audio-base"
    private var loadedModelId: String?

    // Audio player state (manually synced from AudioPlayer via Combine)
    var isPlaying: Bool = false
    var currentTime: TimeInterval = 0
    var duration: TimeInterval = 0

    private var model: SAMAudio?
    private let audioPlayer = AudioPlayer()
    private var cancellables = Set<AnyCancellable>()
    private var separationTask: Task<Void, Never>?

    var isModelLoaded: Bool { model != nil }

    init() {
        setupAudioPlayerObservers()
    }

    // MARK: - AudioPlayer Observers

    private func setupAudioPlayerObservers() {
        audioPlayer.$isPlaying
            .receive(on: DispatchQueue.main)
            .sink { [weak self] value in self?.isPlaying = value }
            .store(in: &cancellables)

        audioPlayer.$currentTime
            .receive(on: DispatchQueue.main)
            .sink { [weak self] value in self?.currentTime = value }
            .store(in: &cancellables)

        audioPlayer.$duration
            .receive(on: DispatchQueue.main)
            .sink { [weak self] value in self?.duration = value }
            .store(in: &cancellables)
    }

    // MARK: - Model Loading

    func loadModel() async {
        guard model == nil || loadedModelId != modelId else { return }

        isLoading = true
        errorMessage = nil
        generationProgress = "Downloading model..."

        do {
            model = try await SAMAudio.fromPretrained(modelId)
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

    // MARK: - Separation

    func startSeparation(inputURL: URL) {
        separationTask = Task {
            await separate(inputURL: inputURL)
        }
    }

    func separate(inputURL: URL) async {
        guard let model else {
            errorMessage = "Model not loaded"
            return
        }

        isGenerating = true
        errorMessage = nil
        generationProgress = "Starting separation..."

        do {
            try Task.checkCancellation()

            generationProgress = "Separating \"\(separationDescription)\"..."

            let result = try await model.separateLong(
                audioPaths: [inputURL.path],
                descriptions: [separationDescription]
            )

            try Task.checkCancellation()

            // Pick target or residual based on user preference
            let outputArray: MLXArray
            if useResidual {
                guard let residual = result.residual.first else {
                    throw NSError(
                        domain: "STSViewModel", code: 2,
                        userInfo: [NSLocalizedDescriptionKey: "No residual audio in result"]
                    )
                }
                outputArray = residual
            } else {
                guard let target = result.target.first else {
                    throw NSError(
                        domain: "STSViewModel", code: 1,
                        userInfo: [NSLocalizedDescriptionKey: "No target audio in result"]
                    )
                }
                outputArray = target
            }

            let samples = outputArray.asArray(Float.self)

            guard !samples.isEmpty else {
                throw NSError(
                    domain: "STSViewModel", code: 3,
                    userInfo: [NSLocalizedDescriptionKey: "Separated audio was empty"]
                )
            }

            if let peakMem = result.peakMemoryGB {
                print("[STS] Peak memory: \(String(format: "%.2f", peakMem)) GB")
            }

            // Write to WAV — same pattern as TTSViewModel
            let sampleRate = Double(model.sampleRate)
            let tempURL = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString)
                .appendingPathExtension("wav")

            let wavWriter = try StreamingWAVWriter(url: tempURL, sampleRate: sampleRate)
            try wavWriter.writeChunk(samples)
            let finalURL = wavWriter.finalize()

            guard wavWriter.framesWritten > 0 else {
                throw NSError(
                    domain: "STSViewModel", code: 4,
                    userInfo: [NSLocalizedDescriptionKey: "No frames written to WAV"]
                )
            }

            Memory.clearCache()

            audioURL = finalURL
            generationProgress = ""

            // Load and play — no streaming since separate() isn't an AsyncSequence
            audioPlayer.loadAudio(from: finalURL)
            audioPlayer.play()

        } catch is CancellationError {
            audioPlayer.stop()
            Memory.clearCache()
            generationProgress = ""
        } catch {
            errorMessage = "Separation failed: \(error.localizedDescription)"
            generationProgress = ""
            Memory.clearCache()
        }

        isGenerating = false
    }

    // MARK: - Playback Controls

    func play()           { audioPlayer.play() }
    func pause()          { audioPlayer.pause() }
    func togglePlayPause(){ audioPlayer.togglePlayPause() }
    func seek(to time: TimeInterval) { audioPlayer.seek(to: time) }

    func stop() {
        separationTask?.cancel()
        separationTask = nil
        audioPlayer.stop()
        if isGenerating {
            isGenerating = false
            generationProgress = ""
        }
    }

    // MARK: - Save

    func saveAudioFile() {
        guard let audioURL else { return }

        #if os(macOS)
        let savePanel = NSSavePanel()
        savePanel.allowedContentTypes = [.wav]
        savePanel.canCreateDirectories = true
        savePanel.isExtensionHidden = false
        savePanel.title = "Save Separated Audio"
        savePanel.nameFieldStringValue = "separated_audio.wav"

        savePanel.begin { response in
            guard response == .OK, let dest = savePanel.url else { return }
            do {
                if FileManager.default.fileExists(atPath: dest.path) {
                    try FileManager.default.removeItem(at: dest)
                }
                try FileManager.default.copyItem(at: audioURL, to: dest)
            } catch {
                DispatchQueue.main.async {
                    self.errorMessage = "Failed to save file: \(error.localizedDescription)"
                }
            }
        }
        #else
        errorMessage = "Save functionality is available on macOS"
        #endif
    }
}
