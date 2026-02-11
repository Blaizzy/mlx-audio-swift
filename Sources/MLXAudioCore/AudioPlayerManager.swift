//
//  AudioPlayerManager.swift
//  MLXAudio
//
//  Created by Rudrank Riyam on 6/11/25.
//

import Foundation
import AVFoundation
import Combine

@MainActor
public class AudioPlayerManager: NSObject, ObservableObject {
    // Published properties for UI binding
    @Published public var isPlaying: Bool = false
    @Published public var currentTime: TimeInterval = 0
    @Published public var duration: TimeInterval = 0
    @Published public var currentAudioURL: URL?

    private var player: AVAudioPlayer?
    private var timer: Timer?

    // Streaming playback components
    private var audioEngine: AVAudioEngine?
    private var playerNode: AVAudioPlayerNode?
    private var streamingFormat: AVAudioFormat?
    private var isStreaming: Bool = false
    private var scheduledSamples: Int = 0

    public override init() {
        super.init()
    }

    @MainActor deinit {
        stop()
    }

    // MARK: - Playback Control

    public func loadAudio(from url: URL) {
        do {
            // Stop any existing playback
            stop()

            // Setup audio session for iOS
            #if os(iOS)
            AudioSessionManager.shared.setupAudioSession()
            #endif

            // Create new player
            player = try AVAudioPlayer(contentsOf: url)
            player?.delegate = self
            player?.prepareToPlay()

            // Update state
            currentAudioURL = url
            duration = player?.duration ?? 0
            currentTime = 0

        } catch {
            print("Failed to load audio: \(error.localizedDescription)")
            currentAudioURL = nil
            duration = 0
            currentTime = 0
        }
    }

    public func play() {
        guard let player = player else { return }

        player.play()
        isPlaying = true
        startTimer()
    }

    public func pause() {
        if isStreaming {
            playerNode?.pause()
        } else {
            player?.pause()
        }
        isPlaying = false
        stopTimer()
    }

    public func togglePlayPause() {
        if isPlaying {
            pause()
        } else {
            if isStreaming {
                playerNode?.play()
                isPlaying = true
                startTimer()
            } else {
                play()
            }
        }
    }

    public func stop() {
        if isStreaming {
            stopStreaming()
        } else {
            player?.stop()
        }
        isPlaying = false
        stopTimer()
        currentTime = 0
    }

    public func seek(to time: TimeInterval) {
        guard let player = player else { return }
        player.currentTime = max(0, min(time, duration))
        currentTime = player.currentTime
    }

    // MARK: - Streaming Playback

    /// Start streaming playback - call this before scheduling chunks
    public func startStreaming(sampleRate: Double) {
        stop()

        // Setup audio session for iOS
        #if os(iOS)
        AudioSessionManager.shared.setupAudioSession()
        #endif

        audioEngine = AVAudioEngine()
        playerNode = AVAudioPlayerNode()

        guard let engine = audioEngine, let node = playerNode else { return }

        streamingFormat = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)
        scheduledSamples = 0

        engine.attach(node)
        engine.connect(node, to: engine.mainMixerNode, format: streamingFormat)

        do {
            try engine.start()
            node.play()
            isStreaming = true
            isPlaying = true
            startStreamingTimer()
        } catch {
            print("Failed to start audio engine: \(error)")
        }
    }

    /// Schedule audio samples for streaming playback
    public func scheduleAudioChunk(_ samples: [Float], withCrossfade: Bool = true) {
        guard isStreaming,
              let node = playerNode,
              let format = streamingFormat else { return }

        var processedSamples = samples

        // Apply fade-in to first chunk, crossfade to subsequent chunks
        if scheduledSamples == 0 {
            // Fade in the first chunk (10ms)
            let fadeInSamples = min(Int(format.sampleRate * 0.01), samples.count)
            for i in 0..<fadeInSamples {
                let factor = Float(i) / Float(fadeInSamples)
                processedSamples[i] *= factor
            }
        } else if withCrossfade {
            // Crossfade: fade in at the start (20ms)
            let crossfadeSamples = min(Int(format.sampleRate * 0.02), samples.count)
            for i in 0..<crossfadeSamples {
                let factor = Float(i) / Float(crossfadeSamples)
                processedSamples[i] *= factor
            }
        }

        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(processedSamples.count)) else {
            return
        }

        buffer.frameLength = AVAudioFrameCount(processedSamples.count)

        if let channelData = buffer.floatChannelData {
            processedSamples.withUnsafeBufferPointer { src in
                channelData[0].update(from: src.baseAddress!, count: processedSamples.count)
            }
        }

        node.scheduleBuffer(buffer)
        scheduledSamples += samples.count
        duration = Double(scheduledSamples) / format.sampleRate
    }

    /// Stop streaming and clean up
    public func stopStreaming() {
        playerNode?.stop()
        audioEngine?.stop()
        audioEngine = nil
        playerNode = nil
        streamingFormat = nil
        isStreaming = false
        scheduledSamples = 0
    }

    /// Check if currently in streaming mode
    public var isStreamingMode: Bool {
        return isStreaming
    }

    // MARK: - Timer Management

    private func startTimer() {
        stopTimer()
        timer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            Task { @MainActor in
                guard let player = self?.player else { return }
                self?.currentTime = player.currentTime
            }
        }
        timer?.tolerance = 0.05
    }

    private func startStreamingTimer() {
        stopTimer()
        timer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            Task { @MainActor in
                guard let node = self?.playerNode,
                      let nodeTime = node.lastRenderTime,
                      let playerTime = node.playerTime(forNodeTime: nodeTime) else { return }
                let time = Double(playerTime.sampleTime) / playerTime.sampleRate
                self?.currentTime = time
            }
        }
        timer?.tolerance = 0.05
    }

    private func stopTimer() {
        timer?.invalidate()
        timer = nil
    }
}

// MARK: - AVAudioPlayerDelegate

extension AudioPlayerManager: @MainActor AVAudioPlayerDelegate {
    public func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        isPlaying = false
        stopTimer()
        currentTime = 0
    }

    public func audioPlayerDecodeErrorDidOccur(_ player: AVAudioPlayer, error: Error?) {
        print("Audio decode error: \(error?.localizedDescription ?? "unknown")")
        isPlaying = false
        stopTimer()
    }
}

