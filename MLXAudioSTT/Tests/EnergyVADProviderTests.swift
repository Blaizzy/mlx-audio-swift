import Foundation
import MLX
import Testing
@testable import MLXAudioSTT

struct EnergyVADProviderTests {
    let sampleRate = 16000

    // MARK: - Speech Detection Tests

    @Test func detectsSpeechSegmentsInAudioWithSpeechSilencePattern() async throws {
        // Create audio with speech-silence-speech pattern
        // 0.5s silence + 1s speech (high energy) + 0.5s silence + 0.5s speech + 0.5s silence
        let silenceSamples = Int(0.5 * Double(sampleRate))
        let speechSamples = Int(1.0 * Double(sampleRate))
        let shortSpeechSamples = Int(0.5 * Double(sampleRate))

        var audioData: [Float] = []

        // First silence (low amplitude noise)
        audioData.append(contentsOf: (0..<silenceSamples).map { _ in Float.random(in: -0.001...0.001) })

        // First speech (high amplitude)
        audioData.append(contentsOf: (0..<speechSamples).map { i in
            Float(Foundation.sin(Double(i) * 0.1)) * 0.3
        })

        // Middle silence
        audioData.append(contentsOf: (0..<silenceSamples).map { _ in Float.random(in: -0.001...0.001) })

        // Second speech (shorter)
        audioData.append(contentsOf: (0..<shortSpeechSamples).map { i in
            Float(Foundation.sin(Double(i) * 0.1)) * 0.25
        })

        // Final silence
        audioData.append(contentsOf: (0..<silenceSamples).map { _ in Float.random(in: -0.001...0.001) })

        let audio = MLXArray(audioData)

        let config = EnergyVADProvider.EnergyVADConfig(
            speechThreshold: 0.05,
            windowDuration: 0.025,
            hopDuration: 0.010,
            smoothingFactor: 0.5
        )
        let segmentConfig = VADSegmentConfig(
            minSpeechDuration: 0.2,
            minSilenceDuration: 0.3,
            speechPadding: 0.05
        )
        let provider = EnergyVADProvider(config: config, segmentConfig: segmentConfig)

        let segments = try await provider.detectSpeech(in: audio, sampleRate: sampleRate)

        #expect(segments.count >= 1)

        // First segment should start around 0.5s (after first silence)
        if let firstSegment = segments.first {
            #expect(firstSegment.start >= 0.3)
            #expect(firstSegment.start <= 0.7)
            #expect(firstSegment.confidence > 0)
        }
    }

    @Test func handlesSilenceOnlyAudioReturnsEmptyArray() async throws {
        // Create 2 seconds of near-silence
        let numSamples = 2 * sampleRate
        let silentAudio = (0..<numSamples).map { _ in Float.random(in: -0.001...0.001) }
        let audio = MLXArray(silentAudio)

        let config = EnergyVADProvider.EnergyVADConfig(
            speechThreshold: 0.05,
            windowDuration: 0.025,
            hopDuration: 0.010,
            smoothingFactor: 0.5
        )
        let segmentConfig = VADSegmentConfig(minSpeechDuration: 0.25)
        let provider = EnergyVADProvider(config: config, segmentConfig: segmentConfig)

        let segments = try await provider.detectSpeech(in: audio, sampleRate: sampleRate)

        #expect(segments.isEmpty)
    }

    @Test func respectsThresholdConfiguration() async throws {
        // Create audio with moderate energy
        let numSamples = sampleRate // 1 second
        let moderateAudio = (0..<numSamples).map { i in
            Float(Foundation.sin(Double(i) * 0.1)) * 0.03
        }
        let audio = MLXArray(moderateAudio)

        let segmentConfig = VADSegmentConfig(minSpeechDuration: 0.1)

        // High threshold - should detect nothing
        let highThresholdConfig = EnergyVADProvider.EnergyVADConfig(
            speechThreshold: 0.1,
            smoothingFactor: 0.0
        )
        let highThresholdProvider = EnergyVADProvider(config: highThresholdConfig, segmentConfig: segmentConfig)
        let highThresholdSegments = try await highThresholdProvider.detectSpeech(in: audio, sampleRate: sampleRate)

        // Low threshold - should detect speech
        let lowThresholdConfig = EnergyVADProvider.EnergyVADConfig(
            speechThreshold: 0.01,
            smoothingFactor: 0.0
        )
        let lowThresholdProvider = EnergyVADProvider(config: lowThresholdConfig, segmentConfig: segmentConfig)
        let lowThresholdSegments = try await lowThresholdProvider.detectSpeech(in: audio, sampleRate: sampleRate)

        #expect(highThresholdSegments.isEmpty)
        #expect(!lowThresholdSegments.isEmpty)
    }

    // MARK: - Speech Probabilities Tests

    @Test func speechProbabilitiesReturnsCorrectTimeValues() async throws {
        let numSamples = sampleRate // 1 second
        let audio = MLXArray.ones([numSamples])

        let config = EnergyVADProvider.EnergyVADConfig(
            windowDuration: 0.025,
            hopDuration: 0.010,
            smoothingFactor: 0.0
        )
        let provider = EnergyVADProvider(config: config)

        let probabilities = try await provider.speechProbabilities(in: audio, sampleRate: sampleRate)

        #expect(!probabilities.isEmpty)

        // First probability should be at time 0
        if let first = probabilities.first {
            #expect(first.time == 0.0)
        }

        // Check time increments by hop duration
        if probabilities.count >= 2 {
            let timeDiff = probabilities[1].time - probabilities[0].time
            #expect(abs(timeDiff - config.hopDuration) < 0.001)
        }
    }

    @Test func resetDoesNotThrow() async {
        let provider = EnergyVADProvider()
        await provider.reset()
    }

    @Test func handlesEmptyAudio() async throws {
        let audio = MLXArray([Float]())
        let provider = EnergyVADProvider()

        let segments = try await provider.detectSpeech(in: audio, sampleRate: sampleRate)
        #expect(segments.isEmpty)

        let probabilities = try await provider.speechProbabilities(in: audio, sampleRate: sampleRate)
        #expect(probabilities.isEmpty)
    }

    @Test func handlesAudioShorterThanWindow() async throws {
        // Create audio shorter than default window (25ms = 400 samples at 16kHz)
        let shortAudio = MLXArray.ones([100])
        let provider = EnergyVADProvider()

        let segments = try await provider.detectSpeech(in: shortAudio, sampleRate: sampleRate)
        #expect(segments.isEmpty)
    }
}
