import Foundation
import Testing
@testable import MLXAudioSTT

struct MLXAudioSTTTests {
    @Test func transcriptionOptionsDefaultsMatchSDK() {
        let options = TranscriptionOptions.default

        #expect(options.language == nil)
        #expect(options.task == .transcribe)
        #expect(options.timeout == 30.0)
        #expect(options.temperature == 0.0)
    }

    @Test func sttGenerationInfoSummaryIncludesKeyFields() {
        let info = STTGenerationInfo(
            promptTokenCount: 12,
            generationTokenCount: 34,
            prefillTime: 0.5,
            generateTime: 1.25,
            tokensPerSecond: 27.2,
            peakMemoryUsage: 1.5
        )

        let summary = info.summary
        #expect(summary.contains("Prompt:"))
        #expect(summary.contains("Generation:"))
        #expect(summary.contains("Peak Memory Usage:"))
    }

    @Test func sttOutputDescriptionIncludesTextAndCounts() {
        let output = STTOutput(
            text: "hello world",
            segments: [STTSegment(text: "hello world", start: 0, end: 1.2)],
            language: "en",
            promptTokens: 10,
            generationTokens: 5,
            totalTokens: 15,
            promptTps: 20.0,
            generationTps: 15.0,
            totalTime: 0.8,
            peakMemoryUsage: 0.9
        )

        let description = output.description
        #expect(description.contains("text: hello world"))
        #expect(description.contains("language: en"))
        #expect(description.contains("prompt_tokens: 10"))
        #expect(description.contains("generation_tokens: 5"))
        #expect(description.contains("total_tokens: 15"))
    }

    @Test func transcriptionProgressComputesProgress() {
        let progress = TranscriptionProgress(
            text: "partial",
            words: nil,
            isFinal: false,
            processedDuration: 5.0,
            audioDuration: 10.0,
            chunkIndex: 0,
            totalChunks: 2
        )

        #expect(abs(progress.progress - 0.5) < 0.001)
    }

    @Test func processingLimitsPresetsAreSane() {
        let conservative = ProcessingLimits.conservative
        let standard = ProcessingLimits.default
        let aggressive = ProcessingLimits.aggressive

        #expect(conservative.maxConcurrentChunks < standard.maxConcurrentChunks)
        #expect(aggressive.maxConcurrentChunks > standard.maxConcurrentChunks)
        #expect(conservative.chunkTimeout <= standard.chunkTimeout)
        #expect(aggressive.chunkTimeout >= standard.chunkTimeout)
    }

    @Test func alignmentHeadsAreDefinedForAllModels() {
        for model in WhisperModel.allCases {
            let heads = WhisperAlignmentHeads.heads(for: model)
            #expect(!heads.isEmpty)
        }
    }
}
