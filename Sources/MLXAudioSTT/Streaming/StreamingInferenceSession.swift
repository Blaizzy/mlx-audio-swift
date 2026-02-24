//
//  StreamingInferenceSession.swift
//  MLXAudioSTT
//
//  Created by Prince Canuma on 07/02/2026.
//

import Foundation
import MLX
import MLXNN
import MLXLMCommon
import Tokenizers
import os

// MARK: - Model Backend

private enum ModelBackend: @unchecked Sendable {
    case qwen3(Qwen3ASRModel)
    case parakeet(ParakeetModel)
}

// MARK: - Shared State

private struct SessionSharedState: Sendable {
    /// Accumulated text from completed encoder windows — frozen, never re-decoded.
    var completedText: String = ""
    /// Streaming decode state — only covers the current pending (partial) window.
    var confirmedTokenIds: [Int] = []
    var provisionalTokenIds: [Int] = []
    var provisionalFirstSeen: [Date] = []
    var provisionalAgreementCounts: [Int] = []
    var confirmedText: String = ""
    var isDecoding: Bool = false

    // Parakeet token-piece state
    var confirmedPieces: [String] = []
    var provisionalPieces: [String] = []
    var pieceFirstSeen: [Date] = []
    var pieceAgreementCounts: [Int] = []
    /// Merged absolute tokens from recent buffered windows (used for boundary stitching).
    var parakeetMergedTokens: [ParakeetAlignedToken] = []
    /// Commit boundary held back by one window to reduce boundary word splits.
    var parakeetPendingCommitEndSample: Int?
}

// MARK: - Decode Pass Parameters

private struct DecodePassParams: Sendable {
    let audioFeatures: UncheckedSendableBox<MLXArray>
    let model: UncheckedSendableBox<Qwen3ASRModel>
    let config: StreamingConfig
    let confirmedTokenIds: [Int]
    /// completedText + confirmedText for display
    let displayPrefix: String
    let prevProvisional: [Int]
    let prevFirstSeen: [Date]
    let prevAgreementCounts: [Int]
    let minAgreementPasses: Int
}

private struct FinalizeWindowsParams: Sendable {
    let windows: UncheckedSendableBox<[MLXArray]>
    let model: UncheckedSendableBox<Qwen3ASRModel>
    let config: StreamingConfig
    let totalSamples: Int
    let encodedWindowCount: Int
}

private struct StopSnapshot: Sendable {
    let continuation: AsyncStream<TranscriptionEvent>.Continuation?
    let completedWindows: UncheckedSendableBox<[MLXArray]>?
    let pendingAudioFeatures: UncheckedSendableBox<MLXArray>?
    let confirmedCount: Int
    let totalSamples: Int
    let encodedWindowCount: Int
    let fallbackFinalText: String?
}

private struct ParakeetBufferedWindow: Sendable {
    /// Absolute sample index where this decode window starts.
    let windowStartSample: Int
    /// Absolute sample interval whose tokens should be frozen.
    let commitStartSample: Int
    let commitEndSample: Int
    /// Audio samples for the decode window.
    let samples: [Float]
}

private struct ParakeetDecodePassParams: Sendable {
    /// Completed buffered windows to freeze (one-shot decode and commit middle chunk)
    let completedWindows: UncheckedSendableBox<[ParakeetBufferedWindow]>
    /// Current active window for streaming decode + word promotion
    let activeAudio: UncheckedSendableBox<[Float]>
    let activeWindowStartSample: Int
    /// Tokens before this absolute sample are already frozen in `completedText`.
    let activeCommitStartSample: Int
    let model: UncheckedSendableBox<ParakeetModel>
    let config: StreamingConfig
    let sampleRate: Int
    let totalSamples: Int
    let prevConfirmedPieces: [String]
    let prevProvisionalPieces: [String]
    let prevFirstSeen: [Date]
    let prevAgreementCounts: [Int]
}

/// Orchestrates streaming speech-to-text inference.
///
/// Streaming decode runs on the current **pending** (partial) encoder window for
/// low-latency feedback. When a full encoder window completes, the session can
/// optionally run a one-shot decode for that completed window
/// (`StreamingConfig.finalizeCompletedWindows`) to improve accuracy, then resets
/// decode state for the next window.
public class StreamingInferenceSession: @unchecked Sendable, StreamingSession {
    private let backend: ModelBackend
    private let config: StreamingConfig

    // Qwen3-specific
    private let melProcessor: IncrementalMelSpectrogram?
    private let encoder: StreamingEncoder?

    // Parakeet-specific
    private var audioBuffer: [Float] = []
    /// Absolute sample index of `audioBuffer[0]`.
    private var audioBufferStartSample: Int = 0
    /// Absolute sample index of the next chunk start to freeze.
    private var frozenSampleCount: Int = 0
    private var parakeetSampleRate: Int = 16000

    private let shared = OSAllocatedUnfairLock(initialState: SessionSharedState())
    private let sessionLock = OSAllocatedUnfairLock(initialState: 0)

    private var isActive: Bool = false
    private var totalSamplesFed: Int = 0
    private var lastDecodeTime: Date?
    private var boundaryFastDecodeUntil: Date?
    private var hasNewEncoderContent: Bool = false
    /// Number of encoder windows whose text has been frozen into completedText.
    private var frozenWindowCount: Int = 0

    private var continuation: AsyncStream<TranscriptionEvent>.Continuation?
    private var decodeTask: Task<Void, Never>?
    private var stopTask: Task<Void, Never>?

    public let events: AsyncStream<TranscriptionEvent>

    public init(model: Qwen3ASRModel, config: StreamingConfig = StreamingConfig()) {
        self.backend = .qwen3(model)
        self.config = config
        let overlapFrames = max(0, Int(round(config.encoderWindowOverlapSeconds * Double(model.sampleRate) / 160.0)))
        self.melProcessor = IncrementalMelSpectrogram(
            sampleRate: model.sampleRate,
            nFft: 400,
            hopLength: 160,
            nMels: model.config.audioConfig.numMelBins
        )
        self.encoder = StreamingEncoder(
            encoder: model.audioTower,
            maxCachedWindows: config.maxCachedWindows,
            overlapFrames: overlapFrames
        )

        var continuation: AsyncStream<TranscriptionEvent>.Continuation!
        self.events = AsyncStream { continuation = $0 }
        self.continuation = continuation
        self.isActive = true
    }

    public init(model: ParakeetModel, config: StreamingConfig = StreamingConfig()) {
        self.backend = .parakeet(model)
        self.config = config
        self.melProcessor = nil
        self.encoder = nil
        self.parakeetSampleRate = model.preprocessConfig.sampleRate

        var continuation: AsyncStream<TranscriptionEvent>.Continuation!
        self.events = AsyncStream { continuation = $0 }
        self.continuation = continuation
        self.isActive = true
    }

    public func feedAudio(samples: [Float]) {
        switch backend {
        case .qwen3:
            feedAudioQwen3(samples: samples)
        case .parakeet:
            feedAudioParakeet(samples: samples)
        }
    }

    private func feedAudioQwen3(samples: [Float]) {
        sessionLock.withLock { _ in
            guard isActive else { return }

            totalSamplesFed += samples.count

            guard let melFrames = melProcessor?.process(samples: samples) else { return }
            guard let encoder else { return }

            let newWindows = encoder.feed(melFrames: melFrames)
            if newWindows > 0 || encoder.hasPendingFrames {
                hasNewEncoderContent = true
            }

            let now = Date()
            if newWindows > 0 {
                let boostSeconds = max(0, config.boundaryBoostSeconds)
                if boostSeconds > 0 {
                    boundaryFastDecodeUntil = now.addingTimeInterval(boostSeconds)
                } else {
                    boundaryFastDecodeUntil = nil
                }
            }

            let effectiveDecodeIntervalSeconds: Double
            if let boundaryFastDecodeUntil,
               now < boundaryFastDecodeUntil
            {
                let fastInterval = max(0.05, config.boundaryDecodeIntervalSeconds)
                let normalInterval = max(0.05, config.decodeIntervalSeconds)
                effectiveDecodeIntervalSeconds = min(fastInterval, normalInterval)
            } else {
                boundaryFastDecodeUntil = nil
                effectiveDecodeIntervalSeconds = max(0.05, config.decodeIntervalSeconds)
            }

            let shouldDecode: Bool
            if config.finalizeCompletedWindows, newWindows > 0 {
                shouldDecode = true
            } else if let lastDecode = lastDecodeTime {
                shouldDecode = now.timeIntervalSince(lastDecode) >= effectiveDecodeIntervalSeconds
            } else {
                shouldDecode = hasNewEncoderContent
            }

            if shouldDecode && hasNewEncoderContent {
                let canDecode = shared.withLock { state in
                    guard !state.isDecoding else { return false }
                    state.isDecoding = true
                    return true
                }

                if canDecode {
                    hasNewEncoderContent = false
                    let isBoundaryFinalizePass = config.finalizeCompletedWindows && newWindows > 0
                    if !isBoundaryFinalizePass {
                        lastDecodeTime = now
                    }
                    launchDecodePassLocked()
                }
            }
        }
    }

    private func feedAudioParakeet(samples: [Float]) {
        sessionLock.withLock { _ in
            guard isActive else { return }

            audioBuffer.append(contentsOf: samples)
            totalSamplesFed += samples.count

            let now = Date()
            let shouldDecode: Bool
            if let lastDecode = lastDecodeTime {
                shouldDecode = now.timeIntervalSince(lastDecode) >= max(0.05, config.decodeIntervalSeconds)
            } else {
                shouldDecode = !audioBuffer.isEmpty
            }

            guard shouldDecode else { return }

            let canDecode = shared.withLock { state in
                guard !state.isDecoding else { return false }
                state.isDecoding = true
                return true
            }
            guard canDecode else { return }

            lastDecodeTime = now

            guard case .parakeet(let model) = backend else { return }

            // NeMo-like buffered RNNT decode:
            // freeze `chunk` spans using windows of `totalBuffer` audio (left+right context).
            let sampleRate = parakeetSampleRate
            let chunkSeconds = max(0.2, config.bufferedChunkSeconds)
            let chunkSamples = max(1, Int(round(chunkSeconds * Double(sampleRate))))
            let totalBufferSeconds = max(config.bufferedTotalWindowSeconds, chunkSeconds)
            let totalBufferSamples = max(chunkSamples, Int(round(totalBufferSeconds * Double(sampleRate))))
            let contextSamples = max(0, totalBufferSamples - chunkSamples)
            let leftContextSamples = contextSamples / 2
            let rightContextSamples = contextSamples - leftContextSamples

            let bufferEndSample = audioBufferStartSample + audioBuffer.count

            var completedWindows: [ParakeetBufferedWindow] = []
            let maxCompletedWindowsPerPass = 3
            while completedWindows.count < maxCompletedWindowsPerPass,
                  frozenSampleCount + chunkSamples + rightContextSamples <= totalSamplesFed {
                let commitStart = frozenSampleCount
                let commitEnd = commitStart + chunkSamples
                let windowStart = max(0, commitStart - leftContextSamples)
                let windowEnd = commitEnd + rightContextSamples

                guard windowStart >= audioBufferStartSample,
                      windowEnd <= bufferEndSample
                else { break }

                let startIdx = windowStart - audioBufferStartSample
                let endIdx = windowEnd - audioBufferStartSample
                guard startIdx >= 0,
                      endIdx <= audioBuffer.count,
                      endIdx > startIdx
                else { break }

                completedWindows.append(
                    ParakeetBufferedWindow(
                        windowStartSample: windowStart,
                        commitStartSample: commitStart,
                        commitEndSample: commitEnd,
                        samples: Array(audioBuffer[startIdx..<endIdx])
                    )
                )
                frozenSampleCount += chunkSamples
            }

            let activeCommitStartSample = frozenSampleCount
            let activeWindowStartSample = max(0, activeCommitStartSample - leftContextSamples)
            let desiredActiveWindowEndSample = activeWindowStartSample + totalBufferSamples
            let activeWindowEndSample = min(desiredActiveWindowEndSample, bufferEndSample)

            var activeAudio: [Float] = []
            if activeWindowEndSample > activeWindowStartSample,
               activeWindowStartSample >= audioBufferStartSample {
                let startIdx = activeWindowStartSample - audioBufferStartSample
                let endIdx = activeWindowEndSample - audioBufferStartSample
                activeAudio = Array(audioBuffer[startIdx..<endIdx])
            }
            if activeAudio.count < totalBufferSamples {
                activeAudio.append(contentsOf: Array(repeating: 0, count: totalBufferSamples - activeAudio.count))
            }

            // Keep only the minimum history needed for the next buffered window.
            let keepFromSample = max(0, activeCommitStartSample - leftContextSamples)
            if keepFromSample > audioBufferStartSample {
                let dropCount = min(keepFromSample - audioBufferStartSample, audioBuffer.count)
                if dropCount > 0 {
                    audioBuffer.removeFirst(dropCount)
                    audioBufferStartSample += dropCount
                    Self.compactFloatBufferIfNeeded(
                        &audioBuffer,
                        minRetainedSamples: max(totalBufferSamples, audioBuffer.count)
                    )
                }
            }

            let snapshot = shared.withLock { state in
                (state.confirmedPieces, state.provisionalPieces,
                 state.pieceFirstSeen, state.pieceAgreementCounts)
            }

            let params = ParakeetDecodePassParams(
                completedWindows: UncheckedSendableBox(completedWindows),
                activeAudio: UncheckedSendableBox(activeAudio),
                activeWindowStartSample: activeWindowStartSample,
                activeCommitStartSample: activeCommitStartSample,
                model: UncheckedSendableBox(model),
                config: config,
                sampleRate: sampleRate,
                totalSamples: totalSamplesFed,
                prevConfirmedPieces: snapshot.0,
                prevProvisionalPieces: snapshot.1,
                prevFirstSeen: snapshot.2,
                prevAgreementCounts: snapshot.3
            )

            let continuation = self.continuation
            let sharedState = self.shared

            decodeTask = Task.detached {
                defer { sharedState.withLock { $0.isDecoding = false } }

                Self.runParakeetDecodePass(
                    params: params,
                    continuation: continuation,
                    sharedState: sharedState
                )
            }
        }
    }

    // MARK: - Window Completion

    /// When the encoder completes a full window, freeze the current streaming
    /// text and reset decode state — next decode starts fresh on new pending.
    private func freezeCompletedWindowsLocked() {
        guard let encoder else { return }
        let currentWindowCount = encoder.encodedWindowCount
        guard currentWindowCount > frozenWindowCount else { return }

        guard case .qwen3(let qwen3Model) = backend else { return }
        let tokenizer = qwen3Model.tokenizer

        shared.withLock { state in
            var allTokens = state.confirmedTokenIds
            allTokens.append(contentsOf: state.provisionalTokenIds)
            if let tokenizer, !allTokens.isEmpty {
                let windowText = tokenizer.decode(tokens: allTokens)
                Self.appendText(windowText, to: &state.completedText)
            }
            state.confirmedTokenIds = []
            state.provisionalTokenIds = []
            state.provisionalFirstSeen = []
            state.provisionalAgreementCounts = []
            state.confirmedText = ""
        }

        frozenWindowCount = currentWindowCount
    }

    private func launchDecodePassLocked() {
        guard let encoder else { return }
        guard case .qwen3(let qwen3Model) = backend else { return }

        if config.finalizeCompletedWindows {
            let windowsToFinalize = encoder.drainNewlyEncodedWindows()
            if !windowsToFinalize.isEmpty {
                frozenWindowCount = encoder.encodedWindowCount

                let params = FinalizeWindowsParams(
                    windows: UncheckedSendableBox(windowsToFinalize),
                    model: UncheckedSendableBox(qwen3Model),
                    config: self.config,
                    totalSamples: totalSamplesFed,
                    encodedWindowCount: encoder.encodedWindowCount
                )

                let continuation = self.continuation
                let sharedState = self.shared

                decodeTask = Task.detached {
                    defer {
                        sharedState.withLock { $0.isDecoding = false }
                    }

                    Self.runFinalizeCompletedWindows(
                        params: params,
                        continuation: continuation,
                        sharedState: sharedState
                    )
                }
                return
            }
        } else {
            freezeCompletedWindowsLocked()
        }

        guard let audioFeatures = encoder.encodePending() else {
            shared.withLock { $0.isDecoding = false }
            return
        }

        let snapshot = shared.withLock { state -> ([Int], String, [Int], [Date], [Int]) in
            let prefix = Self.concatText(state.completedText, state.confirmedText)
            return (state.confirmedTokenIds,
                    prefix,
                    state.provisionalTokenIds,
                    state.provisionalFirstSeen,
                    state.provisionalAgreementCounts)
        }
        let (confirmedTokenIds, displayPrefix, prevProvisional, prevFirstSeen, prevAgreementCounts) = snapshot
        let minAgreementPasses: Int
        if let boundaryFastDecodeUntil,
           Date() < boundaryFastDecodeUntil
        {
            minAgreementPasses = max(1, max(config.minAgreementPasses, config.boundaryMinAgreementPasses))
        } else {
            minAgreementPasses = max(1, config.minAgreementPasses)
        }

        let params = DecodePassParams(
            audioFeatures: UncheckedSendableBox(audioFeatures),
            model: UncheckedSendableBox(qwen3Model),
            config: self.config,
            confirmedTokenIds: confirmedTokenIds,
            displayPrefix: displayPrefix,
            prevProvisional: prevProvisional,
            prevFirstSeen: prevFirstSeen,
            prevAgreementCounts: prevAgreementCounts,
            minAgreementPasses: minAgreementPasses
        )

        let continuation = self.continuation
        let sharedState = self.shared
        let totalSamples = totalSamplesFed
        let encodedWindowCount = encoder.encodedWindowCount

        decodeTask = Task.detached {
            defer {
                sharedState.withLock { $0.isDecoding = false }
            }

            Self.runDecodePass(
                params: params,
                continuation: continuation,
                sharedState: sharedState,
                totalSamples: totalSamples,
                encodedWindowCount: encodedWindowCount
            )
        }
    }

    private static func appendText(_ segment: String, to base: inout String) {
        let normalizedSegment = segment.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalizedSegment.isEmpty else { return }
        if base.isEmpty {
            base = normalizedSegment
            return
        }
        let dedupedSegment = dedupeLeadingWordOverlap(base: base, segment: normalizedSegment)
        let containedTrimmedSegment = trimContainedLeadingOverlap(base: base, segment: dedupedSegment)
        guard !containedTrimmedSegment.isEmpty else { return }
        if shouldSkipDuplicateAppend(base: base, segment: containedTrimmedSegment) {
            return
        }
        if base.last?.isWhitespace == true || containedTrimmedSegment.first?.isWhitespace == true {
            base += containedTrimmedSegment
        } else {
            base += " " + containedTrimmedSegment
        }
    }

    private static func normalizedComparableWord(_ word: String) -> String {
        let asciiApostrophe: UnicodeScalar = "'"
        let smartApostrophe: UnicodeScalar = "’"
        let normalizedScalars = word.lowercased().unicodeScalars.filter { scalar in
            CharacterSet.alphanumerics.contains(scalar) ||
                scalar == asciiApostrophe ||
                scalar == smartApostrophe
        }
        return String(String.UnicodeScalarView(normalizedScalars))
    }

    private static func wordsEquivalent(
        lhsRaw: String,
        lhsNormalized: String,
        rhsRaw: String,
        rhsNormalized: String
    ) -> Bool {
        if !lhsNormalized.isEmpty && !rhsNormalized.isEmpty {
            return lhsNormalized == rhsNormalized
        }
        return lhsRaw.caseInsensitiveCompare(rhsRaw) == .orderedSame
    }

    private static func normalizedWords(_ text: String) -> [String] {
        text.split(whereSeparator: \.isWhitespace)
            .map { normalizedComparableWord(String($0)) }
            .filter { !$0.isEmpty }
    }

    private static func shouldSkipDuplicateAppend(base: String, segment: String) -> Bool {
        let segmentWords = normalizedWords(segment)
        guard !segmentWords.isEmpty else { return true }

        let baseWords = normalizedWords(base)
        guard !baseWords.isEmpty else { return false }

        if baseWords.count < segmentWords.count { return false }
        let lookbackCount = min(baseWords.count, max(segmentWords.count * 2, 48))
        let tailWords = Array(baseWords.suffix(lookbackCount))
        guard tailWords.count >= segmentWords.count else { return false }

        let tailSuffix = Array(tailWords.suffix(segmentWords.count))
        return tailSuffix == segmentWords
    }

    private static func containsContiguousSubsequence(
        haystack: [String],
        needle: [String]
    ) -> Bool {
        guard !needle.isEmpty, needle.count <= haystack.count else { return false }
        let maxStart = haystack.count - needle.count
        if maxStart < 0 { return false }

        for start in 0...maxStart {
            var matches = true
            for idx in 0..<needle.count where haystack[start + idx] != needle[idx] {
                matches = false
                break
            }
            if matches {
                return true
            }
        }

        return false
    }

    private static func trimContainedLeadingOverlap(base: String, segment: String) -> String {
        let segmentRawWords = segment.split(whereSeparator: \.isWhitespace).map(String.init)
        guard segmentRawWords.count >= 8 else { return segment }

        let baseWords = normalizedWords(base)
        guard !baseWords.isEmpty else { return segment }

        let segmentWords = segmentRawWords.map { normalizedComparableWord($0) }
        let lookbackCount = min(baseWords.count, max(segmentWords.count * 4, 160))
        let tailWords = Array(baseWords.suffix(lookbackCount))
        guard !tailWords.isEmpty else { return segment }

        let minOverlapWords = min(12, segmentWords.count)
        guard minOverlapWords >= 8 else { return segment }

        for overlap in stride(from: segmentWords.count, through: minOverlapWords, by: -1) {
            let prefix = Array(segmentWords.prefix(overlap))
            if containsContiguousSubsequence(haystack: tailWords, needle: prefix) {
                let remainder = segmentRawWords.dropFirst(overlap)
                return remainder.joined(separator: " ")
            }
        }

        return segment
    }

    private static func dedupeLeadingWordOverlap(base: String, segment: String, maxWords: Int = 64) -> String {
        let baseWords = base.split(whereSeparator: \.isWhitespace).map(String.init)
        let segmentWords = segment.split(whereSeparator: \.isWhitespace).map(String.init)
        guard !baseWords.isEmpty, !segmentWords.isEmpty else { return segment }
        let baseWordsNormalized = baseWords.map { normalizedComparableWord($0) }
        let segmentWordsNormalized = segmentWords.map { normalizedComparableWord($0) }

        let maxOverlap = min(maxWords, min(baseWords.count, segmentWords.count))
        var overlapCount = 0

        if maxOverlap > 0 {
            for size in stride(from: maxOverlap, through: 1, by: -1) {
                var matches = true
                for idx in 0..<size {
                    let lhsIdx = baseWords.count - size + idx
                    if !wordsEquivalent(
                        lhsRaw: baseWords[lhsIdx],
                        lhsNormalized: baseWordsNormalized[lhsIdx],
                        rhsRaw: segmentWords[idx],
                        rhsNormalized: segmentWordsNormalized[idx]
                    ) {
                        matches = false
                        break
                    }
                }
                if matches {
                    overlapCount = size
                    break
                }
            }
        }

        guard overlapCount > 0 else { return segment }
        let remainder = segmentWords.dropFirst(overlapCount)
        return remainder.joined(separator: " ")
    }

    private static func concatText(_ a: String, _ b: String) -> String {
        var result = a
        appendText(b, to: &result)
        return result
    }

    // MARK: - Decode (identical logic for every pass)

    private static func runDecodePass(
        params: DecodePassParams,
        continuation: AsyncStream<TranscriptionEvent>.Continuation?,
        sharedState: OSAllocatedUnfairLock<SessionSharedState>,
        totalSamples: Int,
        encodedWindowCount: Int
    ) {
        if Task.isCancelled { return }

        let model = params.model.value
        let audioFeatures = params.audioFeatures.value
        guard let tokenizer = model.tokenizer else { return }

        let numAudioTokens = audioFeatures.dim(0)
        guard numAudioTokens > 0 else { return }

        let eosTokenIds = [151645, 151643]
        let confirmedCount = params.confirmedTokenIds.count

        let inputIds = model.buildPrompt(
            numAudioTokens: numAudioTokens,
            language: params.config.language
        )

        let embeds = model.model.embedTokens(inputIds)
        let inputsEmbeds = model.mergeAudioFeatures(
            inputsEmbeds: embeds,
            audioFeatures: audioFeatures.asType(embeds.dtype),
            inputIds: inputIds
        )

        let cache = model.makeCache()
        var logits = model.callAsFunction(
            inputIds: inputIds,
            inputEmbeddings: inputsEmbeds,
            cache: cache
        )
        eval(logits)

        if Task.isCancelled { return }

        let windowedSeconds = Double(numAudioTokens) / 13.0
        let estimatedTotalTokens = max(24, Int(ceil(windowedSeconds * 10.0)))
        let maxTokens = min(
            params.config.maxTokensPerPass,
            max(estimatedTotalTokens, confirmedCount + 24)
        )

        var allTokenIds: [Int] = params.confirmedTokenIds
        let startTime = Date()

        for token in params.confirmedTokenIds {
            if Task.isCancelled { return }

            let tokenArray = MLXArray([Int32(token)]).expandedDimensions(axis: 0)
            logits = model.callAsFunction(inputIds: tokenArray, cache: cache)
            eval(logits)
        }

        if Task.isCancelled { return }

        let remaining = max(0, maxTokens - confirmedCount)
        for _ in 0..<remaining {
            if Task.isCancelled { return }

            var lastLogits = logits[0..., -1, 0...]
            if params.config.temperature > 0 {
                lastLogits = lastLogits / params.config.temperature
            }
            let nextToken = lastLogits.argMax(axis: -1).item(Int.self)

            if eosTokenIds.contains(nextToken) { break }

            allTokenIds.append(nextToken)

            if allTokenIds.count > confirmedCount {
                let newProvisional = Array(allTokenIds.dropFirst(confirmedCount))
                let provText = tokenizer.decode(tokens: newProvisional)
                continuation?.yield(.displayUpdate(
                    confirmedText: params.displayPrefix,
                    provisionalText: provText
                ))
            }

            let nextTokenArray = MLXArray([Int32(nextToken)]).expandedDimensions(axis: 0)
            logits = model.callAsFunction(inputIds: nextTokenArray, cache: cache)
            eval(logits)
        }

        let decodeTime = Date().timeIntervalSince(startTime)
        let genTokenCount = allTokenIds.count

        Memory.clearCache()

        if Task.isCancelled { return }

        promoteTokens(
            allTokenIds: allTokenIds,
            params: params,
            continuation: continuation,
            sharedState: sharedState,
            tokenizer: tokenizer,
            totalSamples: totalSamples,
            decodeTime: decodeTime,
            genTokenCount: genTokenCount,
            encodedWindowCount: encodedWindowCount
        )
    }

    private static func promoteTokens(
        allTokenIds: [Int],
        params: DecodePassParams,
        continuation: AsyncStream<TranscriptionEvent>.Continuation?,
        sharedState: OSAllocatedUnfairLock<SessionSharedState>,
        tokenizer: any Tokenizer,
        totalSamples: Int,
        decodeTime: Double,
        genTokenCount: Int,
        encodedWindowCount: Int
    ) {
        let confirmedCount = params.confirmedTokenIds.count
        let prevProvisional = params.prevProvisional
        let prevFirstSeen = params.prevFirstSeen
        let prevAgreementCounts = params.prevAgreementCounts

        let newProvisional = Array(allTokenIds.dropFirst(confirmedCount))

        let now = Date()
        let delaySeconds = Double(params.config.delayPreset.delayMs) / 1000.0

        var matchLen = 0
        let compareLen = min(prevProvisional.count, newProvisional.count)
        for i in 0..<compareLen {
            if prevProvisional[i] == newProvisional[i] {
                matchLen = i + 1
            } else {
                break
            }
        }

        var nextFirstSeen: [Date] = []
        nextFirstSeen.reserveCapacity(newProvisional.count)
        var nextAgreementCounts: [Int] = []
        nextAgreementCounts.reserveCapacity(newProvisional.count)

        for i in 0..<newProvisional.count {
            if i < matchLen {
                let firstSeen = i < prevFirstSeen.count ? prevFirstSeen[i] : now
                let prevAgreement = i < prevAgreementCounts.count ? prevAgreementCounts[i] : 1
                nextFirstSeen.append(firstSeen)
                nextAgreementCounts.append(max(1, prevAgreement + 1))
            } else {
                nextFirstSeen.append(now)
                nextAgreementCounts.append(1)
            }
        }

        let requiredAgreementPasses = max(1, params.minAgreementPasses)
        var promotionCount = 0
        for i in 0..<newProvisional.count {
            let hasDelay = i < nextFirstSeen.count && now.timeIntervalSince(nextFirstSeen[i]) >= delaySeconds
            let hasAgreement = i < nextAgreementCounts.count && nextAgreementCounts[i] >= requiredAgreementPasses
            if hasDelay && hasAgreement {
                promotionCount = i + 1
            } else {
                break
            }
        }

        let promoteCount = promotionCount
        let finalProvisional = Array(newProvisional.dropFirst(promoteCount))
        let finalFirstSeen = Array(nextFirstSeen.dropFirst(promoteCount))
        let finalAgreementCounts = Array(nextAgreementCounts.dropFirst(promoteCount))

        let displayPrefix: String = sharedState.withLock { state in
            if promoteCount > 0 {
                let promoted = Array(newProvisional.prefix(promoteCount))
                state.confirmedTokenIds.append(contentsOf: promoted)
                state.confirmedText = tokenizer.decode(tokens: state.confirmedTokenIds)
                continuation?.yield(.confirmed(text: Self.concatText(state.completedText, state.confirmedText)))
            }
            state.provisionalTokenIds = finalProvisional
            state.provisionalFirstSeen = finalFirstSeen
            state.provisionalAgreementCounts = finalAgreementCounts
            return Self.concatText(state.completedText, state.confirmedText)
        }

        let finalProvText = tokenizer.decode(tokens: finalProvisional)
        continuation?.yield(.displayUpdate(
            confirmedText: displayPrefix,
            provisionalText: finalProvText
        ))

        let totalAudioSeconds = Double(totalSamples) / 16000.0
        let tps = decodeTime > 0 ? Double(genTokenCount) / decodeTime : 0
        continuation?.yield(.stats(StreamingStats(
            encodedWindowCount: encodedWindowCount,
            totalAudioSeconds: totalAudioSeconds,
            tokensPerSecond: tps,
            realTimeFactor: 0,
            peakMemoryGB: Double(Memory.peakMemory) / 1e9
        )))
    }

    private static func runFinalizeCompletedWindows(
        params: FinalizeWindowsParams,
        continuation: AsyncStream<TranscriptionEvent>.Continuation?,
        sharedState: OSAllocatedUnfairLock<SessionSharedState>
    ) {
        if Task.isCancelled { return }

        let model = params.model.value
        guard let tokenizer = model.tokenizer else { return }

        let windows = params.windows.value
        guard !windows.isEmpty else { return }

        var totalDecodeTime: Double = 0
        var totalGeneratedTokens: Int = 0
        let streamedFallbackForFirstWindow: String? = sharedState.withLock { state in
            var streamTokens = state.confirmedTokenIds
            streamTokens.append(contentsOf: state.provisionalTokenIds)
            guard !streamTokens.isEmpty else { return nil }
            return tokenizer.decode(tokens: streamTokens)
        }

        for (idx, audioFeatures) in windows.enumerated() {
            if Task.isCancelled { return }

            let selectedWindowText: String
            if idx == 0, let streamedFallbackForFirstWindow {
                selectedWindowText = streamedFallbackForFirstWindow
            } else {
                let numAudioTokens = audioFeatures.dim(0)
                if numAudioTokens <= 0 { continue }

                let startTime = Date()
                let tokenIds = decodeAllTokenIds(
                    model: model,
                    audioFeatures: audioFeatures,
                    confirmedCount: 0,
                    config: params.config
                )
                if Task.isCancelled { return }

                let decodeTime = Date().timeIntervalSince(startTime)
                totalDecodeTime += decodeTime
                totalGeneratedTokens += tokenIds.count

                let windowText = tokenizer.decode(tokens: tokenIds)
                selectedWindowText = windowText
            }
            if selectedWindowText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { continue }

            sharedState.withLock { state in
                Self.appendText(selectedWindowText, to: &state.completedText)
                state.confirmedTokenIds = []
                state.provisionalTokenIds = []
                state.provisionalFirstSeen = []
                state.provisionalAgreementCounts = []
                state.confirmedText = ""
            }
        }

        Memory.clearCache()

        let totalAudioSeconds = Double(params.totalSamples) / 16000.0
        let tps = totalDecodeTime > 0 ? Double(totalGeneratedTokens) / totalDecodeTime : 0
        continuation?.yield(.stats(StreamingStats(
            encodedWindowCount: params.encodedWindowCount,
            totalAudioSeconds: totalAudioSeconds,
            tokensPerSecond: tps,
            realTimeFactor: 0,
            peakMemoryGB: Double(Memory.peakMemory) / 1e9
        )))
    }

    // MARK: - Parakeet Decode Pass

    private static func absoluteParakeetTokens(
        from result: ParakeetAlignedResult,
        windowStartSample: Int,
        sampleRate: Int
    ) -> [ParakeetAlignedToken] {
        let offsetSeconds = Double(windowStartSample) / Double(sampleRate)
        var tokens = result.sentences.flatMap(\.tokens)
        guard offsetSeconds != 0 else { return tokens }
        for i in tokens.indices {
            tokens[i].start += offsetSeconds
        }
        return tokens
    }

    private static func parakeetText(from tokens: [ParakeetAlignedToken]) -> String {
        tokens.map(\.text)
            .joined()
    }

    private static func compactFloatBufferIfNeeded(_ buffer: inout [Float], minRetainedSamples: Int) {
        let minSamples = max(4096, minRetainedSamples)
        let maxAllowedCapacity = minSamples * 8
        guard buffer.capacity > maxAllowedCapacity,
              buffer.count < minSamples * 2
        else { return }

        var compacted = Array(buffer)
        compacted.reserveCapacity(max(compacted.count, minSamples * 2))
        buffer = compacted
    }

    private static func mergeParakeetTokenSequences(
        existing: [ParakeetAlignedToken],
        incoming: [ParakeetAlignedToken],
        overlapDuration: Double
    ) -> [ParakeetAlignedToken] {
        if existing.isEmpty { return incoming }
        if incoming.isEmpty { return existing }

        do {
            return try ParakeetAlignment.mergeLongestContiguous(
                existing,
                incoming,
                overlapDuration: overlapDuration
            )
        } catch {
            return ParakeetAlignment.mergeLongestCommonSubsequence(
                existing,
                incoming,
                overlapDuration: overlapDuration
            )
        }
    }

    private static func splitParakeetTokensForCommit(
        _ tokens: [ParakeetAlignedToken],
        commitEndSample: Int,
        sampleRate: Int
    ) -> (committed: [ParakeetAlignedToken], pending: [ParakeetAlignedToken]) {
        let commitEnd = Double(commitEndSample) / Double(sampleRate)
        var commitCount = 0
        while commitCount < tokens.count {
            let token = tokens[commitCount]
            let midpoint = token.start + (token.duration * 0.5)
            if midpoint < commitEnd {
                commitCount += 1
            } else {
                break
            }
        }

        if commitCount <= 0 {
            return ([], tokens)
        }
        if commitCount >= tokens.count {
            return (tokens, [])
        }
        return (
            Array(tokens[..<commitCount]),
            Array(tokens[commitCount...])
        )
    }

    private static func concatParakeetText(_ base: String, _ segment: String) -> String {
        if base.isEmpty { return segment }
        if segment.isEmpty { return base }
        return base + segment
    }

    private static func appendParakeetTextSegment(_ segment: String, to base: inout String) {
        let normalizedSegment = segment.replacingOccurrences(of: "\n", with: " ")
        guard !normalizedSegment.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }

        if base.isEmpty {
            base = normalizedSegment.trimmingCharacters(in: .whitespacesAndNewlines)
        } else {
            base += normalizedSegment
        }
    }

    private static func runParakeetDecodePass(
        params: ParakeetDecodePassParams,
        continuation: AsyncStream<TranscriptionEvent>.Continuation?,
        sharedState: OSAllocatedUnfairLock<SessionSharedState>
    ) {
        if Task.isCancelled { return }

        let model = params.model.value

        // 1. Freeze completed chunk spans from buffered windows.
        let completedWindows = params.completedWindows.value
        var didFreezeWindows = false
        if !completedWindows.isEmpty {
            let chunkSeconds = max(0.2, params.config.bufferedChunkSeconds)
            let totalBufferSeconds = max(params.config.bufferedTotalWindowSeconds, chunkSeconds)
            let overlapDuration = max(0.2, totalBufferSeconds - chunkSeconds)

            for window in completedWindows {
                if Task.isCancelled { return }
                let windowArray = MLXArray(window.samples)
                let result = model.decodeChunk(windowArray)
                eval()

                let windowTokens = absoluteParakeetTokens(
                    from: result,
                    windowStartSample: window.windowStartSample,
                    sampleRate: params.sampleRate
                )
                sharedState.withLock { state in
                    state.parakeetMergedTokens = Self.mergeParakeetTokenSequences(
                        existing: state.parakeetMergedTokens,
                        incoming: windowTokens,
                        overlapDuration: overlapDuration
                    )

                    if let pendingCommitEnd = state.parakeetPendingCommitEndSample {
                        let split = Self.splitParakeetTokensForCommit(
                            state.parakeetMergedTokens,
                            commitEndSample: pendingCommitEnd,
                            sampleRate: params.sampleRate
                        )
                        state.parakeetMergedTokens = split.pending

                        let committedText = parakeetText(from: split.committed)
                        if !committedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                            Self.appendParakeetTextSegment(committedText, to: &state.completedText)
                        }
                    }

                    // Hold back this boundary until the next window arrives.
                    state.parakeetPendingCommitEndSample = window.commitEndSample
                }
                Memory.clearCache()
            }
            didFreezeWindows = true

            // Reset word promotion because the commit boundary advanced.
            sharedState.withLock { state in
                state.confirmedPieces = []
                state.provisionalPieces = []
                state.pieceFirstSeen = []
                state.pieceAgreementCounts = []
            }

            // Emit display update so frozen text appears immediately
            let frozenText = sharedState.withLock { $0.completedText }
            continuation?.yield(.displayUpdate(confirmedText: frozenText, provisionalText: ""))
        }

        // 2. Streaming decode on an active window that includes left context.
        let audio = params.activeAudio.value
        guard !audio.isEmpty else { return }

        let startTime = Date()
        let audioArray = MLXArray(audio)
        let result = model.decodeChunk(audioArray)
        eval()
        let decodeTime = Date().timeIntervalSince(startTime)

        if Task.isCancelled { return }

        let activeTokens = absoluteParakeetTokens(
            from: result,
            windowStartSample: params.activeWindowStartSample,
            sampleRate: params.sampleRate
        ).filter { token in
            let midpoint = token.start + (token.duration * 0.5)
            return midpoint >= Double(params.activeCommitStartSample) / Double(params.sampleRate)
        }

        let currentPieces = activeTokens.map(\.text).filter { !$0.isEmpty }

        // After freezing windows, override stale params with empty piece state.
        let effectiveParams: ParakeetDecodePassParams
        if didFreezeWindows {
            effectiveParams = ParakeetDecodePassParams(
                completedWindows: params.completedWindows,
                activeAudio: params.activeAudio,
                activeWindowStartSample: params.activeWindowStartSample,
                activeCommitStartSample: params.activeCommitStartSample,
                model: params.model,
                config: params.config,
                sampleRate: params.sampleRate,
                totalSamples: params.totalSamples,
                prevConfirmedPieces: [],
                prevProvisionalPieces: [],
                prevFirstSeen: [],
                prevAgreementCounts: []
            )
        } else {
            effectiveParams = params
        }

        promoteParakeetPieces(
            currentPieces: currentPieces,
            params: effectiveParams,
            continuation: continuation,
            sharedState: sharedState,
            decodeTime: decodeTime
        )
    }

    private static func normalizedParakeetPiece(_ piece: String) -> String {
        piece.lowercased()
    }

    private static func parakeetPiecesEquivalent(_ lhs: String, _ rhs: String) -> Bool {
        let lhsNormalized = normalizedParakeetPiece(lhs)
        let rhsNormalized = normalizedParakeetPiece(rhs)
        if !lhsNormalized.isEmpty && !rhsNormalized.isEmpty {
            return lhsNormalized == rhsNormalized
        }
        return lhs == rhs
    }

    private static func promoteParakeetPieces(
        currentPieces: [String],
        params: ParakeetDecodePassParams,
        continuation: AsyncStream<TranscriptionEvent>.Continuation?,
        sharedState: OSAllocatedUnfairLock<SessionSharedState>,
        decodeTime: Double
    ) {
        // After a chunk freeze, confirmed/provisional are empty and currentPieces
        // covers only the new partial chunk. Clamp confirmedCount so we never
        // build an invalid range.
        let confirmedCount = min(params.prevConfirmedPieces.count, currentPieces.count)
        let prevConfirmedPrefix = Array(params.prevConfirmedPieces.prefix(confirmedCount))
        let prevAllPieces = prevConfirmedPrefix + params.prevProvisionalPieces

        let now = Date()
        let delaySeconds = Double(params.config.delayPreset.delayMs) / 1000.0

        let compareLen = min(prevAllPieces.count, currentPieces.count)
        var matchLen = 0
        for i in 0..<compareLen {
            if parakeetPiecesEquivalent(prevAllPieces[i], currentPieces[i]) {
                matchLen = i + 1
            } else {
                break
            }
        }

        let provisionalStart = confirmedCount
        var nextFirstSeen: [Date] = []
        var nextAgreementCounts: [Int] = []

        for i in provisionalStart..<currentPieces.count {
            let prevProvIdx = i - confirmedCount
            if i < matchLen && prevProvIdx >= 0 && prevProvIdx < params.prevFirstSeen.count {
                nextFirstSeen.append(params.prevFirstSeen[prevProvIdx])
                let prevAg = prevProvIdx < params.prevAgreementCounts.count
                    ? params.prevAgreementCounts[prevProvIdx] : 1
                nextAgreementCounts.append(prevAg + 1)
            } else {
                nextFirstSeen.append(now)
                nextAgreementCounts.append(1)
            }
        }

        let minAgreement = max(1, params.config.minAgreementPasses)
        var promotionCount = 0
        for i in 0..<nextFirstSeen.count {
            let wordIdx = provisionalStart + i
            guard wordIdx < matchLen else { break }
            let hasDelay = now.timeIntervalSince(nextFirstSeen[i]) >= delaySeconds
            let hasAgreement = nextAgreementCounts[i] >= minAgreement
            if hasDelay && hasAgreement {
                promotionCount = i + 1
            } else {
                break
            }
        }

        let newConfirmedPieces = prevConfirmedPrefix + Array(currentPieces[confirmedCount..<(confirmedCount + promotionCount)])
        let newProvisionalPieces = Array(currentPieces.dropFirst(confirmedCount + promotionCount))
        let finalFirstSeen = Array(nextFirstSeen.dropFirst(promotionCount))
        let finalAgreementCounts = Array(nextAgreementCounts.dropFirst(promotionCount))

        let confirmedText = newConfirmedPieces.joined()
        let provisionalText = newProvisionalPieces.joined()

        let displayPrefix: String = sharedState.withLock { state in
            state.confirmedPieces = newConfirmedPieces
            state.provisionalPieces = newProvisionalPieces
            state.pieceFirstSeen = finalFirstSeen
            state.pieceAgreementCounts = finalAgreementCounts

            let frozen = state.completedText
            let confirmed = confirmedText
            return concatParakeetText(frozen, confirmed).trimmingCharacters(in: .newlines)
        }

        continuation?.yield(.displayUpdate(
            confirmedText: displayPrefix,
            provisionalText: provisionalText
        ))

        if promotionCount > 0 {
            continuation?.yield(.confirmed(text: displayPrefix))
        }

        let totalAudioSeconds = Double(params.totalSamples) / Double(params.sampleRate)
        let piecesPerSec = decodeTime > 0 ? Double(currentPieces.count) / decodeTime : 0
        continuation?.yield(.stats(StreamingStats(
            encodedWindowCount: 0,
            totalAudioSeconds: totalAudioSeconds,
            tokensPerSecond: piecesPerSec,
            realTimeFactor: totalAudioSeconds > 0 ? decodeTime / totalAudioSeconds : 0,
            peakMemoryGB: Double(Memory.peakMemory) / 1e9
        )))

        Memory.clearCache()
    }

    // MARK: - Stop / Cancel

    public func stop() {
        sessionLock.withLock { _ in
            guard isActive else { return }
            isActive = false

            let inFlightDecode = decodeTask
            decodeTask = nil

            stopTask?.cancel()
            switch backend {
            case .qwen3:
                stopTask = Task.detached { [self] in
                    await finishStopQwen3(waitingFor: inFlightDecode)
                }
            case .parakeet:
                stopTask = Task.detached { [self] in
                    await finishStopParakeet(waitingFor: inFlightDecode)
                }
            }
        }
    }

    private func finishStopParakeet(waitingFor inFlightDecode: Task<Void, Never>?) async {
        if let inFlightDecode {
            _ = await inFlightDecode.value
        }

        guard case .parakeet(let model) = backend else { return }
        let sampleRateForCommit = parakeetSampleRate

        shared.withLock { state in
            if let pendingCommitEnd = state.parakeetPendingCommitEndSample {
                let split = Self.splitParakeetTokensForCommit(
                    state.parakeetMergedTokens,
                    commitEndSample: pendingCommitEnd,
                    sampleRate: sampleRateForCommit
                )
                state.parakeetMergedTokens = split.pending
                let committedText = Self.parakeetText(from: split.committed)
                if !committedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    Self.appendParakeetTextSegment(committedText, to: &state.completedText)
                }
                state.parakeetPendingCommitEndSample = nil
            }
        }

        let finalSnapshot: (audio: [Float], windowStartSample: Int, commitStartSample: Int, sampleRate: Int) = sessionLock.withLock { _ in
            let sampleRate = parakeetSampleRate
            let chunkSeconds = max(0.2, config.bufferedChunkSeconds)
            let chunkSamples = max(1, Int(round(chunkSeconds * Double(sampleRate))))
            let totalBufferSeconds = max(config.bufferedTotalWindowSeconds, chunkSeconds)
            let totalBufferSamples = max(chunkSamples, Int(round(totalBufferSeconds * Double(sampleRate))))
            let contextSamples = max(0, totalBufferSamples - chunkSamples)
            let leftContextSamples = contextSamples / 2

            let windowStartSample = max(0, frozenSampleCount - leftContextSamples)
            let bufferEndSample = audioBufferStartSample + audioBuffer.count

            if windowStartSample >= audioBufferStartSample,
               bufferEndSample > windowStartSample {
                let startIdx = windowStartSample - audioBufferStartSample
                let finalAudio = Array(audioBuffer[startIdx..<audioBuffer.count])
                return (finalAudio, windowStartSample, frozenSampleCount, sampleRate)
            }

            return ([], 0, frozenSampleCount, sampleRate)
        }

        let finalText: String
        if !finalSnapshot.audio.isEmpty {
            let audioArray = MLXArray(finalSnapshot.audio)
            let result = model.decodeChunk(audioArray)
            eval()

            finalText = shared.withLock { state in
                let tokens = Self.absoluteParakeetTokens(
                    from: result,
                    windowStartSample: finalSnapshot.windowStartSample,
                    sampleRate: finalSnapshot.sampleRate
                ).filter { token in
                    let midpoint = token.start + (token.duration * 0.5)
                    return midpoint >= Double(finalSnapshot.commitStartSample) / Double(finalSnapshot.sampleRate)
                }
                let decodedTail = Self.parakeetText(from: tokens)
                if decodedTail.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    let fallbackTail = (state.confirmedPieces + state.provisionalPieces).joined()
                    let mergedTail = Self.parakeetText(from: state.parakeetMergedTokens)
                    let fallbackCombined = Self.concatParakeetText(fallbackTail, mergedTail)
                    return Self.concatParakeetText(state.completedText, fallbackCombined)
                }
                return Self.concatParakeetText(state.completedText, decodedTail)
            }
        } else {
            finalText = shared.withLock { state in
                let allPieces = state.confirmedPieces + state.provisionalPieces
                let current = allPieces.joined()
                let mergedTail = Self.parakeetText(from: state.parakeetMergedTokens)
                let combinedTail = Self.concatParakeetText(current, mergedTail)
                return Self.concatParakeetText(state.completedText, combinedTail)
            }
        }

        continuation?.yield(.ended(fullText: finalText))
        continuation?.finish()

        sessionLock.withLock { _ in
            continuation = nil
            stopTask = nil
            audioBuffer = []
            audioBufferStartSample = 0
            frozenSampleCount = 0
        }
        shared.withLock { $0 = SessionSharedState() }

        Memory.clearCache()
    }

    private func finishStopQwen3(waitingFor inFlightDecode: Task<Void, Never>?) async {
        if let inFlightDecode {
            _ = await inFlightDecode.value
        }

        if Task.isCancelled {
            return
        }

        guard case .qwen3(let qwen3Model) = backend else { return }
        let qwen3Tokenizer = qwen3Model.tokenizer

        let snapshot: StopSnapshot = sessionLock.withLock { _ in
            if let melFrames = melProcessor?.flush() {
                _ = encoder?.feed(melFrames: melFrames)
            }

            let completedWindows: [MLXArray]
            if config.finalizeCompletedWindows, let encoder {
                completedWindows = encoder.drainNewlyEncodedWindows()
                frozenWindowCount = encoder.encodedWindowCount
            } else {
                completedWindows = []
                freezeCompletedWindowsLocked()
            }

            let continuation = self.continuation
            let totalSamples = totalSamplesFed
            let encodedWindowCount = encoder?.encodedWindowCount ?? 0

            if let audioFeatures = encoder?.encodePending(),
               audioFeatures.dim(0) > 0,
               qwen3Tokenizer != nil
            {
                let confirmedCount = shared.withLock { $0.confirmedTokenIds.count }
                return StopSnapshot(
                    continuation: continuation,
                    completedWindows: completedWindows.isEmpty ? nil : UncheckedSendableBox(completedWindows),
                    pendingAudioFeatures: UncheckedSendableBox(audioFeatures),
                    confirmedCount: confirmedCount,
                    totalSamples: totalSamples,
                    encodedWindowCount: encodedWindowCount,
                    fallbackFinalText: nil
                )
            }

            let fallbackFinalText = shared.withLock { state in
                if !state.provisionalTokenIds.isEmpty {
                    state.confirmedTokenIds.append(contentsOf: state.provisionalTokenIds)
                    state.provisionalTokenIds = []
                    state.provisionalFirstSeen = []
                    state.provisionalAgreementCounts = []
                }
                if let qwen3Tokenizer, !state.confirmedTokenIds.isEmpty {
                    state.confirmedText = qwen3Tokenizer.decode(tokens: state.confirmedTokenIds)
                }
                return Self.concatText(state.completedText, state.confirmedText)
            }

            return StopSnapshot(
                continuation: continuation,
                completedWindows: completedWindows.isEmpty ? nil : UncheckedSendableBox(completedWindows),
                pendingAudioFeatures: nil,
                confirmedCount: 0,
                totalSamples: totalSamples,
                encodedWindowCount: encodedWindowCount,
                fallbackFinalText: fallbackFinalText
            )
        }

        if Task.isCancelled {
            return
        }

        if let completedWindows = snapshot.completedWindows?.value,
           !completedWindows.isEmpty,
           let tokenizer = qwen3Model.tokenizer
        {
            for audioFeatures in completedWindows {
                if Task.isCancelled { return }

                if audioFeatures.dim(0) <= 0 { continue }
                let tokenIds = Self.decodeAllTokenIds(
                    model: qwen3Model,
                    audioFeatures: audioFeatures,
                    confirmedCount: 0,
                    config: config
                )
                if Task.isCancelled { return }

                let windowText = tokenizer.decode(tokens: tokenIds)
                if windowText.isEmpty { continue }

                shared.withLock { state in
                    Self.appendText(windowText, to: &state.completedText)
                    state.confirmedTokenIds = []
                    state.provisionalTokenIds = []
                    state.provisionalFirstSeen = []
                    state.provisionalAgreementCounts = []
                    state.confirmedText = ""
                }
            }

            Memory.clearCache()
        }

        let finalText: String
        if let audioFeatures = snapshot.pendingAudioFeatures?.value,
           let tokenizer = qwen3Model.tokenizer
        {
            let startTime = Date()
            let tokenIds = Self.decodeAllTokenIds(
                model: qwen3Model,
                audioFeatures: audioFeatures,
                confirmedCount: snapshot.confirmedCount,
                config: config
            )
            if Task.isCancelled {
                return
            }

            let decodeTime = Date().timeIntervalSince(startTime)
            Memory.clearCache()

            finalText = shared.withLock { state in
                state.confirmedTokenIds = tokenIds
                state.provisionalTokenIds = []
                state.provisionalFirstSeen = []
                state.provisionalAgreementCounts = []
                state.confirmedText = tokenizer.decode(tokens: tokenIds)
                return Self.concatText(state.completedText, state.confirmedText)
            }

            let totalAudioSeconds = Double(snapshot.totalSamples) / 16000.0
            let tps = decodeTime > 0 ? Double(tokenIds.count) / decodeTime : 0
            snapshot.continuation?.yield(.stats(StreamingStats(
                encodedWindowCount: snapshot.encodedWindowCount,
                totalAudioSeconds: totalAudioSeconds,
                tokensPerSecond: tps,
                realTimeFactor: 0,
                peakMemoryGB: Double(Memory.peakMemory) / 1e9
            )))
        } else {
            let tokenizer2 = qwen3Model.tokenizer
            finalText = shared.withLock { state in
                if !state.provisionalTokenIds.isEmpty {
                    state.confirmedTokenIds.append(contentsOf: state.provisionalTokenIds)
                    state.provisionalTokenIds = []
                    state.provisionalFirstSeen = []
                    state.provisionalAgreementCounts = []
                }
                if let tokenizer2, !state.confirmedTokenIds.isEmpty {
                    state.confirmedText = tokenizer2.decode(tokens: state.confirmedTokenIds)
                }
                return Self.concatText(state.completedText, state.confirmedText)
            }
        }

        if Task.isCancelled {
            return
        }

        snapshot.continuation?.yield(.ended(fullText: finalText))
        snapshot.continuation?.finish()

        sessionLock.withLock { _ in
            self.continuation = nil
            stopTask = nil
            encoder?.reset()
            melProcessor?.reset()
            boundaryFastDecodeUntil = nil
        }
        shared.withLock { $0 = SessionSharedState() }

        Memory.clearCache()
    }

    public func cancel() {
        sessionLock.withLock { _ in
            isActive = false
            decodeTask?.cancel()
            decodeTask = nil
            stopTask?.cancel()
            stopTask = nil
            continuation?.finish()
            continuation = nil
            encoder?.reset()
            melProcessor?.reset()
            audioBuffer = []
            audioBufferStartSample = 0
            frozenSampleCount = 0
            boundaryFastDecodeUntil = nil
        }
        shared.withLock { $0 = SessionSharedState() }
        Memory.clearCache()
    }

    private static func decodeAllTokenIds(
        model: Qwen3ASRModel,
        audioFeatures: MLXArray,
        confirmedCount: Int,
        config: StreamingConfig
    ) -> [Int] {
        if Task.isCancelled { return [] }

        let numAudioTokens = audioFeatures.dim(0)
        let eosTokenIds = [151645, 151643]

        let inputIds = model.buildPrompt(
            numAudioTokens: numAudioTokens,
            language: config.language
        )

        let embeds = model.model.embedTokens(inputIds)
        let inputsEmbeds = model.mergeAudioFeatures(
            inputsEmbeds: embeds,
            audioFeatures: audioFeatures.asType(embeds.dtype),
            inputIds: inputIds
        )

        let cache = model.makeCache()
        var logits = model.callAsFunction(
            inputIds: inputIds,
            inputEmbeddings: inputsEmbeds,
            cache: cache
        )
        eval(logits)

        let windowedSeconds = Double(numAudioTokens) / 13.0
        let estimatedTotalTokens = max(24, Int(ceil(windowedSeconds * 10.0)))
        let maxTokens = min(
            config.maxTokensPerPass,
            max(estimatedTotalTokens, confirmedCount + 24)
        )

        var allTokenIds: [Int] = []
        allTokenIds.reserveCapacity(maxTokens)

        for _ in 0..<maxTokens {
            if Task.isCancelled { return [] }

            var lastLogits = logits[0..., -1, 0...]
            if config.temperature > 0 {
                lastLogits = lastLogits / config.temperature
            }
            let nextToken = lastLogits.argMax(axis: -1).item(Int.self)

            if eosTokenIds.contains(nextToken) { break }
            allTokenIds.append(nextToken)

            let nextTokenArray = MLXArray([Int32(nextToken)]).expandedDimensions(axis: 0)
            logits = model.callAsFunction(inputIds: nextTokenArray, cache: cache)
            eval(logits)
        }

        return allTokenIds
    }
}
