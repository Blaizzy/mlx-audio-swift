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

private struct CohereStreamingSharedState: Sendable {
    var pendingSamples: [Float] = []
    var totalSamplesFed: Int = 0
    var lastDecodeTime: Date?
    var isDecoding: Bool = false
    var finalizedWindowCount: Int = 0
}

private enum CohereDecodePassKind: Sendable {
    case partial
    case finalWindow
}

private struct CohereDecodePassParams: Sendable {
    let audio: UncheckedSendableBox<MLXArray>
    let model: UncheckedSendableBox<CohereTranscribeModel>
    let config: StreamingConfig
    let kind: CohereDecodePassKind
    let confirmedTokenIds: [Int]
    let displayPrefix: String
    let prevProvisional: [Int]
    let prevFirstSeen: [Date]
    let prevAgreementCounts: [Int]
    let minAgreementPasses: Int
    let totalSamples: Int
}

private struct CohereDecodeLaunch: Sendable {
    let samples: [Float]
    let kind: CohereDecodePassKind
    let totalSamples: Int
    let encodedWindowCount: Int
}

private struct MossStreamingSharedState: Sendable {
    var completedText: String = ""
    var provisionalText: String = ""
}

private struct MossAudioStreamingSharedState: Sendable {
    var pendingSamples: [Float] = []
    var pendingStartSample: Int = 0
    var totalSamplesFed: Int = 0
    var lastDecodeTime: Date?
    var isDecoding: Bool = false
    var finalizedWindowCount: Int = 0
}

private enum MossDecodePassKind: Sendable {
    case partial
    case finalWindow
}

private struct MossDecodePassParams: Sendable {
    let audio: UncheckedSendableBox<MLXArray>
    let model: UncheckedSendableBox<MossTranscribeDiarizeModel>
    let config: StreamingConfig
    let kind: MossDecodePassKind
    let offsetSeconds: Double
    let totalSamples: Int
    let encodedWindowCount: Int
}

private struct MossDecodeLaunch: Sendable {
    let samples: [Float]
    let kind: MossDecodePassKind
    let offsetSeconds: Double
    let totalSamples: Int
    let encodedWindowCount: Int
}

private protocol StreamingInferenceSessionCore: AnyObject, Sendable {
    var events: AsyncStream<TranscriptionEvent> { get }

    func feedAudio(samples: [Float])
    func stop()
    func cancel()
}

/// Orchestrates streaming speech-to-text inference.
///
/// Streaming decode runs on the current **pending** (partial) encoder window for
/// low-latency feedback. When a full encoder window completes, the session can
/// optionally run a one-shot decode for that completed window
/// (`StreamingConfig.finalizeCompletedWindows`) to improve accuracy, then resets
/// decode state for the next window.
public final class StreamingInferenceSession: @unchecked Sendable {
    private let core: any StreamingInferenceSessionCore

    public let events: AsyncStream<TranscriptionEvent>

    private init(core: any StreamingInferenceSessionCore) {
        self.core = core
        self.events = core.events
    }

    public convenience init(model: Qwen3ASRModel, config: StreamingConfig = StreamingConfig()) {
        self.init(core: QwenStreamingInferenceSessionCore(model: model, config: config))
    }

    public convenience init(model: any STTGenerationModel, config: StreamingConfig = StreamingConfig()) {
        if let qwenModel = model as? Qwen3ASRModel {
            self.init(model: qwenModel, config: config)
        } else if let cohereModel = model as? CohereTranscribeModel {
            self.init(core: CohereStreamingInferenceSessionCore(model: cohereModel, config: config))
        } else if let mossModel = model as? MossTranscribeDiarizeModel {
            self.init(core: MossStreamingInferenceSessionCore(model: mossModel, config: config))
        } else {
            preconditionFailure(
                "StreamingInferenceSession requires Qwen3ASRModel, CohereTranscribeModel, or MossTranscribeDiarizeModel"
            )
        }
    }

    public func feedAudio(samples: [Float]) {
        core.feedAudio(samples: samples)
    }

    public func stop() {
        core.stop()
    }

    public func cancel() {
        core.cancel()
    }
}

private final class MossStreamingInferenceSessionCore: @unchecked Sendable, StreamingInferenceSessionCore {
    private let model: MossTranscribeDiarizeModel
    private let config: StreamingConfig
    private let sampleRate: Int
    private let windowSamples: Int
    private let partialWindowSamples: Int
    private let minimumPartialSamples: Int
    private let shared = OSAllocatedUnfairLock(initialState: MossStreamingSharedState())
    private let audioState = OSAllocatedUnfairLock(initialState: MossAudioStreamingSharedState())
    private let sessionLock = OSAllocatedUnfairLock(initialState: 0)

    private var isActive: Bool = false
    private var continuation: AsyncStream<TranscriptionEvent>.Continuation?
    private var decodeTask: Task<Void, Never>?
    private var stopTask: Task<Void, Never>?

    let events: AsyncStream<TranscriptionEvent>

    init(model: MossTranscribeDiarizeModel, config: StreamingConfig = StreamingConfig()) {
        self.model = model
        self.config = config
        self.sampleRate = model.sampleRate
        let windowSeconds = max(3.0, min(6.0, Double(max(1, config.maxDecodeWindows)) * 4.0))
        self.windowSamples = max(model.sampleRate, Int(round(windowSeconds * Double(model.sampleRate))))
        self.minimumPartialSamples = max(model.sampleRate, Int(round(1.25 * Double(model.sampleRate))))
        self.partialWindowSamples = max(
            self.minimumPartialSamples,
            Int(round(min(windowSeconds, 2.5) * Double(model.sampleRate)))
        )

        var continuation: AsyncStream<TranscriptionEvent>.Continuation!
        self.events = AsyncStream { continuation = $0 }
        self.continuation = continuation
        self.isActive = true
    }

    func feedAudio(samples: [Float]) {
        let launch: MossDecodeLaunch? = sessionLock.withLock { _ in
            guard isActive else { return nil }

            let now = Date()
            audioState.withLock { state in
                state.pendingSamples.append(contentsOf: samples)
                state.totalSamplesFed += samples.count
            }

            return audioState.withLock { state -> MossDecodeLaunch? in
                guard !state.isDecoding else { return nil }

                if state.pendingSamples.count >= windowSamples {
                    let window = Array(state.pendingSamples.prefix(windowSamples))
                    let offsetSeconds = Double(state.pendingStartSample) / Double(sampleRate)
                    state.pendingSamples = Array(state.pendingSamples.dropFirst(windowSamples))
                    state.pendingStartSample += windowSamples
                    state.finalizedWindowCount += 1
                    state.isDecoding = true
                    state.lastDecodeTime = now
                    return MossDecodeLaunch(
                        samples: window,
                        kind: .finalWindow,
                        offsetSeconds: offsetSeconds,
                        totalSamples: state.totalSamplesFed,
                        encodedWindowCount: state.finalizedWindowCount
                    )
                }

                guard state.pendingSamples.count >= minimumPartialSamples else { return nil }
                if let lastDecode = state.lastDecodeTime,
                   now.timeIntervalSince(lastDecode) < max(1.0, config.decodeIntervalSeconds)
                {
                    return nil
                }

                state.isDecoding = true
                state.lastDecodeTime = now
                let pendingCount = state.pendingSamples.count
                let partialCount = min(pendingCount, partialWindowSamples)
                let partialStart = pendingCount - partialCount
                let partialSamples = Array(state.pendingSamples[partialStart..<pendingCount])
                return MossDecodeLaunch(
                    samples: partialSamples,
                    kind: .partial,
                    offsetSeconds: Double(state.pendingStartSample + partialStart) / Double(sampleRate),
                    totalSamples: state.totalSamplesFed,
                    encodedWindowCount: state.finalizedWindowCount
                )
            }
        }

        if let launch {
            launchDecodePass(launch)
        }
    }

    private func launchDecodePass(_ launch: MossDecodeLaunch) {
        let params = MossDecodePassParams(
            audio: UncheckedSendableBox(MLXArray(launch.samples)),
            model: UncheckedSendableBox(model),
            config: config,
            kind: launch.kind,
            offsetSeconds: launch.offsetSeconds,
            totalSamples: launch.totalSamples,
            encodedWindowCount: launch.encodedWindowCount
        )
        let continuation = self.continuation
        let sharedState = self.shared
        let audioState = self.audioState

        decodeTask = Task.detached {
            defer {
                audioState.withLock { $0.isDecoding = false }
            }

            Self.runDecodePass(
                params: params,
                continuation: continuation,
                sharedState: sharedState
            )
        }
    }

    private static func runDecodePass(
        params: MossDecodePassParams,
        continuation: AsyncStream<TranscriptionEvent>.Continuation?,
        sharedState: OSAllocatedUnfairLock<MossStreamingSharedState>
    ) {
        if Task.isCancelled { return }

        let decodeStart = Date()
        let output: STTOutput
        var liveText = ""
        let currentWindowSeconds = Double(params.audio.value.dim(0)) / Double(params.model.value.sampleRate)
        let maxTokens: Int?
        switch params.kind {
        case .partial:
            maxTokens = min(params.config.maxTokensPerPass, max(48, Int(ceil(currentWindowSeconds * 16.0))))
        case .finalWindow:
            maxTokens = nil
        }
        let onText: (String) -> Void = { delta in
            guard !Task.isCancelled, !delta.isEmpty else { return }
            liveText += delta
            let provisional = liveText.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !provisional.isEmpty else { return }
            let completed = sharedState.withLock { state -> String in
                state.provisionalText = provisional
                return state.completedText
            }
            yieldMossDisplayUpdate(
                completedText: completed,
                provisionalText: provisional,
                continuation: continuation
            )
        }

        do {
            output = try params.model.value.streamingTranscribeWindow(
                audio: params.audio.value,
                offsetSeconds: params.offsetSeconds,
                config: params.config,
                maxTokens: maxTokens,
                onText: onText
            )
        } catch {
            return
        }

        if Task.isCancelled { return }

        let text = output.text.trimmingCharacters(in: .whitespacesAndNewlines)
        switch params.kind {
        case .partial:
            let completed = sharedState.withLock { state -> String in
                state.provisionalText = text
                return state.completedText
            }
            yieldMossDisplayUpdate(
                completedText: completed,
                provisionalText: text,
                continuation: continuation
            )

        case .finalWindow:
            let completed = sharedState.withLock { state -> String in
                appendMossText(text, to: &state.completedText)
                state.provisionalText = ""
                return state.completedText
            }
            continuation?.yield(.displayUpdate(
                confirmedText: completed,
                provisionalText: ""
            ))
        }

        let decodeTime = Date().timeIntervalSince(decodeStart)
        let totalAudioSeconds = Double(params.totalSamples) / Double(params.model.value.sampleRate)
        continuation?.yield(.stats(StreamingStats(
            encodedWindowCount: max(params.encodedWindowCount, Int(ceil(totalAudioSeconds / max(currentWindowSeconds, 0.001)))),
            totalAudioSeconds: totalAudioSeconds,
            tokensPerSecond: decodeTime > 0 ? Double(output.generationTokens) / decodeTime : 0,
            realTimeFactor: decodeTime / max(currentWindowSeconds, 0.001),
            peakMemoryGB: output.peakMemoryUsage
        )))
    }

    private static func yieldMossDisplayUpdate(
        completedText: String,
        provisionalText: String,
        continuation: AsyncStream<TranscriptionEvent>.Continuation?
    ) {
        let confirmedText = !completedText.isEmpty
            && !provisionalText.isEmpty
            && !provisionalText.hasPrefix("\n")
            ? completedText + "\n"
            : completedText
        continuation?.yield(.displayUpdate(
            confirmedText: confirmedText,
            provisionalText: provisionalText
        ))
    }

    private static func appendMossText(_ segment: String, to base: inout String) {
        let normalizedSegment = segment.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalizedSegment.isEmpty else { return }
        if base.isEmpty {
            base = normalizedSegment
        } else {
            base += "\n" + normalizedSegment
        }
    }

    func stop() {
        let stopContext: (
            previousStopTask: Task<Void, Never>?,
            inFlightDecode: Task<Void, Never>?,
            pendingSamples: [Float],
            pendingStartSample: Int,
            totalSamples: Int,
            encodedWindowCount: Int
        )? =
            sessionLock.withLock { _ in
                guard isActive else { return nil }
                isActive = false

                let inFlightDecode = decodeTask
                decodeTask = nil

                let snapshot = audioState.withLock { state in
                    (
                        state.pendingSamples,
                        state.pendingStartSample,
                        state.totalSamplesFed,
                        state.finalizedWindowCount
                    )
                }

                let previousStopTask = stopTask
                stopTask = nil
                return (previousStopTask, inFlightDecode, snapshot.0, snapshot.1, snapshot.2, snapshot.3)
            }

        guard let stopContext else { return }
        stopContext.previousStopTask?.cancel()

        let task = Task.detached {
            [self,
             inFlightDecode = stopContext.inFlightDecode,
             pendingSamples = stopContext.pendingSamples,
             pendingStartSample = stopContext.pendingStartSample,
             totalSamples = stopContext.totalSamples,
             encodedWindowCount = stopContext.encodedWindowCount] in
            await finishStop(
                waitingFor: inFlightDecode,
                pendingSamples: pendingSamples,
                pendingStartSample: pendingStartSample,
                totalSamples: totalSamples,
                encodedWindowCount: encodedWindowCount
            )
        }

        sessionLock.withLock { _ in
            if continuation != nil {
                stopTask = task
            }
        }
    }

    private func finishStop(
        waitingFor inFlightDecode: Task<Void, Never>?,
        pendingSamples: [Float],
        pendingStartSample: Int,
        totalSamples: Int,
        encodedWindowCount: Int
    ) async {
        if let inFlightDecode {
            _ = await inFlightDecode.value
        }

        if Task.isCancelled {
            return
        }

        if !pendingSamples.isEmpty {
            let params = MossDecodePassParams(
                audio: UncheckedSendableBox(MLXArray(pendingSamples)),
                model: UncheckedSendableBox(model),
                config: config,
                kind: .finalWindow,
                offsetSeconds: Double(pendingStartSample) / Double(sampleRate),
                totalSamples: totalSamples,
                encodedWindowCount: encodedWindowCount + 1
            )
            Self.runDecodePass(
                params: params,
                continuation: continuation,
                sharedState: shared
            )
        }

        let finalText = shared.withLock { state in
            if state.completedText.isEmpty {
                return state.provisionalText
            }
            return state.completedText
        }

        if Task.isCancelled {
            return
        }

        continuation?.yield(.ended(fullText: finalText))
        continuation?.finish()

        sessionLock.withLock { _ in
            continuation = nil
            stopTask = nil
            audioState.withLock { state in
                state.pendingSamples = []
                state.pendingStartSample = totalSamples
                state.isDecoding = false
            }
        }

        Memory.clearCache()
    }

    func cancel() {
        sessionLock.withLock { _ in
            isActive = false
            decodeTask?.cancel()
            decodeTask = nil
            stopTask?.cancel()
            stopTask = nil
            continuation?.finish()
            continuation = nil
            audioState.withLock { state in
                state.pendingSamples = []
                state.isDecoding = false
            }
        }
    }
}

private final class CohereStreamingInferenceSessionCore: @unchecked Sendable, StreamingInferenceSessionCore {
    private let model: CohereTranscribeModel
    private let config: StreamingConfig
    private let sampleRate: Int
    private let windowSamples: Int
    private let overlapSamples: Int
    private let shared = OSAllocatedUnfairLock(initialState: SessionSharedState())
    private let audioState = OSAllocatedUnfairLock(initialState: CohereStreamingSharedState())
    private let sessionLock = OSAllocatedUnfairLock(initialState: 0)

    private var isActive: Bool = false
    private var continuation: AsyncStream<TranscriptionEvent>.Continuation?
    private var decodeTask: Task<Void, Never>?
    private var stopTask: Task<Void, Never>?

    let events: AsyncStream<TranscriptionEvent>

    init(model: CohereTranscribeModel, config: StreamingConfig = StreamingConfig()) {
        self.model = model
        self.config = config
        self.sampleRate = model.config.sampleRate
        let windowSamples = model.config.sampleRate * 8
        self.windowSamples = windowSamples
        self.overlapSamples = max(
            0,
            min(
                Int(round(config.encoderWindowOverlapSeconds * Double(model.config.sampleRate))),
                max(0, windowSamples - 1)
            )
        )

        var continuation: AsyncStream<TranscriptionEvent>.Continuation!
        self.events = AsyncStream { continuation = $0 }
        self.continuation = continuation
        self.isActive = true
    }

    func feedAudio(samples: [Float]) {
        let launch: CohereDecodeLaunch? = sessionLock.withLock { _ in
            guard isActive else { return nil }

            let now = Date()
            audioState.withLock { state in
                state.pendingSamples.append(contentsOf: samples)
                state.totalSamplesFed += samples.count
            }

            return audioState.withLock { state -> CohereDecodeLaunch? in
                guard !state.isDecoding else { return nil }

                if state.pendingSamples.count >= windowSamples {
                    let window = Array(state.pendingSamples.prefix(windowSamples))
                    let keepStart = max(0, windowSamples - overlapSamples)
                    state.pendingSamples = Array(state.pendingSamples[keepStart...])
                    state.finalizedWindowCount += 1
                    state.isDecoding = true
                    state.lastDecodeTime = now
                    return CohereDecodeLaunch(
                        samples: window,
                        kind: .finalWindow,
                        totalSamples: state.totalSamplesFed,
                        encodedWindowCount: state.finalizedWindowCount
                    )
                }

                guard state.pendingSamples.count >= sampleRate / 2 else { return nil }
                if let lastDecode = state.lastDecodeTime,
                   now.timeIntervalSince(lastDecode) < max(0.2, config.decodeIntervalSeconds)
                {
                    return nil
                }

                state.isDecoding = true
                state.lastDecodeTime = now
                return CohereDecodeLaunch(
                    samples: state.pendingSamples,
                    kind: .partial,
                    totalSamples: state.totalSamplesFed,
                    encodedWindowCount: state.finalizedWindowCount
                )
            }
        }

        if let launch {
            launchDecodePass(launch)
        }
    }

    private func launchDecodePass(_ launch: CohereDecodeLaunch) {
        let snapshot = shared.withLock { state -> ([Int], String, [Int], [Date], [Int]) in
            let prefix = QwenStreamingInferenceSessionCore.concatText(state.completedText, state.confirmedText)
            return (state.confirmedTokenIds,
                    prefix,
                    state.provisionalTokenIds,
                    state.provisionalFirstSeen,
                    state.provisionalAgreementCounts)
        }
        let (confirmedTokenIds, displayPrefix, prevProvisional, prevFirstSeen, prevAgreementCounts) = snapshot
        let params = CohereDecodePassParams(
            audio: UncheckedSendableBox(MLXArray(launch.samples)),
            model: UncheckedSendableBox(model),
            config: config,
            kind: launch.kind,
            confirmedTokenIds: launch.kind == .partial ? confirmedTokenIds : [],
            displayPrefix: launch.kind == .partial ? displayPrefix : "",
            prevProvisional: launch.kind == .partial ? prevProvisional : [],
            prevFirstSeen: launch.kind == .partial ? prevFirstSeen : [],
            prevAgreementCounts: launch.kind == .partial ? prevAgreementCounts : [],
            minAgreementPasses: launch.kind == .partial ? max(1, config.minAgreementPasses) : 1,
            totalSamples: launch.totalSamples
        )
        let continuation = self.continuation
        let sharedState = self.shared
        let audioState = self.audioState
        let encodedWindowCount = launch.encodedWindowCount

        decodeTask = Task.detached {
            defer {
                sharedState.withLock { $0.isDecoding = false }
                audioState.withLock { $0.isDecoding = false }
            }

            Self.runDecodePass(
                params: params,
                continuation: continuation,
                sharedState: sharedState,
                encodedWindowCount: encodedWindowCount
            )
        }
    }

    private static func runDecodePass(
        params: CohereDecodePassParams,
        continuation: AsyncStream<TranscriptionEvent>.Continuation?,
        sharedState: OSAllocatedUnfairLock<SessionSharedState>,
        encodedWindowCount: Int
    ) {
        if Task.isCancelled { return }

        let result = params.model.value.streamingDecodeTokenIds(
            audio: params.audio.value,
            config: params.config,
            confirmedTokenIds: params.confirmedTokenIds
        )

        if Task.isCancelled { return }

        switch params.kind {
        case .partial:
            promoteTokens(
                allTokenIds: result.tokenIds,
                params: params,
                continuation: continuation,
                sharedState: sharedState
            )
        case .finalWindow:
            finalizeWindow(
                tokenIds: result.tokenIds,
                model: params.model.value,
                continuation: continuation,
                sharedState: sharedState
            )
        }

        let totalAudioSeconds = Double(params.totalSamples) / Double(params.model.value.config.sampleRate)
        let tokenCount = max(0, result.tokenIds.count - params.confirmedTokenIds.count)
        continuation?.yield(.stats(StreamingStats(
            encodedWindowCount: max(encodedWindowCount, Int(ceil(totalAudioSeconds / 8.0))),
            totalAudioSeconds: totalAudioSeconds,
            tokensPerSecond: result.decodeTime > 0 ? Double(tokenCount) / result.decodeTime : 0,
            realTimeFactor: result.totalTime / max(totalAudioSeconds, 0.001),
            peakMemoryGB: result.peakMemoryUsage
        )))
    }

    private static func finalizeWindow(
        tokenIds: [Int],
        model: CohereTranscribeModel,
        continuation: AsyncStream<TranscriptionEvent>.Continuation?,
        sharedState: OSAllocatedUnfairLock<SessionSharedState>
    ) {
        let windowText = model.streamingDecodeText(tokens: tokenIds)
        let completedText = sharedState.withLock { state in
            state.completedText = QwenStreamingInferenceSessionCore.concatText(state.completedText, windowText)
            state.confirmedTokenIds = []
            state.provisionalTokenIds = []
            state.provisionalFirstSeen = []
            state.provisionalAgreementCounts = []
            state.confirmedText = ""
            return state.completedText
        }

        continuation?.yield(.displayUpdate(
            confirmedText: completedText,
            provisionalText: ""
        ))
    }

    private static func promoteTokens(
        allTokenIds: [Int],
        params: CohereDecodePassParams,
        continuation: AsyncStream<TranscriptionEvent>.Continuation?,
        sharedState: OSAllocatedUnfairLock<SessionSharedState>
    ) {
        let confirmedCount = params.confirmedTokenIds.count
        let prevProvisional = params.prevProvisional
        let prevFirstSeen = params.prevFirstSeen
        let prevAgreementCounts = params.prevAgreementCounts

        let newProvisional = Array(allTokenIds.dropFirst(min(confirmedCount, allTokenIds.count)))
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

        var promotionCount = 0
        for i in 0..<newProvisional.count {
            let hasDelay = i < nextFirstSeen.count && now.timeIntervalSince(nextFirstSeen[i]) >= delaySeconds
            let hasAgreement = i < nextAgreementCounts.count && nextAgreementCounts[i] >= params.minAgreementPasses
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
                state.confirmedText = params.model.value.streamingDecodeText(tokens: state.confirmedTokenIds)
                continuation?.yield(.confirmed(text: QwenStreamingInferenceSessionCore.concatText(
                    state.completedText,
                    state.confirmedText
                )))
            }
            state.provisionalTokenIds = finalProvisional
            state.provisionalFirstSeen = finalFirstSeen
            state.provisionalAgreementCounts = finalAgreementCounts
            return QwenStreamingInferenceSessionCore.concatText(state.completedText, state.confirmedText)
        }

        let finalProvText = params.model.value.streamingDecodeText(tokens: finalProvisional)
        continuation?.yield(.displayUpdate(
            confirmedText: displayPrefix,
            provisionalText: finalProvText
        ))
    }

    func stop() {
        let stopContext: (
            previousStopTask: Task<Void, Never>?,
            inFlightDecode: Task<Void, Never>?,
            pendingSamples: [Float],
            totalSamples: Int,
            encodedWindowCount: Int
        )? =
            sessionLock.withLock { _ in
                guard isActive else { return nil }
                isActive = false

                let inFlightDecode = decodeTask
                decodeTask = nil

                let snapshot = audioState.withLock { state in
                    (
                        state.pendingSamples,
                        state.totalSamplesFed,
                        state.finalizedWindowCount
                    )
                }

                let previousStopTask = stopTask
                stopTask = nil
                return (previousStopTask, inFlightDecode, snapshot.0, snapshot.1, snapshot.2)
            }

        guard let stopContext else { return }
        stopContext.previousStopTask?.cancel()

        let task = Task.detached {
            [self,
             inFlightDecode = stopContext.inFlightDecode,
             pendingSamples = stopContext.pendingSamples,
             totalSamples = stopContext.totalSamples,
             encodedWindowCount = stopContext.encodedWindowCount] in
            await finishStop(
                waitingFor: inFlightDecode,
                pendingSamples: pendingSamples,
                totalSamples: totalSamples,
                encodedWindowCount: encodedWindowCount
            )
        }

        sessionLock.withLock { _ in
            if continuation != nil {
                stopTask = task
            }
        }
    }

    private func finishStop(
        waitingFor inFlightDecode: Task<Void, Never>?,
        pendingSamples: [Float],
        totalSamples: Int,
        encodedWindowCount: Int
    ) async {
        if let inFlightDecode {
            _ = await inFlightDecode.value
        }

        if Task.isCancelled {
            return
        }

        if !pendingSamples.isEmpty {
            let params = CohereDecodePassParams(
                audio: UncheckedSendableBox(MLXArray(pendingSamples)),
                model: UncheckedSendableBox(model),
                config: config,
                kind: .finalWindow,
                confirmedTokenIds: [],
                displayPrefix: "",
                prevProvisional: [],
                prevFirstSeen: [],
                prevAgreementCounts: [],
                minAgreementPasses: 1,
                totalSamples: totalSamples
            )
            Self.runDecodePass(
                params: params,
                continuation: continuation,
                sharedState: shared,
                encodedWindowCount: encodedWindowCount + 1
            )
        }

        let finalText = shared.withLock { state -> String in
            if !state.provisionalTokenIds.isEmpty {
                state.confirmedTokenIds.append(contentsOf: state.provisionalTokenIds)
                state.provisionalTokenIds = []
                state.provisionalFirstSeen = []
                state.provisionalAgreementCounts = []
            }
            state.confirmedText = model.streamingDecodeText(tokens: state.confirmedTokenIds)
            return QwenStreamingInferenceSessionCore.concatText(state.completedText, state.confirmedText)
        }

        if Task.isCancelled {
            return
        }

        continuation?.yield(.ended(fullText: finalText))
        continuation?.finish()

        sessionLock.withLock { _ in
            continuation = nil
            stopTask = nil
            audioState.withLock { state in
                state.pendingSamples = []
                state.isDecoding = false
            }
        }

        Memory.clearCache()
    }

    func cancel() {
        sessionLock.withLock { _ in
            isActive = false
            decodeTask?.cancel()
            decodeTask = nil
            stopTask?.cancel()
            stopTask = nil
            continuation?.finish()
            continuation = nil
            audioState.withLock { state in
                state.pendingSamples = []
                state.isDecoding = false
            }
        }
    }
}

private final class QwenStreamingInferenceSessionCore: @unchecked Sendable, StreamingInferenceSessionCore {
    private let model: Qwen3ASRModel
    private let config: StreamingConfig
    private let melProcessor: IncrementalMelSpectrogram
    private let encoder: StreamingEncoder

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

    let events: AsyncStream<TranscriptionEvent>

    init(model: Qwen3ASRModel, config: StreamingConfig = StreamingConfig()) {
        self.model = model
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

    func feedAudio(samples: [Float]) {
        sessionLock.withLock { _ in
            guard isActive else { return }

            totalSamplesFed += samples.count

            guard let melFrames = melProcessor.process(samples: samples) else { return }

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

    // MARK: - Window Completion

    /// When the encoder completes a full window, freeze the current streaming
    /// text and reset decode state — next decode starts fresh on new pending.
    private func freezeCompletedWindowsLocked() {
        let currentWindowCount = encoder.encodedWindowCount
        guard currentWindowCount > frozenWindowCount else { return }

        shared.withLock { state in
            // Promote provisional and freeze everything
            var allTokens = state.confirmedTokenIds
            allTokens.append(contentsOf: state.provisionalTokenIds)
            if let tokenizer = model.tokenizer, !allTokens.isEmpty {
                let windowText = tokenizer.decode(tokens: allTokens)
                Self.appendText(windowText, to: &state.completedText)
            }
            // Reset — next decode is a fresh start on new pending frames
            state.confirmedTokenIds = []
            state.provisionalTokenIds = []
            state.provisionalFirstSeen = []
            state.provisionalAgreementCounts = []
            state.confirmedText = ""
        }

        frozenWindowCount = currentWindowCount
    }

    private func launchDecodePassLocked() {
        if config.finalizeCompletedWindows {
            let windowsToFinalize = encoder.drainNewlyEncodedWindows()
            if !windowsToFinalize.isEmpty {
                frozenWindowCount = encoder.encodedWindowCount

                let params = FinalizeWindowsParams(
                    windows: UncheckedSendableBox(windowsToFinalize),
                    model: UncheckedSendableBox(self.model),
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

        // Only decode the current pending (partial) window
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
            model: UncheckedSendableBox(self.model),
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

    fileprivate static func concatText(_ a: String, _ b: String) -> String {
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
        tokenizer: any Tokenizers.Tokenizer,
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

    public func stop() {
        sessionLock.withLock { _ in
            guard isActive else { return }
            isActive = false

            let inFlightDecode = decodeTask
            decodeTask = nil

            stopTask?.cancel()
            stopTask = Task.detached { [self] in
                await finishStop(waitingFor: inFlightDecode)
            }
        }
    }

    private func finishStop(waitingFor inFlightDecode: Task<Void, Never>?) async {
        if let inFlightDecode {
            _ = await inFlightDecode.value
        }

        if Task.isCancelled {
            return
        }

        let snapshot: StopSnapshot = sessionLock.withLock { _ in
            if let melFrames = melProcessor.flush() {
                _ = encoder.feed(melFrames: melFrames)
            }

            let completedWindows: [MLXArray]
            if config.finalizeCompletedWindows {
                completedWindows = encoder.drainNewlyEncodedWindows()
                frozenWindowCount = encoder.encodedWindowCount
            } else {
                completedWindows = []
                // Freeze any windows completed since last decode
                freezeCompletedWindowsLocked()
            }

            let continuation = self.continuation
            let totalSamples = totalSamplesFed
            let encodedWindowCount = encoder.encodedWindowCount

            if let audioFeatures = encoder.encodePending(),
               audioFeatures.dim(0) > 0,
               model.tokenizer != nil
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
                if let tokenizer = model.tokenizer, !state.confirmedTokenIds.isEmpty {
                    state.confirmedText = tokenizer.decode(tokens: state.confirmedTokenIds)
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
           let tokenizer = model.tokenizer
        {
            for audioFeatures in completedWindows {
                if Task.isCancelled { return }

                if audioFeatures.dim(0) <= 0 { continue }
                let tokenIds = Self.decodeAllTokenIds(
                    model: model,
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
           let tokenizer = model.tokenizer
        {
            let startTime = Date()
            let tokenIds = Self.decodeAllTokenIds(
                model: model,
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
            finalText = shared.withLock { state in
                if !state.provisionalTokenIds.isEmpty {
                    state.confirmedTokenIds.append(contentsOf: state.provisionalTokenIds)
                    state.provisionalTokenIds = []
                    state.provisionalFirstSeen = []
                    state.provisionalAgreementCounts = []
                }
                if let tokenizer = model.tokenizer, !state.confirmedTokenIds.isEmpty {
                    state.confirmedText = tokenizer.decode(tokens: state.confirmedTokenIds)
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
            encoder.reset()
            melProcessor.reset()
            boundaryFastDecodeUntil = nil
        }

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
            encoder.reset()
            melProcessor.reset()
            boundaryFastDecodeUntil = nil
        }
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
