import Foundation
import MLX
import os

public final class WhisperSession: @unchecked Sendable {
    /// Loading state for background model initialization
    /// Note: @unchecked Sendable because Error is not Sendable, but we ensure thread-safe access via OSAllocatedUnfairLock
    public enum LoadingState: @unchecked Sendable {
        case loading
        case ready
        case failed(Error)
        case cancelled
    }

    private let modelType: WhisperModel
    private let streamingConfig: StreamingConfig
    private var currentTask: Task<Void, Never>?
    private let taskLock = NSLock()

    private let encoder: AudioEncoder
    private let decoder: TextDecoder
    private let tokenizer: WhisperTokenizer
    private let config: WhisperConfiguration
    private let alignmentHeads: [(layer: Int, head: Int)]

    private let _state = OSAllocatedUnfairLock(initialState: LoadingState.loading)
    private let readyStream: AsyncStream<Void>
    private let readyContinuation: AsyncStream<Void>.Continuation
    private var backgroundTask: Task<Void, Never>?

    /// Preallocated KV caches for compiled decoder (one per layer)
    private let kvCaches: [KVCache]

    /// Compiled encoder function for graph caching
    private let compiledEncode: @Sendable ([MLXArray]) -> [MLXArray]

    /// Compiled decoder function for graph caching
    private let compiledDecode: @Sendable ([MLXArray]) -> [MLXArray]

    public let actualQuantization: WhisperQuantization
    public let didFallback: Bool

    public var state: LoadingState { _state.withLock { $0 } }
    public var isReady: Bool {
        if case .ready = state { return true }
        return false
    }

    private init(
        modelType: WhisperModel,
        streamingConfig: StreamingConfig,
        encoder: AudioEncoder,
        decoder: TextDecoder,
        tokenizer: WhisperTokenizer,
        config: WhisperConfiguration,
        actualQuantization: WhisperQuantization,
        didFallback: Bool
    ) {
        self.modelType = modelType
        self.streamingConfig = streamingConfig
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.config = config
        self.alignmentHeads = WhisperAlignmentHeads.heads(for: modelType)
        self.actualQuantization = actualQuantization
        self.didFallback = didFallback

        // Preallocate KV caches with fixed shapes for compile-friendly decoding
        self.kvCaches = (0..<config.nTextLayer).map { _ in
            KVCache(
                batchSize: 1,
                maxSequenceLength: config.nTextCtx,
                dim: config.nTextState
            )
        }

        // Compile encoder: [mel] -> [encoderOutput]
        // Encoder has fixed input shape (30s audio = 3000 mel frames) and no state
        self.compiledEncode = compile(inputs: [encoder]) { [encoder] inputs in
            let mel = inputs[0]
            return [encoder(mel)]
        }

        // Decoder: Use direct call (KVCache slice operations don't work with compile)
        // The preallocated KVCache pattern still helps avoid shape-change overhead
        let kvCachesRef = self.kvCaches
        self.compiledDecode = { [decoder] inputs in
            let tokens = inputs[0]
            let encoderOutput = inputs[1]
            let (logits, crossQK) = decoder(
                tokens: tokens,
                encoderOutput: encoderOutput,
                kvCache: kvCachesRef
            )
            return [logits] + crossQK
        }

        var continuation: AsyncStream<Void>.Continuation!
        self.readyStream = AsyncStream { continuation = $0 }
        self.readyContinuation = continuation
    }

    deinit {
        backgroundTask?.cancel()
    }

    /// Warmup encoder compilation and decoder path
    /// Called at init to avoid cold-start latency on first transcription
    private func warmup() {
        // Reset caches before warmup
        for cache in kvCaches {
            cache.reset()
        }

        // Warmup compiled encoder with standard 30s mel shape
        let dummyMel = MLXArray.zeros([1, config.nMels, 3000])
        let encoderOut = compiledEncode([dummyMel])[0]

        // Force encoder compilation to complete
        eval(encoderOut)

        // Reset caches for actual use
        for cache in kvCaches {
            cache.reset()
        }
    }

    public func waitUntilReady(timeout: Duration = .seconds(30)) async throws -> Bool {
        try Task.checkCancellation()

        switch state {
        case .ready: return true
        case .failed(let error): throw error
        case .cancelled: throw WhisperError.cancelled
        case .loading: break
        }

        let didComplete = await withTaskGroup(of: Bool.self) { group in
            group.addTask {
                for await _ in self.readyStream { break }
                return true
            }
            group.addTask {
                try? await Task.sleep(for: timeout)
                return false
            }
            let result = await group.next() ?? false
            group.cancelAll()
            return result
        }

        switch state {
        case .ready: return true
        case .failed(let error): throw error
        case .cancelled: throw WhisperError.cancelled
        case .loading: return didComplete
        }
    }

    // MARK: - Factory

    public static func fromPretrained(
        model: WhisperModel = .largeTurbo,
        streaming: StreamingConfig = .default,
        progressHandler: ((WhisperProgress) -> Void)? = nil
    ) async throws -> WhisperSession {
        progressHandler?(.downloading(0))

        let loaded = try await WhisperModelLoader.load(
            model: model,
            progressHandler: { progress in
                let fraction = Float(progress.fractionCompleted)
                progressHandler?(.downloading(fraction * 0.8))
            }
        )

        progressHandler?(.loading(0.9))

        let tokenizer = try await WhisperTokenizer(modelFolder: loaded.tokenizerDirectory)

        progressHandler?(.loading(1.0))

        let session = WhisperSession(
            modelType: model,
            streamingConfig: streaming,
            encoder: loaded.encoder,
            decoder: loaded.decoder,
            tokenizer: tokenizer,
            config: loaded.config,
            actualQuantization: .float16,
            didFallback: false
        )
        // Materialize model weights before warmup
        eval(loaded.encoder, loaded.decoder)
        // Warmup compiled functions to populate compilation cache
        session.warmup()
        session._state.withLock { $0 = .ready }
        return session
    }

    public static func fromPretrained(
        model: WhisperModel = .largeTurbo,
        options: ModelLoadingOptions,
        streaming: StreamingConfig = .default,
        progressHandler: ((WhisperProgress) -> Void)? = nil
    ) async throws -> WhisperSession {
        progressHandler?(.downloading(0))

        let loadResult = try await WhisperModelLoader.load(
            model: model,
            quantization: options.quantization,
            fallbackToFloat16: options.fallbackToFloat16,
            progressHandler: { progress in
                let fraction = Float(progress.fractionCompleted)
                progressHandler?(.downloading(fraction * 0.8))
            }
        )

        progressHandler?(.loading(0.9))

        let tokenizer = try await WhisperTokenizer(modelFolder: loadResult.model.tokenizerDirectory)

        progressHandler?(.loading(1.0))

        let session = WhisperSession(
            modelType: model,
            streamingConfig: streaming,
            encoder: loadResult.model.encoder,
            decoder: loadResult.model.decoder,
            tokenizer: tokenizer,
            config: loadResult.model.config,
            actualQuantization: loadResult.actualQuantization,
            didFallback: loadResult.didFallback
        )

        if options.loadInBackground {
            session.backgroundTask = Task.detached { [weak session] in
                guard let session = session else { return }
                do {
                    try Task.checkCancellation()
                    eval(loadResult.model.encoder, loadResult.model.decoder)
                    session.warmup()
                    session._state.withLock { $0 = .ready }
                } catch is CancellationError {
                    session._state.withLock { $0 = .cancelled }
                } catch {
                    session._state.withLock { $0 = .failed(error) }
                }
                session.readyContinuation.finish()
            }
        } else {
            eval(loadResult.model.encoder, loadResult.model.decoder)
            session.warmup()
            session._state.withLock { $0 = .ready }
            session.readyContinuation.finish()
        }

        return session
    }

    // MARK: - Transcription (Streaming)

    public func transcribe(
        _ audio: MLXArray,
        sampleRate: Int = AudioConstants.sampleRate,
        options: TranscriptionOptions = .default
    ) -> AsyncThrowingStream<StreamingResult, Error> {
        AsyncThrowingStream { continuation in
            continuation.onTermination = { [weak self] _ in
                self?.taskLock.withLock {
                    self?.currentTask?.cancel()
                    self?.currentTask = nil
                }
            }

            self.taskLock.withLock {
                self.currentTask = Task {
                    do {
                        guard sampleRate == AudioConstants.sampleRate else {
                            throw WhisperError.sampleRateMismatch(
                                expected: AudioConstants.sampleRate,
                                got: sampleRate
                            )
                        }

                        try Task.checkCancellation()

                        // Calculate audio duration for timestamps
                        let audioDuration = Double(audio.shape[0]) / Double(sampleRate)

                        // 1. Pad/trim audio to 30 seconds
                        let paddedAudio = AudioUtils.padOrTrim(audio, length: AudioConstants.nSamples)

                        // 2. Compute mel spectrogram
                        let mel = try MelSpectrogram.compute(audio: paddedAudio, nMels: self.config.nMels)
                        let melBatched = mel.expandedDimensions(axis: 0)

                        // 3. Encode audio (COMPILED)
                        let encoderOutput = self.compiledEncode([melBatched])[0]
                        let totalFrames = encoderOutput.shape[1]

                        // 4. Initialize decoder state
                        var tokens = self.tokenizer.initialTokens(
                            language: options.language,
                            task: options.task
                        )

                        // Reset preallocated KV caches for this transcription
                        for cache in self.kvCaches {
                            cache.reset()
                        }
                        var emittedText = ""
                        var lastEmittedIndex = tokens.count
                        var didEmitFinal = false

                        // 5. Decoding loop
                        let maxTokens = self.config.nTextCtx - tokens.count

                        for step in 0..<maxTokens {
                            try Task.checkCancellation()

                            // Create token array for decoder
                            let tokenArray: MLXArray
                            if step == 0 {
                                tokenArray = MLXArray(tokens).expandedDimensions(axis: 0)
                            } else {
                                tokenArray = MLXArray([tokens.last!]).expandedDimensions(axis: 0)
                            }

                            // Decode one step (COMPILED)
                            let outputs = self.compiledDecode([tokenArray, encoderOutput])
                            let logits = outputs[0]
                            let crossQKArrays = Array(outputs.dropFirst())

                            // Convert [MLXArray] to [MLXArray?] for StreamingDecoder
                            let crossQK: [MLXArray?] = crossQKArrays.map { $0 }

                            // Sample next token (greedy)
                            let nextToken = Int(MLX.argMax(logits[0, -1]).item(Int.self))

                            // Check for end of transcription
                            if nextToken == WhisperTokenizer.eotToken {
                                let finalTokens = Array(tokens.dropFirst(lastEmittedIndex))
                                let finalText = self.tokenizer.decode(finalTokens)
                                let fullText = emittedText + finalText
                                if !fullText.isEmpty {
                                    continuation.yield(StreamingResult(
                                        text: fullText.trimmingCharacters(in: .whitespaces),
                                        isFinal: true,
                                        timestamp: 0...audioDuration
                                    ))
                                }
                                didEmitFinal = true
                                break
                            }

                            tokens.append(nextToken)

                            // AlignAtt: Check if we should emit
                            let mostAttendedFrame = StreamingDecoder.getMostAttendedFrame(
                                crossQK: crossQK,
                                alignmentHeads: self.alignmentHeads
                            )

                            let shouldEmit = StreamingDecoder.shouldEmit(
                                mostAttendedFrame: mostAttendedFrame,
                                totalContentFrames: totalFrames,
                                threshold: self.streamingConfig.frameThreshold
                            )

                            if shouldEmit && self.streamingConfig.emitPartial {
                                let newTokens = Array(tokens[lastEmittedIndex...])
                                let newText = self.tokenizer.decode(newTokens)

                                if !newText.isEmpty {
                                    emittedText += newText
                                    lastEmittedIndex = tokens.count

                                    // Convert frame index to seconds using actual audio duration and encoder frame count
                                    let frameTime = audioDuration * Double(mostAttendedFrame) / Double(totalFrames)

                                    continuation.yield(StreamingResult(
                                        text: emittedText.trimmingCharacters(in: .whitespaces),
                                        isFinal: false,
                                        timestamp: 0...frameTime
                                    ))
                                }
                            }
                        }

                        // Emit final result if loop ended without EOT (e.g., maxTokens reached)
                        if !didEmitFinal {
                            let remainingTokens = Array(tokens[lastEmittedIndex...])
                            if !remainingTokens.isEmpty {
                                let remainingText = self.tokenizer.decode(remainingTokens)
                                let fullText = emittedText + remainingText
                                if !fullText.isEmpty {
                                    continuation.yield(StreamingResult(
                                        text: fullText.trimmingCharacters(in: .whitespaces),
                                        isFinal: true,
                                        timestamp: 0...audioDuration
                                    ))
                                }
                            } else if !emittedText.isEmpty {
                                // All tokens already emitted, just mark as final
                                continuation.yield(StreamingResult(
                                    text: emittedText.trimmingCharacters(in: .whitespaces),
                                    isFinal: true,
                                    timestamp: 0...audioDuration
                                ))
                            }
                        }

                        self.taskLock.withLock {
                            self.currentTask = nil
                        }
                        continuation.finish()
                    } catch is CancellationError {
                        self.taskLock.withLock {
                            self.currentTask = nil
                        }
                        continuation.finish(throwing: WhisperError.cancelled)
                    } catch {
                        self.taskLock.withLock {
                            self.currentTask = nil
                        }
                        continuation.finish(throwing: error)
                    }
                }
            }
        }
    }

    // MARK: - Transcription (Blocking)

    public func transcribe(
        _ audio: MLXArray,
        sampleRate: Int = AudioConstants.sampleRate,
        options: TranscriptionOptions = .default
    ) async throws -> String {
        var result = ""
        for try await partial in transcribe(audio, sampleRate: sampleRate, options: options) {
            if partial.isFinal {
                result = partial.text
            }
        }
        return result
    }

    // MARK: - Control

    public func cancel() {
        taskLock.withLock {
            currentTask?.cancel()
            currentTask = nil
        }
    }
}

// MARK: - STTSession Conformance

extension WhisperSession: STTSession {
    public func transcribe(_ audio: MLXArray, sampleRate: Int) -> AsyncThrowingStream<StreamingResult, Error> {
        transcribe(audio, sampleRate: sampleRate, options: .default)
    }

    public func transcribe(_ audio: MLXArray, sampleRate: Int) async throws -> String {
        try await transcribe(audio, sampleRate: sampleRate, options: .default)
    }
}
