import Foundation
import MLX

public final class WhisperSession: @unchecked Sendable {
    private let modelType: WhisperModel
    private let streamingConfig: StreamingConfig
    private var currentTask: Task<Void, Never>?
    private let taskLock = NSLock()

    private let encoder: AudioEncoder
    private let decoder: TextDecoder
    private let tokenizer: WhisperTokenizer
    private let config: WhisperConfiguration
    private let alignmentHeads: [(layer: Int, head: Int)]

    private init(
        modelType: WhisperModel,
        streamingConfig: StreamingConfig,
        encoder: AudioEncoder,
        decoder: TextDecoder,
        tokenizer: WhisperTokenizer,
        config: WhisperConfiguration
    ) {
        self.modelType = modelType
        self.streamingConfig = streamingConfig
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.config = config
        self.alignmentHeads = WhisperAlignmentHeads.heads(for: modelType)
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

        // Load tokenizer from OpenAI repo (mlx-community repos don't have tokenizer files)
        let tokenizer = try await WhisperTokenizer(pretrained: WhisperModelLoader.tokenizerRepoId(for: model))

        progressHandler?(.loading(1.0))

        return WhisperSession(
            modelType: model,
            streamingConfig: streaming,
            encoder: loaded.encoder,
            decoder: loaded.decoder,
            tokenizer: tokenizer,
            config: loaded.config
        )
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

                        // 1. Pad/trim audio to 30 seconds
                        let paddedAudio = AudioUtils.padOrTrim(audio, length: AudioConstants.nSamples)

                        // 2. Compute mel spectrogram
                        let mel = try MelSpectrogram.compute(audio: paddedAudio, nMels: self.config.nMels)
                        // Add batch dimension: [nMels, nFrames] -> [1, nMels, nFrames]
                        let melBatched = mel.expandedDimensions(axis: 0)

                        // 3. Encode audio
                        let encoderOutput = self.encoder(melBatched)
                        let totalFrames = encoderOutput.shape[1]

                        // 4. Initialize decoder state
                        var tokens = self.tokenizer.initialTokens(
                            language: options.language,
                            task: options.task
                        )

                        // Create KV cache for each decoder layer
                        let kvCaches: [KVCache] = (0..<self.config.nTextLayer).map { _ in KVCache() }
                        var emittedText = ""
                        var lastEmittedIndex = tokens.count
                        var didEmitFinal = false

                        // 5. Decoding loop
                        let maxTokens = self.config.nTextCtx - tokens.count
                        let audioDuration = Double(audio.shape[0]) / Double(AudioConstants.sampleRate)

                        for step in 0..<maxTokens {
                            try Task.checkCancellation()

                            // Create token array for decoder
                            let tokenArray: MLXArray
                            if step == 0 {
                                tokenArray = MLXArray(tokens).expandedDimensions(axis: 0)
                            } else {
                                tokenArray = MLXArray([tokens.last!]).expandedDimensions(axis: 0)
                            }

                            // Decode one step
                            let (logits, crossQKArrays) = self.decoder(
                                tokens: tokenArray,
                                encoderOutput: encoderOutput,
                                kvCache: kvCaches
                            )

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
