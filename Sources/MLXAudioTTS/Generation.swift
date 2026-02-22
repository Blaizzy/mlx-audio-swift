@preconcurrency import MLX
import MLXAudioCore
@preconcurrency import MLXLMCommon
#if canImport(AVFoundation)
@preconcurrency import AVFoundation
#endif

public protocol SpeechGenerationModel: AnyObject {
    var sampleRate: Int { get }
    var defaultGenerationParameters: GenerateParameters { get }

    func generate(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray

    func generateStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error>
}

public extension SpeechGenerationModel {
    func generate(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters? = nil
    ) async throws -> MLXArray {
        try await generate(text: text, voice: voice, refAudio: refAudio, refText: refText, language: language, generationParameters: generationParameters ?? defaultGenerationParameters)
    }

    func generateSamplesStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters? = nil
    ) -> AsyncThrowingStream<[Float], Error> {
        let stream = generateStream(
            text: text,
            voice: voice,
            refAudio: refAudio,
            refText: refText,
            language: language,
            generationParameters: generationParameters ?? defaultGenerationParameters
        )
        return proxyAudioStream(stream, extract: {
            guard case .audio(let samples) = $0 else { return nil }
            return samples.asArray(Float.self)
        })
    }

#if canImport(AVFoundation)
    @MainActor
    func generatePCMBufferStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters? = nil
    ) -> AsyncThrowingStream<AVAudioPCMBuffer, Error> {
        let sampleStream = generateSamplesStream(
            text: text,
            voice: voice,
            refAudio: refAudio,
            refText: refText,
            language: language,
            generationParameters: generationParameters
        )

        let (stream, continuation) = AsyncThrowingStream<AVAudioPCMBuffer, Error>.makeStream()
        let sampleRate = self.sampleRate

        Task { @MainActor in
            do {
                for try await samples in sampleStream {
                    let buffer = try makePCMBuffer(samples: samples, sampleRate: sampleRate)
                    continuation.yield(buffer)
                }
                continuation.finish()
            } catch is CancellationError {
                continuation.finish(throwing: CancellationError())
            } catch {
                continuation.finish(throwing: error)
            }
        }

        return stream
    }
#endif
}

private func proxyAudioStream<T: Sendable, U: Sendable>(
    _ upstream: AsyncThrowingStream<T, Error>,
    extract: @Sendable @escaping (T) -> U?
) -> AsyncThrowingStream<U, Error> {
    AsyncThrowingStream<U, Error> { continuation in
        let task = Task { @Sendable in
            do {
                for try await value in upstream {
                    guard let extracted = extract(value) else { continue }
                    continuation.yield(extracted)
                }
                continuation.finish()
            } catch is CancellationError {
                continuation.finish(throwing: CancellationError())
            } catch {
                continuation.finish(throwing: error)
            }
        }
        continuation.onTermination = { @Sendable _ in task.cancel() }
    }
}

#if canImport(AVFoundation)
@MainActor
private func makePCMBuffer(samples: [Float], sampleRate: Int) throws -> AVAudioPCMBuffer {
    let frameCount = AVAudioFrameCount(samples.count)
    guard
        let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(sampleRate),
            channels: 1,
            interleaved: false
        ),
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount),
        let channel = buffer.floatChannelData?[0]
    else {
        throw AudioGenerationError.audioDecodingFailed("Failed to create AVAudioPCMBuffer")
    }

    buffer.frameLength = frameCount
    for i in 0 ..< samples.count {
        channel[i] = samples[i]
    }
    return buffer
}
#endif
