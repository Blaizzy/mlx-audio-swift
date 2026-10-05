@preconcurrency import MLX
import MLXAudioCore
@preconcurrency import MLXLMCommon
#if canImport(AVFoundation)
@preconcurrency import AVFoundation
#endif

/// Reference tensors are transferred into generation via `sending`. After a call,
/// the caller must not access the reference tensor or any aliases to it. Create a
/// fresh tensor for each generation when reusing reference samples. Returned audio
/// is transferred back to the caller. Model instances still require serialized use.
public protocol SpeechGenerationModel: AnyObject {
    var sampleRate: Int { get }
    var defaultGenerationParameters: GenerateParameters { get }

    func generate(
        text: String,
        voice: String?,
        refAudio: sending MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) async throws -> sending MLXArray

    func generateStream(
        text: String,
        voice: String?,
        refAudio: sending MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> sending AsyncThrowingStream<AudioGeneration, Error>

    func generateStream(
        text: String,
        voice: String?,
        refAudio: sending MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters,
        streamingInterval: Double
    ) -> sending AsyncThrowingStream<AudioGeneration, Error>
}

public extension SpeechGenerationModel {
    func generate(
        text: String,
        voice: String?,
        refAudio: sending MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters? = nil
    ) async throws -> sending MLXArray {
        try await generate(text: text, voice: voice, refAudio: refAudio, refText: refText, language: language, generationParameters: generationParameters ?? defaultGenerationParameters)
    }

    func generateSamplesStream(
        text: String,
        voice: String?,
        refAudio: sending MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters? = nil,
        streamingInterval: Double = 2.0
    ) -> AsyncThrowingStream<[Float], Error> {
        let stream = generateStream(
            text: text,
            voice: voice,
            refAudio: refAudio,
            refText: refText,
            language: language,
            generationParameters: generationParameters ?? defaultGenerationParameters,
            streamingInterval: streamingInterval
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
        refAudio: sending MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters? = nil,
        streamingInterval: Double = 2.0
    ) -> AsyncThrowingStream<AVAudioPCMBuffer, Error> {
        let sampleStream = generateSamplesStream(
            text: text,
            voice: voice,
            refAudio: refAudio,
            refText: refText,
            language: language,
            generationParameters: generationParameters,
            streamingInterval: streamingInterval
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

    func generateStream(
        text: String,
        voice: String?,
        refAudio: sending MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters,
        streamingInterval: Double = 2.0
    ) -> sending AsyncThrowingStream<AudioGeneration, Error> {
        _ = streamingInterval
        return generateStream(
            text: text,
            voice: voice,
            refAudio: refAudio,
            refText: refText,
            language: language,
            generationParameters: generationParameters
        )
    }
}

private func proxyAudioStream<T, U: Sendable>(
    _ upstream: sending AsyncThrowingStream<T, Error>,
    extract: @Sendable @escaping (T) -> U?
) -> AsyncThrowingStream<U, Error> {
    let upstream = SendingBox(upstream)
    return AsyncThrowingStream<U, Error> { continuation in
        let task = Task { @Sendable in
            let upstream = upstream.take()
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
