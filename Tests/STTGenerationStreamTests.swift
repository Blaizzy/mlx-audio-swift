import Foundation
import MLX
import Testing

@testable import MLXAudioSTT

@Suite("STT generation streams", .serialized, .timeLimit(.minutes(5)))
struct STTGenerationStreamTests {
    private enum Failure: Error, Equatable { case expected }

    @Test @MainActor func returnsBeforeInferenceFinishesAndDeliversEventsIncrementally() async throws {
        let release = DispatchSemaphore(value: 0)
        defer { release.signal() }
        let stream = makeSTTStream { @Sendable continuation in
            // Fail before waiting if inference regresses to running inline.
            guard !Thread.isMainThread else {
                Issue.record("Inference ran on the caller's main thread")
                return
            }
            continuation.yield(.token("first"))
            // The consumer must receive the first token before the producer can
            // finish. Other suites may delay MainActor for tens of seconds, so
            // synchronize explicitly and let the test's time limit bound hangs.
            release.wait()
            continuation.yield(.token(" second"))
            continuation.yield(.info(STTGenerationInfo(
                promptTokenCount: 1, generationTokenCount: 2, prefillTime: 0,
                generateTime: 1, tokensPerSecond: 2, peakMemoryUsage: 0
            )))
            continuation.yield(.result(STTOutput(text: "first second")))
        }

        var iterator = stream.makeAsyncIterator()
        let first = try await iterator.next()
        guard case .token("first") = first else {
            Issue.record("Expected a token before inference completed")
            return
        }
        release.signal()

        var events: [String] = []
        while let event = try await iterator.next() {
            switch event {
            case .token(let text): events.append(text)
            case .info(let info):
                #expect(info.generationTokenCount == 2)
                events.append("info")
            case .result(let output):
                #expect(output.text == "first second")
                events.append("result")
            }
        }
        #expect(events == [" second", "info", "result"])
    }

    @Test func propagatesProducerErrorsAfterBufferedTokens() async throws {
        let stream = makeSTTStream { continuation in
            continuation.yield(.token("partial"))
            throw Failure.expected
        }
        var iterator = stream.makeAsyncIterator()
        let first = try await iterator.next()
        guard case .token("partial") = first else {
            Issue.record("Lost the token emitted before the error")
            return
        }
        do {
            _ = try await iterator.next()
            Issue.record("Expected the producer error")
        } catch {
            #expect(error as? Failure == .expected)
        }
    }

    @Test @MainActor func cancellingConsumerCancelsProducer() async throws {
        let release = DispatchSemaphore(value: 0)
        defer { release.signal() }
        let (received, receivedContinuation) = AsyncStream<Void>.makeStream()
        let (finished, finishedContinuation) = AsyncStream<Void>.makeStream()
        let stream = makeSTTStream { @Sendable continuation in
            defer { finishedContinuation.finish() }
            guard !Thread.isMainThread else {
                Issue.record("Inference ran on the caller's main thread")
                return
            }
            continuation.yield(.token("first"))
            release.wait()
            #expect(Task.isCancelled)
            try Task.checkCancellation()
            Issue.record("Cancelled producer continued generation")
        }
        let consumer = Task {
            defer { receivedContinuation.finish() }
            for try await _ in stream { receivedContinuation.yield(()) }
        }
        defer { consumer.cancel() }
        var receivedIterator = received.makeAsyncIterator()
        guard let _ = await receivedIterator.next() else {
            Issue.record("Expected a token before cancelling generation")
            return
        }
        consumer.cancel()
        _ = await consumer.result
        release.signal()
        for await _ in finished {}
    }

    @Test @MainActor func singleResultModelPreservesItsTranscript() async throws {
        let config = Wav2Vec2STTConfig(
            vocabSize: 6, hiddenSize: 8, numHiddenLayers: 0,
            numAttentionHeads: 2, intermediateSize: 16, hiddenDropout: 0,
            activationDropout: 0, featProjDropout: 0, convDim: [4],
            convStride: [2], convKernel: [4], numConvPosEmbeddings: 4,
            numConvPosEmbeddingGroups: 2
        )
        let model = Wav2Vec2CTCModel(config: config)
        model.train(false)
        let audio = MLXArray.zeros([64])
        let parameters = STTGenerateParameters()
        let expected = model.generate(audio: audio, generationParameters: parameters)
        var results: [STTOutput] = []
        for try await event in model.generateStream(audio: audio, generationParameters: parameters) {
            if case .result(let output) = event { results.append(output) }
        }
        #expect(results.count == 1)
        #expect(results.first?.text == expected.text)
        #expect(results.first?.generationTokens == expected.generationTokens)
    }
}
