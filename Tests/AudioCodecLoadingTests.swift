import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXAudioCodecs

@Suite("Shared codec loading", .serialized)
struct AudioCodecLoadingTests {
    @Test func metadataTakesPrecedenceOverRepositoryName() throws {
        #expect(try AudioCodecs.resolveFamily(config: ["model_type": " encodec "], source: "snac") == .encodec)
        #expect(throws: AudioCodecError.self) {
            try AudioCodecs.resolveFamily(config: ["model_type": "unimplemented_codec"], source: "snac")
        }
        #expect(try AudioCodecs.resolveFamily(config: ["vq_strides": [8, 4, 2]], source: "snapshot") == .snac)
        #expect(try AudioCodecs.resolveFamily(config: ["encoder_rates": [2], "n_codebooks": 1], source: "snapshot") == .descript)
        #expect(AudioCodecs.inferredFamily("kyutai/an-unrelated-model") == nil)
    }

    @Test func commonWaveformContractPreservesChannelsAndTrimsPadding() throws {
        let model = AnyAudioCodec(sampleRate: 48_000, audioChannels: 2) {
            concatenated([$0, MLXArray.zeros([$0.dim(0), 3, 2])], axis: 1)
        }
        let waveform = MLXArray((0..<20).map(Float.init), [2, 5, 2])
        let result = try model.reconstruct(waveform)
        #expect(model.sampleRate == 48_000)
        #expect(model.audioChannels == 2)
        #expect(result.shape == [2, 5, 2])
        #expect(result.asArray(Float.self) == waveform.asArray(Float.self))
    }

    @Test func malformedWaveformsAndDecoderOutputsThrow() throws {
        let model = AnyAudioCodec(sampleRate: 24_000) { $0 }
        #expect(throws: AudioCodecError.self) { try model.reconstruct(MLXArray.zeros([1, 0, 1])) }
        #expect(throws: AudioCodecError.self) { try model.reconstruct(MLXArray.zeros([1, 8])) }
        #expect(throws: AudioCodecError.self) { try model.reconstruct(MLXArray.zeros([1, 8, 2])) }
        let short = AnyAudioCodec(sampleRate: 24_000) { $0[0..., 0..<2, 0...] }
        #expect(throws: AudioCodecError.self) { try short.reconstruct(MLXArray.zeros([1, 8, 1])) }
        let failure = AnyAudioCodec(sampleRate: 24_000) { _ in
            throw CocoaError(.fileReadCorruptFile)
        }
        #expect(throws: CocoaError.self) { try failure.reconstruct(MLXArray.zeros([1, 8, 1])) }
    }

    @Test func arbitraryLocalDirectoryAndSymlinkUseCheckpointMetadata() async throws {
        let temporary = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let checkpoint = temporary.appendingPathComponent("snapshot")
        try FileManager.default.createDirectory(at: checkpoint, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temporary) }
        let config = DescriptDACConfig(
            encoderDim: 4, encoderRates: [2], latentDim: 8, decoderDim: 8,
            decoderRates: [2], nCodebooks: 1, codebookSize: 8, codebookDim: 2, sampleRate: 16_000
        )
        let original = DescriptDAC(config: config)
        try JSONEncoder().encode(config).write(to: checkpoint.appendingPathComponent("config.json"))
        try save(
            arrays: Dictionary(uniqueKeysWithValues: original.parameters().flattened()),
            url: checkpoint.appendingPathComponent("model.safetensors")
        )
        let alias = temporary.appendingPathComponent("unrelated-name")
        try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: checkpoint)
        let loaded = try await AudioCodecs.loadModel(modelRepo: alias.path)
        let waveform = MLXArray((0..<9).map { Float($0) / 100 }, [1, 9, 1])
        let expected = original.reconstruct(waveform)
        let actual = try loaded.reconstruct(waveform)
        #expect(loaded.sampleRate == 16_000)
        #expect(actual.shape == [1, 9, 1])
        #expect(abs(expected - actual).max().item(Float.self) < 1e-5)
    }

    @Test func missingLocalPathFailsWithoutHubResolution() async {
        await #expect(throws: AudioCodecError.self) {
            try await AudioCodecs.loadModel(modelRepo: "/private/tmp/\(UUID().uuidString)/missing")
        }
    }
}
