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
        #expect(try AudioCodecs.resolveFamily(
            config: ["encoder_rates": [2], "n_codebooks": 9, "semantic_codebook_size": 4096],
            source: "snapshot") == .fishS1DAC)
        #expect(try AudioCodecs.resolveFamily(
            config: ["model_type": "mimo_audio_tokenizer"], source: "nemotron-voicechat") == .mimoAudioTokenizer)
        #expect(try AudioCodecs.resolveFamily(
            config: ["model_type": "nemotron_voicechat"], source: "mimo-audio-tokenizer") == .nemotronVoiceChat)
        #expect(try AudioCodecs.resolveFamily(
            config: ["encoder_attention_heads": 4, "vocoder_attention_heads": 4], source: "snapshot") == .mimoAudioTokenizer)
        #expect(try AudioCodecs.resolveFamily(
            config: ["codec_config": ["base_channels": 4]], source: "snapshot") == .nemotronVoiceChat)
        #expect(AudioCodecs.inferredFamily("XiaomiMiMo/MiMo-Audio-Tokenizer") == .mimoAudioTokenizer)
        #expect(AudioCodecs.inferredFamily("mlx-community/NemotronLabs-VoiceChat-11B-4bit") == .nemotronVoiceChat)
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

    @Test func channelsFirstCodecAdapterReconstructsSNACWithoutTransposingSamples() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let config: [String: Any] = [
            "sampling_rate": 24_000, "encoder_dim": 4, "encoder_rates": [2], "latent_dim": 8,
            "decoder_dim": 8, "decoder_rates": [2], "attn_window_size": NSNull(),
            "codebook_size": 8, "codebook_dim": 2, "vq_strides": [1], "noise": false, "depthwise": false,
        ]
        try JSONSerialization.data(withJSONObject: config).write(to: directory.appendingPathComponent("config.json"))
        let native = SNAC(samplingRate: 24_000, encoderDim: 4, encoderRates: [2], latentDim: 8,
                          decoderDim: 8, decoderRates: [2], attnWindowSize: nil, codebookSize: 8,
                          codebookDim: 2, vqStrides: [1], noise: false, depthwise: false)
        try save(arrays: Dictionary(uniqueKeysWithValues: native.parameters().flattened()),
                 url: directory.appendingPathComponent("model.safetensors"))
        let common = try await AudioCodecs.loadModel(modelRepo: directory.path)
        let input = MLXArray((0..<9).map { Float($0) / 100 }, [1, 9, 1])
        let expected = native.reconstruct(input.transposed(0, 2, 1)).transposed(0, 2, 1)[0..., 0..<9, 0...]
        let actual = try common.reconstruct(input)
        #expect(actual.shape == [1, 9, 1])
        #expect(abs(actual - expected).max().item(Float.self) < 1e-5)
    }

    @Test func missingLocalPathFailsWithoutHubResolution() async {
        await #expect(throws: AudioCodecError.self) {
            try await AudioCodecs.loadModel(modelRepo: "/private/tmp/\(UUID().uuidString)/missing")
        }
    }
}
