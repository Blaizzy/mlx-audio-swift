import Foundation
import Testing

@testable import MLXAudioTTS

@Suite("Dia TTS")
struct DiaTTSTests {
    private static let configJSON = """
    {
      "version": "0.1",
      "model": {
        "encoder": {"n_layer": 12, "n_embd": 1024, "n_hidden": 4096, "n_head": 16, "head_dim": 128},
        "decoder": {"n_layer": 18, "n_embd": 2048, "n_hidden": 8192, "gqa_query_heads": 16,
                    "cross_query_heads": 16, "kv_heads": 4, "gqa_head_dim": 128, "cross_head_dim": 128},
        "src_vocab_size": 256,
        "tgt_vocab_size": 1028,
        "dropout": 0.0
      },
      "training": {},
      "data": {
        "text_length": 1024, "audio_length": 3072, "channels": 9,
        "text_pad_value": 0, "audio_eos_value": 1024, "audio_pad_value": 1025, "audio_bos_value": 1026,
        "delay_pattern": [0, 8, 9, 10, 11, 12, 13, 14, 15]
      }
    }
    """

    @Test func configDecodesArchitectureAndDataSettings() throws {
        let config = try JSONDecoder().decode(DiaConfig.self, from: Data(Self.configJSON.utf8))
        #expect(config.model.encoder.nLayer == 12)
        #expect(config.model.encoder.nEmbd == 1024)
        #expect(config.model.decoder.nLayer == 18)
        #expect(config.model.decoder.gqaQueryHeads == 16)
        #expect(config.model.decoder.kvHeads == 4)
        #expect(config.model.decoder.crossHeadDim == 128)
        #expect(config.model.srcVocabSize == 256)
        #expect(config.model.tgtVocabSize == 1028)
        #expect(config.model.sampleRate == 44_100)
        #expect(config.data.textLength == 1024)
        #expect(config.data.audioLength == 3072)
        #expect(config.data.channels == 9)
        #expect(config.data.audioBosValue == 1026)
        #expect(config.data.delayPattern == [0, 8, 9, 10, 11, 12, 13, 14, 15])
    }

    @Test func buildsModelFromDecodedConfig() throws {
        let config = try JSONDecoder().decode(DiaConfig.self, from: Data(Self.configJSON.utf8))
        let model = DiaModel(config)
        #expect(model.decoder.numLayers == 18)
        #expect(model.decoder.numChannels == 9)
    }

    @Test func ttsModelResolutionIncludesDia() {
        #expect(TTS.resolveModelType(modelRepo: "mlx-community/Dia-1.6B") == "dia")
        #expect(TTS.resolveModelType(modelRepo: "nari-labs/Dia-1.6B") == "dia")
        #expect(TTS.resolveModelType(modelRepo: "anything", modelType: "dia") == "dia")
        #expect(TTS.resolveModelType(modelRepo: "mlx-community/parakeet-diarization") != "dia")
    }
}
