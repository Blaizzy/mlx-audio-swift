import Testing

@testable import MLXAudioSTT

@Suite("MiMo V2 ASR Config Tests")
struct MiMoV2ASRConfigTests {
    @Test func mimoAudioTokenizerConfigDecodesOfficialShape() throws {
        let json = """
        {
          "max_audio_seconds": 1800,
          "stride_size": 2,
          "avg_pooler": 2,
          "d_model": 1280,
          "encoder_layers": 32,
          "encoder_skip_layer_id": 3,
          "encoder_attention_heads": 20,
          "encoder_ffn_dim": 5120,
          "nfft": 960,
          "n_mels": 128,
          "sampling_rate": 24000,
          "hop_length": 240,
          "window_size": 960,
          "num_quantizers": 20,
          "codebook_size": [1024, 1024, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128],
          "threshold_ema_dead_code": 2,
          "position_embedding_type": "rope",
          "rope_theta": 10000,
          "rope_type": "default",
          "ln_type": "LayerNorm",
          "vocoder_attention_heads": 16,
          "vocoder_attn_window_size": [40, 10]
        }
        """
        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(MiMoAudioTokenizerConfig.self, from: data)

        #expect(config.dModel == 1280)
        #expect(config.encoderLayers == 32)
        #expect(config.nMels == 128)
        #expect(config.samplingRate == 24000)
        #expect(config.numQuantizers == 20)
        #expect(config.activeCodebookSizes(prefixCount: 8) == [1024, 1024, 128, 128, 128, 128, 128, 128])
    }

    @Test func mimoV2ASRConfigDecodesOfficialShape() throws {
        let json = """
        {
          "architectures": ["MiMoV2ASRForCausalLM"],
          "model_type": "qwen2",
          "hidden_size": 4096,
          "intermediate_size": 11008,
          "num_hidden_layers": 36,
          "num_attention_heads": 32,
          "num_key_value_heads": 8,
          "head_dim": 128,
          "hidden_act": "silu",
          "max_position_embeddings": 8192,
          "vocab_size": 151680,
          "rope_theta": 640000,
          "rms_norm_eps": 1e-6,
          "attention_bias": true,
          "attention_dropout": 0.0,
          "use_cache": true,
          "tie_word_embeddings": false,
          "group_size": 4,
          "audio_channels": 8,
          "delay_pattern": "0-1-2-3-4-5-6-7",
          "speech_vocab_size": "1025-1025-129-129-129-129-129-129",
          "speech_zeroemb_idx": "1024-1024-128-128-128-128-128-128",
          "local_dim": 1024,
          "local_layers": 16,
          "local_attn_heads": 64,
          "local_ffn_dim": 4096,
          "local_attn_dropout": 0.1,
          "input_local_layers": 6,
          "input_local_dim": 1024,
          "input_full_attention": true,
          "n_rvq": 20,
          "add_input_local_transformer": true,
          "add_speech_sosp_eosp": false,
          "audio_config": {
            "tokenizer_version": "v1",
            "speech_vocab_size": "1025-1025-129-129-129-129-129-129",
            "speech_zeroemb_idx": "1024-1024-128-128-128-128-128-128",
            "group_size": 4,
            "audio_channels": 8,
            "input_local_layers": 6,
            "input_local_dim": 1024,
            "input_full_attention": true,
            "input_local_attn_heads": 64,
            "input_local_head_dim": 16,
            "input_local_intermediate_size": 4096,
            "input_local_hidden_dropout": 0.1,
            "out_hidden_size": 4096,
            "rope_theta": 640000,
            "partial_rotary_factor": 1.0,
            "projection_layers": 1,
            "add_post_norm": true,
            "audio_segment_size": 6000
          },
          "quantization_config": {
            "group_size": 64,
            "bits": 4,
            "mode": "affine"
          }
        }
        """
        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(MiMoV2ASRConfig.self, from: data)

        #expect(config.isMiMoV2ASR)
        #expect(config.modelType == "qwen2")
        #expect(config.hiddenSize == 4096)
        #expect(config.audioChannels == 8)
        #expect(config.groupSize == 4)
        #expect(config.nRVQ == 20)
        #expect(config.parsedDelayPattern == [0, 1, 2, 3, 4, 5, 6, 7])
        #expect(config.activeSpeechCodebookSizes == [1025, 1025, 129, 129, 129, 129, 129, 129])
        #expect(config.audioConfig?.audioSegmentSize == 6000)
        #expect(config.quantization?.bits == 4)
        #expect(config.quantization?.groupSize == 64)
    }

}
