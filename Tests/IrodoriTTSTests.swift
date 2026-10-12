import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXAudioTTS

// Goldens come from the Python mlx-audio implementation (irodori_tts) run on the same
// tiny configurations. Parameters are filled by a shape-only deterministic rule
// (`irodoriTestFill`) so both sides agree without a key mapping.

/// scale·(0.05·sin(0.37·i + 0.13·(n mod 97)) + 0.02·cos(0.11·i)) over the flattened parameter.
private func irodoriTestFill(_ module: Module, scale: Float = 1) {
    let filled = module.parameters().flattened().map { key, p -> (String, MLXArray) in
        let n = p.size
        let i = MLXArray(0 ..< n).asType(.float32)
        let v = 0.05 * MLX.sin(i * 0.37 + Float(n % 97) * 0.13) + 0.02 * MLX.cos(i * 0.11)
        return (key, (scale * v).reshaped(p.shape))
    }
    module.update(parameters: ModuleParameters.unflattened(filled))
}

private func irodoriTestDecode<T: Decodable>(_ type: T.Type, _ json: String) throws -> T {
    try JSONDecoder().decode(type, from: Data(json.utf8))
}

private let tinyModernBertJSON = """
    {"model_type": "modernbert", "vocab_size": 32, "hidden_size": 16, "intermediate_size": 24,
     "num_hidden_layers": 4, "num_attention_heads": 2, "global_attn_every_n_layers": 3,
     "local_attention": 4, "norm_bias": false, "attention_bias": false, "mlp_bias": false,
     "rope_parameters": {"full_attention": {"rope_type": "default", "rope_theta": 160000.0},
                         "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0}}}
    """

/// Tiny v4.1-MF DiT: pretrained (ModernBERT) text/caption encoder, residual_mlp
/// projectors, speaker patch size 4, MeanFlow delta conditioning.
private func tinyMeanFlowDiTConfig() throws -> IrodoriDiTConfig {
    var cfg = IrodoriDiTConfig()
    cfg.flowParameterization = "meanflow"
    cfg.latentDim = 4
    cfg.modelDim = 32
    cfg.numLayers = 2
    cfg.numHeads = 4
    cfg.mlpRatio = 2
    cfg.textMlpRatio = 2
    cfg.speakerMlpRatio = 2
    cfg.textVocabSize = 32
    cfg.textDim = 16
    cfg.textLayers = 1
    cfg.textHeads = 2
    cfg.textEncoderType = "pretrained"
    cfg.textEncoderConfig = try irodoriTestDecode(IrodoriModernBertConfig.self, tinyModernBertJSON)
    cfg.pretrainedProjectorType = "residual_mlp"
    cfg.speakerDim = 16
    cfg.speakerLayers = 1
    cfg.speakerHeads = 2
    cfg.speakerPatchSize = 4
    cfg.timestepEmbedDim = 16
    cfg.adalnRank = 4
    cfg.useCaptionCondition = true
    cfg.useSpeakerCondition = true
    cfg.captionVocabSize = 32
    cfg.captionDim = 16
    cfg.useDurationPredictor = false
    return cfg
}

private func expectClose(_ got: MLXArray, _ want: [Float], atol: Float = 1e-4) {
    let values = got.asType(.float32).asArray(Float.self)
    #expect(values.count == want.count)
    let maxDiff = zip(values, want).map { abs($0 - $1) }.max() ?? .infinity
    #expect(maxDiff <= atol, "max |Δ| = \(maxDiff)")
}

@Suite("Irodori-TTS v4", .serialized)
struct IrodoriTTSV4Tests {
    @Test func configDecodesV41MeanFlowAndKeepsV3Defaults() throws {
        let v41mf = try irodoriTestDecode(IrodoriTTSConfig.self, """
            {"model_type": "irodori_tts", "max_text_length": 256, "ref_max_seconds": 120.0,
             "dit": {"flow_parameterization": "meanflow", "text_vocab_size": 102400,
                     "text_tokenizer_repo": "sbintuitions/modernbert-ja-310m",
                     "text_encoder_type": "pretrained",
                     "text_encoder_config": {"model_type": "modernbert", "hidden_size": 768,
                        "num_hidden_layers": 25, "num_attention_heads": 12, "local_attention": 128,
                        "rope_parameters": {"full_attention": {"rope_theta": 50000.0},
                                            "sliding_attention": {"rope_theta": 2500.0}}},
                     "pretrained_projector_type": "residual_mlp",
                     "pretrained_projector_hidden_ratio": 2.0, "speaker_patch_size": 4,
                     "use_caption_condition": true, "use_speaker_condition": true,
                     "caption_layers": null, "caption_heads": null, "caption_mlp_ratio": null},
             "sampler": {"num_steps": 4, "min_seconds": 0.5, "max_seconds": 30.0}}
            """)
        #expect(v41mf.dit.useMeanflow)
        #expect(v41mf.dit.usePretrainedTextEncoder)
        #expect(v41mf.dit.pretrainedProjectorType == "residual_mlp")
        #expect(v41mf.dit.speakerPatchSize == 4)
        #expect(v41mf.refMaxSeconds == 120)
        #expect(v41mf.sampler.numSteps == 4)
        let bert = try #require(v41mf.dit.textEncoderConfig)
        #expect(bert.numHiddenLayers == 25 && bert.localAttention == 128)
        // transformers >= 5 nests the per-layer-type RoPE thetas
        #expect(bert.globalRopeTheta == 50_000 && bert.localRopeTheta == 2_500)

        let v3 = try irodoriTestDecode(IrodoriTTSConfig.self, """
            {"model_type": "irodori_tts", "dit": {"text_vocab_size": 99574, "speaker_patch_size": 1,
             "use_caption_condition": true, "caption_layers": 10}, "sampler": {"num_steps": 40}}
            """)
        #expect(!v3.dit.useMeanflow)
        #expect(!v3.dit.usePretrainedTextEncoder)
        #expect(v3.dit.textEncoderConfig == nil)
        #expect(v3.refMaxSeconds == nil)
        #expect(v3.dit.captionLayersResolved == 10)
    }

    @Test func modernBertMatchesPython() throws {
        let config = try irodoriTestDecode(IrodoriModernBertConfig.self, tinyModernBertJSON)
        let encoder = try IrodoriModernBertEncoder(config: config)
        // Large enough weights that attention is not near-uniform, so RoPE thetas matter.
        irodoriTestFill(encoder, scale: 16)
        // 7 real tokens + 1 pad: exercises the key mask and the sliding window (local_attention 4).
        let ids = MLXArray([1, 5, 9, 13, 17, 21, 25, 3] as [Int32]).reshaped(1, 8)
        let mask = MLXArray([true, true, true, true, true, true, true, false]).reshaped(1, 8)
        let out = encoder(ids, mask: mask)
        #expect(out.shape == [1, 8, 16])
        expectClose(out[0, 0 ..< 7, 0 ..< 4], [
            -1.663344, 0.259008, -0.123047, -0.211456, 0.967362, -0.157313, -0.356682,
            0.341810, 1.050628, 0.290418, -0.253410, 0.388281, -1.839053, 0.670116,
            -0.217902, -0.156326, -1.768652, 0.216591, -0.113051, -0.255352, -1.608138,
            0.274052, -0.950824, 0.202779, 1.140644, 0.342895, -0.237972, 0.375593,
        ], atol: 1e-3)
        #expect(abs(out[0, 0 ..< 7].sum().item(Float.self) - -6.32924) < 1e-2)
    }

    @Test func meanFlowSamplerMatchesPython() throws {
        let dit = try IrodoriDiT(cfg: try tinyMeanFlowDiTConfig())
        // Large enough weights that the delta_t conditioning visibly moves the output.
        irodoriTestFill(dit, scale: 4)
        let enc = dit.encodeConditionsFull(
            textInputIDs: MLXArray([1, 4, 8, 15, 16, 23] as [Int32]).reshaped(1, 6),
            textMask: MLXArray.ones([1, 6], dtype: .bool),
            refLatent: MLXArray.zeros([1, 4, 4]),
            refMask: MLXArray.zeros([1, 4], dtype: .bool),
            captionInputIDs: MLXArray([1, 7, 11] as [Int32]).reshaped(1, 3),
            captionMask: MLXArray.ones([1, 3], dtype: .bool))
        // Python: mx.random.seed(0); mx.random.normal((1, 5, 4))
        let noise = MLXArray([
            0.519983, -0.273471, 0.311575, 0.730221, 2.001783, 0.492771, 0.452304, 0.884373,
            0.000688, -0.732530, 0.833127, -0.526324, -0.339867, -0.426560, 0.667879, -1.136015,
            -0.305548, -1.919324, -2.106826, 0.579773,
        ] as [Float]).reshaped(1, 5, 4)
        let out = try irodoriSampleEulerMeanFlow(
            model: dit, encoded: enc, latentDim: 4, sequenceLength: 5, rngSeed: 0, numSteps: 4,
            initialNoise: noise)
        // Python: sample_euler_meanflow(..., sequence_length=5, num_steps=4)
        expectClose(out, [
            0.194378, -0.669000, 0.023873, 0.463021, 1.744015, 0.082289, 0.201642, 0.560532,
            -0.336010, -1.091838, 0.554811, -0.766227, -0.680117, -0.762569, 0.400826, -1.372530,
            -0.595871, -2.154732, -2.358053, 0.335465,
        ], atol: 1e-3)
    }

    @Test func zeroReferenceSpansOneSpeakerPatch() throws {
        var config = IrodoriTTSConfig()
        config.dit = try tinyMeanFlowDiTConfig()
        let model = try IrodoriTTSModel(config: config)
        let latent = model.zeroReferenceLatent()
        let enc = model.model.encodeConditionsFull(
            textInputIDs: MLXArray([1, 4] as [Int32]).reshaped(1, 2),
            textMask: MLXArray.ones([1, 2], dtype: .bool),
            refLatent: latent,
            refMask: MLXArray.zeros([1, latent.shape[1]], dtype: .bool),
            captionInputIDs: nil, captionMask: nil)
        // speaker_patch_size 4: a shorter placeholder would patch to an empty sequence.
        #expect(enc.speakerState?.shape == [1, 1, 16])
        #expect(enc.speakerMask?.shape == [1, 1])
    }

    @Test func upstreamNormalizerMatchesPython() {
        let cases: [(String, String)] = [
            ("「前半」と「後半」", "「前半」と「後半」"),
            ("（(x)）", "x"),
            ("ｷﾞｭｳﾆｭｳ。", "ギュウニュウ。"),
            ("えっと...そう..です、はい。", "えっと…そう…です、はい。"),
            ("Ｈｅｌｌｏ\u{3000}Ｗｏｒｌｄ！", "HelloWorld!"),
            ("わ〜い♥ ㈱Ⅲ", "わーい♡ (株)III"),
        ]
        for (input, want) in cases {
            let got = irodoriNormalizeTextUpstream(input).irodoriPythonStripped()
            #expect(Array(got.unicodeScalars) == Array(want.unicodeScalars), "\(input) -> \(got)")
        }
    }
}
