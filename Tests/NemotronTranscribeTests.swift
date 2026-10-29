import Foundation
import Hub
import MLX
import MLXNN
import Testing
import Tokenizers

@testable import MLXAudioCore
@testable import MLXAudioSTT

@Suite("Nemotron Transcribe", .serialized)
struct NemotronTranscribeTests {
    static let configuration = #"""
        {"model_type":"nemotron_transcribe","llm_config":{
          "model_type":"nemotron_h","vocab_size":64,"hidden_size":16,"intermediate_size":24,
          "num_attention_heads":2,"num_key_value_heads":1,"head_dim":8,
          "mamba_num_heads":2,"mamba_head_dim":8,"ssm_state_size":8,"conv_kernel":4,"n_groups":1,
          "layers_block_type":["mamba","attention","moe"],"n_routed_experts":4,
          "num_experts_per_tok":2,"n_shared_experts":1,"moe_intermediate_size":24,
          "moe_shared_expert_intermediate_size":32,"n_group":1,"topk_group":1},
         "perception":{"output_dim":16,"preprocessor":{"features":8},
           "modality_adapter":{"_target_":"nemo.IdentityConnector","d_model":16}},
         "pe_encoder_config":{"asr_encoder_type":"transformer",
           "speaker_feature_config_version":1,"speaker_feature_mode":"thresholded","speaker_activity_threshold":0.5,
           "online_inference_length":4,"chunk_left_context":1,"chunk_right_context":1,
           "diar_spkcache_len":8,"diar_fifo_len":2,"diar_spkcache_update_period":2,
           "asr_encoder_cfg":{"feat_in":8,"d_model":16,"n_layers":2,"n_heads":2,
             "subsampling_factor":2,"ff_expansion":2,"qk_norm":true,"rotary_fraction":0.5},
           "diarization_model_cfg":{"high_resolution":true,"preprocessor":{"features":8},
             "encoder":{"feat_in":8,"d_model":16,"n_layers":2,"n_heads":2,
               "subsampling_factor":2,"ff_expansion":2,"qk_norm":false,"rotary_fraction":1},
             "sortformer_modules":{"num_spks":2,"fc_d_model":16,"tf_d_model":8}}}}
        """#

    func config(timestamps: Bool = false) throws -> NemotronTranscribeConfiguration {
        let source =
            timestamps
            ? Self.configuration.replacingOccurrences(
                of: "\"model_type\":\"nemotron_transcribe\",",
                with: #"""
                    "model_type":"nemotron_transcribe","ctc_timestamps":{"decoder_config":{
                    "feat_in":16,"num_classes":64,"n_layers":2,"n_heads":2,"ff_expansion":0.5,
                    "qk_norm":true,"rotary_fraction":0.5,"upsampling_factor":2,"residual_scale":0.1}},
                    """#) : Self.configuration
        return try JSONDecoder().decode(
            NemotronTranscribeConfiguration.self, from: Data(source.utf8))
    }
    func model(timestamps: Bool = false) throws -> NemotronTranscribeModel {
        let m = NemotronTranscribeModel(try config(timestamps: timestamps))
        var values: [String: MLXArray] = [:]
        for (key, value) in m.parameters().flattened() {
            let tensor: MLXArray
            if key.contains(".preprocessor.") {
                tensor = value
            } else if key.hasSuffix(".A_log") || key.hasSuffix(".dt_bias") {
                tensor = .full(value.shape, values: MLXArray(Float(-1)))
            } else if key.hasSuffix(".D") {
                tensor = .full(value.shape, values: MLXArray(Float(0.5)))
            } else if key.hasSuffix(".e_score_correction_bias") {
                tensor = MLXArray(0 ..< value.size).asType(.float32).reshaped(value.shape) / 1024
            } else if key.hasSuffix(".weight")
                && key.split(separator: ".").dropLast().last!.contains("norm")
            {
                tensor = .ones(value.shape)
            } else {
                let seed = key.utf8.reduce(0) { $0 + Int($1) } % 17
                tensor =
                    ((MLXArray(0 ..< value.size) + seed) % 17 - 8).asType(.float32).reshaped(
                        value.shape) / 64
            }
            values[key] = tensor
        }
        try m.update(parameters: ModuleParameters.unflattened(values), verify: .all)
        m.train(false)
        return m
    }
    func audio() -> MLXArray {
        let samples: [Float] = (0 ..< 2400).map { index in
            let sine = sin(Double(index) * 0.07) * 0.1
            let cosine = cos(Double(index) * 0.03) * 0.05
            return Float(sine + cosine)
        }
        return MLXArray(samples)
    }
    func close(
        _ actual: MLXArray, _ expected: [Float], tolerance: Float = 2e-5,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        let first = actual.reshaped(-1)[..<expected.count]
        #expect(
            abs(first - MLXArray(expected)).max().item(Float.self) < tolerance,
            sourceLocation: sourceLocation)
    }

    @Test func configurationAndLoaderAliases() throws {
        let c = try config()
        #expect(c.text.blockTypes == ["M", "*", "E"])
        #expect(c.speaker.modules.spkcacheLen == 8 && c.speaker.modules.fifoLen == 2)
        #expect(c.speaker.modules.chunkLen == 4)
        #expect(
            STT.inferModelType(from: "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B")
                == "nemotron_transcribe")
        #expect(
            STT.inferModelType(from: "mlx-community/nemotron-speech-streaming-en-0.6b-bf16")
                == "nemotron_asr")
        let tagged = Self.configuration.replacingOccurrences(
            of: "\"conv_kernel\":4",
            with: "\"conv_kernel\":4,\"time_step_limit\":[0,{\"__float__\":\"Infinity\"}]")
        let decoded = try JSONDecoder().decode(
            NemotronTranscribeConfiguration.self, from: Data(tagged.utf8))
        #expect(decoded.text.maximumTimeStep == .infinity)
        var invalid = c
        invalid.speaker.encoder.factor = 4
        #expect(throws: STTError.self) { try invalid.validate() }
        invalid = c
        invalid.speakerThreshold = nil
        #expect(throws: STTError.self) { try invalid.validate() }
        invalid = c
        invalid.text.groups = 3
        #expect(throws: (any Error).self) { try invalid.validate() }
    }

    @Test func sourceMappingsAreStrictAndIdempotent() throws {
        let m = try model()
        let original = Dictionary(uniqueKeysWithValues: m.parameters().flattened())
        // Parameter inventory of the equivalent Python model, including its mel buffers.
        #expect(original.count == 107)
        var source: [String: MLXArray] = [:]
        for (key, tensor) in original {
            var key = key.replacingOccurrences(
                of: "llm.backbone.embeddings.", with: "llm.model.embed_tokens."
            )
            .replacingOccurrences(of: "llm.backbone.norm_f.", with: "llm.model.norm.")
            .replacingOccurrences(of: "llm.backbone.layers.", with: "llm.model.layers.")
            .replacingOccurrences(of: ".ffn.linear1.", with: ".ffn.net.0.")
            .replacingOccurrences(of: ".ffn.linear2.", with: ".ffn.net.3.")
            .replacingOccurrences(of: ".preprocessor.", with: ".preprocessor.featurizer.")
            var value = tensor
            if key.hasSuffix(".switch_mlp.fc1.weight") {
                key = key.replacingOccurrences(
                    of: ".switch_mlp.fc1.weight", with: ".experts.gate_and_up_projs")
                value = value.transposed(0, 2, 1)
            } else if key.hasSuffix(".switch_mlp.fc2.weight") {
                key = key.replacingOccurrences(
                    of: ".switch_mlp.fc2.weight", with: ".experts.down_projs")
                value = value.transposed(0, 2, 1)
            } else if key.hasSuffix(".conv1d.weight") || key.hasSuffix(".subpixel_upsample.weight")
            {
                value = value.transposed(0, 2, 1)
            }
            if key.hasSuffix(".A_log") || key.hasSuffix(".dt_bias") || key.hasSuffix(".D") {
                let split = key.lastIndex(of: ".")!
                key.insert(contentsOf: "._fp32_params", at: split)
            }
            source[key] = value
        }
        source["llm.mtp.layers.0.norm.weight"] = .ones([16])
        source["ctc_timestamp.decoder.residual_scale"] = MLXArray(Float(0.1))
        source["llm.lm_head._extra_state"] = .zeros([0])
        let once = NemotronTranscribeModel.sanitize(weights: source)
        let twice = NemotronTranscribeModel.sanitize(weights: once)
        #expect(Set(once.keys) == Set(original.keys))
        for (key, value) in original {
            #expect(arrayEqual(once[key]!, value).item(Bool.self))
            #expect(arrayEqual(twice[key]!, value).item(Bool.self))
        }
        try m.update(parameters: ModuleParameters.unflattened(once), verify: .all)
    }

    // Filled with CPU reference outputs, using the exact binary weights above.
    static let golden: [String: [Float]] = [
        "mel": [
            -6.6448917389, -6.3265833855, -6.3259840012, -6.3246173859, -6.3234314919,
            -6.3223252296, -6.3214392662, -6.3206596375, -6.3201150894, -6.3197050095,
            -6.3195309639, -6.319507122, -6.3197193146, -6.3200774193, -6.3204708099, -9.1924848557,
            -14.113671303, -16.600828171, -16.590581894, -16.600767136, -16.593009949,
            -16.597686768, -16.597232819, -16.593992233, -16.600477219, -16.592325211,
            -16.600618362, -16.593839645, -16.597600937, -15.232591629, -9.4499759674,
            -14.166947365,
        ],
        "stacked": [
            -4.0404171944, -1.2465054989, -0.21764308214, 0.13451468945, 0.41827660799,
            0.67964828014, 0.93079400063, 1.1767376661, 1.4201129675, 1.6622880697, 2.7757012844,
            1.8206689358, 0.8514854908, -0.12360525131, -1.1013278961, -2.0804548264, -3.0477795601,
            0.78765130043, 2.9427416325, 2.3685770035, 1.7852578163, 1.2018780708, 0.61849522591,
            0.035112380981, -0.54827070236, -1.1316536665, 1.0238002539, 0.45235705376,
            -0.13087773323, -0.71424913406, -1.2976303101, -1.8810133934,
        ],
        "speech": [
            -2.7671339512, -0.72716957331, 0.63765907288, 1.3212502003, 1.1291414499, 0.67704629898,
            0.3441901803, -0.12225929648, -0.49200034142, -0.56037080288, 0.46431815624,
            0.88248217106, 0.8152281642, 0.2074598968, -0.55869394541, -1.2667729855, -1.3669759035,
            1.0478770733, 2.4644796848, 1.5763976574, 0.63459122181, -0.0092525035143,
            0.013868145645, -0.0012721270323, -0.30922165513, -0.91514188051, 0.44713789225,
            0.12129443884, -0.45585268736, -1.0261135101, -1.2759796381, -0.9614610076,
        ],
        "speaker": [
            -0.16516630352, -0.85293155909, -1.5688285828, -2.0664243698, -0.82787597179,
            0.10121931136, 0.80082839727, 1.3296263218, 1.2950286865, 0.88941240311, 0.30907270312,
            -0.26152133942, -0.54520726204, 0.17294128239, 0.42419800162, 0.90312826633,
            -0.77453845739, -1.0904010534, -1.1293967962, -0.949223876, 0.82659351826, 1.8558050394,
            1.5317171812, 1.0938520432, 0.59321677685, -0.01763304323, -0.60469758511,
            -0.8385515213, -0.95977908373, 0.182419613, 0.14229649305, 0.075820609927,
        ],
        "head": [
            0.48820620775, 0.4924197495, 0.48827001452, 0.49222818017, 0.48817655444, 0.49250885844,
            0.48828339577, 0.49218809605, 0.48826003075, 0.49225834012, 0.48828339577,
            0.49218809605, 0.48826035857, 0.49225720763, 0.48828339577, 0.49218809605,
            0.48826161027, 0.49225339293, 0.48828339577, 0.49218809605, 0.48826104403,
            0.49225512147, 0.48828339577, 0.49218809605, 0.48815655708, 0.49256879091, 0.4880862236,
            0.49277991056, 0.48817354441, 0.49251776934, 0.48828339577, 0.49218809605,
        ],
        "parallel": [
            -0.66399914026, -1.0343177319, -2.2832531929, -0.71165549755, 1.0183372498,
            1.5699379444, 1.2049065828, 0.71187269688, 0.15209257603, -0.066822908819,
            -1.0206182003, -0.41459834576, 0.36518391967, 1.1114275455, 0.63716262579,
            0.14857296646, -1.2626001835, -0.0072279572487, 2.4869158268, 1.6171176434,
            0.59985685349, 0.083379544318, 0.089197203517, 0.071484185755, -0.49739718437,
            -1.2402523756, 1.0416742563, 0.66026455164, -0.335958004, -0.83487170935, -1.0839855671,
            -0.66336834431,
        ],
        "perception": [
            0.23827113211, 0.41182637215, 0.45560002327, 0.45990896225, 0.28784307837,
            0.017411321402, -0.58476889133, -0.76949346066, -0.4946886301, -0.33898997307,
            -0.014627769589, 0.17877233028, 0.22348082066, 0.21003994346, -0.056752853096,
            -0.16257162392, 0.035637170076, -0.24798756838, -0.41988584399, -0.41557693481,
            -0.74664616585, -0.74425721169, -0.079361215234, 0.35449457169, 0.51814049482,
            0.27897208929, 0.30697402358, 0.33027091622, 0.20245873928, -0.12267438322,
            0.15832926333, 0.33802098036,
        ],
        "language": [
            -0.12709161639, -0.14654333889, -0.046250384301, -0.3701980412, -0.1625970602,
            -0.46269848943, -0.42020589113, -0.33388134837, -0.39781600237, 0.30482965708,
            0.18474219739, 0.19676119089, 0.3285741806, 0.39067623019, 0.66975665092, 0.25336495042,
            0.13857711852, -0.12709161639, -0.14654333889, -0.046250384301, -0.3701980412,
            -0.1625970602, -0.46269848943, -0.42020589113, -0.33388134837, -0.39781600237,
            0.30482965708, 0.18474219739, 0.19676119089, 0.3285741806, 0.39067623019, 0.66975665092,
        ],
        "mamba": [
            0.133635059, 0.30336117744, 0.20533953607, 0.087662398815, -0.26722887158,
            -0.43851608038, -0.28564542532, -0.42380607128, -0.29450878501, -0.13341020048,
            0.045862346888, 0.093630045652, 0.13324426115, 0.21499162912, 0.35767030716,
            0.17040780187, 0.02567589283, -0.27537834644, -0.23966212571, -0.11843892932,
            0.0094506368041, 0.087790966034, -0.019326664507, -0.0051303002983, -0.091614946723,
            -0.21596558392, -0.30905383825, -0.35445076227, -0.1363863945, 0.30187302828,
            0.5169197917, 0.50729548931,
        ],
        "router_indices": [3, 2, 0, 3, 3, 2, 3, 2, 0, 1, 3, 2, 3, 0],
        "router_weights": [
            1.3429813385, 1.1570187807, 1.2954573631, 1.2045426369, 1.3290708065, 1.1709291935,
            1.3126689196, 1.1873309612, 1.3126689196, 1.1873309612, 1.3456648588, 1.154335022, 1.25,
            1.25,
        ],
        "moe": [
            0.11761648208, -0.60348635912, 0.15516942739, -1.2475332022, 0.51449304819,
            0.57965105772, 0.17018593848, 0.86614978313, -0.56788557768, 0.16265641153,
            -0.61100143194, -0.5795570612, 0.22288411856, 0.075251758099, 0.93632870913,
            -0.35734438896, 1.4040411711, 0.51674997807, 0.46806550026, -0.56157505512,
            -0.56092703342, 0.067441791296, -0.47346040606, 0.11418530345, 0.2822830081,
            0.35464167595, -0.038300514221, 0.05288977176, 0.57552444935, -0.45807915926,
            -0.52033054829, -1.2580797672,
        ],
    ]

    @Test(arguments: [Device.cpu, Device.gpu])
    func languageBackboneMatchesPythonAndCachedChunks(device: Device) throws {
        try Device.withDefaultDevice(device) {
            let m = try model()
            let tokens = MLXArray([Int32(1), 8, 4, 5, 9, 2, 7]).reshaped(1, 7)
            let whole = m.llm(tokens)
            // GPU matrix kernels accumulate differently; the CPU golden remains tight.
            close(whole, Self.golden["language"]!, tolerance: device == .cpu ? 3e-6 : 5e-4)
            let cache = m.llm.makeCache()
            let a = m.llm(tokens[0..., ..<3], cache: cache)
            let b = m.llm(tokens[0..., 3 ..< 5], cache: cache)
            let c = m.llm(tokens[0..., 5...], cache: cache)
            #expect(abs(concatenated([a, b, c], axis: 1) - whole).max().item(Float.self) < 2e-5)
            #expect(cache.recurrent.compactMap { $0 }.allSatisfy { $0.dtype == .float32 })
            #expect(cache.convolution[0]?.shape == [1, 3, 32])
            let embedded = m.llm.backbone.embeddings(tokens)
            #expect(abs(m.llm(embeddings: embedded) - whole).max().item(Float.self) < 1e-6)
        }
    }

    @Test func mambaAndRouterMatchPythonCPU() throws {
        try Device.withDefaultDevice(.cpu) {
            let m = try model()
            let tokens = MLXArray([Int32(1), 8, 4, 5, 9, 2, 7]).reshaped(1, 7)
            let first = m.llm.backbone.layers[0]
            let x = first.norm(m.llm.backbone.embeddings(tokens))
            let mamba = first.mixer as! NemotronHMambaMixer
            close(mamba(x, cache: nil, layer: 0), Self.golden["mamba"]!, tolerance: 2e-6)
            let moe = m.llm.backbone.layers[2].mixer as! NemotronHMoE
            let normalized = m.llm.backbone.layers[2].norm(x)
            let routing = moe.gate(normalized)
            close(routing.0.asType(.float32), Self.golden["router_indices"]!, tolerance: 1e-6)
            close(routing.1, Self.golden["router_weights"]!, tolerance: 2e-7)
            close(moe(normalized), Self.golden["moe"]!, tolerance: 2e-6)
        }
    }

    @Test(arguments: [Device.cpu, Device.gpu])
    func perceptionMatchesPython(device: Device) throws {
        try Device.withDefaultDevice(device) {
            let m = try model()
            let features = m.perception.preprocessor(audio())
            #expect(features.shape == [1, 8, 15])
            // Log-mels amplify GPU FFT rounding; CPU reference checks stay tight.
            close(features, Self.golden["mel"]!, tolerance: device == .cpu ? 3e-5 : 2e-3)
            let speech = m.perception.encoder.speech
            let stacked = speech.stacking(features)
            close(stacked, Self.golden["stacked"]!, tolerance: device == .cpu ? 1e-5 : 6e-3)
            close(
                speech(stacked, length: 8), Self.golden["speech"]!,
                tolerance: device == .cpu ? 3e-5 : 4e-3)
            let speaker = m.perception.encoder.speaker
            let speakerStates = speaker.encoder(speaker.encoder.stacking(features), length: 8)
            close(speakerStates, Self.golden["speaker"]!, tolerance: device == .cpu ? 3e-5 : 4e-3)
            close(speaker.head(speakerStates), Self.golden["head"]!, tolerance: 2e-6)
            close(
                m.perception.encoder(features), Self.golden["parallel"]!,
                tolerance: device == .cpu ? 4e-5 : 1e-2)
            close(
                m.perception(audio()), Self.golden["perception"]!,
                tolerance: device == .cpu ? 1e-5 : 3e-3)
        }
    }

    @Test(arguments: [1, 7, 8, 9, 16, 17, 41])
    func windowedPerceptionIncludesPartialFinalFrame(frames: Int) throws {
        let m = try model()
        let features = (MLXArray(0 ..< (8 * frames)).asType(.float32) / 64).reshaped(1, 8, frames)
        let output = m.perception.encoder(features)
        #expect(output.shape == [1, (frames + 1) / 2, 16])
        #expect(MLX.all(isFinite(output)).item(Bool.self))
    }

    @Test func leftContextIsExcludedFromSpeakerHistory() throws {
        let m = try model().perception.encoder.speaker
        let state = m.makeState()
        let features = (MLXArray(0 ..< 80).asType(.float32) / 32).reshaped(1, 8, 10)
        let chunk = m.encoder.stacking(features)
        let result = m.step(
            features, state: state, centralFrames: 6, featureLength: 10, leftContextFrames: 2)
        #expect(result.shape == [6, 2])
        let history = concatenated([state.cache, state.fifo], axis: 1)
        #expect(arrayEqual(history, chunk[0..., 1 ..< 4, 0...]).item(Bool.self))
    }

    @Test func bfloat16DynamicsRetainFloat32Memory() throws {
        let m = try model()
        let weights = Dictionary(
            uniqueKeysWithValues: m.parameters().flattened().map { key, value in
                let sensitive =
                    key.hasSuffix(".A_log") || key.hasSuffix(".D") || key.hasSuffix(".dt_bias")
                    || key.hasSuffix(".gate.weight") || key.hasSuffix(".e_score_correction_bias")
                return (key, value.asType(sensitive ? .float32 : .bfloat16))
            })
        try m.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        let cache = m.llm.makeCache()
        let output = m.llm(MLXArray([Int32(1), 2, 3]).reshaped(1, 3), cache: cache)
        #expect(output.dtype == .bfloat16)
        #expect(cache.recurrent[0]?.dtype == .float32)
        #expect(cache.convolution[0]?.dtype == .bfloat16)
        #expect(MLX.all(isFinite(output)).item(Bool.self))
    }

    @Test func expertSortingAndQuantizedRoundTrip() throws {
        var c = try config().text
        c.hiddenSize = 64
        c.expertIntermediateSize = 64
        let model = NemotronHExperts(c)
        let weights = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().map { key, value in
                (
                    key,
                    ((MLXArray(0 ..< value.size) % 17) - 8).asType(.float32).reshaped(value.shape)
                        / 64
                )
            })
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        quantize(model: model, groupSize: 32, bits: 4)
        #expect(model.fc1 is QuantizedExpertLinear)
        #expect(model.fc2 is QuantizedExpertLinear)
        let x = ((MLXArray(0 ..< (40 * 64)) % 61) - 30).asType(.float32).reshaped(1, 40, 64) / 64
        var choices: [Int32] = []
        for i in 0 ..< 40 {
            choices.append(Int32(i % 4))
            choices.append(Int32((i + 2) % 4))
        }
        let indices = MLXArray(choices).reshaped(1, 40, 2)
        let sortedResult: MLXArray = model(x, indices: indices)
        var pieces: [MLXArray] = []
        for i in 0 ..< 40 {
            let input: MLXArray = x[0..., i ..< (i + 1), 0...]
            let selection: MLXArray = indices[0..., i ..< (i + 1), 0...]
            pieces.append(model(input, indices: selection))
        }
        let unsorted: MLXArray = concatenated(pieces, axis: 1)
        let difference: MLXArray = sortedResult - unsorted
        let errorNorm: Float = sqrt((difference * difference).sum()).item(Float.self)
        let referenceNorm: Float = sqrt((unsorted * unsorted).sum()).item(Float.self)
        let relative = errorNorm / max(referenceNorm, 1e-12)
        #expect(relative < 1e-3)
    }

    @Test(arguments: [Device.cpu, Device.gpu])
    func boundedMambaTimeStepsMatchReference(device: Device) {
        Device.withDefaultDevice(device) {
            let x = MLXArray([Float(1), 2, 3, 4]).reshaped(1, 2, 1, 2)
            let scan = Mamba2Scan.callAsFunction(
                x: x, aLog: .zeros([1]),
                b: .ones([1, 2, 1, 3]), c: .ones([1, 2, 1, 3]), d: MLXArray([Float(0.5)]),
                dt: .zeros([1, 2, 1]), dtBias: .zeros([1]), maximumTimeStep: 0.01)
            close(scan.output, [0.53, 1.06, 1.6197015, 2.179403], tolerance: 5e-7)
            #expect(scan.state.dtype == .float32)
        }
    }

    @Test func nativePromptsAndSpeakerSegments() throws {
        let defaults = STTGenerateParameters()
        #expect(
            NemotronTranscribeModel.resolvePrompt(
                .init(), parameters: defaults, audioToken: "<|audio|>")?.hasSuffix(" <|audio|>")
                == true)
        let target = STTGenerateParameters(language: "German", task: "translate")
        #expect(
            NemotronTranscribeModel.resolvePrompt(
                .init(), parameters: target, audioToken: "<|audio|>")
                == "Translate the speech in this audio into German. <|audio|>")
        #expect(
            NemotronTranscribeModel.resolvePrompt(
                .init(task: .questionAnswering), parameters: defaults, audioToken: "<|audio|>")
                == "<|audio|>")
        #expect(
            NemotronTranscribeModel.resolvePrompt(
                .init(prompt: "<|audio|><|audio|>"), parameters: defaults, audioToken: "<|audio|>")
                == nil)
        let turns = NemotronTranscribeModel.turns(prompt: "hello <|audio|>", options: .init())
        #expect(turns[0] == "<|im_start|>system\n<|im_end|>\n")
        #expect(turns[2] == "<|im_start|>assistant\n<think></think>")
        let segments = try #require(
            NemotronTranscribeModel.speakerSegments("<spk:0>Hello <ev:laugh> <spk:1>Hi"))
        #expect(segments.count == 2)
        #expect(segments[0]["speaker"] as? Int == 0)
        #expect(segments[0]["text"] as? String == "Hello <ev:laugh>")
        #expect(segments[0]["start"] == nil)
        #expect(NemotronTranscribeModel.speakerSegments("no speaker markup") == nil)
    }

    @Test func localCheckpointAliasesUseTheSharedFactory() async throws {
        for alias in ["nemotron_transcribe", "nemo_speechlm", "nemotron3_5_transcribe"] {
            let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
                UUID().uuidString)
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: directory) }
            let source = Self.configuration.replacingOccurrences(
                of: "\"model_type\":\"nemotron_transcribe\"",
                with: "\"model_type\":\"\(alias)\"")
            try Data(source.utf8).write(to: directory.appendingPathComponent("config.json"))
            do {
                _ = try await STT.loadModel(modelRepo: directory.path)
                Issue.record("An empty checkpoint loaded")
            } catch STTError.modelNotInitialized(let message) {
                #expect(message == "No Nemotron Transcribe safetensors found")
            }
        }
    }

    @Test(arguments: [Device.cpu, Device.gpu])
    func ctcHeadMatchesPython(device: Device) throws {
        try Device.withDefaultDevice(device) {
            let m = try model(timestamps: true)
            let head = try #require(m.ctcTimestamp)
            let states = ((MLXArray(0 ..< 112) % 23) - 11).asType(.float32).reshaped(1, 7, 16) / 32
            let logits = head.decoder(states)
            #expect(logits.shape == [1, 14, 65])
            // Python CPU reference with the same exact generated parameters.
            close(
                logits,
                [
                    -0.0105101299, 0.0064385701, 0.0202217717, 0.0190904215,
                    0.0069566965, -0.0047822446, -0.0108795092, -0.0021039546, 0.0291477591,
                    0.0890451819, 0.1439055055, -0.0899884626, -0.0696466640, -0.0565057695,
                    -0.0377994590, -0.0100390837, -0.0225506239,
                ], tolerance: device == .cpu ? 2e-6 : 5e-5)
            let original = Dictionary(
                uniqueKeysWithValues: head.parameters().flattened().map {
                    ("ctc_timestamp." + $0.0, $0.1)
                })
            var source = original
            source["ctc_timestamp.decoder.upsample.weight"] = head.decoder.upsample!.weight
                .transposed(2, 0, 1)
            source["ctc_timestamp.decoder.decoder_layers.0.weight"] = head.decoder.projections[0]
                .weight.expandedDimensions(axis: -1)
            source["ctc_timestamp.tokenizer.model_proto"] = .zeros([5], dtype: .uint8)
            source["ctc_timestamp.decoder.upsample_factor"] = MLXArray(Int32(2))
            let once = NemotronTranscribeModel.sanitize(
                weights: source, includeTimestamps: true, timestampUpsamplingFactor: 2)
            let twice = NemotronTranscribeModel.sanitize(
                weights: once, includeTimestamps: true, timestampUpsamplingFactor: 2)
            #expect(Set(once.keys) == Set(original.keys))
            for (key, value) in original { #expect(arrayEqual(twice[key]!, value).item(Bool.self)) }
            let aligned = m.perception.withAlignment(audio())
            #expect(aligned.states.shape == [1, 8, 16])
            #expect(aligned.probabilities.shape == [1, 16, 2])
            #expect(arrayEqual(aligned.embeddings, m.perception(audio())).item(Bool.self))
            let columns = [64, 1, 2]
            let compact = head.compactLogProbabilities(
                states: states, columns: columns, configuration: m.config)
            #expect(compact.count == 14 * 3)
            let first = head.decoder(states[0..., ..<5, 0...])[0, ..<8, 0...].asType(.float32)
            let last = head.decoder(states[0..., 3..., 0...])[0, 2..., 0...].asType(.float32)
            let joined = concatenated([first, last], axis: 0)
            let expected =
                joined[0..., MLXArray(columns.map(Int32.init))]
                - logSumExp(joined, axis: -1, keepDims: true)
            #expect(abs(MLXArray(compact).reshaped(14, 3) - expected).max().item(Float.self) < 1e-6)
        }
    }

    @Test func ctcRepeatsAndImpossibleAlignment() throws {
        let labels = [0, 1, 1, 0, 1, 0, 2, 2, 0]
        var logits = Array(repeating: Float(-20), count: labels.count * 3)
        for (i, label) in labels.enumerated() { logits[i * 3 + label] = 0 }
        let path = try NemotronWordAlignment.ctcPath(
            logProbabilities: logits, classes: 3, target: [1, 1, 2],
            activity: Array(repeating: 1, count: 9))
        #expect(path.path == [0, 1, 1, 2, 3, 4, 5, 5, 6])
        #expect(path.frames == Array(0 ..< 9))
        #expect(throws: NemotronWordAlignment.Failure.self) {
            try NemotronWordAlignment.ctcPath(
                logProbabilities: Array(logits.prefix(6)), classes: 3, target: [1, 1],
                activity: [1, 1])
        }
        #expect(throws: NemotronWordAlignment.Failure.self) {
            try NemotronWordAlignment.ctcPath(
                logProbabilities: logits, classes: 3, target: [1],
                activity: Array(repeating: 0, count: 9))
        }
    }

    @Test func longAlignmentUsesBoundedSearch() throws {
        let target = (0 ..< 400).map { ($0 * 31) % 39 + 1 }
        var logits = Array(repeating: Float(-20), count: 1600 * 40)
        for frame in 0 ..< 1600 {
            logits[frame * 40 + (frame % 4 == 1 ? target[frame / 4] : 0)] = 0
        }
        let path = try NemotronWordAlignment.ctcPath(
            logProbabilities: logits, classes: 40, target: target,
            activity: Array(repeating: 1, count: 1600))
        var allAligned = true
        for i in 0 ..< 400 {
            let observed: Int = (path.path[i * 4 + 1] - 1) / 2
            if observed != i { allAligned = false }
        }
        #expect(allAligned)
    }

    @Test func timestampsPreserveTranscriptSpeakerIDs() throws {
        let labels = [0, 1, 1, 0, 2, 2, 0, 0]
        var logits = Array(repeating: Float(-20), count: 24)
        for (frame, label) in labels.enumerated() { logits[frame * 3 + label] = 0 }
        let probabilities = MLXArray([Float(0), 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0])
            .reshaped(8, 2)
        let streams = [
            NemotronWordAlignment.Stream(speaker: 0, words: [.init(text: "Hello", ids: [1])]),
            .init(speaker: 1, words: [.init(text: "Hi", ids: [2])]),
        ]
        let result = try NemotronWordAlignment.alignCompact(
            streams: streams, logProbabilities: logits, classes: 3,
            nativeProbabilities: probabilities, duration: 0.08, configuration: config(),
            upsamplingFactor: 2)
        #expect(result.count == 2)
        #expect(result[0]["text"] as? String == "Hello" && result[0]["speaker"] as? Int == 0)
        #expect(result[1]["text"] as? String == "Hi" && result[1]["speaker"] as? Int == 1)
        for word in result {
            let start = try #require(word["start"] as? Double)
            let end = try #require(word["end"] as? Double)
            #expect(0 <= start && start <= end && end <= 0.08)
        }
    }

    @Test func characterAlignmentAndUnsupportedThai() throws {
        let config: Config = ["tokenizer_class": "TokenizersBackend", "unk_token": "<unk>"]
        let data: Config = [
            "model": [
                "type": "BPE", "vocab": ["<unk>": 0, "你": 1, "好": 2, "あ": 3, "▁": 4],
                "merges": [], "unk_token": "<unk>",
            ], "added_tokens": [],
        ]
        let tokenizer = try AutoTokenizer.from(tokenizerConfig: config, tokenizerData: data)
        let chinese = try NemotronWordAlignment.tokenize(
            "你好。", tokenizer: .huggingFace(tokenizer), blank: 5, language: nil)
        #expect(chinese[0].words.map(\.text) == ["你", "好"])
        let japanese = try NemotronWordAlignment.tokenize(
            "あ", tokenizer: .huggingFace(tokenizer), blank: 5, language: nil)
        #expect(japanese[0].words.map(\.text) == ["あ"])
        #expect(throws: NemotronWordAlignment.Failure.self) {
            try NemotronWordAlignment.tokenize(
                "ไทย", tokenizer: .huggingFace(tokenizer), blank: 5, language: nil)
        }
    }

    @Test(arguments: [false, true])
    func olderExportsLoadWithoutTimestampWeights(configuredHead: Bool) async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        var source =
            try JSONSerialization.jsonObject(with: Data(Self.configuration.utf8)) as! [String: Any]
        if configuredHead {
            source["ctc_timestamps"] = [
                "decoder_config": ["feat_in": 16, "num_classes": 64, "n_layers": 1, "n_heads": 2]
            ]
        }
        try JSONSerialization.data(withJSONObject: source).write(
            to: directory.appendingPathComponent("config.json"))
        try writeTinyTokenizer(directory)
        let weights = Dictionary(uniqueKeysWithValues: try model().parameters().flattened())
        try MLX.save(arrays: weights, url: directory.appendingPathComponent("model.safetensors"))
        let loaded = try await NemotronTranscribeModel.fromModelDirectory(directory)
        #expect(!loaded.supportsWordTimestamps)
        #expect(loaded.ctcTimestamp == nil)
        #expect(loaded.parameters().flattened().count == 107)
    }

    private func writeTinyTokenizer(_ directory: URL) throws {
        let vocabulary = Dictionary(
            uniqueKeysWithValues: (0 ..< 64).map { ($0 == 0 ? "<unk>" : "t\($0)", $0) })
        let tokenizer: [String: Any] = [
            "model": ["type": "BPE", "vocab": vocabulary, "merges": [], "unk_token": "<unk>"],
            "added_tokens": [
                [
                    "id": 64, "content": "<|audio|>", "special": true, "single_word": false,
                    "lstrip": false, "rstrip": false, "normalized": false,
                ]
            ],
        ]
        try JSONSerialization.data(withJSONObject: tokenizer).write(
            to: directory.appendingPathComponent("tokenizer.json"))
        try Data(#"{"tokenizer_class":"TokenizersBackend","unk_token":"<unk>"}"#.utf8).write(
            to: directory.appendingPathComponent("tokenizer_config.json"))
    }

    private func tinyAlignmentProto() -> Data {
        func varint(_ value: Int) -> [UInt8] {
            var value = value
            var bytes: [UInt8] = []
            while value >= 128 {
                bytes.append(UInt8(value & 127) | 128)
                value >>= 7
            }
            bytes.append(UInt8(value))
            return bytes
        }
        func field(_ number: Int, _ bytes: [UInt8]) -> [UInt8] {
            varint(number * 8 + 2) + varint(bytes.count) + bytes
        }
        var proto: [UInt8] = []
        for (i, piece) in ["<unk>", "▁", "h", "i", "▁h", "▁hi"].enumerated() {
            var score = Float(-i).bitPattern.littleEndian
            let scoreBytes = withUnsafeBytes(of: &score) { Array($0) }
            let value = field(1, Array(piece.utf8)) + [0x15] + scoreBytes + [0x18, i == 0 ? 2 : 1]
            proto += field(1, value)
        }
        proto += field(2, [0x18, 2])  // SentencePiece BPE trainer model type.
        return Data(proto)
    }

    @Test(arguments: [false, true])
    func embeddedAndSidecarAlignmentTokenizers(sidecar: Bool) async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        var source =
            try JSONSerialization.jsonObject(with: Data(Self.configuration.utf8)) as! [String: Any]
        source["ctc_timestamps"] = [
            "decoder_config": [
                "feat_in": 16, "num_classes": 64, "n_layers": 2, "n_heads": 2,
                "ff_expansion": 0.5, "qk_norm": true, "rotary_fraction": 0.5,
                "upsampling_factor": 2, "residual_scale": 0.1,
            ]
        ]
        try JSONSerialization.data(withJSONObject: source).write(
            to: directory.appendingPathComponent("config.json"))
        try writeTinyTokenizer(directory)
        let proto = tinyAlignmentProto()
        var weights = Dictionary(
            uniqueKeysWithValues: try model(timestamps: true).parameters().flattened())
        if sidecar {
            let path = directory.appendingPathComponent("alignment_tokenizer")
            try FileManager.default.createDirectory(at: path, withIntermediateDirectories: true)
            try proto.write(to: path.appendingPathComponent("tokenizer.model"))
        } else {
            weights["ctc_timestamp.tokenizer.model_proto"] = MLXArray(Array(proto))
        }
        try MLX.save(arrays: weights, url: directory.appendingPathComponent("model.safetensors"))
        let loaded = try await NemotronTranscribeModel.fromModelDirectory(directory)
        #expect(loaded.supportsWordTimestamps)
        let tokenizer = try #require(loaded.alignmentTokenizer?.value)
        #expect(tokenizer.encode(text: " ｈｉ", addSpecialTokens: false) == [1, 5])
        #expect(tokenizer.convertTokenToId("▁hi") == 5)
    }
}
