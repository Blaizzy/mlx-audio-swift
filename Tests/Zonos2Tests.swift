import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXAudioCore
@testable import MLXAudioTTS

@Suite(.serialized)
struct Zonos2Tests {
    func tiny() -> Zonos2Configuration {
        var c = Zonos2Configuration()
        c.nLayers = 4
        c.dim = 8
        c.headDim = 4
        c.nHeads = 2
        c.nKVHeads = 1
        c.multipleOf = 4
        c.nCodebooks = 2
        c.codebookSize = 8
        c.eoaID = 8
        c.audioPadID = 9
        c.speakerEnabled = false
        c.moeNExperts = 3
        c.moeRouterDim = 4
        c.moeStartFromLayer = 1
        c.moeEndFromLayer = 1
        c.specialTopKLayers = ["2": 2]
        return c
    }
    func synthetic(_ model: Zonos2Model) throws {
        let weights = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().map { key, value in
                if key.hasSuffix("temp") || key.hasSuffix("router_states_scale") {
                    return (key, MLXArray.ones(value.shape))
                }
                var a = ((MLXArray(0..<value.size) % 17).asType(.float32) - 8) / 64
                if key.hasSuffix("norm.weight") || key.hasSuffix("rmsnorm_eda.weight") { a = a + 1 }
                return (key, a.reshaped(value.shape))
            })
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
    }
    @Test func configurationAndNulls() throws {
        let c = try JSONDecoder().decode(
            Zonos2Configuration.self,
            from: Data(
                #"{"n_kv_heads":null,"speaker_lda_dim":null,"speaker_encoder_path":null,"special_topk_layers":{"26":2}}"#
                    .utf8))
        #expect(c.numKVHeads == 16)
        #expect(c.speakerLDADim == nil)
        #expect(c.speakerEncoderPath == nil)
        #expect(c.intermediateSize == 3072)
        #expect(c.qualityBucketCounts == [12, 12, 12, 8, 8, 8])
        try c.validate()
        var bad = c
        bad.nCodebooks = 10
        #expect(throws: Zonos2Error.self) { try bad.validate() }
        #expect(TTS.resolveModelType(modelRepo: "test/Zyphra-ZONOS2-4bit") == "zonos2")
    }
    @Test func conditioningPromptAndDelay() throws {
        let c = Zonos2Configuration()
        var options = Zonos2Options()
        options.speakingRateBucket = 3
        options.cleanSpeakerBackground = true
        let rows = try Zonos2Prompt.build(text: "é", config: c, options: options, speaker: true)
        #expect(rows.prefix(5).map { $0.last! } == [519, 516, 518, 451, 511])
        #expect(rows[5...8].map { $0.last! } == [2, 387, 361, 3])
        #expect(rows.count == 26)
        #expect(Array(rows[9].prefix(3)) == [568, 1025, 1025])
        #expect(Array(rows[11].prefix(3)) == [568, 778, 338])
        options.qualityBuckets = ["invalid": 0]
        #expect(throws: Zonos2Error.self) {
            try Zonos2Prompt.build(text: "a", config: c, options: options, speaker: false)
        }
        var small = tiny()
        small.nCodebooks = 3
        let codes = Zonos2Prompt.codes(
            [[1, 9, 9], [2, 3, 9], [4, 5, 6], [7, 0, 1]], config: small, eosFrame: 2)
        #expect(codes.asArray(Int32.self) == [1, 3, 6, 2, 5, 1])
    }
    @Test func endOfAudioDrain() {
        var state = Zonos2GenerationState(codebooks: 3, eoaID: 8)
        state.append([1, 2, 3], ignoreEOS: false)
        state.append([1, 8, 8], ignoreEOS: false)
        #expect(state.eosFrame == 0)
        #expect(!state.finished)
        for _ in 0..<3 { state.append([8, 8, 8], ignoreEOS: false) }
        #expect(state.finished)
        var ignored = Zonos2GenerationState(codebooks: 3, eoaID: 8)
        ignored.append([8, 8, 8], ignoreEOS: true)
        #expect(ignored.eosFrame == nil)
    }
    @Test func cpuTransformerAndRouterMatchPython() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try Zonos2Model(config: tiny())
            try synthetic(model)
            let ids = MLXArray([1, 2, 3, 3, 2, 4, 2, 4, 5]).reshaped(1, 3, 3)
            let result = model(ids)
            let expected: [Float] = [
                -0.07252727448940277, 0.33125826716423035, 0.14222711324691772, 0.28079932928085327,
                -0.1635666787624359, 0.23033399879932404, -0.3726981580257416, 0.17986349761486053,
                -0.4663761258125305, 0.12938891351222992, -0.40712830424308777, 0.07891141623258591,
                -0.2582070529460907, 0.028432101011276245, -0.020217115059494972, -0.02204783633351326,
                0.38170960545539856, -0.07252727448940277, 0.33125826716423035, 0.14222711324691772,
                0.14884823560714722, -0.2083427906036377, 0.5476237535476685, -0.16369855403900146,
                0.6005523800849915, -0.11905138194561005, 0.2523118257522583, -0.0744020938873291,
                -0.017566196620464325, -0.02975151687860489, -0.209550678730011, 0.014899606816470623,
                -0.32827314734458923, 0.05955046787858009, -0.32479819655418396, 0.10420025140047073,
                -0.25298333168029785, 0.14884823560714722, -0.2083427906036377, 0.5476237535476685,
                0.1994221955537796, -0.24468167126178741, 0.5666895508766174, -0.18917618691921234,
                0.6016181707382202, -0.1336655169725418, 0.2566136419773102, -0.078151173889637,
                -0.014486159197986126, -0.02263469807803631, -0.21074168384075165, 0.03288239613175392,
                -0.3431997299194336, 0.08839859068393707, -0.35303980112075806, 0.14391236007213593,
                -0.30018046498298645, 0.1994222104549408, -0.24468167126178741, 0.5666896104812622,
            ]
            #expect(result.shape == [1, 3, 2, 10])
            #expect(max(abs(result.reshaped(-1) - MLXArray(expected))).item(Float.self) < 2e-6)
            let x = MLXArray(0..<24).asType(.float32).reshaped(1, 3, 8) / 32
            let routed = model.layers[2].feedForward.router!(x, state: MLXArray.ones([1, 3, 4]) / 16)
            let weights: [Float] = [
                0.32937777042388916, 0.3333176374435425, 0.32943639159202576, 0.3333180546760559,
                0.32945123314857483, 0.33331823348999023,
            ]
            #expect(routed.1.asArray(Int32.self) == [2, 1, 2, 1, 2, 1])
            var quantileConfig = tiny()
            quantileConfig.moeBalancingStrategy = " Quantile "
            let quantileModel = try Zonos2Model(config: quantileConfig)
            try synthetic(quantileModel)
            let quantileRoute = quantileModel.layers[2].feedForward.router!(
                x, state: MLXArray.ones([1, 3, 4]) / 16)
            #expect(quantileRoute.1.asArray(Int32.self) == [0, 1, 0, 1, 0, 1])
            #expect(max(abs(routed.0.reshaped(-1) - MLXArray(weights))).item(Float.self) < 1e-7)
            let cache: [KVCache] = (0..<4).map { _ in KVCacheSimple() }
            var parts: [MLXArray] = []
            for i in 0..<3 {
                let part = model(ids[0..., i..<(i + 1), 0...], cache: cache)
                eval(part)
                parts.append(part)
            }
            #expect(max(abs(result - concatenated(parts, axis: 1))).item(Float.self) < 2e-6)
        }
    }
    @Test func mixedQuantizedDirectDirectoryRoundTrip() async throws {
        var c = tiny()
        c.dim = 32
        c.headDim = 16
        c.multipleOf = 32
        c.ffnDimMultiplier = 2
        let model = try Zonos2Model(config: c)
        try synthetic(model)
        var specification: [String: Any] = ["bits": 4, "group_size": 32]
        quantize(model: model) { path, _ in
            if path.hasSuffix("attention.wq") {
                specification[path] = ["bits": 8, "group_size": 32]
                return (32, 8, .affine)
            }
            if path.contains(".experts.") { return (32, 4, .affine) }
            return nil
        }
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        var object = try JSONSerialization.jsonObject(with: JSONEncoder().encode(c)) as! [String: Any]
        object["quantization"] = specification
        try JSONSerialization.data(withJSONObject: object).write(
            to: folder.appendingPathComponent("config.json"))
        var weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        // Published chunked projections retain a leading two-way axis.
        for (key, value) in weights where key.hasSuffix("wkv.weight") || key.hasSuffix("w_in.weight") {
            weights[key] = value.reshaped(2, value.dim(0) / 2, value.dim(1))
        }
        try save(arrays: weights, url: folder.appendingPathComponent("model.safetensors"))
        let loaded = try await TTS.loadModel(modelRepo: folder.path)
        let second = try #require(loaded as? Zonos2Model)
        let ids = MLXArray([1, 2, 3, 3, 2, 4]).reshaped(1, 2, 3)
        #expect(max(abs(model(ids) - second(ids))).item(Float.self) < 1e-6)
        #expect(second.layers[1].feedForward.experts!.gate is MingTinyQuantizedSwitchLinear)
    }
    @Test func samplingFiltersAndRepetition() {
        Device.withDefaultDevice(.cpu) {
            var state = Zonos2GenerationState(codebooks: 2, eoaID: 8)
            state.append([3, 0, 519], ignoreEOS: false)
            let logits = MLXArray([0, 1, 2, 3, 3, 2, 1, 0]).asType(.float32).reshaped(2, 4)
            var p = GenerateParameters(temperature: 0, repetitionPenalty: 2, repetitionContextSize: 4)
            let sampled = Zonos2Sampling.sample(
                logits, state: state, parameters: p, options: .init(), textVocab: 519, step: 0)
            #expect(sampled == [2, 1, 519])
            p.temperature = 1
            p.topK = 2
            p.minP = 0.5
            p.topP = 0.8
            let probabilities = Zonos2Sampling.probabilities(
                logits, state: state, parameters: p, repetitionCodebooks: 2)
            #expect(max(abs(sum(probabilities, axis: -1) - 1)).item(Float.self) < 1e-6)
            #expect(probabilities[0, 0].item(Float.self) == 0)
            #expect(probabilities[1, 3].item(Float.self) == 0)
            p.temperature = 1.15
            p.topK = 3
            p.topP = 0.8
            p.minP = 0.18
            p.seed = 17
            #expect(
                Zonos2Sampling.sample(
                    logits, state: state, parameters: p, options: .init(), textVocab: 519, step: 0) == [
                        3, 1, 519,
                    ])
            #expect(
                Zonos2Sampling.sample(
                    logits, state: state, parameters: p, options: .init(), textVocab: 519, step: 5) == [
                        1, 1, 519,
                    ])
        }
    }
    @Test func englishNormalization() {
        #expect(
            Zonos2Text.normalize(
                "On 2026-10-26, I paid $12.50 for 2 kg at 9:05 PM. NASA & AI.", language: "en")
                == "On october twenty sixth twenty twenty six, I paid twelve dollars and fifty cents for two kilograms at nine oh five p m. n a s a and a i."
        )
        #expect(
            Zonos2Text.normalize("Call (415) 555-0123, 1/2 or 3/4, 21st, 03.14, 50%.", language: "en_us")
                == "Call four one five five five five zero one two three, one half or three quarters, twenty first, three point one four, fifty percent."
        )
        #expect(Zonos2Text.normalize("Hello 42", language: "fr") == "Hello 42")
        #expect(Zonos2Text.normalize("-9223372036854775808th", language: "en") == "-9223372036854775808th")
    }
    @Test func speakerFrontendMatchesPython() throws {
        try Device.withDefaultDevice(.cpu) {
            let waveform = ((MLXArray(0..<2048) % 64).asType(.float32) - 32) / 64
            let mel = try Zonos2Model.speakerMel(waveform, sampleRate: 24000)
            let expected: [Float] = [
                -0.8722888231277466, -0.9891712665557861, -0.9716832637786865, -0.9459160566329956,
                -0.910423219203949, -0.8680707216262817, -0.7862192392349243, -0.6460476517677307,
                -0.6107012033462524, -0.4731525778770447, -0.26173943281173706, 0.12641793489456177,
            ]
            #expect(mel.shape == [1, 8, 128])
            #expect(max(abs(mel[0, 0, 0..<12] - MLXArray(expected))).item(Float.self) < 2e-5)
        }
        #expect(throws: Zonos2Error.self) {
            try Zonos2Model.speakerMel(MLXArray.zeros([384]), sampleRate: 24000)
        }
    }
    @Test func batchInputValidation() async throws {
        let model = try Zonos2Model(config: tiny())
        let empty = try await model.generateBatch(texts: [])
        #expect(empty.isEmpty)
        await #expect(throws: Zonos2Error.self) {
            try await model.generateBatch(texts: ["one", "two"], refAudios: [MLXArray.zeros([500])])
        }
        await #expect(throws: Zonos2Error.self) {
            try await model.generateBatch(texts: ["one"], speakerEmbeddings: MLXArray.zeros([2, 2048]))
        }
    }

}
