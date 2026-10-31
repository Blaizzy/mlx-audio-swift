import Foundation
import Testing
import MLX
import MLXNN
@testable import MLXAudioCore
@testable import MLXAudioCodecs
@testable import MLXAudioTTS

@Suite(.serialized)
struct Maya2Tests {
    private func configuration() -> Maya2Configuration {
        var c = Maya2Configuration()
        c.backbone = Qwen3Configuration(hiddenSize: 16, hiddenLayers: 2, intermediateSize: 32,
            attentionHeads: 4, kvHeads: 2, headDim: 4, vocabularySize: 64, rmsNormEps: 1e-6,
            ropeTheta: 1_000_000, tieWordEmbeddings: true, maxPositionEmbeddings: 256)
        c.depth.hiddenSize = 16; c.depth.numAttentionHeads = 4
        c.depth.intermediateSize = 32; c.depth.numHiddenLayers = 2; c.depth.embeddingRank = 4
        c.codebookSize = 8; c.speakerAnchorHidden = 8
        return c
    }
    private func binaryWeights(_ module: Module) throws {
        let suffixes = ["n1.weight", "n2.weight", "n.weight", "norm.weight", "layernorm.weight", "q_norm.weight", "k_norm.weight"]
        let weights = Dictionary(uniqueKeysWithValues: module.parameters().flattened().map { key, value in
            (key, suffixes.contains(where: { key.hasSuffix($0) }) ? MLXArray.ones(value.shape)
                : ((MLXArray(0..<value.size) % 17).asType(.float32) - 8).reshaped(value.shape) / 64)
        })
        try module.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
    }
    private func close(_ actual: MLXArray, _ values: [Float], tolerance: Float = 3e-6) {
        #expect(max(abs(actual.reshaped(-1) - MLXArray(values))).item(Float.self) < tolerance)
    }

    @Test func backboneAudioAndSpeakerMatchPythonCPU() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = Maya2Model(configuration())
            try binaryWeights(model)
            let x = ((MLXArray(0..<48) % 13).asType(.float32) - 6).reshaped(1, 3, 16) / 16
            let (hidden, logits) = model(x)
            close(hidden[0..., -1, 0...], [-0.11390943080186844, -0.2806151211261749, -0.5840604305267334, -0.7922290563583374, 1.0928423404693604, 1.8270334005355835, 1.6074765920639038, -0.1116081178188324, -0.5325703024864197, -0.9974082112312317, -1.3382889032363892, -1.4141027927398682, -0.5961816906929016, 1.2396703958511353, 0.7653059363365173, 0.5134117603302002])
            close(logits[0..., -1, 0...], [0.022942543029785156, -0.011764124035835266, -0.09075199067592621, -0.2503424882888794, -0.4652278423309326, -0.1793910562992096, 0.3014652132987976, 0.7240016460418701, 0.6899063587188721])
            let codes = (MLXArray(0..<32) % 8).reshaped(1, 32)
            close(model.audioIn.frames(codes), [-0.19821177423000336, -0.5946353673934937, -0.991058886051178, -1.3874825239181519, 1.5856941938400269, 1.1892707347869873, 0.7928470969200134, 0.3964235484600067, 0.0, -0.3964235484600067, -0.7928470969200134, -1.1892707347869873, -1.5856941938400269, 1.3874825239181519, 0.991058886051178, 0.5946353673934937])
            let refs = MLXArray(0..<192).asType(.float32).reshaped(1, 1, 192) / 256
            let anchor = try #require(model.speakerAnchor)
            close(anchor(refs), [-0.09799927473068237, -0.11750322580337524, -0.08439333736896515, -0.08186210691928864, -0.07415808737277985, -0.046220988035202026, -0.0369047150015831, -0.010579870082437992, -0.007286418229341507, 0.02506124973297119, 0.005940726026892662, 0.0607023686170578, 0.033055488020181656, 0.09634348750114441, 0.09133943915367126, 0.13198460638523102])
            let cache: [KVCache] = [KVCacheSimple(), KVCacheSimple()]
            var chunks: [MLXArray] = []
            for i in 0..<3 { chunks.append(model(x[0..., i..<(i+1), 0...], cache: cache).0) }
            #expect(max(abs(hidden - concatenated(chunks, axis: 1))).item(Float.self) < 3e-6)
        }
    }
    @Test func groupedDepthCacheMatchesPythonAndFullPrefixes() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = Maya2Model(configuration())
            try binaryWeights(model)
            let hidden = ((MLXArray(32..<48) % 13).asType(.float32) - 6).reshaped(1, 16) / 16
            let codes = (MLXArray(0..<32) % 8).reshaped(1, 32)
            let logits = model.depth(hidden: hidden, codes: codes)
            close(logits[0], [-0.028500676155090332, 0.12320993095636368, 0.0735938772559166, -0.049944259226322174, -0.28714367747306824, -0.4303073287010193, -0.24442335963249207, 0.192292720079422])
            close(logits[30], [0.1154392883181572, 0.14464813470840454, 0.0656004399061203, -0.1478678584098816, -0.44021841883659363, -0.521386444568634, -0.21067702770233154, 0.2593778669834137])
            let actual = model.depth.generate(hidden: hidden, cb0: MLXArray([Int32(2)]), memory: model.depth.speakerMemory(), temperature: 0, topK: 0)
            #expect(actual.asArray(Int32.self) == [2, 7, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 1, 1, 2, 2, 3, 4, 4, 4, 5, 6, 6, 6, 7, 7, 7, 7])
            let memory = ((MLXArray(0..<32) % 13).asType(.float32) - 6).reshaped(1, 2, 16) / 16
            let cached = model.depth.generate(hidden: hidden, cb0: MLXArray([Int32(2)]), memory: memory, temperature: 0, topK: 0)
            var expected = [MLXArray([Int32(2)])]
            for level in 1..<32 {
                let prefix = model.depth.prefix(hidden: hidden, codes: stacked(expected, axis: 1), level: level)
                let h = model.depth.run(prefix, group: Maya2DepthDecoder.groupOfLevel[level], memory: memory)
                expected.append(argMax(model.depth.heads[level - 1](h), axis: -1))
            }
            #expect(cached.asArray(Int32.self) == stacked(expected, axis: 1).asArray(Int32.self))
        }
    }
    @Test func sourceAndExportWeightsLoadStrictly() throws {
        let model = Maya2Model(configuration())
        try binaryWeights(model)
        var source = Dictionary(uniqueKeysWithValues: model.parameters().flattened().map { ("model." + $0.0, $0.1) })
        source["model.backbone.lm.lm_head.weight"] = model.backbone.lm.model.embedTokens.weight
        let weights = model.sanitize(weights: source)
        let restored = Maya2Model(configuration())
        try restored.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        #expect(weights["depth.null_mem"]?.dtype == .float16)
        #expect(weights["backbone.lm.lm_head.weight"] == nil)
        #expect(weights.count == model.parameters().flattened().count)
    }
    @Test func promptLanguageBoundariesAndFeedbackSlots() throws {
        let tokens = ["<|audio|>": 2, "<|lang:en|>": 3, "<|lang:hi|>": 4]
        var encoded: [String] = []
        let named = try Maya2Prompt.build(text: " Hello. ", language: "en", voice: "Ahana", referenceText: nil,
            referenceCodes: nil, referenceLanguage: nil, numCodebooks: 32,
            encode: { encoded.append($0); return [5, 6] }, tokenID: { tokens[$0] })
        #expect(encoded == ["{Ahana}: Hello."])
        #expect(named.tokens == [3, 5, 6, 2])
        #expect(named.sentinels == [3])
        let codes = (MLXArray(0..<64) % 8).reshaped(32, 2)
        for lang in ["en", "hi"] {
            let p = try Maya2Prompt.build(text: "Hello.", language: "en", voice: nil, referenceText: "Reference.",
                referenceCodes: codes, referenceLanguage: lang, numCodebooks: 32,
                encode: { _ in [5, 6] }, tokenID: { tokens[$0] })
            #expect(p.tokens == (lang == "en" ? [3,5,6,2,0,0,0,5,6,2] : [4,5,6,2,0,0,0,3,5,6,2]))
            #expect(Array(p.kinds[4...6]) == [1, 1, 3])
            #expect(p.codes[4] == codes[0..., 0].asArray(Int32.self))
        }
        #expect(throws: Error.self) {
            try Maya2Prompt.build(text: "Hello", language: "en", voice: nil, referenceText: nil,
                referenceCodes: nil, referenceLanguage: nil, numCodebooks: 32, encode: { _ in [] }, tokenID: { _ in nil })
        }
    }
    @Test func publishedConfigurationAndTypeAliases() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory.appendingPathComponent("backbone"), withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data(#"{"model_type":"maya3","sample_rate":24000}"#.utf8).write(to: directory.appendingPathComponent("config.json"))
        var b = try #require(JSONSerialization.jsonObject(with: JSONEncoder().encode(configuration().backbone)) as? [String: Any])
        b.removeValue(forKey: "rope_theta"); b["rope_parameters"] = ["rope_theta": 500000]
        try JSONSerialization.data(withJSONObject: b).write(to: directory.appendingPathComponent("backbone/config.json"))
        try Data(#"{"model":{"speaker_anchor":{"enabled":true,"hidden":8,"input_dim":192}}}"#.utf8).write(to: directory.appendingPathComponent("maya3_config.json"))
        let c = try Maya2Model.configuration(from: directory)
        #expect(c.backbone.ropeTheta == 500000)
        #expect(c.speakerAnchorHidden == 8)
        #expect(TTS.resolveModelType(modelRepo: "maya-research/maya2-native") == "maya2")
        #expect(TTS.resolveModelType(modelRepo: "local", modelType: "maya3") == "maya3")
        var bad = c; bad.numCodebooks = 33
        #expect(throws: Error.self) { try bad.validate() }
    }
    @Test func speechBrainFrontendMatchesPythonCPU() throws {
        try Device.withDefaultDevice(.cpu) {
            let signal = sin(MLXArray(0..<1600).asType(.float32) * 0.125) * 0.25
            let mel = EcapaSpeakerEncoder.logMel(signal)
            #expect(mel.shape == [11, 80])
            let selection = mel[MLXArray([0, 5, 10])][0..., MLXArray([0, 1, 20, 60, 79])]
            close(selection, [-0.3681882619857788, 5.746517658233643, -4.0463547706604, -30.26392936706543, -33.74787521362305, -40.28010559082031, -34.16539764404297, -32.868316650390625, -44.51681137084961, -45.05073928833008, -7.206303596496582, -1.0915977954864502, 1.38624906539917, -10.267485618591309, -10.808938980102539], tolerance: 0.002)
        }
    }
    @Test func centeredSpeakerStatisticsPreserveSmallVariance() throws {
        try Device.withDefaultDevice(.cpu) {
            let pooling = AttentiveStatisticsPooling(channels: 2, attentionChannels: 2,
                globalContext: true, centeredStatistics: true)
            let weights = Dictionary(uniqueKeysWithValues: pooling.parameters().flattened().map {
                ($0.0, MLXArray.zeros($0.1.shape))
            })
            try pooling.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
            pooling.train(false)
            let delta: Float = 1.0 / 64
            let x = MLXArray([Float(1024), -1024, 1024 + delta, -1024 + delta,
                1024 - delta, -1024 - delta]).reshaped(1, 3, 2)
            let deviation = (Float(2) / 3).squareRoot() / 64
            close(pooling(x), [1024, -1024, deviation, deviation])
        }
    }
    @Test func samplingKeepsEOSAndPresenceSupport() {
        var options = Maya2GenerationOptions(); options.presencePenalty = 0.8; options.presenceWindow = 2
        let raw = MLXArray([Float(1), 3, 2, -1]).reshaped(1, 4)
        let filtered = Maya2Model.filteredLogits(raw, history: [0, 1, 1], parameters: .init(temperature: 1, topK: 2), options: options)
        close(filtered[0..., 1...2], [2.2, 2])
        #expect(!isFinite(filtered[0, 0]).item(Bool.self))
    }
}
