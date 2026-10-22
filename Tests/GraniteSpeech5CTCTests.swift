import Foundation
import MLX
import MLXNN
import Testing
import MLXAudioCore
@testable import MLXAudioSTT

@Suite(.serialized)
struct GraniteSpeech5CTCTests {
    private func tinyConfig() -> GraniteSpeech5CTCConfig {
        var config = GraniteSpeech5CTCConfig()
        config.vocabSize = 16
        config.encoderConfig.vocabSize = 16
        config.encoderConfig.hiddenSize = 16
        config.encoderConfig.intermediateSize = 32
        config.encoderConfig.numHiddenLayers = 4
        config.encoderConfig.numAttentionHeads = 2
        config.encoderConfig.numKeyValueHeads = 2
        config.encoderConfig.numMelBins = 4
        config.encoderConfig.headDim = 8
        config.encoderConfig.maxPositionEmbeddings = 16
        config.encoderConfig.contextSize = 8
        config.encoderConfig.convKernelSize = 3
        return config
    }
    private func fixture() throws -> (GraniteSpeech5CTCModel, [String:MLXArray]) {
        let dataURL = try #require(Bundle.module.url(forResource: "granite_speech5_ctc_reference", withExtension: "safetensors", subdirectory: "media"))
        let config = tinyConfig()
        try config.validate()
        let arrays = try MLX.loadArrays(url: dataURL)
        let model = GraniteSpeech5CTCModel(config)
        let weights = Dictionary(uniqueKeysWithValues: arrays.compactMap { key,value in
            key.hasPrefix("weights.") ? (String(key.dropFirst(8)),value) : nil
        })
        #expect(Set(model.parameters().flattened().map(\.0)) == Set(weights.keys))
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        model.train(false)
        return (model,arrays)
    }
    private func compare(_ actual: MLXArray, _ expected: MLXArray, tolerance: Float) {
        #expect(actual.shape == expected.shape)
        let error = abs(actual.asType(.float32) - expected.asType(.float32))
        print("Granite tiny reference max_abs=\(error.max().item(Float.self)) mean_abs=\(error.mean().item(Float.self))")
        #expect(error.max().item(Float.self) < tolerance)
    }
    private func writeTokenizer(_ directory: URL, size: Int) throws {
        let pieces = ["<unk>", "▁caf", "<0xC3>", "<0xA9>", ".", "<0xE2>", "<0x96>", "<0x81>"] + (8..<size).map { "piece\($0)" }
        let vocabulary = Dictionary(uniqueKeysWithValues: pieces.enumerated().map { ($0.element,$0.offset) })
        let json: [String:Any] = ["model": ["type":"BPE", "vocab":vocabulary],
            "added_tokens":[["id":0,"content":"<unk>","special":true]]]
        try JSONSerialization.data(withJSONObject: json).write(to: directory.appendingPathComponent("tokenizer.json"))
    }
    @Test func pythonFrontendAndFullConformerWithOddLengthsAndMasks() throws {
        let (model,f) = try fixture()
        compare(try GraniteSpeech5CTCFrontend.features(f["reference.audio"]!, numMelBins: 4), f["reference.features"]!, tolerance: 5e-5)
        compare(model(f["reference.input"]!), f["reference.logits"]!, tolerance: 2e-4)
        compare(model(f["reference.input"]!, attentionMask: f["reference.mask"]!), f["reference.masked_logits"]!, tolerance: 2e-4)
        #expect(model(f["reference.input"]!).shape == [2,4,16])
    }
    @Test func deltaEdgesCTCCollapseAndShortFrontend() throws {
        compare(try GraniteSpeech5CTCFrontend.deltas(MLXArray([Float(1),3,7]).reshaped(3,1)), MLXArray([Float(1),3,2]).reshaped(3,1), tolerance: 1e-6)
        #expect(GraniteSpeech5CTCModel.collapse([0,1,1,0,1,2,2,0]) == [1,1,2])
        #expect(GraniteSpeech5CTCModel.collapse([]).isEmpty)
        #expect(try GraniteSpeech5CTCFrontend.features(.zeros([321]), numMelBins: 4).shape == [1,16])
        #expect(throws: STTError.self) { try GraniteSpeech5CTCFrontend.features(.zeros([0])) }
        #expect(throws: STTError.self) { try GraniteSpeech5CTCFrontend.deltas(.ones([3,1]), windowLength: 2) }
    }
    @Test func configDefaultsAndInvalidArchitecture() throws {
        let implicitEncoder = try JSONDecoder().decode(GraniteSpeech5CTCConfig.self, from: Data(#"{"vocab_size":16}"#.utf8))
        #expect(implicitEncoder.encoderConfig.vocabSize == 16)
        let json = #"{"vocab_size":16,"encoder_config":{"vocab_size":16,"hidden_size":24,"num_attention_heads":3,"num_hidden_layers":2,"subsample_layers":[0]}}"#
        var config = try JSONDecoder().decode(GraniteSpeech5CTCConfig.self, from: Data(json.utf8))
        #expect(config.encoderConfig.headDim == 8)
        #expect(config.encoderConfig.numKeyValueHeads == 3)
        try config.validate()
        config.encoderConfig.subsampleLayers = [0,0]
        #expect(throws: STTError.self) { try config.validate() }
        config.encoderConfig.subsampleLayers = [0]
        config.encoderConfig.vocabSize = 17
        #expect(throws: STTError.self) { try config.validate() }
    }
    @Test func tokenizerByteFallbackAndStrictLoading() throws {
        let (model,_) = try fixture()
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("granite-ctc-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try JSONEncoder().encode(model.config).write(to: directory.appendingPathComponent("config.json"))
        try writeTokenizer(directory, size: 16)
        let tokenizer = try GraniteSpeech5CTCTokenizer(directory: directory, vocabSize: 16)
        #expect(tokenizer.decode([0,1,2,3,4]) == "café.")
        #expect(tokenizer.decode([5,6,7]) == "▁")
        var weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        try MLX.save(arrays: weights, url: directory.appendingPathComponent("model.safetensors"))
        let loaded = try GraniteSpeech5CTCModel.fromModelDirectory(directory)
        let result = try loaded.transcribe(audio: .ones([1]))
        #expect(result.language == "en")
        #expect(throws: STTError.self) { try loaded.transcribe(audio: .zeros([0])) }
        weights.removeValue(forKey: "encoder.input_linear.weight")
        try MLX.save(arrays: weights, url: directory.appendingPathComponent("model.safetensors"))
        #expect(throws: (any Error).self) { try GraniteSpeech5CTCModel.fromModelDirectory(directory) }
    }
    @Test func sanitizationPreservesPreparedConvolutions() {
        let sanitized = GraniteSpeech5CTCModel.sanitize(weights: [
            "encoder.layers.0.conv.depthwise_conv.weight": .zeros([32,1,3]),
            "encoder.layers.1.conv.depthwise_conv.weight": .zeros([32,3,1]),
            "encoder.layers.0.conv.norm.num_batches_tracked": MLXArray(0),
            "encoder.out.weight": .ones([16,16])])
        #expect(sanitized["encoder.layers.0.conv.depthwise_conv.weight"]?.shape == [32,3,1])
        #expect(sanitized["encoder.layers.1.conv.depthwise_conv.weight"]?.shape == [32,3,1])
        #expect(sanitized["encoder.layers.0.conv.norm.num_batches_tracked"] == nil)
    }
    @Test func quantizedInputProjectionKeepsFloatingPointFeatures() throws {
        let (fixtureModel,_) = try fixture()
        var config = fixtureModel.config
        config.encoderConfig.numMelBins = 8
        config.encoderConfig.hiddenSize = 64
        config.encoderConfig.headDim = 32
        let model = GraniteSpeech5CTCModel(config)
        model.encoder.update(modules: ModuleChildren.unflattened([
            ("input_linear", QuantizedLinear(model.encoder.input, groupSize: 32, bits: 4))
        ]))
        #expect(model.encoder.input.weight.dtype == .uint32)
        let logits = model(.ones([1,19,32]))
        #expect(logits.dtype.isFloatingPoint)
        #expect(MLX.all(isFinite(logits)).item(Bool.self))
    }
    @Test func commonLoaderAndFullBufferStreaming() async throws {
        let (fixtureModel,_) = try fixture()
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("granite-stream-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try JSONEncoder().encode(fixtureModel.config).write(to: directory.appendingPathComponent("config.json"))
        try writeTokenizer(directory, size: 16)
        try MLX.save(arrays: Dictionary(uniqueKeysWithValues: fixtureModel.parameters().flattened()),
            url: directory.appendingPathComponent("model.safetensors"))
        let model = try await STT.loadModel(modelRepo: directory.path, modelType: "granite_speech5_ctc")
        #expect(model is GraniteSpeech5CTCModel)
        var finalCount = 0
        for try await event in model.generateStream(audio: .ones([1280])) {
            if case .result = event { finalCount += 1 }
        }
        #expect(finalCount == 1)
        await #expect(throws: STTError.self) {
            for try await _ in model.generateStream(audio: .zeros([0])) {}
        }
    }
}

@Suite(.serialized)
struct GraniteSpeech5CTCCheckpointTests {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["GRANITE_TURBOCTC_MODEL_DIR"] != nil))
    func realCheckpointParityAndMultiwordTranscription() async throws {
        let env = ProcessInfo.processInfo.environment
        let modelPath = try #require(env["GRANITE_TURBOCTC_MODEL_DIR"])
        let references = try #require(env["GRANITE_TURBOCTC_REFERENCES"]).split(separator: ":").map(String.init)
        let model = try await GraniteSpeech5CTCModel.fromPretrained(modelPath)
        for path in references {
            let directory = URL(fileURLWithPath: path)
            let f = try MLX.loadArrays(url: directory.appendingPathComponent("reference.safetensors"))
            let metadata = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("reference.json"))) as? [String:Any])
            let features = try GraniteSpeech5CTCFrontend.features(f["audio"]!).expandedDimensions(axis: 0)
            let logits = model(features).asType(.float32), expected = f["logits"]!.asType(.float32)
            let error = logits - expected
            let relative = (sqrt(mean(error.square())) / maximum(sqrt(mean(expected.square())),1e-8)).item(Float.self)
            let cosine = (sum(logits * expected) / sqrt(sum(logits.square()) * sum(expected.square()))).item(Float.self)
            let frontendError = abs(features - f["features"]!).max().item(Float.self)
            print("Granite checkpoint \(directory.lastPathComponent): frontend_max=\(frontendError), logits_max=\(abs(error).max().item(Float.self)), RMS_relative=\(relative), cosine=\(cosine)")
            #expect(frontendError < 1e-3)
            #expect(relative < 0.01)
            #expect(cosine > 0.9999)
            let expectedText = try #require(metadata["text"] as? String)
            let output = try model.transcribe(audio: f["audio"]!)
            print("Granite Swift: \(output.text)\nGranite Python: \(expectedText)")
            #expect(!output.text.isEmpty)
            #expect(wordErrorRate(output.text, expectedText) <= 0.1)
            if let audioPath = metadata["audio"] as? String {
                let (_,audio) = try loadAudioArray(from: URL(fileURLWithPath: audioPath), sampleRate: 16000)
                let independent = try model.transcribe(audio: audio)
                print("Granite independent Swift resampling: \(independent.text)")
                #expect(wordErrorRate(independent.text, expectedText) <= 0.1)
            }
        }
    }
    private func wordErrorRate(_ actual: String, _ expected: String) -> Double {
        func words(_ text: String) -> [String] {
            text.lowercased().components(separatedBy: CharacterSet.alphanumerics.inverted).filter { !$0.isEmpty }
        }
        let a = words(actual), b = words(expected)
        var costs = Array(0...b.count)
        for (i,word) in a.enumerated() {
            var row = [i + 1]
            for (j,ref) in b.enumerated() { row.append(min(row[j] + 1, costs[j + 1] + 1, costs[j] + (word == ref ? 0 : 1))) }
            costs = row
        }
        return Double(costs.last!) / Double(max(b.count,1))
    }
}
