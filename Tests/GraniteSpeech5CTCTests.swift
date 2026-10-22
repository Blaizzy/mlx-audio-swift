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
    private func tinyModel() throws -> GraniteSpeech5CTCModel {
        let config = tinyConfig()
        try config.validate()
        let model = GraniteSpeech5CTCModel(config)
        model.train(false)
        return model
    }
    private func writeTokenizer(_ directory: URL, size: Int) throws {
        let pieces = ["<unk>", "▁caf", "<0xC3>", "<0xA9>", ".", "<0xE2>", "<0x96>", "<0x81>"] + (8..<size).map { "piece\($0)" }
        let vocabulary = Dictionary(uniqueKeysWithValues: pieces.enumerated().map { ($0.element,$0.offset) })
        let json: [String:Any] = ["model": ["type":"BPE", "vocab":vocabulary],
            "added_tokens":[["id":0,"content":"<unk>","special":true]]]
        try JSONSerialization.data(withJSONObject: json).write(to: directory.appendingPathComponent("tokenizer.json"))
    }
    @Test func maskedPaddingDoesNotAffectOddLengthBatch() throws {
        let model = try tinyModel()
        let features = arange(2 * 19 * 16).asType(.float32).reshaped(2,19,16) / 100
        let mask = MLXArray(Array(repeating: true, count: 32) + Array(repeating: false, count: 6)).reshaped(2,19)
        let changedPadding = MLX.where(mask.expandedDimensions(axis: -1), features, features + 1000)
        let logits = model(features, attentionMask: mask)
        let changedPaddingLogits = model(changedPadding, attentionMask: mask)
        #expect(logits.shape == [2,4,16])
        #expect(MLX.all(isFinite(logits)).item(Bool.self))
        #expect(abs(logits - changedPaddingLogits).max().item(Float.self) == 0)
    }
    @Test func deltaEdgesCTCCollapseAndShortFrontend() {
        let deltas = GraniteSpeech5CTCFrontend.deltas(MLXArray([Float(1),3,7]).reshaped(3,1))
        #expect(deltas.shape == [3,1])
        #expect(deltas.asArray(Float.self) == [1,3,2])
        #expect(GraniteSpeech5CTCModel.collapse([0,1,1,0,1,2,2,0]) == [1,1,2])
        #expect(GraniteSpeech5CTCModel.collapse([]).isEmpty)
        #expect(GraniteSpeech5CTCFrontend.features(.zeros([321]), numMelBins: 4).shape == [1,16])
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
        let model = try tinyModel()
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
        let features = MLXArray.ones([1,19,16])
        #expect(abs(model(features) - loaded(features)).max().item(Float.self) == 0)
        let result = loaded.generate(audio: .ones([1]))
        #expect(result.language == "en")
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
        var config = tinyConfig()
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
        let nativeModel = try tinyModel()
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("granite-stream-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try JSONEncoder().encode(nativeModel.config).write(to: directory.appendingPathComponent("config.json"))
        try writeTokenizer(directory, size: 16)
        try MLX.save(arrays: Dictionary(uniqueKeysWithValues: nativeModel.parameters().flattened()),
            url: directory.appendingPathComponent("model.safetensors"))
        let model = try await STT.loadModel(modelRepo: directory.path, modelType: "granite_speech5_ctc")
        #expect(model is GraniteSpeech5CTCModel)
        let audio = MLXArray.ones([1280])
        let expected = model.generate(audio: audio)
        var finalCount = 0
        var streamedText = ""
        for try await event in model.generateStream(audio: audio) {
            switch event {
            case .token(let text): streamedText += text
            case .info: break
            case .result(let result):
                finalCount += 1
                #expect(result.text == expected.text)
                #expect(result.language == expected.language)
                #expect(result.generationTokens == expected.generationTokens)
            }
        }
        #expect(finalCount == 1)
        #expect(streamedText == expected.text)
    }
}

@Suite(.serialized)
struct GraniteSpeech5CTCCheckpointTests {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["GRANITE_TURBOCTC_MODEL_DIR"] != nil))
    func localCheckpointTranscription() throws {
        let modelPath = try #require(ProcessInfo.processInfo.environment["GRANITE_TURBOCTC_MODEL_DIR"])
        let model = try GraniteSpeech5CTCModel.fromModelDirectory(URL(fileURLWithPath: modelPath))
        let audioURL = try #require(Bundle.module.url(forResource: "intention", withExtension: "wav", subdirectory: "media"))
        let (_,audio) = try loadAudioArray(from: audioURL, sampleRate: model.sampleRate)
        let output = model.generate(audio: audio)
        let words = output.text.lowercased().components(separatedBy: CharacterSet.alphanumerics.inverted).filter { !$0.isEmpty }
        #expect(words == ["intention"])
        #expect(output.language == "en")
    }
}
