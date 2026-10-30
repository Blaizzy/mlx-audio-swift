import Foundation
@preconcurrency import MLX
import MLXNN
import MLXAudioCore
import MLXAudioCodecs
import Testing
import Tokenizers
import Hub
@testable import MLXAudioTTS

@Suite("OuteTTS", .serialized)
struct OuteTTSTests {
    private func speaker(text: String = "Hello") -> OuteTTSSpeaker {
        .init(text: text, words: [.init(word: text, duration: 0.2, c1: [1, 2], c2: [3, 4],
                                      features: .init(energy: 10, spectralCentroid: 20, pitch: 30))])
    }

    @Test func promptNormalizationAndConditioning() throws {
        let profile = speaker()
        let prompt = try OuteTTSPrompt.completion(text: "  “Hi”\tthere… — yes  ", speaker: profile)
        #expect(prompt == "<|im_start|>\n<|text_start|>Hello. \"Hi\" there... - yes<|text_end|>\n<|audio_start|>\n<|word_start|>Hello.<|features|><|t_0.20|><|energy_10|><|spectral_centroid_20|><|pitch_30|><|code|><|c1_1|><|c2_3|><|c1_2|><|c2_4|><|word_end|>\n<|word_start|>")
        #expect(profile.words[0].word == "Hello")
        #expect(try OuteTTSPrompt.completion(text: "こんにちは", speaker: speaker(text: "猫"))
            .contains("猫。こんにちは"))
        #expect(try OuteTTSPrompt.completion(text: "Again", speaker: profile) ==
                OuteTTSPrompt.completion(text: "Again", speaker: profile))
    }

    @Test func chunksAndIncompleteFrames() {
        #expect(OuteTTSPrompt.chunks("one two. three four! five?", maxWords: 3) == ["one two", "three four five"])
        #expect(OuteTTSPrompt.chunks("one two three four", maxWords: 2) == ["one two three four"])
        #expect(OuteTTSPrompt.chunks(" .！？ ").isEmpty)
        let map = OuteTTSTokenMap(c1: [10: 2, 11: 3], c2: [20: 4], stop: [30])
        #expect(map.extract([99, 10, 20, 11]) == [[2], [4]])
        #expect(map.extract([11]) == [[], []])
    }

    @Test func samplingExcludesInvalidCodecAndChatTokens() throws {
        let strings = ["<unk>", "<|c1_0|>", "<|c1_1|>", "<|c2_0|>", "<|c2_1|>",
                       "<|c1_1024|>", "<|im_end|>", "<|word_start|>", "<|features|>",
                       "<|code|>", "<|im_start|>"]
        let vocab = Dictionary(uniqueKeysWithValues: strings.enumerated().map { ($0.element, $0.offset) })
        let added: [[String: Any]] = strings.enumerated().map { ["id": $0.offset, "content": $0.element, "special": true] }
        let tokenizerData: [NSString: Any] = ["model": ["type": "BPE", "unk_token": "<unk>", "merges": [], "vocab": vocab], "added_tokens": added]
        let tokenizer = try AutoTokenizer.from(
            tokenizerConfig: ["tokenizer_class": "GPT2Tokenizer", "unk_token": "<unk>"],
            tokenizerData: Config(tokenizerData))
        let map = try OuteTTSTokenMap(tokenizer: tokenizer, vocabularySize: 12, codebookSize: 2)
        #expect(map.forbidden == [5, 10, 11])
        #expect(map.stop == [6])
        #expect(map.extract([1, 3, 2, 4]) == [[0, 1], [0, 1]])
    }

    @Test func speakerPersistenceAndValidation() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("voice.json")
        try speaker().save(to: url)
        #expect(try OuteTTSSpeaker.load(from: url) == speaker())
        var invalid = speaker()
        invalid.words[0].c1.append(1024)
        #expect(throws: (any Error).self) { try invalid.validate() }
        invalid = speaker()
        invalid.words[0].duration = .nan
        #expect(throws: (any Error).self) { try invalid.validate() }
    }

    @Test func genericBackboneDispatchUsesOuteVocabulary() throws {
        #expect(TTS.resolveModelType(modelRepo: "mlx-community/OuteTTS-1.0-0.6B-fp16", modelType: "qwen3") == "outetts")
        #expect(TTS.resolveModelType(modelRepo: "mlx-community/VyvoTTS", modelType: "qwen3") == "qwen3")
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data("{\"model_type\":\"qwen3\"}".utf8).write(to: directory.appendingPathComponent("config.json"))
        #expect(try !TTS.isOuteTTS(directory))
        try Data("{\"<|c1_0|>\":1,\"<|c2_0|>\":2,\"<|word_start|>\":3,\"<|features|>\":4}".utf8)
            .write(to: directory.appendingPathComponent("added_tokens.json"))
        #expect(try TTS.isOuteTTS(directory))
    }

    @Test func descriptSpeechConfigAliasesAndNativeRoundTrip() throws {
        let alias = try JSONDecoder().decode(DescriptDACConfig.self, from: Data("""
        {"encoder_hidden_size":8,"downsampling_ratios":[2],"hidden_size":16,
         "decoder_hidden_size":16,"upsampling_ratios":[2],"sampling_rate":24000,
         "n_codebooks":2,"codebook_size":8,"codebook_dim":2,"hop_length":512}
        """.utf8))
        #expect(alias.sampleRate == 24_000)
        #expect(alias.encoderDim == 8 && alias.latentDim == 16)
        #expect(DescriptDAC(config: alias).hopLength == 2)
        let native = try JSONDecoder().decode(DescriptDACConfig.self, from: JSONEncoder().encode(alias))
        #expect(native.encoderRates == alias.encoderRates)
        #expect(native.decoderDim == alias.decoderDim && native.sampleRate == alias.sampleRate)
    }

    private func tinyModel() throws -> OuteTTSModel {
        let config = Qwen3Configuration(hiddenSize: 8, hiddenLayers: 1, intermediateSize: 16,
            attentionHeads: 2, kvHeads: 1, headDim: 4, vocabularySize: 4,
            rmsNormEps: 1e-6, ropeTheta: 10_000)
        let model = OuteTTSModel(config)
        var weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened().map {
            ($0.0, MLXArray.zeros($0.1.shape))
        })
        weights["model.model.embed_tokens.weight"] = MLXArray([
            Float(1), 0, 0, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 0,
        ]).reshaped(4, 8)
        weights["model.model.norm.weight"] = MLXArray.ones([8])
        weights["model.lm_head.weight"] = MLXArray([
            Float(0), 0, 0, 0, 0, 0, 0, 0,
            1, 0, 1, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 0,
        ]).reshaped(4, 8)
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        model.tokenizer = try AutoTokenizer.from(
            tokenizerConfig: ["tokenizer_class": "GPT2Tokenizer", "unk_token": "<unk>", "fuse_unk": true],
            tokenizerData: ["model": ["type": "BPE", "unk_token": "<unk>", "merges": [], "vocab": ["<unk>": 0]],
                            "added_tokens": [["id": 0, "content": "<unk>", "special": true]]])
        model.tokenMap = OuteTTSTokenMap(c1: [1: 0], c2: [2: 1], stop: [3])
        model.codec = DescriptDAC(config: .init(encoderDim: 4, encoderRates: [2], latentDim: 8,
            decoderDim: 8, decoderRates: [2], nCodebooks: 2, codebookSize: 4, codebookDim: 2, sampleRate: 24_000))
        model.train(false)
        return model
    }

    @Test func tinyGenerationStreamingAndReferenceError() async throws {
        let model = try tinyModel()
        let parameters = GenerateParameters(maxTokens: 4, temperature: 0)
        let audio = try await model.generate(text: "Hello", voice: nil, refAudio: nil, refText: nil,
                                             language: nil, generationParameters: parameters)
        #expect(audio.shape == [4])
        #expect(audio.asArray(Float.self).allSatisfy { $0.isFinite })
        var samples = 0, tokens = [Int]()
        for try await event in model.generateStream(text: "Hello", voice: nil, refAudio: nil, refText: nil,
            language: nil, generationParameters: parameters, streamingInterval: 0.01) {
            if case .audio(let chunk) = event { samples += chunk.size }
            if case .token(let id) = event { tokens.append(id) }
        }
        #expect(samples == audio.size)
        #expect(tokens == [1, 2, 1, 2])
        await #expect(throws: (any Error).self) {
            try await model.generate(text: "Hi", voice: nil, refAudio: MLXArray.ones([100]).materialized(),
                                     refText: "Reference", language: nil, generationParameters: parameters)
        }
    }

    @Test func sanitizerPreservesPublishedAndUnwrappedWeights() throws {
        let model = try tinyModel()
        let value = MLXArray.ones([8])
        #expect(model.sanitize(weights: ["model.norm.weight": value])["model.model.norm.weight"] != nil)
        #expect(model.sanitize(weights: ["model.model.norm.weight": value])["model.model.norm.weight"] != nil)
    }

    @Test func quantizedBackboneWeightsLoadWithoutCasting() throws {
        let config = Qwen3Configuration(hiddenSize: 32, hiddenLayers: 1, intermediateSize: 64,
            attentionHeads: 2, kvHeads: 1, headDim: 16, vocabularySize: 32,
            rmsNormEps: 1e-6, ropeTheta: 10_000)
        let original = OuteTTSModel(config)
        quantize(model: original, groupSize: 32, bits: 4)
        let weights = Dictionary(uniqueKeysWithValues: original.parameters().flattened())
        #expect(weights["model.model.embed_tokens.weight"]?.dtype == .uint32)
        let copy = OuteTTSModel(config)
        quantize(model: copy, groupSize: 32, bits: 4)
        try copy.update(parameters: ModuleParameters.unflattened(copy.sanitize(weights: weights)), verify: .all)
        let input = MLXArray([Int32(0), 1]).reshaped(1, 2)
        let delta = abs(original.backbone(input) - copy.backbone(input)).max().item(Float.self)
        #expect(delta == 0)
    }

    @Test func speakerCreationUsesCodecFramesAndWordAlignment() throws {
        let model = try tinyModel()
        model.codec = DescriptDAC(config: .init(encoderDim: 2, encoderRates: [2, 4, 5, 8], latentDim: 32,
            decoderDim: 32, decoderRates: [8, 5, 4, 2], nCodebooks: 2, codebookSize: 4, codebookDim: 2,
            sampleRate: 24_000))
        let time = MLXArray(0..<24_000).asType(.float32) / 24_000
        let audio = 0.1 * sin(time * Float.pi * 440)
        let profile = try model.createSpeaker(audio: audio, text: "Hello world", words: [
            .init(word: "Hello", start: 0.1, end: 0.4), .init(word: "world", start: 0.4, end: 0.8),
        ])
        #expect(profile.words.map { $0.c1.count } == [30, 45])
        #expect(profile.words.map { $0.c2.count } == [30, 45])
        #expect(profile.words[0].duration == 0.4)
        #expect(profile.words[1].duration == 0.6)
        #expect(profile.globalFeatures?.energy == 7)
        #expect(throws: (any Error).self) {
            try model.createSpeaker(audio: audio, text: "Hello", words: [.init(word: "Hello", start: 0.5, end: 0.2)])
        }
    }

    @Test func finiteNormalizationAndFeatureBins() throws {
        let t = MLXArray(0..<24_000).asType(.float32) / 24_000
        let audio = 0.1 * sin(t * Float.pi * 440)
        let normalized = try OuteTTSAudio.normalize(audio)
        #expect(normalized.size == 24_000)
        #expect(normalized.asArray(Float.self).allSatisfy { $0.isFinite })
        let features = OuteTTSAudio.features(audio)
        #expect(features.energy == 7)
        #expect(features.spectralCentroid < 3)
        #expect((0...100).contains(features.pitch))
    }
}
