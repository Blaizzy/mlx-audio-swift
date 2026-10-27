import Foundation
import HuggingFace
import MLX
import MLXNN
import Testing
@testable import MLXAudioCore
@testable import MLXAudioTTS

@Suite("LongCat AudioDiT", .serialized)
struct LongCatTests {
    func configuration() -> LongCatConfiguration {
        LongCatConfiguration(ditDim: 8, ditDepth: 2, ditHeads: 2, ditFfMult: 2, ditTextDim: 8,
            repaDitLayer: 1, latentDim: 2, samplingRate: 8000, latentHop: 2,
            maxWavDuration: 3,
            vaeConfig: .init(channels: 4, cMults: [1], strides: [2], latentDim: 2,
                encoderLatentDim: 4, downsamplingRatio: 2, sampleRate: 8000),
            textEncoderConfig: .init(vocabSize: 32, dModel: 8, dKv: 4, dFf: 16, numLayers: 2, numHeads: 2))
    }

    @Test func defaultsAndRouting() throws {
        let c = try JSONDecoder().decode(LongCatConfiguration.self, from: Data(#"{"model_type":"longcat_audiodit","vae_config":{},"text_encoder_config":{}}"#.utf8))
        #expect(c.ditDim == 1536)
        #expect(c.vaeConfig.strides == [2, 4, 4, 8, 8])
        #expect(c.textEncoderConfig.vocabSize == 256384)
        #expect(TTS.resolveModelType(modelRepo: "mlx-community/LongCat-AudioDiT-1B-bf16") == "longcat_audiodit")
        try LongCatModel.validate(configuration())
        var invalid = configuration(); invalid.ditHeads = 3
        #expect(throws: (any Error).self) { try LongCatModel.validate(invalid) }
    }

    @Test func relativeBucketsAndFrontend() {
        let positions = [-129, -128, -16, -8, -7, -1, 0, 1, 7, 8, 16, 128]
        let actual = positions.map { LongCatT5Attention.relativeBucket($0, buckets: 32, maxDistance: 128) }
        #expect(actual == [15, 15, 10, 8, 7, 1, 0, 17, 23, 24, 26, 31])
        #expect(LongCatModel.normalizedText("  Hello, ‘WORLD’!\n") == "hello, world !")
        #expect(abs(LongCatModel.approximateDuration("你好!", maximum: 30) - 0.63) < 1e-6)
        let promptTime = Double(12 * 2048) / 24000
        let ratio = promptTime / LongCatModel.approximateDuration("Hello, this is", maximum: 30)
        let duration = LongCatModel.approximateDuration("today is warm turning to rain.", maximum: 30) * ratio
        #expect(Int(duration * 24000 / 2048) == 25)
    }

    @Test func pixelShuffleAndShortcuts() {
        let x = MLXArray(0..<16).asType(.float32).reshaped(1, 4, 4)
        let packed = longCatPixelUnshuffle(x, factor: 2)
        #expect(packed.shape == [1, 2, 8])
        #expect(packed.asArray(Float.self) == [0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15])
        #expect(longCatPixelShuffle(packed, factor: 2).asArray(Float.self) == x.asArray(Float.self))
        let down = LongCatShortcut(4, 4, factor: 2, upsample: false)(x)
        #expect(down.asArray(Float.self) == [2, 3, 4, 5, 10, 11, 12, 13])
    }

    @Test func architectureShapesAndReferencePadding() throws {
        let model = LongCatModel(configuration())
        let keys = Set(model.parameters().flattened().map(\.0))
        #expect(keys.contains("text_encoder.shared.weight"))
        #expect(keys.contains("text_encoder.block.0.SelfAttention.relative_attention_bias.weight"))
        #expect(keys.contains("transformer.blocks.0.adaln_scale_shift"))
        #expect(keys.contains("transformer.text_conv_layer.0.dwconv_weight"))
        #expect(keys.contains("vae.decoder.layers.1.layers.1.weight"))
        let weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened().map { key, value in
            (key, MLXArray.zeros(value.shape))
        })
        try model.update(parameters: .unflattened(weights), verify: .all)
        let encoded = model.encodeText(MLXArray([1, 2, 3]).reshaped(1, 3), mask: .ones([1, 3]))
        #expect(encoded.shape == [1, 3, 8])
        let prompt = try model.encodePromptAudio(MLXArray([Float(0), 0, 0]), noise: .zeros([1, 5, 2]))
        #expect(prompt.shape == [1, 2, 2])
        let output = model.transformer(.ones([1, 4, 2]), text: encoded, textLength: MLXArray([Float(3)]),
            time: MLXArray([Float(0.5)]), mask: .ones([1, 4], dtype: .bool),
            textMask: .ones([1, 3], dtype: .bool), latentCondition: .zeros([1, 4, 2]), returnLayer: 1)
        #expect(output.0.shape == [1, 4, 2])
        #expect(output.1?.shape == [1, 4, 8])
        #expect(model.vae.decode(output.0).shape == [1, 8, 1])
    }

    @Test func overlapStreamingEmitsEachSampleOnce() throws {
        let model = LongCatModel(configuration())
        let latent = MLXArray.zeros([1, 20, 2])
        var chunks: [Int] = []
        // 8-frame windows with 2-frame overlap; the last window starts before
        // the previous final window's end, so termination must be explicit.
        try model.decodeChunks(latent, options: .init(chunkSeconds: 0.002, overlapSeconds: 0.0005)) {
            chunks.append($0.size)
        }
        #expect(chunks == [12, 12, 16])
        #expect(chunks.reduce(0, +) == 40)
    }

    @Test func binaryWeightStagesMatchPythonCPU() throws {
        func stages(device: Device) throws -> [MLXArray] {
            try Device.withDefaultDevice(device) {
                let model = LongCatModel(configuration())
                let weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened().map { key, value in
                    let values = (0..<value.size).map { Float(($0 * 13 + value.size) % 17 - 8) / 64 }
                    return (key, MLXArray(values).reshaped(value.shape))
                })
                try model.update(parameters: .unflattened(weights), verify: .all)
                let text = model.encodeText(MLXArray([1, 2, 3]).reshaped(1, 3), mask: .ones([1, 3]))
                let latent = MLXArray(0..<8).asType(.float32).reshaped(1, 4, 2) / 16
                let result = model.transformer(latent, text: text, textLength: MLXArray([Float(3)]),
                    time: MLXArray([Float(0.5)]), mask: .ones([1, 4], dtype: .bool),
                    textMask: .ones([1, 3], dtype: .bool), latentCondition: .zeros([1, 4, 2]), returnLayer: 1)
                let waveform = model.vae.decode(latent)
                let mean = model.vae.encode(MLXArray(0..<10).asType(.float32).reshaped(1, 10, 1) / 16,
                    noise: .zeros([1, 5, 2]))
                let stages = [text, result.0, result.1!, waveform, mean]
                stages.forEach { eval($0) }
                return stages
            }
        }
        // Python CPU, weights ((i * 13 + tensor.size) % 17 - 8) / 64.
        let expected: [[Float]] = [
            [-0.5339537263, 0.482311964, -0.4450075626, 0.4556770623, -0.5029114485, -0.01324772835, -0.4034422636, 0.9605740309, -0.5013720989, -0.2119386196, -0.399643302, 0.8789111972, -0.3405721784, -0.5583003163, -0.2325555086, 1.365470648, -0.4592234492, -0.5431665778, -0.6792361736, 1.139293909, -0.2398474067, -0.753266871, -0.105553031, 1.640999556],
            [-0.3156888485, -0.1417248249, -0.3156758547, -0.1417450309, -0.3156625032, -0.1417656839, -0.3156473637, -0.1417867839],
            [-0.02772817574, -0.1276248395, -0.2956870496, 0.09544943273, 0.05866420269, -0.09958322346, -0.1382605731, 0.2420877516, -0.02771393023, -0.1275996417, -0.2956638634, 0.09548117965, 0.05865076557, -0.09959173203, -0.1382919997, 0.2420631796, -0.02769924328, -0.1275738329, -0.2956402898, 0.09551335871, 0.05863726139, -0.09960024059, -0.1383240819, 0.2420380265, -0.02768335491, -0.1275462061, -0.2956170738, 0.09554569423, 0.05862310901, -0.09960909188, -0.1383575648, 0.2420119196],
            [-0.1883828044, -0.271386981, -0.3610068262, -0.4487870634, -0.4842054248, -0.4810942113, -0.4648024738, -0.3975982368],
            [-0.6055149436, -0.3762910664, -0.7119745016, -0.3030982018, -0.6885576844, -0.306637615, -0.6226952076, -0.3971369863, -0.4605501294, -0.5294211507],
        ]
        let cpu = try stages(device: .cpu)
        let gpu = try stages(device: .gpu)
        for i in expected.indices {
            let error = abs(cpu[i].flattened() - MLXArray(expected[i])).max().item(Float.self)
            #expect(error < 2e-6, "Python CPU stage \(i) error: \(error)")
            let rounding = abs(gpu[i] - cpu[i]).max().item(Float.self)
            #expect(rounding < 5e-4, "GPU stage \(i) rounding: \(rounding)")
        }
    }

    @Test func projectedGuidanceUsesMomentumAndProjection() {
        let conditional = MLXArray([Float(0.125), 0.25, 0.5, 0.75]).reshaped(1, 2, 2)
        let unconditional = MLXArray([Float(0.0625), 0.125, 0.25, 0.5]).reshaped(1, 2, 2)
        var momentum: MLXArray?
        let first = LongCatModel.projectedGuidance(conditional, unconditional, scale: 4, momentum: &momentum)
        let second = LongCatModel.projectedGuidance(conditional, unconditional, scale: 4, momentum: &momentum)
        #expect(abs(first.flattened() - MLXArray([Float]([0.2763157785, 0.552631557, 1.105263114, 1.157894611]))).max().item(Float.self) < 1e-6)
        #expect(abs(second.flattened() - MLXArray([Float]([0.2309210598, 0.4618421197, 0.9236842394, 1.035526276]))).max().item(Float.self) < 1e-6)
    }

    @Test func syntheticCheckpointLoadsAndSynthesizesThroughSharedLoader() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        var c = configuration()
        c.samplingRate = 20; c.vaeConfig.sampleRate = 20
        c.textEncoderModel = "mlx-audio-tests/longcat-umt5"
        let model = LongCatModel(c)
        let weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened().map { key, value in
            (key, MLXArray.zeros(value.shape))
        })
        try JSONEncoder().encode(c).write(to: directory.appendingPathComponent("config.json"))
        try MLX.save(arrays: weights, url: directory.appendingPathComponent("model.safetensors"))
        let tokenizer = #"{"model":{"type":"Unigram","unk_id":3,"vocab":[["<pad>",0],["</s>",0],["<s>",0],["<unk>",0],["▁",-1],["h",-1],["i",-1]]}}"#
        try Data(tokenizer.utf8).write(to: directory.appendingPathComponent("tokenizer.json"))
        let loaded = try await TTS.loadModel(modelRepo: directory.path)
        #expect(loaded is LongCatModel)
        let audio = try await loaded.generate(text: "hi", voice: nil, refAudio: nil, refText: nil,
            language: nil, generationParameters: .init(seed: 10))
        #expect(audio.shape == [2])
        #expect(audio.asArray(Float.self) == [0, 0])
        var samples = 0, information = 0
        for try await event in loaded.generateStream(text: "hi", voice: nil, refAudio: nil,
            refText: nil, language: nil, generationParameters: .init(seed: 10)) {
            switch event {
            case .audio(let chunk): samples += chunk.size
            case .info: information += 1
            default: break
            }
        }
        #expect(samples == 2)
        #expect(information == 1)
        // Resolve the required external tokenizer from the caller's cache,
        // with a fake repository ID that cannot silently use the global cache.
        try FileManager.default.removeItem(at: directory.appendingPathComponent("tokenizer.json"))
        let cache = HubCache(cacheDirectory: directory.appendingPathComponent("hub"))
        let repo: Repo.ID = "mlx-audio-tests/longcat-umt5"
        let commit = String(repeating: "a", count: 40)
        let snapshot = try cache.snapshotPath(repo: repo, kind: .model, commitHash: commit)
        try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
        try Data(tokenizer.utf8).write(to: snapshot.appendingPathComponent("tokenizer.json"))
        try cache.updateRef(repo: repo, kind: .model, ref: "main", commit: commit)
        let cached = try await LongCatModel.fromModelDirectory(directory, cache: cache)
        #expect(try cached.tokenize("hi") == [4, 5, 6, 1])
    }

    @Test func tokenizerHonorsAddedTokensAndEOS() throws {
        let json = #"{"model":{"type":"Unigram","unk_id":3,"vocab":[["<pad>",0],["</s>",0],["<s>",0],["<unk>",0],["▁",-1],["hello",-1],["world",-1],["<extra_id_0>",0]]}}"#
        let tokenizer = try SentencePieceTokenizer(tokenizerJSONData: Data(json.utf8))
        let model = LongCatModel(configuration(), tokenizer: tokenizer)
        #expect(try model.tokenize("hello<pad>world") == [4, 5, 0, 4, 6, 1])
        #expect(try model.tokenize("<extra_id_0> hello") == [7, 4, 5, 1])
        #expect(try model.tokenize("hello </s> world") == [4, 5, 1, 4, 6, 1])
    }

    @Test func publishedQuantizationModesRoundTripWithoutQuantizingVAE() async throws {
        for mode: QuantizationMode in [.affine, .mxfp4] {
            let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: directory) }
            var c = configuration()
            c.ditDim = 64; c.ditTextDim = 64; c.latentDim = 64
            c.textEncoderConfig = .init(vocabSize: 64, dModel: 64, dKv: 16, dFf: 128, numLayers: 1, numHeads: 4)
            c.vaeConfig.latentDim = 64; c.vaeConfig.encoderLatentDim = 128
            c.vaeConfig.downsampleShortcut = "none"; c.vaeConfig.upsampleShortcut = "none"
            c.vaeConfig.outShortcut = "none"; c.vaeConfig.inShortcut = "none"
            c.quantization = .init(groupSize: 32, bits: 4, mode: mode)
            let original = LongCatModel(c)
            let weights = Dictionary(uniqueKeysWithValues: original.parameters().flattened().map { key, value in
                (key, MLXArray((0..<value.size).map { Float($0 % 17 - 8) / 64 }).reshaped(value.shape))
            })
            try original.update(parameters: .unflattened(weights), verify: .all)
            quantize(model: original, groupSize: 32, bits: 4, mode: mode) { path, _ in
                !path.hasPrefix("vae.") && !path.hasSuffix("relative_attention_bias")
            }
            try MLX.save(arrays: Dictionary(uniqueKeysWithValues: original.parameters().flattened()),
                url: directory.appendingPathComponent("model.safetensors"))
            try JSONEncoder().encode(c).write(to: directory.appendingPathComponent("config.json"))
            let tokenizer = #"{"model":{"type":"Unigram","unk_id":3,"vocab":[["<pad>",0],["</s>",0],["<s>",0],["<unk>",0],["▁",-1],["hi",-1]]}}"#
            try Data(tokenizer.utf8).write(to: directory.appendingPathComponent("tokenizer.json"))
            let loaded = try await LongCatModel.fromModelDirectory(directory)
            let a = original.encodeText(MLXArray([1, 2]).reshaped(1, 2), mask: .ones([1, 2]))
            let b = loaded.encodeText(MLXArray([1, 2]).reshaped(1, 2), mask: .ones([1, 2]))
            #expect(abs(a - b).max().item(Float.self) < 1e-6)
            let parameters = Dictionary(uniqueKeysWithValues: loaded.parameters().flattened())
            #expect(parameters["text_encoder.shared.weight"]?.dtype == .uint32)
            #expect(parameters["text_encoder.shared.scales"]?.dtype == (mode == .affine ? .float32 : .uint8))
            #expect(parameters["vae.decoder.layers.0.weight"]?.dtype == .float32)
        }
    }

    @Test func bfloatTextFeatureNormalizationMatchesPythonRounding() {
        Device.withDefaultDevice(.cpu) {
            let input = MLXArray([Float(0.125), 0.25, 0.5, 1, 2, 3, 4, 8]).asType(.bfloat16).reshaped(1, 8)
            let actual = LongCatModel.normalizeTextFeatures(input)
            let expected: [Float] = [-0.8984375, -0.84765625, -0.75, -0.546875, -0.14453125, 0.2578125, 0.66015625, 2.265625]
            #expect(actual.dtype == .bfloat16)
            #expect(actual.asArray(Float.self) == expected)
            let wide = MLXArray((0..<768).map { Float(($0 * 13) % 57 - 28) / 8 }).asType(.bfloat16).reshaped(1, 768)
            let wideExpected: [Float] = [-1.6953125, -0.90625, -0.1201171875, 0.6640625, 1.453125, -1.2109375, -0.423828125, 0.36328125]
            #expect(LongCatModel.normalizeTextFeatures(wide).flattened()[0..<8].asArray(Float.self) == wideExpected)
        }
    }
}
