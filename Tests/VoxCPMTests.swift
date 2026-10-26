import Foundation
import MLX
import MLXNN
import Testing
import Tokenizers

@testable import MLXAudioCore
@testable import MLXAudioTTS

@Suite("VoxCPM", .serialized)
struct VoxCPMTests {
    private func tinyConfig() -> VoxCPMConfiguration {
        VoxCPMConfiguration(
            lmConfig: VoxCPM2LMConfig(hiddenSize: 32, numHiddenLayers: 1,
                numAttentionHeads: 4, numKeyValueHeads: 2, intermediateSize: 64,
                vocabSize: 128, useMup: false),
            encoderConfig: VoxCPM2EncoderConfig(hiddenDim: 32, ffnDim: 64, numHeads: 4, numLayers: 1),
            ditConfig: VoxCPM2DiTConfig(hiddenDim: 32, ffnDim: 64, numHeads: 4, numLayers: 1),
            audioVaeConfig: VoxCPMAudioVAEConfig(encoderDim: 4, encoderRates: [2],
                latentDim: 4, decoderDim: 8, decoderRates: [2], sampleRate: 8000),
            patchSize: 2, featDim: 4, scalarQuantizationLatentDim: 8,
            residualLmNumLayers: 1, maxLength: 128)
    }

    @Test func olderConfigurationDefaultsSurvivePartialNestedConfigs() throws {
        let config = try JSONDecoder().decode(VoxCPMConfiguration.self,
            from: Data(#"{"architecture":"voxcpm","lm_config":{},"encoder_config":{},"dit_config":{},"audio_vae_config":{}}"#.utf8))
        #expect(!config.lmConfig.useMup)
        #expect(config.encoderConfig.numLayers == 8)
        #expect(config.ditConfig.numLayers == 8)
        #expect(config.scalarQuantizationLatentDim == 256)
        #expect(config.audioVaeConfig.sampleRate == 44100)
        #expect(config.audioVaeConfig.encoderRates == [2, 3, 6, 7, 7])
        #expect(config.audioVaeConfig.decoderRates == [7, 7, 6, 3, 2])
        let quantized = try JSONDecoder().decode(VoxCPMConfiguration.self,
            from: Data(#"{"architecture":"voxcpm","quantization":{"bits":4,"group_size":64}}"#.utf8))
        #expect(quantized.quantization?.bits == 4)
        #expect(TTS.resolveModelType(modelRepo: "mlx-community/VoxCPM1.5-4bit") == "voxcpm")
        #expect(TTS.resolveModelType(modelRepo: "mlx-community/VoxCPM2-4bit") == "voxcpm2")
        let model = VoxCPMModel(tinyConfig())
        #expect(!model.parameters().flattened().contains { $0.0.contains("fusion_concat_proj") })
        #expect(model.audioVAE.decoder.srCondLayers.isEmpty)
    }

    @Test func bothDiTConditioningModesMatchPythonReference() throws {
        let config = VoxCPM2LMConfig(hiddenSize: 8, numHiddenLayers: 1,
            numAttentionHeads: 2, numKeyValueHeads: 1, intermediateSize: 16, vocabSize: 0,
            ropeLongFactor: [1, 1], ropeShortFactor: [1, 1], useMup: false)
        // Python CPU references with weights ((i * 17 + size) % 31 - 15) / 128.
        let expected: [[Float]] = [
            [-0.227741897, -0.215427861, -0.202871650, 0.145720899, 0.115450203, 0.086226530],
            [-0.214767396, -0.201279581, -0.188007340, 0.090439394, 0.059648253, 0.031368062],
        ]
        func output(version: Int, on device: Device) throws -> MLXArray {
            try Device.withDefaultDevice(device) {
                // The default must retain VoxCPM2's separate mu and time tokens.
                let dit = version == 1
                    ? VoxCPMLocDiT(config, inChannels: 2, conditioning: .summed)
                    : VoxCPMLocDiT(config, inChannels: 2)
                let weights = Dictionary(uniqueKeysWithValues: dit.parameters().flattened().map { key, value in
                    if key.contains("rope") || key.hasSuffix("time_embeddings.freqs") { return (key, value) }
                    // Binary fractions avoid GPU-dependent transcendental rounding in the weights.
                    let values: [Float] = (0..<value.size).map { Float(($0 * 17 + value.size) % 31 - 15) / 128 }
                    let generated = key.contains("norm.weight") ? MLXArray.ones(value.shape)
                        : MLXArray(values).reshaped(value.shape)
                    return (key, generated)
                })
                try dit.update(parameters: .unflattened(weights), verify: .all)
                let result = dit(MLXArray(0..<6).asType(.float32).reshaped(1, 2, 3) / 10,
                    mu: MLXArray(0..<(8 * version)).asType(.float32).reshaped(1, -1) / 20,
                    t: MLXArray([Float(0.4)]), cond: MLXArray.ones([1, 2, 3]) * 0.2,
                    dt: MLXArray([Float(0)]))
                eval(result)
                return result
            }
        }
        for version in 1...2 {
            let cpu = try output(version: version, on: .cpu)
            #expect(cpu.shape == [1, 2, 3])
            let reference = MLXArray(expected[version - 1]).reshaped(1, 2, 3)
            let referenceError = MLX.abs(cpu - reference).max().item(Float.self)
            #expect(referenceError < 2e-5, "VoxCPM \(version) CPU reference error: \(referenceError)")

            // Retain the tight architecture check above, then allow GPU matmul
            // rounding, including TF32, when comparing against the CPU result.
            let gpu = try output(version: version, on: .gpu)
            #expect(gpu.shape == cpu.shape)
            let gpuError = MLX.abs(gpu - cpu).max().item(Float.self)
            #expect(gpuError < 1e-3, "VoxCPM \(version) GPU rounding error: \(gpuError)")
        }
    }

    @Test func referenceAlignmentPadsOnTheLeft() throws {
        let model = VoxCPMModel(tinyConfig())
        let input = MLXArray([Float(0.1), 0.2, 0.3, 0.4, 0.5])
        let actual = try model.encodePromptAudio(input)
        let padded = MLXArray([Float(0), 0, 0, 0.1, 0.2, 0.3, 0.4, 0.5]).reshaped(1, 8, 1)
        let expected = model.audioVAE.encode(padded).reshaped(2, 2, 4)
        #expect(actual.shape == [2, 2, 4])
        #expect(MLX.max(MLX.abs(actual - expected)).item(Float.self) < 1e-6)
    }

    @Test func oddStrideDecoderPaddingMatchesPythonWithoutChangingExistingDefault() throws {
        let input = MLXArray([Float(1), 2]).reshaped(1, 2, 1)
        for applyPadding in [false, true] {
            let conv = VoxCausalTransposeConv1d(inChannels: 1, outChannels: 1,
                kernelSize: 6, stride: 3, padding: 2, outputPadding: 1,
                applyOutputPadding: applyPadding)
            try conv.update(parameters: .unflattened([
                "conv.weight": MLXArray.ones([1, 6, 1]), "conv.bias": MLXArray.zeros([1]),
            ]), verify: .all)
            let expected: [Float] = applyPadding ? [1, 1, 1, 3, 3, 3, 2] : [1, 1, 1, 3, 3, 3]
            #expect(conv(input).asArray(Float.self) == expected)
        }
    }

    private func writeCheckpoint(_ directory: URL, quantized: Bool) throws {
        var config = tinyConfig()
        let model = VoxCPMModel(config)
        if quantized {
            config.quantization = .init(groupSize: 32, bits: 4)
            quantize(model: model) { path, _ in
                path.hasPrefix("base_lm.") || path.hasPrefix("residual_lm.") ? (32, 4, .affine) : nil
            }
        }
        try JSONEncoder().encode(config).write(to: directory.appendingPathComponent("config.json"))
        var weights: [String: MLXArray] = [:]
        for (key, value) in model.parameters().flattened() {
            var python = key.replacingOccurrences(of: "specialToken", with: "special_token")
                .replacingOccurrences(of: ".invFreq", with: ".inv_freq")
                .replacingOccurrences(of: ".longFactor", with: ".long_factor")
                .replacingOccurrences(of: ".shortFactor", with: ".short_factor")
            if key.hasPrefix("audio_vae.") {
                python = python.replacingOccurrences(of: ".blocks.", with: ".blocks.layers.")
                for suffix in ["weight", "bias"] {
                    python = python.replacingOccurrences(of: ".conv.\(suffix)", with: ".\(suffix)")
                }
            }
            weights[python] = value
        }
        try MLX.save(arrays: weights, url: directory.appendingPathComponent("model.safetensors"))
        try Data(#"{"model":{"type":"BPE","vocab":{"<unk>":0,"<s>":1,"H":3,"i":4},"merges":[],"unk_token":"<unk>"},"added_tokens":[{"id":1,"content":"<s>","special":true}],"post_processor":{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"<s>","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],"pair":[{"SpecialToken":{"id":"<s>","type_id":0}},{"Sequence":{"id":"A","type_id":0}},{"Sequence":{"id":"B","type_id":1}}],"special_tokens":{"<s>":{"id":"<s>","ids":[1],"tokens":["<s>"]}}}}"#.utf8)
            .write(to: directory.appendingPathComponent("tokenizer.json"))
        try Data(#"{"tokenizer_class":"PreTrainedTokenizerFast","bos_token":"<s>","unk_token":"<unk>"}"#.utf8)
            .write(to: directory.appendingPathComponent("tokenizer_config.json"))
    }

    @Test func sharedLocalLoadingGeneratesDenseAndQuantizedReferenceAudio() async throws {
        for quantized in [false, true] {
            let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: directory) }
            try writeCheckpoint(directory, quantized: quantized)
            let loaded = try await TTS.loadModel(modelRepo: directory.path)
            let model = try #require(loaded as? VoxCPMModel)
            #expect(try model.tokenize("Hi") == [1, 3, 4])
            #expect(model.sampleRate == 8000)
            #expect((model.baseLM.embedTokens is QuantizedEmbedding) == quantized)
            let audio = try await loaded.generate(text: "Hi", voice: nil,
                refAudio: MLXArray.ones([5]) * 0.01, refText: "Hi", language: nil,
                generationParameters: GenerateParameters(maxTokens: 2))
            #expect(audio.shape == [8])
            #expect(audio.asArray(Float.self).allSatisfy { $0.isFinite })
            var audioEvents = 0
            var infoEvents = 0
            for try await event in loaded.generateStream(text: "Hi", voice: nil, refAudio: nil,
                refText: nil, language: nil, generationParameters: GenerateParameters(maxTokens: 1)) {
                switch event {
                case .audio(let samples):
                    audioEvents += 1
                    #expect(samples.shape == [4])
                case .info: infoEvents += 1
                default: break
                }
            }
            #expect(audioEvents == 1 && infoEvents == 1)
        }
    }
}
