import Foundation
import MLX
import MLXAudioCore
import Testing
@testable import MLXAudioTTS

@Suite(.serialized)
struct MeloTTSTests {
    func tinyConfiguration() -> MeloTTSConfiguration {
        var c = MeloTTSConfiguration()
        c.interChannels = 4; c.hiddenChannels = 4; c.filterChannels = 8; c.ginChannels = 2
        c.layers = 3; c.flowLayers = 3; c.nSpeakers = 2; c.vocabularySize = 3
        c.symbols = ["_", "aa", "UNK"]; c.spk2id = ["EN-US": 0]
        c.upsampleChannels = 8; c.upsampleRates = [2]; c.upsampleKernels = [4]
        c.resblockKernels = [3]; c.resblockDilations = [[1, 3]]
        return c
    }
    func binary(_ shape: [Int], denominator: Float = 32) -> MLXArray {
        MLXArray((0..<shape.reduce(1, *)).map { Float($0 % 7 - 3) / denominator }).reshaped(shape)
    }
    func close(_ a: MLXArray, _ expected: [Float], tolerance: Float) {
        #expect(a.size == expected.count)
        #expect(abs(a.reshaped(-1) - MLXArray(expected)).max().item(Float.self) < tolerance)
    }
    @Test func normalizationAndAlignment() throws {
        #expect(MeloEnglishFrontend.normalize("Dr. Smith has 1,234.05 apples!") == "doctor smith has one thousand two hundred thirty four point  five apples!")
        let durations = MLXArray([Float(2), 1, 3]).reshaped(1, 1, 3)
        let path = MeloTTSModel.path(durations, frames: 6)
        #expect(path.shape == [1, 3, 6])
        #expect(path.asArray(Float.self) == [1,1,0,0,0,0, 0,0,1,0,0,0, 0,0,0,1,1,1])
        #expect(TTS.resolveModelType(modelRepo: "mlx-community/MeloTTS-English-MLX") == "melotts")
    }
    @Test func phoneStressWordpieceAndBlankAlignment() throws {
        let frontend = MeloEnglishFrontend(vocabulary: ["hello": 7592, "act": 2552, "##iv": 1001, "##ist": 1002, ".": 1012],
            pronunciations: ["hello": ["HH", "AH0", "L", "OW1"], "activist": ["AE1", "K", "T", "IH0", "V", "IH0", "S", "T"]],
            homographs: [:], tags: [:], tagWeights: [:], tagClasses: [], gru: MeloLayers(weights: [:]))
        let symbols = ["_", "hh", "ah", "l", "ow", "ae", "k", "t", "ih", "V", "s", ".", "UNK"]
        let result = try frontend.process("Hello activist.", symbols: symbols, addBlank: true)
        #expect(result.tokenIDs == [101, 7592, 2552, 1001, 1002, 1012, 102])
        #expect(result.phoneIDs.count == 31)
        #expect(result.phones[19] == "V")
        #expect(result.tones[3] == 7)
        #expect(result.tones[5] == 8)
        #expect(result.tones[9] == 9)
        #expect(result.featureIndices == [0,0,0] + Array(repeating: 1, count: 8) + Array(repeating: 2, count: 6) + Array(repeating: 3, count: 6) + Array(repeating: 4, count: 4) + [5,5,6,6])
        let plain = try frontend.process("Hello.", symbols: symbols, addBlank: false)
        #expect(plain.phoneIDs == [0,1,2,3,4,11,0])
        #expect(plain.tones == [7,7,8,7,9,7,7])
    }
    @Test func validatesConfigurationsOptionsAndRequiredTensors() throws {
        var c = tinyConfiguration(); try c.validate()
        c.heads = 3
        #expect(throws: AudioGenerationError.self) { try c.validate() }
        #expect(throws: AudioGenerationError.self) { try MeloTTSGenerationOptions(speed: 0).validate() }
        #expect(throws: AudioGenerationError.self) { try MeloTTSGenerationOptions(stochasticDurationRatio: 2).validate() }
        #expect(throws: AudioGenerationError.self) { try MeloTTSModel.validate([:], shapes: ["x": [2]]) }
        #expect(throws: AudioGenerationError.self) { try MeloTTSModel.validate(["x": MLXArray.zeros([3])], shapes: ["x": [2]]) }
        #expect(throws: AudioGenerationError.self) { try MeloEnglishFrontend.validateGRU([:]) }
    }
    // The reference GPU kernels differ from CPU by 5.01e-5 (BERT), 8.20e-6
    // (attention) and 7.64e-8 (vocoder); CPU goldens retain their tight bounds.
    @Test(arguments: [Device.cpu, Device.gpu])
    func bertPostNormAndThirdLastFeatures(device: Device) {
        Device.withDefaultDevice(device) {
            var shapes = ["embeddings.word_embeddings.weight": [8, 4], "embeddings.position_embeddings.weight": [8, 4], "embeddings.token_type_embeddings.weight": [2, 4], "embeddings.norm.weight": [4], "embeddings.norm.bias": [4]]
            let p = "encoder.layers.0"
            for name in ["query", "key", "value", "out"] {
                shapes[p + ".attention.\(name)_proj.weight"] = [4, 4]
                shapes[p + ".attention.\(name)_proj.bias"] = [4]
            }
            for name in ["ln1", "ln2"] { shapes[p + ".\(name).weight"] = [4]; shapes[p + ".\(name).bias"] = [4] }
            shapes[p + ".linear1.weight"] = [8,4]; shapes[p + ".linear1.bias"] = [8]
            shapes[p + ".linear2.weight"] = [4,8]; shapes[p + ".linear2.bias"] = [4]
            let weights = shapes.mapValues { binary($0, denominator: 16) }.map { key, value in
                (key, key.hasSuffix("norm.weight") || key.hasSuffix("ln1.weight") || key.hasSuffix("ln2.weight") ? MLXArray.ones(value.shape) : value)
            }
            let layer = MeloLayers(weights: Dictionary(uniqueKeysWithValues: weights))
            let y = layer.bert(MLXArray([Int32(1), 2, 3]).reshaped(1, 3), layers: 1, heads: 2)
            close(y, [-1.5940434933,-0.1455363780,1.3591716290,0.0054082130,-1.5940434933,-0.1455363780,1.3591716290,0.0054082130,-1.1094777584,0.9468082190,-1.1348923445,0.9225618243], tolerance: device == .cpu ? 1e-6 : 6e-5)
        }
    }
    @Test(arguments: [Device.cpu, Device.gpu])
    func relativeAttentionAndVocoderGoldens(device: Device) {
        Device.withDefaultDevice(device) {
            let config = tinyConfiguration()
            let shapes = MeloTTSModel.synthesisShapes(config).filter { $0.key.hasPrefix("dec.") || $0.key.hasPrefix("enc_p.encoder.attn_layers.0.") }
            let weights = shapes.mapValues { binary($0) }
            let prefix = "enc_p.encoder.attn_layers.0."
            let named = Dictionary(uniqueKeysWithValues: weights.filter { $0.key.hasPrefix(prefix) }.map { ("a." + $0.key.dropFirst(prefix.count), $0.value) })
            let layer = MeloLayers(weights: weights), x = MLXArray((0..<12).map { Float($0) / 16 }).reshaped(1, 4, 3)
            let y = MeloLayers(weights: named).attention(x, "a", heads: 2)
            close(y, [-0.0789304227,-0.0763508976,-0.0782819018,-0.0686931312,-0.0654067546,-0.0644414946,-0.0228656791,-0.0235236455,-0.0241647996,-0.0074746045,-0.0097325854,-0.0097449524], tolerance: device == .cpu ? 2e-7 : 1e-5)
            let audio = layer.decode(x, g: MLXArray([Float(0.25), -0.5]).reshaped(1, 2, 1), config: config)
            close(audio, [-0.0006659614,-0.0009491517,-0.0015725611,-0.0013225902,0.0021339322,-0.0027703980], tolerance: device == .cpu ? 2e-8 : 1e-7)
        }
    }
    @Test(arguments: [Device.cpu, Device.gpu])
    func gruCellMatchesNumpy(device: Device) {
        Device.withDefaultDevice(device) {
            func pattern(_ n: Int, modulus: Int, scale: Float) -> MLXArray { MLXArray((0..<n).map { Float($0 % modulus - modulus / 2) / scale }) }
            let weights = MeloLayers(weights: ["enc_w_ih": pattern(384 * 64, modulus: 5, scale: 128).reshaped(384, 64), "enc_w_hh": pattern(384 * 128, modulus: 7, scale: 128).reshaped(384, 128), "enc_b_ih": pattern(384, modulus: 3, scale: 64), "enc_b_hh": pattern(384, modulus: 5, scale: 64)])
            let x = MLXArray((0..<64).map { Float($0) / 128 }).reshaped(1, 64)
            let h = MLXArray((0..<128).map { Float($0) / 256 }).reshaped(1, 128)
            let result = MeloEnglishFrontend.gruCell(x, h: h, prefix: "enc", weights: weights)
            close(result[0, ..<8], [-0.0075071417,0.0068012327,0.0038220587,0.0166314170,0.0057781516,-0.0027679438,0.0109643862,0.0233368278], tolerance: 1e-7)
            #expect(abs(result.sum().item(Float.self) - 15.8851966858) < 3e-6)
        }
    }
    @Test(arguments: [Device.cpu, Device.gpu])
    func rationalQuadraticInverseRoundTripsAndTails(device: Device) {
        Device.withDefaultDevice(device) {
            let x = MLXArray([Float(-6), -5, -4.25, -2, -0.125, 0, 1.5, 4.125, 5, 6]).reshaped(1, 1, -1)
            let raw = MLXArray([Float(-0.5), 0.25, 1]).reshaped(1, 1, 1, 3)
            let widths = broadcast(raw, to: [1, 1, 10, 3]), heights = broadcast(-raw, to: [1, 1, 10, 3])
            let derivatives = MLXArray.ones([1, 1, 10, 2]) * Float(0.25)
            let y = MeloLayers.inverseSpline(x, widths: widths, heights: heights, derivatives: derivatives, inverse: false)
            close(y, [-6,-5,-1.2966003418,2.5939064026,3.9872381687,4.0109348297,4.1719660759,4.4838776588,5,6], tolerance: 2e-6)
            let restored = MeloLayers.inverseSpline(y, widths: widths, heights: heights, derivatives: derivatives)
            #expect(abs(restored - x).max().item(Float.self) < 2e-6)
            let inverse = MeloLayers.inverseSpline(x, widths: widths, heights: heights, derivatives: derivatives)
            let forward = MeloLayers.inverseSpline(inverse, widths: widths, heights: heights, derivatives: derivatives, inverse: false)
            #expect(abs(forward - x).max().item(Float.self) < 3e-6)
            #expect(y[0, 0, 0].item(Float.self) == -6)
            #expect(y[0, 0, -1].item(Float.self) == 6)
        }
    }
}
