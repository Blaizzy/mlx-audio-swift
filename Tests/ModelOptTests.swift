import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXAudioCore
@testable import MLXAudioSTT

@Suite(.serialized)
struct ModelOptTests {
    @Test func fp8SafetensorTagsLoadAsBytes() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString + ".safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        let header = Data(
            #"{"weight":{"dtype":"F8_E4M3","shape":[2,2],"data_offsets":[0,4]}}"#.utf8)
        var length = UInt64(header.count).littleEndian
        var data = withUnsafeBytes(of: &length) { Data($0) }
        data.append(header)
        data.append(contentsOf: [56, 64, 68, 72])
        try data.write(to: url)
        let weights = try MLX.loadArrays(url: url)
        #expect(weights["weight"]?.dtype == .uint8)
        #expect(weights["weight"]?.asArray(UInt8.self) == [56, 64, 68, 72])
    }

    @Test func fp8ScalarChannelAndExpertScales() throws {
        for shape in [[4, 8], [3, 4, 8]] {
            for scaleShape in [[], Array(shape.dropLast())] {
                let decoded = try ModelOpt.decodeFP8(
                    .full(shape, values: MLXArray(UInt8(68))),
                    scale: .full(scaleShape, values: MLXArray(Float(0.25))))
                #expect(decoded.dtype == .bfloat16)
                #expect(MLX.all(decoded .== 0.75).item(Bool.self))
            }
        }
        #expect(throws: (any Error).self) {
            try ModelOpt.decodeFP8(.zeros([4, 8], dtype: .uint8), scale: .ones([3]))
        }
    }

    @Test func modelOptConfigAndPackedWeights() throws {
        var weights: [String: MLXArray] = [
            "layer.weight": .full([4, 32], values: MLXArray(UInt8(0x77))),
            "layer.weight_scale": .full([4, 4], values: MLXArray(UInt8(56))),
            "layer.weight_scale_2": MLXArray(Float(0.137)),
            "layer.input_scale": MLXArray(Float(0.25)),
            "fp8.weight": MLXArray([UInt8(56), 64, 68, 72]).reshaped(2, 2),
            "fp8.weight_scale": MLXArray([Float(0.5), 0.25]),
        ]
        var config: [String: Any] = [
            "quantization_config": ["quant_method": "modelopt", "quant_algo": "MIXED_PRECISION"]
        ]
        try ModelOpt.prepare(weights: &weights, configuration: &config)
        #expect(weights["layer.weight"]?.shape == [4, 8])
        #expect(weights["layer.weight"]?.dtype == .uint32)
        #expect(weights["layer.scales"]?.dtype == .uint8)
        #expect(weights["layer.weight_scale_2"]?.dtype == .float32)
        #expect(weights["layer.input_scale"] == nil)
        #expect((config["quantization"] as? [String: Any])?["mode"] as? String == "nvfp4")
        #expect(
            arrayEqual(weights["fp8.weight"]!, MLXArray([Float(0.5), 1, 0.75, 1]).reshaped(2, 2))
                .item(Bool.self))
        let before = weights["layer.weight"]!
        try ModelOpt.prepare(weights: &weights, configuration: &config)
        #expect(arrayEqual(before, weights["layer.weight"]!).item(Bool.self))
    }

    @Test(arguments: [Device.cpu, Device.gpu])
    func nvfp4GlobalAndInputScales(device: Device) throws {
        try Device.withDefaultDevice(device) {
            let packed = MLXArray.full([3, 4, 8], values: MLXArray(UInt32(0x7777_7777)))
            let scales = MLXArray.full([3, 4, 4], values: MLXArray(UInt8(56)))
            let indices = MLXArray([Int32(0), 2, 2, 1]).reshaped(2, 2)
            let x = (MLXArray(0 ..< 128).asType(.float32) / 128 + 0.1).reshaped(2, 1, 1, 64)
            let global = MLXArray([Float(0.137), 0.348, 0.00012715657])
            let pre = (MLXArray(0 ..< 192).asType(.float32) / 192 + 0.5).reshaped(3, 64)
            for name in ["weight_scale_2", "weight_global_scale"] {
                let factor = name == "weight_scale_2" ? global : 1 / global
                let expert = QuantizedExpertLinear(
                    weight: .ones([3, 4, 64]), groupSize: 16, bits: 4, mode: .nvfp4)
                let model = ModelOptExpertLinear(expert, scale: global, name: name, pre: pre)
                try model.update(
                    parameters: ModuleParameters.unflattened([
                        "weight": packed, "scales": scales, name: global, "pre_quant_scale": pre,
                    ]), verify: .all)
                let decoded = broadcast(
                    (6 * factor).reshaped(3, 1, 1) * pre.expandedDimensions(axis: -2),
                    to: [3, 4, 64])
                let expected = gatherMM(x, decoded.swappedAxes(-1, -2), rhsIndices: indices)
                #expect(
                    (abs(model(x, indices: indices) - expected) / (1 + abs(expected))).max().item(
                        Float.self) < 2e-5)
                let linear = QuantizedLinear(
                    weight: packed[0], scales: scales[0], biases: nil, groupSize: 16, bits: 4,
                    mode: .nvfp4)
                let dense = ModelOptLinear(linear, scale: global[0], name: name, pre: pre[0])
                let actual = dense(x.reshaped(2, 64))
                let reference = matmul(x.reshaped(2, 64), decoded[0].T)
                #expect(
                    (abs(actual - reference) / (1 + abs(reference))).max().item(Float.self)
                        < (device == .gpu ? 1e-3 : 2e-5))
            }
        }
    }

    @Test func splitExpertsAndSourceBackboneAreMapped() {
        var source: [String: MLXArray] = [:]
        for expert in 0 ..< 4 {
            let prefix = "backbone.layers.2.mixer.experts.\(expert).up_proj"
            source[prefix + ".weight"] = .ones([8, 8])
            source[prefix + ".scales"] = .ones([8, 1])
            source[prefix + ".weight_scale_2"] = MLXArray(Float(expert + 1))
        }
        let mapped = NemotronTranscribeModel.sanitize(weights: source, expertCount: 4)
        let path = "llm.backbone.layers.2.mixer.switch_mlp.fc1"
        #expect(mapped[path + ".weight"]?.shape == [4, 8, 8])
        #expect(mapped[path + ".weight_scale_2"]?.shape == [4])
        #expect(mapped.keys.count == 3)
    }

    @Test func fullTinyModelLoadsMixedModelOptWeightsStrictly() throws {
        var c = try NemotronTranscribeTests().config()
        c.text.hiddenSize = 64
        c.text.intermediateSize = 64
        c.text.expertIntermediateSize = 64
        c.text.sharedIntermediateSize = 64
        c.outputDimension = 64
        let model = NemotronTranscribeModel(c)
        var source = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        let projection = "llm.backbone.layers.1.mixer.q_proj"
        let denseWeight = source.removeValue(forKey: projection + ".weight")!
        let packed = MLX.quantized(denseWeight, groupSize: 16, bits: 4, mode: .nvfp4)
        source[projection + ".weight"] = packed.0.view(dtype: .uint8)
        source[projection + ".weight_scale"] = packed.1
        source[projection + ".weight_scale_2"] = MLXArray(Float(0.137))
        for name in ["fc1", "fc2"] {
            let path = "llm.backbone.layers.2.mixer.switch_mlp.\(name)"
            let original = source.removeValue(forKey: path + ".weight")!
            for i in 0 ..< c.text.experts {
                let prefix =
                    "backbone.layers.2.mixer.experts.\(i).\(name == "fc1" ? "up_proj" : "down_proj")"
                let q = MLX.quantized(original[i], groupSize: 16, bits: 4, mode: .nvfp4)
                source[prefix + ".weight"] = q.0.view(dtype: .uint8)
                source[prefix + ".weight_scale"] = q.1
                source[prefix + ".weight_scale_2"] = MLXArray(Float(0.1 + Double(i) * 0.1))
            }
        }
        var dictionary: [String: Any] = [
            "quantization_config": ["quant_method": "modelopt", "quant_algo": "MIXED_PRECISION"]
        ]
        try ModelOpt.prepare(weights: &source, configuration: &dictionary)
        let weights = NemotronTranscribeModel.sanitize(weights: source, expertCount: c.text.experts)
        try ModelOpt.quantize(model: model) { path, _ in
            weights[path + ".scales"] == nil ? nil : (16, 4, .nvfp4)
        }
        try ModelOpt.installScaledLinears(in: model, weights: weights)
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        let output = model.llm(MLXArray([Int32(1), 2, 3]).reshaped(1, 3))
        #expect(output.shape == [1, 3, 64])
        #expect(MLX.all(isFinite(output)).item(Bool.self))
    }
}
