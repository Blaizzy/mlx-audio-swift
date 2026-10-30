import Foundation
import MLX
import MLXNN

/// Weight-only loading of NVIDIA ModelOpt FP8 and NVFP4 checkpoints.
public enum ModelOpt {
    private static func invalid(_ message: String) -> NSError {
        NSError(domain: "ModelOpt", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
    }

    public static func decodeFP8(_ weight: MLXArray, scale: MLXArray) throws -> MLXArray {
        guard weight.dtype == .uint8, [2, 3].contains(weight.ndim), scale.dtype.isFloatingPoint
        else {
            throw invalid("Expected FP8 byte matrices and floating-point scales")
        }
        let dtype = scale.dtype == .float32 ? DType.bfloat16 : scale.dtype
        var scale = scale.asType(.float32)
        if scale.size == 1 {
            scale = scale.reshaped([])
        } else if scale.shape == Array(weight.shape.dropLast()) {
            scale = scale.expandedDimensions(axis: -1)
        } else if weight.ndim == 3 && scale.shape == Array(weight.shape.dropLast(2)) {
            scale = scale.reshaped(-1, 1, 1)
        } else if scale.shape != Array(weight.shape.dropLast()) + [1] {
            throw invalid("Invalid ModelOpt FP8 scale shape for \(weight.shape)")
        }
        let table: [Float] = (0 ..< 256).map { byte in
            let exponent = (byte >> 3) & 15
            let mantissa = byte & 7
            let value: Float
            if exponent == 0 {
                value = Float(mantissa) / 512
            } else if exponent == 15 && mantissa == 7 {
                value = .nan
            } else {
                value = (1 + Float(mantissa) / 8) * pow(2, Float(exponent - 7))
            }
            return byte & 128 == 0 ? value : -value
        }
        return (MLXArray(table)[weight.asType(.uint32)] * scale).asType(dtype)
    }

    /// Consumes source tensors in bounded batches; NVFP4 remains packed.
    public static func prepare(
        weights: inout [String: MLXArray], configuration: inout [String: Any],
        directory: URL? = nil
    ) throws {
        if (configuration["quantization"] as? [String: Any])?["bits"] != nil { return }
        var source = configuration["quantization_config"] as? [String: Any]
        for name in ["text_config", "llm_config"] where source == nil {
            source =
                (configuration[name] as? [String: Any])?["quantization_config"] as? [String: Any]
        }
        if source == nil, let directory,
            let data = try? Data(
                contentsOf: directory.appendingPathComponent("hf_quant_config.json")),
            let sidecar = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            (sidecar["producer"] as? [String: Any])?["name"] as? String == "modelopt",
            weights.keys.contains(where: { $0.hasSuffix(".weight_scale") })
        {
            source = sidecar["quantization"] as? [String: Any]
            source?["quant_method"] = "modelopt"
        }
        guard let source,
            ["modelopt", "modelopt_mixed"].contains(source["quant_method"] as? String ?? "")
        else { return }
        let algorithm = source["quant_algo"] as? String ?? ""
        guard ["FP8", "NVFP4", "NVFP4_AWQ", "W4A16_NVFP4", "MIXED_PRECISION"].contains(algorithm)
        else {
            throw invalid("Unsupported ModelOpt quantization algorithm: \(algorithm)")
        }
        let prefixes = weights.keys.filter { $0.hasSuffix(".weight_scale") }.map {
            String($0.dropLast(13))
        }
        guard !prefixes.isEmpty else { throw invalid("ModelOpt checkpoint has no weight scales") }
        var pending: [MLXArray] = []
        var pendingBytes = 0
        var hasNVFP4 = false
        for prefix in prefixes.sorted() {
            guard let weight = weights.removeValue(forKey: prefix + ".weight"),
                var scale = weights.removeValue(forKey: prefix + ".weight_scale")
            else {
                throw invalid("Missing ModelOpt tensors for \(prefix)")
            }
            if let global = weights.removeValue(forKey: prefix + ".weight_scale_2") {
                guard weight.dtype == .uint8, scale.dtype == .uint8,
                    [2, 3].contains(weight.ndim), scale.ndim == weight.ndim,
                    Array(weight.shape.dropLast()) == Array(scale.shape.dropLast()),
                    weight.dim(-1) == scale.dim(-1) * 8, weight.dim(-1) % 4 == 0,
                    global.dtype.isFloatingPoint,
                    global.size == 1 || (weight.ndim == 3 && global.size == weight.dim(0))
                else {
                    throw invalid("Invalid ModelOpt NVFP4 tensors for \(prefix)")
                }
                if algorithm == "NVFP4_AWQ" {
                    scale = MLX.where((scale & 127) .== 127, MLXArray(UInt8(0)), scale)
                }
                weights[prefix + ".weight"] = weight.view(dtype: .uint32)
                weights[prefix + ".scales"] = scale
                weights[prefix + ".weight_scale_2"] = global.asType(.float32).squeezed()
                hasNVFP4 = true
            } else {
                let decoded = try decodeFP8(weight, scale: scale)
                weights[prefix + ".weight"] = decoded
                pending.append(decoded)
                pendingBytes += decoded.size * decoded.dtype.size
                if pendingBytes >= 64 * 1024 * 1024 {
                    eval(pending)
                    pending.removeAll()
                    pendingBytes = 0
                    // FP8 buffers cannot generally back their larger BF16 replacements.
                    // Release cached source allocations while expanding large checkpoints.
                    Memory.clearCache()
                }
            }
        }
        eval(pending)
        Memory.clearCache()
        for key in weights.keys.filter({ key in
            [".input_scale", ".input_scale_2", ".k_scale", ".v_scale"].contains {
                key.hasSuffix($0)
            }
        }) { weights.removeValue(forKey: key) }
        configuration.removeValue(forKey: "quantization_config")
        for name in ["text_config", "llm_config"] {
            if var nested = configuration[name] as? [String: Any] {
                nested.removeValue(forKey: "quantization_config")
                configuration[name] = nested
            }
        }
        if hasNVFP4 {
            configuration["quantization"] = ["group_size": 16, "bits": 4, "mode": "nvfp4"]
        } else {
            configuration.removeValue(forKey: "quantization")
        }
    }

    /// Preserve untouched layers when only selected entries in a module array are quantized.
    public static func quantize(
        model: Module,
        filter: (String, Module) -> (groupSize: Int, bits: Int, mode: QuantizationMode)?
    ) throws {
        var replacements: [String: Module] = [:]
        for (path, module) in model.leafModules().flattened() {
            if let q = filter(path, module),
                let quantized = quantizeSingle(
                    layer: module, groupSize: q.groupSize, bits: q.bits, mode: q.mode)
            {
                replacements[path] = quantized
            }
        }
        try replaceModules(in: model, replacements: replacements)
    }

    public static func installScaledLinears(in model: Module, weights: [String: MLXArray]) throws {
        var replacements: [(String, Module)] = []
        for (path, module) in model.leafModules().flattened() {
            let names = ["weight_scale_2", "weight_global_scale"].filter {
                weights[path + "." + $0] != nil
            }
            guard let name = names.first else { continue }
            guard names.count == 1, let global = weights[path + "." + name],
                global.dtype.isFloatingPoint
            else { throw invalid("Conflicting NVFP4 scales: \(path)") }
            let pre = weights[path + ".pre_quant_scale"]
            let shape: [Int]
            if let linear = module as? QuantizedLinear, linear.mode == .nvfp4 {
                shape = [linear.shape.1]
                guard global.size == 1 else { throw invalid("Invalid NVFP4 global scale: \(path)") }
                replacements.append(
                    (path, ModelOptLinear(linear, scale: global, name: name, pre: pre)))
            } else if let expert = module as? QuantizedExpertLinear, expert.mode == .nvfp4 {
                shape = [expert.weight.dim(0), expert.scales.dim(-1) * expert.groupSize]
                guard global.size == 1 || global.size == shape[0] else {
                    throw invalid("Invalid NVFP4 expert scale: \(path)")
                }
                replacements.append(
                    (path, ModelOptExpertLinear(expert, scale: global, name: name, pre: pre)))
            } else {
                throw invalid("NVFP4 scales require a quantized linear: \(path)")
            }
            if let pre,
                !pre.dtype.isFloatingPoint || (pre.shape != shape && pre.shape != [shape.last!])
            {
                throw invalid("Invalid NVFP4 pre-quant scale: \(path)")
            }
        }
        let byPath = Dictionary(uniqueKeysWithValues: replacements)
        try replaceModules(in: model, replacements: byPath)
    }

    private static func replaceModules(in model: Module, replacements: [String: Module]) throws {
        // Empty dictionaries recurse through untouched array entries. MLXNN's
        // sparse unflattening uses `.none` there, which cannot update block arrays.
        func rewrite(_ item: NestedItem<String, Module>, path: String) -> (
            NestedItem<String, Module>, Bool
        ) {
            switch item {
            case .none: return (.none, false)
            case .value:
                if let replacement = replacements[path] { return (.value(replacement), true) }
                return (.dictionary([:]), false)
            case .array(let items):
                let rewritten = items.enumerated().map {
                    rewrite($0.element, path: path + "." + String($0.offset))
                }
                guard rewritten.contains(where: { $0.1 }) else { return (.dictionary([:]), false) }
                let directModules = items.allSatisfy {
                    if case .value = $0 { return true }
                    return false
                }
                let values = items.indices.map { index in
                    directModules && !rewritten[index].1 ? items[index] : rewritten[index].0
                }
                return (.array(values), true)
            case .dictionary(let items):
                var result: [String: NestedItem<String, Module>] = [:]
                for (key, value) in items {
                    let next = rewrite(value, path: path.isEmpty ? key : path + "." + key)
                    if next.1 { result[key] = next.0 }
                }
                return (.dictionary(result), !result.isEmpty)
            }
        }
        let tree = rewrite(model.leafModules().asItem(), path: "").0
        try model.update(modules: ModuleChildren(item: tree), verify: .none)
    }
}

final class ModelOptLinear: Linear, Quantized {
    var groupSize: Int { 16 }
    var bits: Int { 4 }
    var mode: QuantizationMode { .nvfp4 }
    let scales: MLXArray
    @ParameterInfo(key: "weight_scale_2") var scale2: MLXArray?
    @ParameterInfo(key: "weight_global_scale") var global: MLXArray?
    @ParameterInfo(key: "pre_quant_scale") var pre: MLXArray?
    init(_ layer: QuantizedLinear, scale: MLXArray, name: String, pre: MLXArray?) {
        scales = layer.scales
        _scale2.wrappedValue = name == "weight_scale_2" ? scale : nil
        _global.wrappedValue = name == "weight_global_scale" ? scale : nil
        _pre.wrappedValue = pre
        super.init(weight: layer.weight, bias: layer.bias)
        freeze()
    }
    override var shape: (Int, Int) { (weight.dim(0), scales.dim(-1) * 16) }
    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        var input = x.asType(.float32)
        if let pre { input = input * pre.asType(.float32) }
        var y = quantizedMM(
            input, weight, scales: scales, biases: nil, groupSize: 16, bits: 4, mode: .nvfp4)
        y = (global.map { y / $0 } ?? (y * scale2!)).asType(x.dtype)
        return bias.map { y + $0 } ?? y
    }
}

final class ModelOptExpertLinear: ExpertLinear, Quantized {
    var groupSize: Int { 16 }
    var bits: Int { 4 }
    var mode: QuantizationMode { .nvfp4 }
    let scales: MLXArray
    @ParameterInfo(key: "weight_scale_2") var scale2: MLXArray?
    @ParameterInfo(key: "weight_global_scale") var global: MLXArray?
    @ParameterInfo(key: "pre_quant_scale") var pre: MLXArray?
    init(_ layer: QuantizedExpertLinear, scale: MLXArray, name: String, pre: MLXArray?) {
        scales = layer.scales
        _scale2.wrappedValue = name == "weight_scale_2" ? scale : nil
        _global.wrappedValue = name == "weight_global_scale" ? scale : nil
        _pre.wrappedValue = pre
        super.init(weight: layer.weight)
        freeze()
    }
    override func callAsFunction(_ x: MLXArray, indices: MLXArray, sorted: Bool = false) -> MLXArray
    {
        var input = x.asType(.float32)
        if let pre {
            let scale = pre.ndim == 2 ? pre[indices].expandedDimensions(axis: -2) : pre
            input = input * scale.asType(.float32)
        }
        let y = gatherQuantizedMM(
            input, weight, scales: scales, biases: nil, rhsIndices: indices,
            groupSize: 16, bits: 4, mode: .nvfp4, sortedIndices: sorted)
        let raw = global ?? scale2!
        let scale =
            raw.size == 1
            ? raw : raw.reshaped(weight.dim(0), -1)[indices].expandedDimensions(axis: -2)
        return (global == nil ? y * scale : y / scale).asType(x.dtype)
    }
}
