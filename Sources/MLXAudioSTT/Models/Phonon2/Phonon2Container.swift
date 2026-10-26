// Container decoding adapted from fermion-research, Copyright 2026 Fermion Research.
// SPDX-License-Identifier: Apache-2.0
import Foundation
import MLX

/// Lossless decoding of the published five-value container into MLX parameter layouts.
enum Phonon2Container {
    static let format = "fermion-five-value-parakeet-v1"

    private struct Header: Decodable {
        struct Record: Decodable {
            let n: String
            let k: String
            let shape: [Int]
            let b: Int
        }
        let format: String
        let index: [Record]
    }

    static func read(_ url: URL) throws -> (
        weights: [String: MLXArray], modules: [Phonon2Config.PackedModule]
    ) {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        guard data.count >= 8 else { throw invalid("Truncated container") }
        let length = data.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self).littleEndian }
        guard length > 0, length <= min(16 << 20, data.count - 8) else {
            throw invalid("Invalid container header size")
        }
        var cursor = 8 + Int(length)
        let header = try JSONDecoder().decode(Header.self, from: data.subdata(in: 8 ..< cursor))
        guard header.format == format else { throw invalid("Unsupported container format") }
        var weights: [String: MLXArray] = [:]
        var dense: [String: MLXArray] = [:]
        var modules: [Phonon2Config.PackedModule] = []
        var seen = Set<String>()
        for row in header.index {
            guard !row.n.isEmpty, seen.insert(row.n).inserted,
                !row.shape.isEmpty || row.k == "fp16",
                row.shape.allSatisfy({ $0 > 0 }), row.b >= 0, row.b <= data.count - cursor
            else { throw invalid("Duplicate, invalid or truncated record: \(row.n)") }
            let total = try elementCount(row.shape)
            let blob = data.subdata(in: cursor ..< (cursor + row.b))
            cursor += row.b
            switch row.k {
            case "five_value":
                let target = mlxName(row.n)
                modules.append(.init(name: target, shape: row.shape))
                for (suffix, tensor) in try fiveValue(blob, shape: row.shape) {
                    let key = "\(target).\(suffix)"
                    guard weights.updateValue(tensor, forKey: key) == nil else {
                        throw invalid("Duplicate tensor: \(key)")
                    }
                }
            case "int6", "int8":
                dense[row.n] = try integerTable(
                    blob, shape: row.shape, total: total, bits: row.k == "int6" ? 6 : 8)
            case "fp16":
                guard total <= Int.max / 2, blob.count == total * 2 else {
                    throw invalid("Invalid fp16 size: \(row.n)")
                }
                dense[row.n] = MLXArray(blob, row.shape, dtype: .float16)
            default: throw invalid("Unsupported record kind: \(row.k)")
            }
        }
        guard cursor == data.count else { throw invalid("Trailing container bytes") }
        for (name, source) in dense {
            if name.hasSuffix("num_batches_tracked") { continue }
            var target = mlxName(name)
            var value = source
            if name.hasPrefix("decoder.lstm.") {
                let part = name.dropFirst("decoder.lstm.".count).split(separator: "_")
                guard part.count == 3, ["weight", "bias"].contains(part[0]),
                    ["ih", "hh"].contains(part[1]), part[2].hasPrefix("l"),
                    let layer = Int(part[2].dropFirst()), layer >= 0
                else { throw invalid("Invalid LSTM parameter: \(name)") }
                target = "decoder.prediction.dec_rnn.lstm.\(layer)."
                if part[0] == "weight" {
                    target += part[1] == "ih" ? "Wx" : "Wh"
                } else {
                    if part[1] == "hh" { continue }
                    guard let other = dense["decoder.lstm.bias_hh_l\(layer)"],
                        other.shape == value.shape
                    else {
                        throw invalid("Missing LSTM recurrent bias")
                    }
                    target += "bias"
                    value = (value.asType(.float32) + other.asType(.float32)).asType(value.dtype)
                }
            } else if name.hasPrefix("encoder.subsampling.layers."), name.hasSuffix(".weight") {
                guard value.ndim == 4 else { throw invalid("Expected a 4D subsampling kernel") }
                value = value.transposed(0, 2, 3, 1)
            } else if name.hasSuffix(".conv.depthwise_conv.weight") {
                guard value.ndim == 3 else { throw invalid("Expected a 3D depthwise kernel") }
                value = value.transposed(0, 2, 1)
            } else if name.hasSuffix(".conv.pointwise_conv1.weight")
                || name.hasSuffix(".conv.pointwise_conv2.weight")
            {
                guard value.ndim == 2 || value.ndim == 3 else {
                    throw invalid("Invalid pointwise kernel")
                }
                value =
                    value.ndim == 2 ? value.expandedDimensions(axis: 1) : value.transposed(0, 2, 1)
            }
            guard weights[target] == nil else { throw invalid("Duplicate tensor: \(target)") }
            weights[target] = value.asType(.bfloat16)
            eval(weights[target]!)
        }
        return (weights, modules)
    }

    static func mlxName(_ name: String) -> String {
        var name = name.replacingOccurrences(
            of: "encoder.subsampling.layers.", with: "encoder.pre_encode.conv."
        )
        .replacingOccurrences(of: "encoder.subsampling.linear.", with: "encoder.pre_encode.out.")
        .replacingOccurrences(of: ".conv.norm.", with: ".conv.batch_norm.")
        for (source, target) in [
            ("q_proj", "linear_q"), ("k_proj", "linear_k"), ("v_proj", "linear_v"),
            ("o_proj", "linear_out"), ("relative_k_proj", "linear_pos"),
            ("bias_u", "pos_bias_u"), ("bias_v", "pos_bias_v"),
        ] {
            name = name.replacingOccurrences(
                of: ".self_attn.\(source)", with: ".self_attn.\(target)")
        }
        for (source, target) in [
            ("encoder_projector.", "joint.enc."), ("decoder.decoder_projector.", "joint.pred."),
            ("joint.head.", "joint.joint_net.2."),
            ("decoder.embedding.", "decoder.prediction.embed."),
        ] where name.hasPrefix(source) { return target + name.dropFirst(source.count) }
        return name
    }

    private static func fiveValue(_ data: Data, shape: [Int]) throws -> [String: MLXArray] {
        guard shape.count == 2, shape[1] % 128 == 0 else {
            throw invalid("Invalid five-value shape")
        }
        let (outputs, inputs) = (shape[0], shape[1])
        let rowBytes = (inputs + 4) / 5
        let tritBytes = outputs * rowBytes
        guard data.count >= tritBytes + 4 * outputs else {
            throw invalid("Truncated five-value record")
        }
        let packed = MLXArray(
            data.subdata(in: 0 ..< tritBytes), [outputs, rowBytes], dtype: .uint8)
        guard packed.max().item(UInt8.self) < 243 else { throw invalid("Invalid ternary codes") }
        let powers = MLXArray([UInt32(1), 3, 9, 27, 81])
        let digits = remainder(
            floorDivide(packed.asType(.uint32).expandedDimensions(axis: -1), powers), 3)
        let codes = digits.reshaped(outputs, rowBytes * 5)[0..., 0 ..< inputs]
        let nonzero = codes .!= 1
        let count = nonzero.sum().item(Int.self)
        let maskBytes = (count + 7) / 8
        let offset = tritBytes + maskBytes
        guard data.count == offset + 4 * outputs else {
            throw invalid("Invalid five-value record size")
        }
        let high: MLXArray
        if count == 0 {
            high = MLXArray.zeros(shape, dtype: .bool)
        } else {
            let mask = MLXArray(data.subdata(in: tritBytes ..< offset), dtype: .uint8).asType(
                .uint32)
            let bitPowers = MLXArray([UInt32(1), 2, 4, 8, 16, 32, 64, 128])
            let bits = remainder(floorDivide(mask.expandedDimensions(axis: -1), bitPowers), 2)
                .flattened()
            let indices = maximum(cumsum(nonzero.flattened().asType(.int32)) - 1, 0)
            high = take(bits, indices, axis: 0).reshaped(shape).asType(.bool)
        }
        let shifts: [UInt32] = (0 ..< 16).map { UInt32(1) << (2 * $0) }
        let factors = MLXArray(shifts)
        func pack(_ plane: MLXArray) -> MLXArray {
            (plane.reshaped(outputs, inputs / 16, 16) * factors).sum(axis: -1).asType(.uint32)
        }
        let result = [
            "base_q": pack(codes),
            "residual_q": pack(MLX.where(high, codes, MLXArray(UInt32(1)))),
            "row_lo": MLXArray(
                data.subdata(in: offset ..< (offset + 2 * outputs)), dtype: .float16),
            "row_hi": MLXArray(
                data.subdata(in: (offset + 2 * outputs) ..< data.count), dtype: .float16),
        ]
        eval(Array(result.values))
        return result
    }

    private static func integerTable(_ data: Data, shape: [Int], total: Int, bits: Int) throws
        -> MLXArray
    {
        let outputs = shape[0]
        let codeBytes = bits == 8 ? total : ((total + 3) / 4) * 3
        guard data.count == codeBytes + 2 * outputs else {
            throw invalid("Invalid integer table size")
        }
        let body = data.subdata(in: 0 ..< codeBytes)
        let codes: MLXArray
        if bits == 8 {
            codes = MLXArray(body, dtype: .int8).asType(.float32)
        } else {
            let triples = MLXArray(body, [body.count / 3, 3], dtype: .uint8).asType(.uint32)
            let packed = (triples * MLXArray([UInt32(1), 256, 65536])).sum(axis: -1)
            let powers = MLXArray([UInt32(1), 64, 4096, 262144])
            let unsigned = remainder(floorDivide(packed.expandedDimensions(axis: -1), powers), 64)
                .flattened()
            codes = unsigned[0 ..< total].asType(.float32) - 32
        }
        let scales = MLXArray(data.subdata(in: codeBytes ..< data.count), dtype: .float16).asType(
            .float32)
        let result = (codes.reshaped(outputs, -1) * scales.expandedDimensions(axis: -1)).reshaped(
            shape)
        eval(result)
        return result
    }

    private static func elementCount(_ shape: [Int]) throws -> Int {
        var total = 1
        for dimension in shape {
            let next = total.multipliedReportingOverflow(by: dimension)
            guard !next.overflow, next.partialValue <= Int32.max else {
                throw invalid("Oversized tensor")
            }
            total = next.partialValue
        }
        return total
    }

    private static func invalid(_ message: String) -> Phonon2Error { .invalidCheckpoint(message) }
}
