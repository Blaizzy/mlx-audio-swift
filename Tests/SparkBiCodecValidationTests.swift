import Foundation
@preconcurrency import MLX
import MLXNN
import Testing

@testable import MLXAudioTTS

private func readI32(_ path: String) -> [Int32] {
    let d = try! Data(contentsOf: URL(fileURLWithPath: path))
    return d.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
}
private func readF32(_ path: String) -> [Float] {
    let d = try! Data(contentsOf: URL(fileURLWithPath: path))
    return d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
}

private let bicodecConfigJSON = """
{
 "mel_params":{"sample_rate":16000,"n_fft":1024,"win_length":640,"hop_length":320,"mel_fmin":10,"mel_fmax":null,"num_mels":128},
 "encoder":{"input_channels":1024,"vocos_dim":384,"vocos_intermediate_dim":2048,"vocos_num_layers":12,"out_channels":1024,"sample_ratios":[1,1]},
 "decoder":{"input_channel":1024,"channels":1536,"rates":[8,5,4,2],"kernel_sizes":[16,11,8,4]},
 "quantizer":{"input_dim":1024,"codebook_size":8192,"codebook_dim":8,"commitment":0.25,"use_l2_normlize":true},
 "speaker_encoder":{"input_dim":128,"out_dim":1024,"latent_dim":128,"token_num":32,"fsq_levels":[4,4,4,4,4,4],"fsq_num_quantizers":1},
 "prenet":{"input_channels":1024,"vocos_dim":384,"vocos_intermediate_dim":2048,"vocos_num_layers":12,"out_channels":1024,"condition_dim":1024,"sample_ratios":[1,1],"use_tanh_at_final":false},
 "postnet":{"input_channels":1024,"vocos_dim":384,"vocos_intermediate_dim":2048,"vocos_num_layers":6,"out_channels":1024,"use_tanh_at_final":false}
}
"""

@Test func sparkBiCodecDetokenizeMatchesReference() throws {
    let home = FileManager.default.homeDirectoryForCurrentUser.path
    let bicodec = "\(home)/.cache/huggingface/hub/models--mlx-community--Spark-TTS-0.5B-bf16/snapshots/cac753e3cf9d9d92524eb15fe256efc7db45ce7e/BiCodec/model.safetensors"
    let scratch = "/private/tmp/claude-501/-Users-alazarmanakelew/d86e2411-3fdf-4630-9669-36a1eb1362ee/scratchpad"
    guard FileManager.default.fileExists(atPath: bicodec) else {
        print("⚠️ skip: BiCodec weights not present"); return
    }

    let config = try JSONDecoder().decode(BiCodecConfiguration.self, from: Data(bicodecConfigJSON.utf8))
    let model = SparkBiCodec(config)

    let raw = try MLX.loadArrays(url: URL(fileURLWithPath: bicodec))
    let sanitized = model.sanitize(raw)
    let modelKeys = Set(model.parameters().flattened().map(\.0))
    let weightKeys = Set(sanitized.keys)
    let missingInModel = weightKeys.subtracting(modelKeys).sorted()
    let missingInWeights = modelKeys.subtracting(weightKeys).sorted()
    print("keys: model=\(modelKeys.count) weights=\(weightKeys.count) missingInModel=\(missingInModel.count) missingInWeights=\(missingInWeights.count)")
    for k in missingInModel.prefix(15) { print("  weight-not-in-model: \(k)") }
    for k in missingInWeights.prefix(15) { print("  model-not-in-weight: \(k)") }
    #expect(missingInModel.isEmpty && missingInWeights.isEmpty)

    try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: .none)

    let g = MLXArray(readI32(scratch + "/ref_global.i32")).reshaped([1, 32])
    let s = MLXArray(readI32(scratch + "/ref_semantic.i32")).reshaped([1, 38])
    let wav = model.detokenize(semanticTokens: s, globalTokens: g)
    eval(wav)
    let out = wav.asType(.float32).asArray(Float.self)
    let ref = readF32(scratch + "/ref_wav.f32")
    print("out samples=\(out.count) ref samples=\(ref.count)")
    #expect(out.count == ref.count)

    let n = min(out.count, ref.count)
    var maxAbs: Float = 0, sumSq: Float = 0, refSq: Float = 0
    for i in 0..<n {
        maxAbs = max(maxAbs, abs(out[i] - ref[i]))
        sumSq += (out[i] - ref[i]) * (out[i] - ref[i])
        refSq += ref[i] * ref[i]
    }
    let rmse = (sumSq / Float(n)).squareRoot()
    let relerr = (sumSq / max(refSq, 1e-9)).squareRoot()
    print("maxAbsDiff=\(maxAbs) rmse=\(rmse) relErr=\(relerr)")
    print("out[:5]=\(Array(out.prefix(5)))  ref[:5]=\(Array(ref.prefix(5)))")
    #expect(relerr < 0.05)
}
