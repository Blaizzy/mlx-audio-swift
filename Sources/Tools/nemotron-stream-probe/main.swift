import Foundation
import MLX
import MLXAudioSTT

// On-ANE harness for the cache-aware streaming CoreML encoder. `swift test` can't load MLX's
// metallib, so this runs as an executable (which does run Metal).
//
//   swift run nemotron-stream-probe <stream.mlpackage>            # smoke test
//   swift run nemotron-stream-probe --bench 80 <stream.mlpackage> # perf bench (median ms/step + phases)

#if canImport(CoreML)
import CoreML

var args = Array(CommandLine.arguments.dropFirst())
var benchN = 0
var attnCache = 70  // EN model; pass --attn-cache 56 for the 3.5 multilingual model
if let i = args.firstIndex(of: "--bench"), i + 1 < args.count {
    benchN = Int(args[i + 1]) ?? 0
    args.removeSubrange(i...(i + 1))
}
if let i = args.firstIndex(of: "--attn-cache"), i + 1 < args.count {
    attnCache = Int(args[i + 1]) ?? 70
    args.removeSubrange(i...(i + 1))
}
let path = args.first ?? ProcessInfo.processInfo.environment["NEMOTRON_STREAM_MLPACKAGE"] ?? ""
guard FileManager.default.fileExists(atPath: path) else {
    print("usage: nemotron-stream-probe [--bench N] <stream.mlpackage>")
    exit(2)
}

let enc = try NemotronCoreMLStreamingEncoder(
    modelURL: URL(fileURLWithPath: path),
    featIn: 128, dModel: 1024, subsamplingFactor: 8,
    preFrames: 9, newFrames: 112, layers: 24, attnCache: attnCache, convCache: 8)

let vals = (0..<(121 * 128)).map { Float($0 % 97) * 0.01 - 0.5 }
let window = MLXArray(vals, [1, 121, 128])

func checksum(_ a: MLXArray) -> Double {
    a.asArray(Float.self).reduce(0.0) { $0 + Double(abs($1)) }
}

if benchN > 0 {
    // Warmup (ANE/Metal spin-up, ignored), then time N steps. Checksum a warmed step so a change
    // that alters the encoder output is caught. Profiling showed ~99.6% of each step is the CoreML
    // prediction (ANE compute); the Swift marshaling is ~0.1 ms — there is no Swift-side perf win.
    enc.reset()
    for _ in 0..<10 { _ = try enc.step(window) }
    enc.reset()
    let guardSum = checksum(try enc.step(window))
    enc.reset()
    var ms = [Double]()
    ms.reserveCapacity(benchN)
    for _ in 0..<benchN {
        let t0 = DispatchTime.now().uptimeNanoseconds
        _ = try enc.step(window)
        ms.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6)
    }
    ms.sort()
    let med = ms[ms.count / 2], p10 = ms[ms.count / 10], p90 = ms[(ms.count * 9) / 10]
    print(String(format: "checksum=%.4f", guardSum))
    print(String(format: "median=%.3f p10=%.3f p90=%.3f ms/step (n=%d)", med, p10, p90, benchN))
    print(String(format: "%.3f", med))  // LAST line = Verify metric (lower is better)
    exit(0)
}

// --- smoke test ---
print("fixedFrames=\(enc.fixedFrames)  (expect 121)")
let o1 = try enc.step(window)
let f1 = o1.asArray(Float.self)
print("chunk1: shape=\(o1.shape) finite=\(f1.allSatisfy { $0.isFinite }) first3=\(Array(f1.prefix(3)))")
let f2 = try enc.step(window).asArray(Float.self)
print("chunk2 (same input): differs=\(f1 != f2)  → caches are threaded")
enc.reset()
let f3 = try enc.step(window).asArray(Float.self)
let maxDiff = zip(f1, f3).map { abs($0 - $1) }.max() ?? 1
print("after reset(): maxDiff vs chunk1=\(maxDiff)  (≈0 expected)")
let pass = f1.allSatisfy { $0.isFinite } && (f1 != f2) && maxDiff < 1e-3
    && o1.shape == [1, 1024, o1.shape[2]] && o1.shape[2] >= 1 && o1.shape[2] <= 16
print(pass ? "PROBE PASS ✅" : "PROBE FAIL ❌")
exit(pass ? 0 : 1)
#else
print("CoreML unavailable")
exit(2)
#endif
