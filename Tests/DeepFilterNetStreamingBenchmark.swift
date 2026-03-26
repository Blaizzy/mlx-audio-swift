import Foundation
import Testing
import MLX
@testable import MLXAudioCore
@testable import MLXAudioSTS

/// Streaming latency benchmark for DeepFilterNet.
///
/// Measures per-hop wall-clock time with `materializeEveryHops: 1` to simulate
/// real-time streaming (eval forced every hop so output is immediately available).
///
/// Run with:
///   xcodebuild test \
///     -scheme MLXAudio-Package \
///     -destination 'platform=macOS' \
///     -parallel-testing-enabled NO \
///     -only-testing:'MLXAudioTests/DeepFilterNetStreamingBenchmark' \
///     CODE_SIGNING_ALLOWED=NO \
///     2>&1 | grep -E '(Test.*started|passed|failed|BENCHMARK|Stream profile|hop)'
struct DeepFilterNetStreamingBenchmark {

    @Test func streamingPerHopLatency() async throws {
        // --- Load model ---
        // Try local cached model first, fall back to HuggingFace download.
        let localPaths = [
            ProcessInfo.processInfo.environment["MLXAUDIO_DFN_MODEL_DIR"],
            "\(NSHomeDirectory())/.cache/huggingface/hub/deepfilternet-mlx/iky1e_DeepFilterNet3-MLX",
            "\(NSHomeDirectory())/.cache/huggingface/hub/mlx-audio/iky1e_DeepFilterNet3-MLX",
        ].compactMap { $0 }
        var loadedModel: DeepFilterNetModel?
        for path in localPaths {
            if FileManager.default.fileExists(atPath: "\(path)/config.json") {
                print("BENCHMARK: Loading model from local: \(path)")
                loadedModel = try DeepFilterNetModel.fromLocal(URL(fileURLWithPath: path))
                break
            }
        }
        if loadedModel == nil {
            let modelRepo = "mlx-community/DeepFilterNet-mlx"
            print("BENCHMARK: Downloading model from \(modelRepo)...")
            loadedModel = try await DeepFilterNetModel.fromPretrained(modelRepo, subfolder: "v3")
        }
        let model = loadedModel!
        print("BENCHMARK: Model loaded (\(model.modelVersion), sr=\(model.sampleRate))")

        // --- Load test audio ---
        let audioURL = Bundle.module.url(
            forResource: "noisy_audio_10s",
            withExtension: "wav",
            subdirectory: "media"
        )!
        let (sr, rawAudio) = try loadAudioArray(from: audioURL, sampleRate: 48_000)
        #expect(sr == 48_000)
        let totalSamples = rawAudio.shape[0]
        let durationSec = Double(totalSamples) / Double(sr)
        print("BENCHMARK: Audio loaded: \(totalSamples) samples (\(String(format: "%.1f", durationSec))s at \(sr)Hz)")

        let hopSize = model.config.hopSize  // 480 samples = 10ms
        let totalHops = totalSamples / hopSize

        // =====================================================================
        // Test 1: Real-time simulation (materialize every hop)
        // =====================================================================
        print("\n=== TEST 1: Real-time mode (materializeEveryHops=1) ===")
        print("BENCHMARK: Budget per hop: \(String(format: "%.1f", Double(hopSize) / Double(sr) * 1000.0))ms")

        let rtStreamer = model.createStreamer(
            config: DeepFilterNetStreamingConfig(
                padEndFrames: 0,
                compensateDelay: true,
                enableProfiling: true,
                profilingForceEvalPerStage: false,
                materializeEveryHops: 1
            )
        )

        var hopTimesRT = [Double]()
        hopTimesRT.reserveCapacity(totalHops)
        var offset = 0

        // Warm-up: first few hops may include JIT compilation
        let warmupHops = min(10, totalHops)
        for _ in 0..<warmupHops {
            let end = min(offset + hopSize, totalSamples)
            let chunk = rawAudio[offset..<end]
            let out = try rtStreamer.processChunk(chunk)
            if out.shape[0] > 0 {
                let _ = out.asArray(Float.self)
            }
            offset = end
        }
        print("BENCHMARK: Warm-up done (\(warmupHops) hops)")

        // Timed hops
        let timedStart = CFAbsoluteTimeGetCurrent()
        while offset < totalSamples {
            let end = min(offset + hopSize, totalSamples)
            let chunk = rawAudio[offset..<end]

            let t0 = CFAbsoluteTimeGetCurrent()
            let out = try rtStreamer.processChunk(chunk)
            if out.shape[0] > 0 {
                // Force GPU->CPU transfer to measure true end-to-end latency
                let _ = out.asArray(Float.self)
            }
            let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0  // ms
            hopTimesRT.append(elapsed)

            offset = end
        }
        let totalTimedSec = CFAbsoluteTimeGetCurrent() - timedStart

        printStats("Real-time mode", hopTimesRT, budgetMs: Double(hopSize) / Double(sr) * 1000.0)
        let timedAudioSec = Double(hopTimesRT.count * hopSize) / Double(sr)
        let rtf = totalTimedSec / timedAudioSec
        print("BENCHMARK: RTF (real-time factor): \(String(format: "%.3f", rtf))x  (\(String(format: "%.1f", 1.0/rtf))x faster than real-time)")

        if let summary = rtStreamer.profilingSummary() {
            print(summary)
        }

        // =====================================================================
        // Test 2: Batched mode (materialize every 512 hops) for comparison
        // =====================================================================
        print("\n=== TEST 2: Batched mode (materializeEveryHops=512) ===")

        let batchStreamer = model.createStreamer(
            config: DeepFilterNetStreamingConfig(
                padEndFrames: 0,
                compensateDelay: true,
                enableProfiling: true,
                profilingForceEvalPerStage: false,
                materializeEveryHops: 512
            )
        )

        offset = 0
        let batchStart = CFAbsoluteTimeGetCurrent()
        while offset < totalSamples {
            let end = min(offset + hopSize, totalSamples)
            let chunk = rawAudio[offset..<end]
            let out = try batchStreamer.processChunk(chunk)
            if out.shape[0] > 0 {
                let _ = out.asArray(Float.self)
            }
            offset = end
        }
        let tail = try batchStreamer.flushMLX()
        if tail.shape[0] > 0 {
            let _ = tail.asArray(Float.self)
        }
        let batchTotalSec = CFAbsoluteTimeGetCurrent() - batchStart
        let batchRtf = batchTotalSec / durationSec
        print("BENCHMARK: Total: \(String(format: "%.3f", batchTotalSec))s for \(String(format: "%.1f", durationSec))s audio")
        print("BENCHMARK: RTF: \(String(format: "%.3f", batchRtf))x  (\(String(format: "%.1f", 1.0/batchRtf))x faster than real-time)")

        if let summary = batchStreamer.profilingSummary() {
            print(summary)
        }

        // =====================================================================
        // Test 3: Offline enhance() for baseline comparison
        // =====================================================================
        print("\n=== TEST 3: Offline enhance() baseline ===")
        let offlineStart = CFAbsoluteTimeGetCurrent()
        let enhanced = try model.enhance(rawAudio)
        let _ = enhanced.asArray(Float.self)
        let offlineSec = CFAbsoluteTimeGetCurrent() - offlineStart
        let offlineRtf = offlineSec / durationSec
        print("BENCHMARK: Total: \(String(format: "%.3f", offlineSec))s for \(String(format: "%.1f", durationSec))s audio")
        print("BENCHMARK: RTF: \(String(format: "%.3f", offlineRtf))x  (\(String(format: "%.1f", 1.0/offlineRtf))x faster than real-time)")
    }

    private func printStats(_ label: String, _ times: [Double], budgetMs: Double) {
        guard !times.isEmpty else {
            print("BENCHMARK: \(label): no data")
            return
        }
        let sorted = times.sorted()
        let count = sorted.count
        let avg = sorted.reduce(0, +) / Double(count)
        let median = sorted[count / 2]
        let p95 = sorted[Int(Double(count) * 0.95)]
        let p99 = sorted[Int(Double(count) * 0.99)]
        let maxVal = sorted.last!
        let minVal = sorted.first!
        let overBudget = sorted.filter { $0 > budgetMs }.count
        let pctOverBudget = Double(overBudget) / Double(count) * 100.0

        print("BENCHMARK: \(label) — \(count) hops measured (budget=\(String(format: "%.1f", budgetMs))ms)")
        print("BENCHMARK:   min=\(String(format: "%.3f", minVal))ms  avg=\(String(format: "%.3f", avg))ms  median=\(String(format: "%.3f", median))ms")
        print("BENCHMARK:   p95=\(String(format: "%.3f", p95))ms  p99=\(String(format: "%.3f", p99))ms  max=\(String(format: "%.3f", maxVal))ms")
        print("BENCHMARK:   over budget: \(overBudget)/\(count) (\(String(format: "%.1f", pctOverBudget))%)")
    }
}
