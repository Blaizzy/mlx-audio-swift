//  Per-append cost of rebuilding the periodic Hann window against reusing one.
//  Both paths run in a single process so they share a Metal queue.
//
//  `swift test` cannot run this: SwiftPM does not compile MLX's Metal shaders,
//  so that process has no metallib and the first eval aborts.
//
//    VOXTRAL_MEL_BENCH=1 xcodebuild test -scheme MLXAudio-Package \
//      -destination 'platform=macOS' \
//      -only-testing:'MLXAudioTests/VoxtralRealtimeMelWindowBenchTests' \
//      CODE_SIGNING_ALLOWED=NO

import Foundation
import Testing
import MLX

@testable import MLXAudioSTT

private let benchEnabled = ProcessInfo.processInfo.environment["VOXTRAL_MEL_BENCH"] == "1"

@Suite(.serialized, .enabled(if: benchEnabled))
struct VoxtralRealtimeMelWindowBenchTests {
    private static let chunkSamples = 1600
    private static let windowSize = 400
    private static let hopLength = 160
    /// Frames a 100 ms chunk completes once the carry reaches steady state.
    private static let framesPerAppend = chunkSamples / hopLength

    private static let warmup = 200
    private static let iterations = 2000

    private static func elapsedNanos(_ body: () -> Void) -> Double {
        let start = DispatchTime.now().uptimeNanoseconds
        body()
        return Double(DispatchTime.now().uptimeNanoseconds - start)
    }

    private static func report(_ label: String, _ samples: [Double]) -> Double {
        let sorted = samples.sorted()
        let median = sorted[sorted.count / 2]
        let mean = samples.reduce(0, +) / Double(samples.count)
        let p90 = sorted[Int(Double(sorted.count) * 0.9)]
        let padded = label.padding(toLength: 34, withPad: " ", startingAt: 0)
        print(String(format: "%@ median %8.2f us   mean %8.2f us   p90 %8.2f us",
                     padded, median / 1000, mean / 1000, p90 / 1000))
        return median
    }

    private static func sweep(_ count: Int) -> [Float] {
        (0..<count).map { i -> Float in
            let t = Float(i) / 16000
            return 0.5 * sin(2 * .pi * (200 + 1500 * t) * t) + 0.25 * sin(2 * .pi * 45 * t)
        }
    }

    @Test func perAppendHannWindowCost() {
        let filters = VoxtralRealtimeAudio.computeMelFilters().asType(.float32)
        let cached = VoxtralRealtimeAudio.periodicHannWindow(size: Self.windowSize)
        eval(filters, cached)

        let span = (Self.framesPerAppend - 1) * Self.hopLength + Self.windowSize
        let frames = asStrided(
            MLXArray(Self.sweep(span)),
            [Self.framesPerAppend, Self.windowSize],
            strides: [Self.hopLength, 1],
            offset: 0
        )
        eval(frames)

        func melColumns(window: MLXArray) -> MLXArray {
            VoxtralRealtimeAudio.melColumns(
                frames: frames, melFilters: filters,
                window: window, globalLogMelMax: 1.5
            )
        }
        func rebuilt() {
            eval(melColumns(window: VoxtralRealtimeAudio.periodicHannWindow(size: Self.windowSize)))
        }
        func reused() {
            eval(melColumns(window: cached))
        }

        for _ in 0..<Self.warmup { rebuilt(); reused() }

        var rebuiltNs: [Double] = []
        var reusedNs: [Double] = []

        // ABBA per iteration so any drift lands on both equally.
        for _ in 0..<Self.iterations {
            rebuiltNs.append(Self.elapsedNanos(rebuilt))
            reusedNs.append(Self.elapsedNanos(reused))
            reusedNs.append(Self.elapsedNanos(reused))
            rebuiltNs.append(Self.elapsedNanos(rebuilt))
        }

        print("")
        print("melColumns, \(Self.framesPerAppend) frames/append, \(Self.iterations) iterations")
        let rebuiltMedian = Self.report("window rebuilt", rebuiltNs)
        let reusedMedian = Self.report("window reused", reusedNs)
        let delta = rebuiltMedian - reusedMedian
        print(String(format: "delta %.2f us per append (%.1f%% of the rebuilding path)",
                     delta / 1000, 100 * delta / rebuiltMedian))
        print(String(format: "at 10 appends/s that is %.2f ms per minute of audio",
                     delta * 10 * 60 / 1_000_000))
        print("")
    }

    /// Denominator for that delta: one full append, end to end.
    @Test func perAppendTotalCost() {
        let filters = VoxtralRealtimeAudio.computeMelFilters().asType(.float32)
        eval(filters)
        var stream = VoxtralRealtimeMelStream(leftPadSamples: 1280, melFilters: filters)
        let chunk = Self.sweep(Self.chunkSamples)

        for _ in 0..<Self.warmup { eval(stream.append(chunk)) }

        var totals: [Double] = []
        for _ in 0..<Self.iterations {
            totals.append(Self.elapsedNanos { eval(stream.append(chunk)) })
        }

        print("")
        _ = Self.report("melStream.append", totals)
        print("")
    }
}
