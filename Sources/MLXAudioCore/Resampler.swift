#if canImport(FoundationEssentials)
import FoundationEssentials
#else
import Foundation
#endif
// `sin`/`cos` come from the platform C math library, not Foundation, on non-Darwin platforms.
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#elseif canImport(Musl)
import Musl
#endif

/// Pure-Swift band-limited resampler used on platforms without AVFoundation's
/// `AVAudioConverter` (Linux).
///
/// Uses windowed-sinc interpolation with a low-pass cutoff at the lower of the two
/// Nyquist frequencies, so downsampling is anti-aliased. This does not reproduce
/// `AVAudioConverter` bit-for-bit — no resampler does — but it is a high-quality,
/// deterministic band-limited resample suitable for speech/audio model I/O.
enum SincResampler {
    /// Resample mono `Float` samples from `sourceRate` to `targetRate`.
    static func resample(
        _ input: [Float],
        from sourceRate: Int,
        to targetRate: Int,
        halfTaps: Int = 16
    ) -> [Float] {
        if input.isEmpty || sourceRate == targetRate { return input }
        guard sourceRate > 0, targetRate > 0 else { return input }

        let ratio = Double(targetRate) / Double(sourceRate)
        let outputCount = max(1, Int((Double(input.count) * ratio).rounded()))

        // Normalised low-pass cutoff (cycles/sample of the source signal).
        let cutoff = 0.5 * min(1.0, ratio)
        // Window half-width in source samples grows when downsampling to preserve taps.
        let filterHalfWidth = Double(halfTaps) / min(1.0, ratio)

        var output = [Float](repeating: 0, count: outputCount)
        let n = input.count

        for outIndex in 0 ..< outputCount {
            // Continuous position in the source signal for this output sample.
            let pos = Double(outIndex) / ratio
            let first = Int((pos - filterHalfWidth).rounded(.up))
            let last = Int((pos + filterHalfWidth).rounded(.down))

            var acc = 0.0
            var weightSum = 0.0
            var i = max(first, 0)
            let end = min(last, n - 1)
            while i <= end {
                let t = pos - Double(i)
                let w = blackman(t, halfWidth: filterHalfWidth) * lowpass(t, cutoff: cutoff)
                acc += Double(input[i]) * w
                weightSum += w
                i += 1
            }
            output[outIndex] = weightSum != 0 ? Float(acc / weightSum) : 0
        }
        return output
    }

    /// Ideal low-pass impulse: 2*fc*sinc(2*fc*t).
    private static func lowpass(_ t: Double, cutoff fc: Double) -> Double {
        2.0 * fc * sinc(2.0 * fc * t)
    }

    private static func sinc(_ x: Double) -> Double {
        if x == 0 { return 1 }
        let px = Double.pi * x
        return sin(px) / px
    }

    /// Blackman window over [-halfWidth, halfWidth]; 0 outside.
    private static func blackman(_ t: Double, halfWidth: Double) -> Double {
        if abs(t) > halfWidth { return 0 }
        // Map t ∈ [-halfWidth, halfWidth] to phase ∈ [0, 2π].
        let phase = Double.pi * (t / halfWidth + 1.0)
        return 0.42 - 0.5 * cos(phase) + 0.08 * cos(2.0 * phase)
    }
}
