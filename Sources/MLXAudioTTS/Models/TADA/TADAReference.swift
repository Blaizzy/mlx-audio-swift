import Foundation
import MLX

public struct TADAReference: Sendable {
    public let text: String
    public let tokens: [Int]
    public let positions: [Int]
    public let features: MaterializedArray

    public init(text: String, tokens: [Int], positions: [Int], features: MLXArray) {
        self.text = text
        self.tokens = tokens
        self.positions = positions
        self.features = features.materialized()
    }
}

/// Hann-windowed sinc interpolation, matching the standard 6-lobe audio resampler.
func tadaResample(_ audio: MLXArray, from source: Int, to target: Int) throws -> MLXArray {
    guard source > 0, target > 0 else { throw TADAError.invalidInput("Sample rates must be positive") }
    if source == target { return audio.reshaped([-1]) }
    func gcd(_ a: Int, _ b: Int) -> Int { b == 0 ? a : gcd(b, a % b) }
    let divisor = gcd(source, target), original = source / divisor, next = target / divisor
    let base = Float(min(original, next)) * 0.99
    let width = Int(ceil(6 * Float(original) / base))
    let kernelSize = 2 * width + original
    var kernel = [Float](); kernel.reserveCapacity(next * kernelSize)
    for phase in 0 ..< next {
        for index in -width ..< width + original {
            let t = min(6, max(-6, (Float(index) / Float(original) - Float(phase) / Float(next)) * base))
            let window = pow(cos(t * .pi / 12), 2)
            let sinc = t == 0 ? Float(1) : sin(t * .pi) / (t * .pi)
            kernel.append(sinc * window * base / Float(original))
        }
    }
    let input = padded(audio.reshaped(1, -1, 1), widths: [.init(0), .init((width, width + original)), .init(0)])
    let filter = MLXArray(kernel).reshaped(next, kernelSize, 1)
    let result = MLX.conv1d(input, filter, stride: original).reshaped([-1])
    let length = Int(ceil(Double(audio.size) * Double(target) / Double(source)))
    return result[0 ..< length]
}

func tadaGray(_ values: MLXArray) -> Int {
    var gray = 0
    for bit in values.asArray(Float.self) { gray = (gray << 1) | (bit > 0 ? 1 : 0) }
    var binary = gray, mask = gray >> 1
    while mask > 0 { binary ^= mask; mask >>= 1 }
    return binary
}

func tadaNormalize(_ text: String) -> String {
    var value = text
    for (a, b) in [("“", "\""), ("”", "\""), ("„", "\""), ("‟", "\""),
        ("‘", "'"), ("’", "'"), ("‚", "'"), ("‛", "'"), ("–", "-"), ("—", "-"),
        ("―", "-"), ("‐", "-"), ("‑", "-"), ("…", "..."), ("‹", "<"), ("›", ">"), ("«", "<<"), ("»", ">>") ] {
        value = value.replacingOccurrences(of: a, with: b)
    }
    for (a, b) in [("; ", ". "), ("\"", ""), (":", ","), ("(", ""), (")", ""),
        ("--", "-"), ("-", ", "), (",,", ","), (" '", " "), ("' ", " "), ("  ", " ")] {
        value = value.replacingOccurrences(of: a, with: b)
    }
    value = value.replacingOccurrences(of: "\\s+([.,?!])", with: "$1", options: .regularExpression).lowercased()
    var capitalize = true
    return value.map { character -> String in
        if character == "." || character == "!" || character == "?" { capitalize = true; return String(character) }
        if capitalize, !character.isWhitespace { capitalize = false; return String(character).uppercased() }
        return String(character)
    }.joined()
}

/// Globally optimal strict monotonic token/frame alignment used by the published
/// checkpoint. Raw CTC frame indices already use the codec's 50 Hz time base.
func tadaMonotonicAlign(scores: MLXArray, audioLength: Int, sampleRate: Int) throws -> (positions: [Int], masks: [Int32]) {
    let tokens = scores.dim(0), frames = scores.dim(1)
    let codecFrames = Int(ceil(Double(audioLength) / Double(sampleRate) * 50))
    var masks = [Int32](repeating: 0, count: codecFrames)
    if tokens == 0 { return ([], masks) }
    guard frames >= tokens, codecFrames >= frames else { throw TADAError.invalidInput("Reference audio is too short for its transcript") }
    let values = scores.asArray(Float.self)
    var previous = [Float](repeating: -.infinity, count: frames)
    var firstPositions = [Int](repeating: 0, count: frames)
    var best: Float = -.infinity, bestFrame = 0
    for frame in 0 ..< frames {
        if values[frame] > best { best = values[frame]; bestFrame = frame }
        previous[frame] = best; firstPositions[frame] = bestFrame
    }
    var uses = [Bool](repeating: false, count: tokens * frames)
    if tokens > 1 {
        for token in 1 ..< tokens {
            var current = [Float](repeating: -.infinity, count: frames)
            for frame in token ..< frames {
                let skip = current[frame - 1]
                let use = previous[frame - 1] + values[token * frames + frame]
                if use >= skip { current[frame] = use; uses[token * frames + frame] = true }
                else { current[frame] = skip }
            }
            previous = current
        }
    }
    var positions = [Int](repeating: 0, count: tokens), frame = frames - 1
    if tokens > 1 {
        for token in (1 ..< tokens).reversed() {
            while frame > token && !uses[token * frames + frame] { frame -= 1 }
            positions[token] = frame; frame -= 1
        }
    }
    positions[0] = firstPositions[frame]
    for position in positions { masks[position] = 1 }
    return (positions.map { $0 + 1 }, masks)
}
