import Foundation
import MLX
import MLXAudioCore
import MLXRandom

struct Zonos2Prompt {
    static let silence: [[Int]] =
        [
            [568, 778, 338, 524, 967, 360, 728, 550, 90],
            [568, 778, 10, 674, 364, 981, 741, 378, 731],
        ] + Array(repeating: [568, 804, 10, 674, 364, 981, 568, 378, 731], count: 14)
        + [[568, 778, 721, 842, 264, 974, 989, 507, 308]]

    static func build(
        text: String, config c: Zonos2Configuration, options o: Zonos2Options, speaker: Bool
    ) throws -> [[Int]] {
        let counts = c.qualityBucketCounts
        let base =
            c.textVocab - c.speakingRateNumBuckets - counts.reduce(0, +) - c.speakerBackgroundCount
            - c.accurateModeCount
        func row(_ token: Int) -> [Int] {
            Array(repeating: c.audioPadID, count: c.nCodebooks) + [token]
        }
        var rows: [[Int]] = []
        if speaker {
            guard c.speakerEnabled else {
                throw Zonos2Error.invalidInput("Speaker conditioning is disabled in this checkpoint")
            }
            rows.append(row(c.textVocab))
            if c.speakerBackgroundCount > 0 {
                rows.append(
                    row(
                        base + c.speakingRateNumBuckets + counts.reduce(0, +)
                            + (o.cleanSpeakerBackground ? 0 : 1)))
                if o.accurateMode, c.accurateModeCount > 0 {
                    rows.append(row(c.textVocab - c.accurateModeCount))
                }
            }
        }
        if let bucket = o.speakingRateBucket {
            guard (0..<c.speakingRateNumBuckets).contains(bucket) else {
                throw Zonos2Error.invalidInput("Speaking-rate bucket out of range")
            }
            rows.append(row(base + bucket))
        }
        guard Set(o.qualityBuckets.keys).isSubset(of: Set(c.qualityFeatures)) else {
            throw Zonos2Error.invalidInput("Unknown quality feature")
        }
        var offset = base + c.speakingRateNumBuckets
        for (i, feature) in c.qualityFeatures.enumerated() {
            if let bucket = o.qualityBuckets[feature] {
                guard (0..<counts[i]).contains(bucket) else {
                    throw Zonos2Error.invalidInput("Quality bucket out of range for \(feature)")
                }
                rows.append(row(offset + bucket))
            }
            offset += counts[i]
        }
        rows += ([2] + text.utf8.map { Int($0) + 192 } + [3]).map(row)
        for t in 0..<silence.count {
            rows.append(
                (0..<c.nCodebooks).map { cb in t >= cb ? silence[t - cb][cb] : c.audioPadID } + [
                    c.textVocab
                ])
        }
        return rows
    }

    static func codes(
        _ delayed: [[Int]], config c: Zonos2Configuration, eosFrame: Int?, frameLimit: Int? = nil
    ) -> MLXArray {
        let count = max(0, min(eosFrame ?? frameLimit ?? delayed.count, delayed.count))
        let values: [Int32] = (0..<count).flatMap { t in
            (0..<c.nCodebooks).map { cb in
                Int32(
                    min(
                        c.codebookSize - 1,
                        max(0, t + cb < delayed.count ? delayed[t + cb][cb] : c.audioPadID))
                )
            }
        }
        return MLXArray(values).reshaped(count, c.nCodebooks)
    }
}

struct Zonos2GenerationState {
    let codebooks: Int
    let eoaID: Int
    var generated: [[Int]] = []
    var eosFrame: Int?
    var countdown = 0
    var finished: Bool { eosFrame != nil && countdown <= 0 }
    mutating func append(_ frame: [Int], ignoreEOS: Bool) {
        generated.append(Array(frame.prefix(codebooks)))
        guard !ignoreEOS else { return }
        if eosFrame == nil, let last = (0..<codebooks).last(where: { frame[$0] == eoaID }) {
            eosFrame = max(0, generated.count - 1 - last)
            countdown = codebooks + 1
        }
        if eosFrame != nil, countdown > 0 { countdown -= 1 }
    }
}

enum Zonos2Sampling {
    static func probabilities(
        _ input: MLXArray, state: Zonos2GenerationState, parameters p: GenerateParameters,
        repetitionCodebooks: Int
    ) -> MLXArray {
        let logits = penalized(input, state: state, parameters: p, codebooks: repetitionCodebooks)
        var filtered = logits / max(p.temperature, 1e-8)
        if p.topK > 0, p.topK < input.dim(-1) {
            let threshold = sorted(filtered, axis: -1)[0..., (input.dim(-1) - p.topK)].expandedDimensions(
                axis: -1)
            filtered = MLX.where(filtered .< threshold, MLXArray(-Float.infinity), filtered)
        }
        var probabilities = softmax(filtered, axis: -1)
        if p.topP > 0, p.topP < 1 {
            let order = argSort(-probabilities, axis: -1)
            let sorted = takeAlong(probabilities, order, axis: -1)
            let kept = MLX.where(cumsum(sorted, axis: -1) - sorted .> p.topP, MLXArray(Float(0)), sorted)
            probabilities = takeAlong(kept, argSort(order, axis: -1), axis: -1)
            probabilities = probabilities / maximum(sum(probabilities, axis: -1, keepDims: true), 1e-12)
        }
        if p.minP > 0 {
            let threshold = max(probabilities, axis: -1, keepDims: true) * p.minP
            probabilities = MLX.where(probabilities .< threshold, MLXArray(Float(0)), probabilities)
            probabilities = probabilities / maximum(sum(probabilities, axis: -1, keepDims: true), 1e-12)
        }
        return probabilities
    }
    static func penalized(
        _ input: MLXArray, state: Zonos2GenerationState, parameters p: GenerateParameters,
        codebooks: Int
    ) -> MLXArray {
        let input = input.asType(.float32)
        let penalty = p.repetitionPenalty ?? 1.2
        guard penalty > 1, p.repetitionContextSize > 0, !state.generated.isEmpty else { return input }
        let limit = codebooks < 0 ? input.dim(0) : min(codebooks, input.dim(0))
        let recent = state.generated.suffix(p.repetitionContextSize)
        let vocabulary = MLXArray(0..<input.dim(-1))
        return stacked(
            (0..<input.dim(0)).map { cb in
                let row = input[cb]
                guard cb < limit else { return row }
                let seen = Set(recent.map { $0[cb] }.filter { $0 >= 0 && $0 < input.dim(-1) }).sorted()
                guard !seen.isEmpty else { return row }
                let mask = any(
                    vocabulary.expandedDimensions(axis: 0) .== MLXArray(seen).expandedDimensions(axis: -1),
                    axis: 0)
                return MLX.where(mask, MLX.where(row .> 0, row / penalty, row * penalty), row)
            })
    }
    static func sample(
        _ logits: MLXArray, state: Zonos2GenerationState, parameters p: GenerateParameters,
        options: Zonos2Options, textVocab: Int, step: Int
    ) -> [Int] {
        let ids: MLXArray
        if p.temperature <= 1e-8 {
            ids = argMax(
                penalized(logits, state: state, parameters: p, codebooks: options.repetitionCodebooks),
                axis: -1)
        } else {
            let probs = probabilities(
                logits, state: state, parameters: p, repetitionCodebooks: options.repetitionCodebooks)
            let valid = all(isFinite(probs), axis: -1) .&& (sum(probs, axis: -1) .> 0)
            let safe = MLX.where(isFinite(probs), probs, MLXArray(Float(0)))
            let scores = MLX.where(
                valid.expandedDimensions(axis: -1), log(maximum(safe, 1e-20)), MLXArray.zeros(probs.shape))
            let sampled = MLXRandom.categorical(
                scores, axis: -1, key: p.seed.map { MLXRandom.key($0 &+ UInt64(step)) })
            ids = MLX.where(
                valid, sampled,
                argMax(
                    penalized(logits, state: state, parameters: p, codebooks: options.repetitionCodebooks),
                    axis: -1))
        }
        return ids.asType(.int32).asArray(Int32.self).map(Int.init) + [textVocab]
    }
}
