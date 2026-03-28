import Foundation
import MLX

// MARK: - Feature Normalization

extension DeepFilterNetModel {

    func bandMeanNorm(_ x: MLXArray) -> MLXArray {
        let frames = x.shape[0]
        let a = normAlphaValue
        let oneMinusA = Float(1.0) - a
        let time = MLXArray.arange(frames).asType(.float32)
        let powers = MLX.pow(MLXArray(a), time) // [T]
        let invPowers = MLXArray(Float(1.0)) / powers

        let scaled = x * invPowers.expandedDimensions(axis: 1)
        let accum = cumsum(scaled, axis: 0)

        // libDF default EMA init state for ERB mean normalization
        let initState = MLXArray(Self.linspace(start: -60.0, end: -90.0, count: x.shape[1]))
            .expandedDimensions(axis: 0)
        let state = powers.expandedDimensions(axis: 1) * (initState + MLXArray(oneMinusA) * accum)
        return (x - state) / MLXArray(Float(40.0))  // libDF fixed normalization divisor
    }

    func bandUnitNorm(real: MLXArray, imag: MLXArray) -> (MLXArray, MLXArray) {
        let frames = real.shape[0]
        let a = normAlphaValue
        let oneMinusA = Float(1.0) - a
        let time = MLXArray.arange(frames).asType(.float32)
        let powers = MLX.pow(MLXArray(a), time) // [T]
        let invPowers = MLXArray(Float(1.0)) / powers

        let mag = MLX.sqrt(real.square() + imag.square())
        let scaled = mag * invPowers.expandedDimensions(axis: 1)
        let accum = cumsum(scaled, axis: 0)

        // libDF default EMA init state for DF complex unit normalization
        let initState = MLXArray(Self.linspace(start: 0.001, end: 0.0001, count: real.shape[1]))
            .expandedDimensions(axis: 0)
        let state = powers.expandedDimensions(axis: 1) * (initState + MLXArray(oneMinusA) * accum)
        let denom = MLX.sqrt(MLX.maximum(state, MLXArray(Float(1e-12))))
        return (real / denom, imag / denom)
    }

    // Exact sequential EMA path (libDF-style), primarily for DF1 parity.
    func bandMeanNormExact(_ x: MLXArray) -> MLXArray {
        let frames = x.shape[0]
        let bands = x.shape[1]
        let a = normAlphaValue
        let oneMinusA = Float(1.0) - a

        let xVals = x.asArray(Float.self)
        var out = Array<Float>(repeating: 0, count: xVals.count)
        var state = Self.linspace(start: -60.0, end: -90.0, count: bands)  // libDF default EMA init

        for t in 0..<frames {
            let base = t * bands
            for e in 0..<bands {
                let idx = base + e
                let xv = xVals[idx]
                state[e] = xv * oneMinusA + state[e] * a
                out[idx] = (xv - state[e]) / 40.0  // libDF fixed normalization divisor
            }
        }
        return MLXArray(out).reshaped([frames, bands])
    }

    // Exact sequential complex unit-norm path (libDF-style), primarily for DF1 parity.
    func bandUnitNormExact(real: MLXArray, imag: MLXArray) -> (MLXArray, MLXArray) {
        let frames = real.shape[0]
        let freqs = real.shape[1]
        let a = normAlphaValue
        let oneMinusA = Float(1.0) - a

        let rVals = real.asArray(Float.self)
        let iVals = imag.asArray(Float.self)
        var outR = Array<Float>(repeating: 0, count: rVals.count)
        var outI = Array<Float>(repeating: 0, count: iVals.count)
        var state = Self.linspace(start: 0.001, end: 0.0001, count: freqs)  // libDF default EMA init

        for t in 0..<frames {
            let base = t * freqs
            for f in 0..<freqs {
                let idx = base + f
                let rr = rVals[idx]
                let ii = iVals[idx]
                let mag = (rr * rr + ii * ii).squareRoot()
                state[f] = mag * oneMinusA + state[f] * a
                let den = state[f].squareRoot()
                outR[idx] = rr / den
                outI[idx] = ii / den
            }
        }

        return (
            MLXArray(outR).reshaped([frames, freqs]),
            MLXArray(outI).reshaped([frames, freqs])
        )
    }

    static func computeNormAlpha(hopSize: Int, sampleRate: Int) -> Float {
        let aRaw = exp(-Float(hopSize) / Float(sampleRate))
        var precision = 3
        var a: Float = 1.0
        while a >= 1.0 {
            let scale = powf(10, Float(precision))
            a = (aRaw * scale).rounded() / scale
            precision += 1
        }
        return a
    }

    func applyLookahead(feature: MLXArray, lookahead: Int) -> MLXArray {
        guard lookahead > 0 else { return feature }
        let t = feature.shape[2]
        guard t > lookahead else { return feature }
        let shifted = feature[0..., 0..., lookahead..<t, 0...]
        let pad = MLXArray.zeros([feature.shape[0], feature.shape[1], lookahead, feature.shape[3]], type: Float.self)
        return MLX.concatenated([shifted, pad], axis: 2)
    }
}

// MARK: - ERB & Utility

extension DeepFilterNetModel {

    func erbEnergies(_ specMagSq: MLXArray) -> MLXArray {
        if erbFB.shape.count == 2,
           erbFB.shape[0] == config.freqBins,
           erbFB.shape[1] == config.nbErb
        {
            return MLX.matmul(specMagSq, erbFB.asType(specMagSq.dtype))
        }

        var bands = [MLXArray]()
        bands.reserveCapacity(erbBandWidths.count)
        var start = 0
        for width in erbBandWidths {
            let stop = min(start + width, config.freqBins)
            if stop > start {
                bands.append(MLX.mean(specMagSq[0..., start..<stop], axis: 1))
            } else {
                bands.append(MLXArray.zeros([specMagSq.shape[0]], type: Float.self))
            }
            start = stop
        }
        return MLX.stacked(bands, axis: 1)
    }

    static func libdfFreqToErb(_ freqHz: Float) -> Float {
        9.265 * log1p(freqHz / (24.7 * 9.265))
    }

    static func libdfErbToFreq(_ erb: Float) -> Float {
        24.7 * 9.265 * (exp(erb / 9.265) - 1.0)
    }

    static func libdfErbBandWidths(
        sampleRate: Int,
        fftSize: Int,
        nbBands: Int,
        minNbFreqs: Int
    ) -> [Int] {
        guard sampleRate > 0, fftSize > 0, nbBands > 0 else { return [] }

        let nyq = sampleRate / 2
        let freqWidth = Float(sampleRate) / Float(fftSize)
        let erbLow = libdfFreqToErb(0)
        let erbHigh = libdfFreqToErb(Float(nyq))
        let step = (erbHigh - erbLow) / Float(nbBands)

        var widths = [Int](repeating: 0, count: nbBands)
        var prevFreq = 0
        var freqOver = 0
        let minBins = max(1, minNbFreqs)

        for i in 1...nbBands {
            let f = libdfErbToFreq(erbLow + Float(i) * step)
            let fb = Int((f / freqWidth).rounded())
            var nbFreqs = fb - prevFreq - freqOver
            if nbFreqs < minBins {
                freqOver = minBins - nbFreqs
                nbFreqs = minBins
            } else {
                freqOver = 0
            }
            widths[i - 1] = max(1, nbFreqs)
            prevFreq = fb
        }

        widths[nbBands - 1] += 1  // fft_size/2 + 1 bins
        let target = fftSize / 2 + 1
        let total = widths.reduce(0, +)
        if total > target {
            widths[nbBands - 1] -= (total - target)
        } else if total < target {
            widths[nbBands - 1] += (target - total)
        }
        return widths
    }

    static func vorbisWindow(size: Int) -> MLXArray {
        let half = max(1, size / 2)
        var window = [Float](repeating: 0, count: size)
        for i in 0..<size {
            let inner = sin(0.5 * Float.pi * (Float(i) + 0.5) / Float(half))
            window[i] = sin(0.5 * Float.pi * inner * inner)
        }
        return MLXArray(window)
    }

    static func linspace(start: Float, end: Float, count: Int) -> [Float] {
        guard count > 1 else { return [start] }
        let step = (end - start) / Float(count - 1)
        return (0..<count).map { start + Float($0) * step }
    }

    func w(_ key: String) throws -> MLXArray {
        guard let value = weights[key] else {
            throw DeepFilterNetError.missingWeightKey(key)
        }
        return value
    }
}

// MARK: - Weight Cache Builders

extension DeepFilterNetModel {

    static func buildBatchNormAffine(weights: [String: MLXArray]) -> ([String: MLXArray], [String: MLXArray]) {
        var scaleByPrefix = [String: MLXArray]()
        var biasByPrefix = [String: MLXArray]()

        for (key, mean) in weights where key.hasSuffix(".running_mean") {
            let prefix = String(key.dropLast(".running_mean".count))
            guard let gamma = weights["\(prefix).weight"],
                  let beta = weights["\(prefix).bias"],
                  let variance = weights["\(prefix).running_var"]
            else {
                continue
            }
            let scale = gamma / MLX.sqrt(variance + MLXArray(Float(1e-5)))
            let shift = beta - mean * scale
            scaleByPrefix[prefix] = scale.reshaped([1, scale.shape[0], 1, 1])
            biasByPrefix[prefix] = shift.reshaped([1, shift.shape[0], 1, 1])
        }

        return (scaleByPrefix, biasByPrefix)
    }

    static func buildConv2dWeightCache(weights: [String: MLXArray]) -> [String: MLXArray] {
        var cache = [String: MLXArray]()
        for (key, weight) in weights where key.hasSuffix(".weight") && weight.ndim == 4 {
            cache[key] = weight.transposed(0, 2, 3, 1)
        }
        return cache
    }

    static func buildGRUTransposedWeightCache(weights: [String: MLXArray]) -> [String: MLXArray] {
        var cache = [String: MLXArray]()
        cache.reserveCapacity(24)
        for (key, weight) in weights where key.contains(".gru.weight_") && weight.ndim == 2 {
            cache[key] = weight.transposed()
        }
        return cache
    }

    static func buildV1GroupedLinearPacks(
        weights: [String: MLXArray],
        groups: Int
    ) -> [String: V1GroupedLinearPack] {
        guard groups > 1 else { return [:] }
        var groupedKeys = [String: [Int]]()
        for key in weights.keys where key.hasSuffix(".weight") {
            let parts = key.split(separator: ".")
            guard parts.count >= 3, let idx = Int(parts[parts.count - 2]) else { continue }
            let prefix = parts.dropLast(2).joined(separator: ".")
            groupedKeys[prefix, default: []].append(idx)
        }

        var packs = [String: V1GroupedLinearPack]()
        for (prefix, idxs) in groupedKeys {
            let unique = Array(Set(idxs)).sorted()
            guard unique.count == groups, unique.first == 0, unique.last == groups - 1 else { continue }

            var weightSlices = [MLXArray]()
            var biasSlices = [MLXArray]()
            weightSlices.reserveCapacity(groups)
            biasSlices.reserveCapacity(groups)
            var valid = true
            for g in 0..<groups {
                guard let wg = weights["\(prefix).\(g).weight"],
                      let bg = weights["\(prefix).\(g).bias"],
                      wg.ndim == 2, bg.ndim == 1
                else {
                    valid = false
                    break
                }
                weightSlices.append(wg.transposed())  // [I, O]
                biasSlices.append(bg)
            }
            guard valid else { continue }

            let weightGIO = MLX.stacked(weightSlices, axis: 0)
            let biasGO = MLX.stacked(biasSlices, axis: 0)
            packs[prefix] = V1GroupedLinearPack(
                weightGIO: weightGIO,
                biasGO: biasGO,
                groups: groups,
                inputPerGroup: weightGIO.shape[1],
                outputPerGroup: weightGIO.shape[2]
            )
        }
        return packs
    }

    static func buildV1GroupedGRUPacks(
        weights: [String: MLXArray],
        groups: Int,
        prefixes: [String]
    ) -> [String: V1GroupedGRUPack] {
        guard groups > 1 else { return [:] }
        var packs = [String: V1GroupedGRUPack]()

        for prefix in prefixes {
            var layers = [V1GroupedGRULayerPack]()
            var layerIndex = 0
            while true {
                var wih = [MLXArray]()
                var whh = [MLXArray]()
                var bih = [MLXArray]()
                var bhh = [MLXArray]()
                wih.reserveCapacity(groups)
                whh.reserveCapacity(groups)
                bih.reserveCapacity(groups)
                bhh.reserveCapacity(groups)

                var hasLayer = true
                for g in 0..<groups {
                    let base = "\(prefix).\(layerIndex).layers.\(g)"
                    guard let wihG = weights["\(base).weight_ih_l0"],
                          let whhG = weights["\(base).weight_hh_l0"],
                          let bihG = weights["\(base).bias_ih_l0"],
                          let bhhG = weights["\(base).bias_hh_l0"],
                          wihG.ndim == 2, whhG.ndim == 2, bihG.ndim == 1, bhhG.ndim == 1
                    else {
                        hasLayer = false
                        break
                    }
                    wih.append(wihG.transposed())  // [I, 3H]
                    whh.append(whhG.transposed())  // [H, 3H]
                    bih.append(bihG)
                    bhh.append(bhhG)
                }
                guard hasLayer else { break }

                let wihGI3H = MLX.stacked(wih, axis: 0)
                let whhGH3H = MLX.stacked(whh, axis: 0)
                let bihG3H = MLX.stacked(bih, axis: 0)
                let bhhG3H = MLX.stacked(bhh, axis: 0)
                layers.append(
                    V1GroupedGRULayerPack(
                        weightIHGI3H: wihGI3H,
                        weightHHGH3H: whhGH3H,
                        biasIHG3H: bihG3H,
                        biasHHG3H: bhhG3H,
                        inputPerGroup: wihGI3H.shape[1],
                        hiddenPerGroup: wihGI3H.shape[2] / 3
                    )
                )
                layerIndex += 1
            }

            if !layers.isEmpty {
                packs[prefix] = V1GroupedGRUPack(groups: groups, layers: layers)
            }
        }

        return packs
    }

    static func buildGroupedTransposeWeights(
        weights: [String: MLXArray],
        groups: Int
    ) -> [String: [MLXArray]] {
        guard groups > 1 else { return [:] }
        var cache = [String: [MLXArray]]()
        for (key, weight) in weights where key.hasSuffix(".0.weight") && weight.ndim == 4 {
            guard weight.shape[0] % groups == 0 else { continue }
            let inPerGroup = weight.shape[0] / groups
            var grouped = [MLXArray]()
            grouped.reserveCapacity(groups)
            for g in 0..<groups {
                let inStart = g * inPerGroup
                let inEnd = inStart + inPerGroup
                let wg = weight[inStart..<inEnd, 0..., 0..., 0...]
                grouped.append(wg.transposed(1, 2, 3, 0))
            }
            cache[key] = grouped
        }
        return cache
    }

    static func buildDenseTransposeWeights(
        weights: [String: MLXArray],
        groups: Int
    ) -> [String: MLXArray] {
        guard groups > 1 else { return [:] }
        var cache = [String: MLXArray]()
        for (key, weight) in weights where key.hasSuffix(".0.weight") && weight.ndim == 4 {
            guard weight.shape[0] % groups == 0 else { continue }
            let inPerGroup = weight.shape[0] / groups
            let outPerGroup = weight.shape[1]
            let kT = weight.shape[2]
            let kF = weight.shape[3]
            let totalIn = inPerGroup * groups

            var outBlocks = [MLXArray]()
            outBlocks.reserveCapacity(groups)
            for g in 0..<groups {
                let inStart = g * inPerGroup
                let inEnd = inStart + inPerGroup
                let wg = weight[inStart..<inEnd, 0..., 0..., 0...].transposed(1, 2, 3, 0)
                let leftChannels = g * inPerGroup
                let rightChannels = totalIn - leftChannels - inPerGroup
                let left = MLXArray.zeros([outPerGroup, kT, kF, leftChannels], type: Float.self)
                let right = MLXArray.zeros([outPerGroup, kT, kF, rightChannels], type: Float.self)
                outBlocks.append(MLX.concatenated([left, wg, right], axis: 3))
            }
            cache[key] = MLX.concatenated(outBlocks, axis: 0)
        }
        return cache
    }
}
