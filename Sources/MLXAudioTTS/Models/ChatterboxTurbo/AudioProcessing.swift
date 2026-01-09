#if canImport(AVFoundation)
import AVFoundation
#endif

func chatterboxResample(_ wav: [Float], from: Int, to: Int) -> [Float] {
    guard from != to, wav.count > 1 else { return wav }

    if let converted = resampleUsingAVAudioConverter(wav, from: from, to: to) {
        return converted
    }

    return resampleLinearFallback(wav, from: from, to: to)
}

func chatterboxNormalizeLoudness(
    _ wav: [Float],
    sampleRate: Int,
    targetLufs: Float
) -> [Float] {
    guard !wav.isEmpty else { return wav }

    let sr = Float(sampleRate)
    let highPass = biquadHighPass(sampleRate: sr, frequency: 60.0, q: 0.707)
    let shelf = biquadHighShelf(sampleRate: sr, frequency: 4000.0, gainDb: 4.0, slope: 1.0)

    var filtered = applyBiquad(wav, coefficients: highPass)
    filtered = applyBiquad(filtered, coefficients: shelf)

    let rms = sqrt(filtered.reduce(0) { $0 + $1 * $1 } / Float(filtered.count))
    guard rms > 0 else { return wav }

    let target = powf(10.0, targetLufs / 20.0)
    var gain = target / rms
    guard gain.isFinite, gain > 0 else { return wav }
    gain = min(max(gain, 0.1), 10.0)

    return wav.map { $0 * gain }
}

#if canImport(AVFoundation)
private func resampleUsingAVAudioConverter(_ wav: [Float], from: Int, to: Int) -> [Float]? {
    guard let inFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: Double(from),
        channels: 1,
        interleaved: false
    ) else {
        return nil
    }
    guard let outFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: Double(to),
        channels: 1,
        interleaved: false
    ) else {
        return nil
    }
    guard let inBuffer = AVAudioPCMBuffer(
        pcmFormat: inFormat,
        frameCapacity: AVAudioFrameCount(wav.count)
    ) else {
        return nil
    }
    inBuffer.frameLength = AVAudioFrameCount(wav.count)

    guard let channelData = inBuffer.floatChannelData?[0] else { return nil }
    wav.withUnsafeBufferPointer { buffer in
        channelData.assign(from: buffer.baseAddress!, count: buffer.count)
    }

    guard let converter = AVAudioConverter(from: inFormat, to: outFormat) else {
        return nil
    }
    converter.sampleRateConverterQuality = .max

    let outCapacity = AVAudioFrameCount(ceil(Double(wav.count) * Double(to) / Double(from)))
    guard let outBuffer = AVAudioPCMBuffer(
        pcmFormat: outFormat,
        frameCapacity: max(outCapacity, 1)
    ) else {
        return nil
    }

    var inputConsumed = false
    var error: NSError?
    let status = converter.convert(to: outBuffer, error: &error) { _, outStatus in
        if inputConsumed {
            outStatus.pointee = .endOfStream
            return nil
        }
        inputConsumed = true
        outStatus.pointee = .haveData
        return inBuffer
    }

    guard status != .error, error == nil else {
        return nil
    }

    let frames = Int(outBuffer.frameLength)
    guard frames > 0, let outData = outBuffer.floatChannelData?[0] else {
        return nil
    }

    return Array(UnsafeBufferPointer(start: outData, count: frames))
}
#else
private func resampleUsingAVAudioConverter(_ wav: [Float], from: Int, to: Int) -> [Float]? {
    nil
}
#endif

private func resampleLinearFallback(_ wav: [Float], from: Int, to: Int) -> [Float] {
    guard from != to, wav.count > 1 else { return wav }
    let ratio = Float(to) / Float(from)
    let newLength = Int(round(Float(wav.count) * ratio))
    guard newLength > 1 else { return wav }

    var output = [Float](repeating: 0, count: newLength)
    for i in 0..<newLength {
        let pos = Float(i) / ratio
        let idx = Int(floor(pos))
        let frac = pos - Float(idx)
        let idxNext = min(idx + 1, wav.count - 1)
        output[i] = wav[idx] * (1 - frac) + wav[idxNext] * frac
    }
    return output
}

private struct BiquadCoefficients {
    let b0: Float
    let b1: Float
    let b2: Float
    let a1: Float
    let a2: Float
}

private func biquadHighPass(sampleRate: Float, frequency: Float, q: Float) -> BiquadCoefficients {
    let w0 = 2.0 * Float.pi * frequency / sampleRate
    let cosw0 = cos(w0)
    let sinw0 = sin(w0)
    let alpha = sinw0 / (2.0 * q)

    var b0 = (1 + cosw0) / 2.0
    var b1 = -(1 + cosw0)
    var b2 = (1 + cosw0) / 2.0
    var a0 = 1 + alpha
    var a1 = -2.0 * cosw0
    var a2 = 1 - alpha

    b0 /= a0
    b1 /= a0
    b2 /= a0
    a1 /= a0
    a2 /= a0

    return BiquadCoefficients(b0: b0, b1: b1, b2: b2, a1: a1, a2: a2)
}

private func biquadHighShelf(
    sampleRate: Float,
    frequency: Float,
    gainDb: Float,
    slope: Float
) -> BiquadCoefficients {
    let w0 = 2.0 * Float.pi * frequency / sampleRate
    let cosw0 = cos(w0)
    let sinw0 = sin(w0)
    let a = powf(10.0, gainDb / 40.0)
    let sqrtA = sqrt(a)
    let alpha = sinw0 / 2.0 * sqrt((a + 1.0 / a) * (1.0 / slope - 1.0) + 2.0)

    var b0 = a * ((a + 1) + (a - 1) * cosw0 + 2 * sqrtA * alpha)
    var b1 = -2 * a * ((a - 1) + (a + 1) * cosw0)
    var b2 = a * ((a + 1) + (a - 1) * cosw0 - 2 * sqrtA * alpha)
    var a0 = (a + 1) - (a - 1) * cosw0 + 2 * sqrtA * alpha
    var a1 = 2 * ((a - 1) - (a + 1) * cosw0)
    var a2 = (a + 1) - (a - 1) * cosw0 - 2 * sqrtA * alpha

    b0 /= a0
    b1 /= a0
    b2 /= a0
    a1 /= a0
    a2 /= a0

    return BiquadCoefficients(b0: b0, b1: b1, b2: b2, a1: a1, a2: a2)
}

private func applyBiquad(_ input: [Float], coefficients: BiquadCoefficients) -> [Float] {
    var output = [Float](repeating: 0, count: input.count)
    var x1: Float = 0
    var x2: Float = 0
    var y1: Float = 0
    var y2: Float = 0

    for i in 0..<input.count {
        let x0 = input[i]
        let y0 = coefficients.b0 * x0
            + coefficients.b1 * x1
            + coefficients.b2 * x2
            - coefficients.a1 * y1
            - coefficients.a2 * y2
        output[i] = y0
        x2 = x1
        x1 = x0
        y2 = y1
        y1 = y0
    }
    return output
}
