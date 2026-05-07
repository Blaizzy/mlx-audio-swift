import Foundation
import HuggingFace
@preconcurrency import MLX
import MLXAudioCore
import MLXNN

public struct SileroVADTimestamp: Sendable, Equatable {
    public let start: Int
    public let end: Int

    public init(start: Int, end: Int) {
        self.start = start
        self.end = end
    }
}

public struct SileroVADStreamingState: Sendable {
    public var lstmState: MLXArray?
    public var context: MLXArray
    public var sampleRate: Int

    public init(lstmState: MLXArray?, context: MLXArray, sampleRate: Int) {
        self.lstmState = lstmState
        self.context = context
        self.sampleRate = sampleRate
    }
}

public enum SileroVADError: Error, LocalizedError {
    case invalidRepositoryID(String)
    case unsupportedSampleRate(Int)
    case stateSampleRateMismatch(expected: Int, got: Int)
    case unexpectedChunkSize(expected: Int, got: Int)
    case insufficientReflectPadInput(samples: Int, pad: Int)

    public var errorDescription: String? {
        switch self {
        case .invalidRepositoryID(let r): return "Invalid repository ID: \(r)"
        case .unsupportedSampleRate(let s): return "Silero VAD supports 8000 Hz and 16000 Hz audio (got \(s))"
        case .stateSampleRateMismatch(let exp, let got):
            return "Streaming state is for \(exp) Hz, got \(got) Hz"
        case .unexpectedChunkSize(let exp, let got):
            return "Expected \(exp) samples per chunk, got \(got)"
        case .insufficientReflectPadInput(let s, let p):
            return "Reflect padding of \(p) requires more than \(p) samples (got \(s))"
        }
    }
}

private func reflectPadRight(_ x: MLXArray, pad: Int) -> MLXArray {
    if pad <= 0 { return x }
    let n = x.dim(-1)
    precondition(n > pad, "reflect pad needs more than \(pad) samples (got \(n))")
    let indices = MLXArray(Array(stride(from: Int32(n - 2), to: Int32(n - pad - 2), by: -1)))
    let reflected = take(x, indices, axis: -1)
    return concatenated([x, reflected], axis: -1)
}

private final class SileroVADBranch: Module {
    let config: SileroVADBranchConfig

    @ModuleInfo(key: "stft_conv") var stftConv: Conv1d
    @ModuleInfo(key: "conv1") var conv1: Conv1d
    @ModuleInfo(key: "conv2") var conv2: Conv1d
    @ModuleInfo(key: "conv3") var conv3: Conv1d
    @ModuleInfo(key: "conv4") var conv4: Conv1d
    @ModuleInfo(key: "lstm") var lstm: LSTM
    @ModuleInfo(key: "final_conv") var finalConv: Conv1d

    init(_ config: SileroVADBranchConfig) {
        self.config = config
        self._stftConv.wrappedValue = Conv1d(
            inputChannels: 1, outputChannels: config.cutoff * 2,
            kernelSize: config.filterLength, stride: config.hopLength,
            padding: 0, bias: false
        )
        self._conv1.wrappedValue = Conv1d(
            inputChannels: config.cutoff, outputChannels: 128,
            kernelSize: 3, padding: 1
        )
        self._conv2.wrappedValue = Conv1d(
            inputChannels: 128, outputChannels: 64,
            kernelSize: 3, stride: 2, padding: 1
        )
        self._conv3.wrappedValue = Conv1d(
            inputChannels: 64, outputChannels: 64,
            kernelSize: 3, stride: 2, padding: 1
        )
        self._conv4.wrappedValue = Conv1d(
            inputChannels: 64, outputChannels: 128,
            kernelSize: 3, padding: 1
        )
        self._lstm.wrappedValue = LSTM(inputSize: 128, hiddenSize: 128)
        self._finalConv.wrappedValue = Conv1d(
            inputChannels: 128, outputChannels: 1,
            kernelSize: 1
        )
    }

    func callAsFunction(_ x: MLXArray, state: MLXArray?) -> (MLXArray, MLXArray) {
        var x = x
        if x.ndim == 1 { x = x[.newAxis, 0...] }

        let (hidden, cell) = splitState(state)
        x = reflectPadRight(x, pad: config.pad)
        x = stftConv(x[.ellipsis, .newAxis])
        let real = x[.ellipsis, 0 ..< config.cutoff]
        let imag = x[.ellipsis, config.cutoff ..< (config.cutoff * 2)]
        x = sqrt(real * real + imag * imag)

        x = MLXNN.relu(conv1(x))
        x = MLXNN.relu(conv2(x))
        x = MLXNN.relu(conv3(x))
        x = MLXNN.relu(conv4(x))

        let (hSeq, cSeq) = lstm(x, hidden: hidden, cell: cell)
        let lastH = hSeq[0..., -1, 0...]
        let lastC = cSeq[0..., -1, 0...]
        let newState = stacked([lastH, lastC], axis: 0)

        var out = MLXNN.relu(hSeq)
        out = sigmoid(finalConv(out))
        let prob = mean(out.squeezed(axis: -1), axis: 1, keepDims: true)
        return (prob, newState)
    }

    private func splitState(_ s: MLXArray?) -> (MLXArray?, MLXArray?) {
        guard let s else { return (nil, nil) }
        precondition(s.ndim == 3 && s.dim(0) == 2, "expected state shape (2, batch, 128)")
        return (s[0], s[1])
    }
}

public final class SileroVAD: Module {
    public let config: SileroVADConfig
    fileprivate let branch16k: SileroVADBranch
    fileprivate let branch8k: SileroVADBranch

    public init(_ config: SileroVADConfig) {
        self.config = config
        self.branch16k = SileroVADBranch(config.branch16k)
        self.branch8k = SileroVADBranch(config.branch8k)
    }

    private func branch(forSampleRate sr: Int) throws -> SileroVADBranch {
        switch sr {
        case 16000: return branch16k
        case 8000: return branch8k
        default: throw SileroVADError.unsupportedSampleRate(sr)
        }
    }

    public func callAsFunction(
        _ x: MLXArray,
        state: MLXArray? = nil,
        sampleRate: Int = 16000
    ) throws -> (MLXArray, MLXArray) {
        let b = try branch(forSampleRate: sampleRate)
        return b(x, state: state)
    }

    public func initialState(batchSize: Int = 1, sampleRate: Int = 16000) throws -> SileroVADStreamingState {
        let b = try branch(forSampleRate: sampleRate)
        let context = MLXArray.zeros([batchSize, b.config.contextSize])
        return SileroVADStreamingState(lstmState: nil, context: context, sampleRate: sampleRate)
    }

    public func feed(
        chunk: MLXArray,
        state: SileroVADStreamingState? = nil,
        sampleRate: Int = 16000
    ) throws -> (MLXArray, SileroVADStreamingState) {
        let b = try branch(forSampleRate: sampleRate)
        var c = chunk
        if c.ndim == 1 { c = c[.newAxis, 0...] }
        if c.dim(-1) != b.config.chunkSize {
            throw SileroVADError.unexpectedChunkSize(expected: b.config.chunkSize, got: c.dim(-1))
        }

        var st = try state ?? initialState(batchSize: c.dim(0), sampleRate: sampleRate)
        if st.sampleRate != sampleRate {
            throw SileroVADError.stateSampleRateMismatch(expected: st.sampleRate, got: sampleRate)
        }

        let window = concatenated([st.context, c], axis: -1)
        let (prob, lstmState) = b(window, state: st.lstmState)
        let newContext = c[0..., (c.dim(-1) - b.config.contextSize)...]
        st = SileroVADStreamingState(lstmState: lstmState, context: newContext, sampleRate: sampleRate)
        return (prob, st)
    }

    public func predictProba(
        _ audio: MLXArray,
        sampleRate: Int = 16000,
        evalEvery: Int = 16
    ) throws -> MLXArray {
        let b = try branch(forSampleRate: sampleRate)
        let cs = b.config.chunkSize
        let ctx = b.config.contextSize
        var a = audio
        let originalNDim = a.ndim
        if originalNDim == 1 { a = a[.newAxis, 0...] }

        if a.dim(-1) == 0 {
            return originalNDim == 1
                ? MLXArray.zeros([0])
                : MLXArray.zeros([a.dim(0), 0])
        }

        let pad = (cs - a.dim(-1) % cs) % cs
        if pad > 0 {
            a = padded(a, widths: [.init((0, 0)), .init((0, pad))])
        }
        let preCtx = MLXArray.zeros([a.dim(0), ctx])
        a = concatenated([preCtx, a], axis: -1)

        var outputs: [MLXArray] = []
        var state: MLXArray? = nil
        var step = 0
        var pos = ctx
        while pos < a.dim(-1) {
            let window = a[0..., (pos - ctx) ..< (pos + cs)]
            let (out, newState) = b(window, state: state)
            outputs.append(out)
            state = newState
            step += 1
            pos += cs
            if step % evalEvery == 0 {
                asyncEval([out, newState])
            }
        }
        if !outputs.isEmpty, outputs.count % evalEvery != 0 {
            asyncEval([outputs.last!, state!])
        }

        var probs = concatenated(outputs, axis: 1)
        if originalNDim == 1 {
            probs = probs[0]
        }
        return probs
    }

    public func getSpeechTimestamps(
        _ audio: MLXArray,
        sampleRate: Int = 16000,
        threshold: Float? = nil,
        minSpeechDurationMs: Int? = nil,
        minSilenceDurationMs: Int? = nil,
        speechPadMs: Int? = nil
    ) throws -> [SileroVADTimestamp] {
        let probs = try predictProba(audio, sampleRate: sampleRate)
        eval(probs)
        let audioLen = audio.ndim == 1 ? audio.dim(0) : audio.dim(-1)
        return SileroVAD.probsToTimestamps(
            probs,
            audioLen: audioLen,
            sampleRate: sampleRate,
            threshold: threshold ?? config.threshold,
            minSpeechDurationMs: minSpeechDurationMs ?? config.minSpeechDurationMs,
            minSilenceDurationMs: minSilenceDurationMs ?? config.minSilenceDurationMs,
            speechPadMs: speechPadMs ?? config.speechPadMs
        )
    }

    public static func probsToTimestamps(
        _ probabilities: MLXArray,
        audioLen: Int,
        sampleRate: Int,
        threshold: Float,
        minSpeechDurationMs: Int,
        minSilenceDurationMs: Int,
        speechPadMs: Int
    ) -> [SileroVADTimestamp] {
        let probsRow = probabilities.ndim == 2 ? probabilities[0] : probabilities
        let probs = probsRow.asArray(Float.self)
        let chunkSize = sampleRate == 16000 ? 512 : 256
        let minSpeechSamples = Float(sampleRate) * Float(minSpeechDurationMs) / 1000
        let minSilenceSamples = Float(sampleRate) * Float(minSilenceDurationMs) / 1000
        let speechPadSamples = Int(Float(sampleRate) * Float(speechPadMs) / 1000)
        let negThreshold = max(threshold - 0.15, 0.01)

        struct Run { var start: Int; var end: Int }
        var speeches: [Run] = []
        var triggered = false
        var currentStart = 0
        var tempEnd = 0

        for (idx, p) in probs.enumerated() {
            let chunkStart = idx * chunkSize
            if p >= threshold && !triggered {
                triggered = true
                currentStart = chunkStart
                tempEnd = 0
                continue
            }
            if triggered && p >= threshold {
                tempEnd = 0
                continue
            }
            if triggered && p < negThreshold {
                if tempEnd == 0 { tempEnd = chunkStart }
                if Float(chunkStart - tempEnd) >= minSilenceSamples {
                    if Float(tempEnd - currentStart) >= minSpeechSamples {
                        speeches.append(Run(start: currentStart, end: tempEnd))
                    }
                    triggered = false
                    tempEnd = 0
                }
            }
        }
        if triggered {
            let end = min(audioLen, probs.count * chunkSize)
            if Float(end - currentStart) >= minSpeechSamples {
                speeches.append(Run(start: currentStart, end: end))
            }
        }

        var padded: [Run] = []
        for s in speeches {
            let start = max(0, s.start - speechPadSamples)
            let end = min(audioLen, s.end + speechPadSamples)
            if !padded.isEmpty, start <= padded[padded.count - 1].end {
                padded[padded.count - 1].end = max(padded[padded.count - 1].end, end)
            } else {
                padded.append(Run(start: start, end: end))
            }
        }
        return padded.map { SileroVADTimestamp(start: $0.start, end: $0.end) }
    }

    public static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (k, v) in weights {
            if k.hasPrefix("val_") { continue }
            var key = k
            if key.hasPrefix("vad_16k.") {
                key = "branch16k." + String(key.dropFirst("vad_16k.".count))
            } else if key.hasPrefix("vad_8k.") {
                key = "branch8k." + String(key.dropFirst("vad_8k.".count))
            }
            out[key] = v
        }
        return out
    }

    public static func fromPretrained(_ repoId: String) async throws -> SileroVAD {
        guard let repoID = Repo.ID(rawValue: repoId) else {
            throw SileroVADError.invalidRepositoryID(repoId)
        }
        let modelURL = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: "safetensors"
        )
        return try fromModelDirectory(modelURL)
    }

    public static func fromModelDirectory(_ modelURL: URL) throws -> SileroVAD {
        let configURL = modelURL.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(SileroVADConfig.self, from: configData)

        let model = SileroVAD(config)
        let weightFiles = try FileManager.default.contentsOfDirectory(
            at: modelURL,
            includingPropertiesForKeys: nil
        ).filter { $0.pathExtension == "safetensors" }
        var allWeights: [String: MLXArray] = [:]
        for url in weightFiles {
            let w = try MLX.loadArrays(url: url)
            for (k, v) in w { allWeights[k] = v }
        }
        let sanitized = sanitize(weights: allWeights)
        let parameters = ModuleParameters.unflattened(sanitized)
        try model.update(parameters: parameters, verify: [.all])
        eval(model)
        return model
    }
}
