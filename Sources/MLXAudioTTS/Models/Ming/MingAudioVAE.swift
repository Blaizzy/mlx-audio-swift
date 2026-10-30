import Foundation
import MLX
import MLXNN
import MLXAudioCore

final class MingISTFTWindow: Module {
    @ParameterInfo var window: MLXArray
    init(_ count: Int) {
        _window.wrappedValue = 0.5 - 0.5 * cos(2 * Float.pi * MLXArray(0..<count).asType(.float32) / Float(count))
    }
}
public final class MingISTFTHead: Module {
    @ModuleInfo var out: Linear
    let istft: MingISTFTWindow
    let nFFT: Int
    let hop: Int
    public init(dim: Int, nFFT: Int, hop: Int) {
        self.nFFT = nFFT; self.hop = hop
        _out.wrappedValue = Linear(dim, nFFT + 2)
        istft = MingISTFTWindow(nFFT)
    }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let projected = out(x).asType(.float32)
        let half = nFFT / 2 + 1
        let magnitude = minimum(exp(projected[0..., 0..., ..<half]), 100)
        let phase = projected[0..., 0..., half...]
        let complex = magnitude * cos(phase) + MLXArray(real: Float(0), imaginary: Float(1)) * magnitude * sin(phase)
        let frames = MLXFFT.irfft(complex, n: nFFT, axis: -1) * istft.window
        let count = frames.dim(1), length = (count - 1) * hop + nFFT
        let window = istft.window.asArray(Float.self)
        let values = frames.asArray(Float.self)
        let pad = (nFFT - hop) / 2
        var result: [MLXArray] = []
        for batch in 0..<x.dim(0) {
            var audio = [Float](repeating: 0, count: length)
            var envelope = [Float](repeating: 0, count: length)
            for t in 0..<count {
                for j in 0..<nFFT {
                    audio[t * hop + j] += values[(batch * count + t) * nFFT + j]
                    envelope[t * hop + j] += window[j] * window[j]
                }
            }
            for j in pad..<(length - pad) { audio[j] /= max(envelope[j], 1e-11) }
            result.append(MLXArray(Array(audio[pad..<(length - pad)])))
        }
        return stacked(result, axis: 0)
    }
}

public final class MingAudioEncoder: Module {
    public let encoder: Qwen2ModelInner
    public let aggregator: Qwen2ModelInner?
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear
    @ModuleInfo var fc3: Linear
    let norm: LayerNorm
    @ParameterInfo(key: "cls_embed") var clsEmbed: MLXArray?
    let config: MingAudioBackboneConfiguration
    let patchSize: Int
    public init(_ config: MingAudioBackboneConfiguration, patchSize: Int) {
        self.config = config; self.patchSize = patchSize
        let a = config.backbone
        encoder = Qwen2ModelInner(a, slidingWindow: config.useSlidingWindow ? config.slidingWindow : nil,
                                 slidingLayerStart: config.maxWindowLayers)
        _fc1.wrappedValue = Linear(config.inputDim, a.hiddenSize, bias: false)
        _fc2.wrappedValue = Linear(a.hiddenSize, a.hiddenSize)
        _fc3.wrappedValue = Linear(a.hiddenSize, 2 * config.latentDim)
        norm = LayerNorm(dimensions: a.hiddenSize)
        if patchSize > 0 {
            var ag = a; ag.hiddenLayers = 4
            aggregator = Qwen2ModelInner(ag, slidingWindow: config.useSlidingWindow ? config.slidingWindow : nil,
                                       slidingLayerStart: config.maxWindowLayers)
            _clsEmbed.wrappedValue = MLXArray.zeros([1, 1, a.hiddenSize])
        } else { aggregator = nil }
    }
    public func callAsFunction(_ waveform: MLXArray) -> MLXArray {
        let frames = (waveform.dim(1) + config.hopSize - 1) / config.hopSize
        let expected = (frames - 1) * config.hopSize + config.inputDim
        let padded = MLX.padded(waveform.asType(.float32), widths: [.init(0), .init((0, max(0, expected - waveform.dim(1))))])
        let framed = asStrided(padded, [waveform.dim(0), frames, config.inputDim],
                               strides: [expected, config.hopSize, 1])
        let dtype = (fc1 as? QuantizedLinear)?.scales.dtype ?? fc1.weight.dtype
        var x = encoder(inputEmbeddings: fc2(fc1(framed.asType(dtype))))
        if let aggregator, let clsEmbed {
            let padding = (patchSize - x.dim(1) % patchSize) % patchSize
            x = MLX.padded(x, widths: [.init(0), .init((0, padding)), .init(0)])
            let b = x.dim(0), dim = x.dim(2)
            x = x.reshaped(-1, patchSize, dim)
            x = concatenated([x, broadcast(clsEmbed.asType(x.dtype), to: [x.dim(0), 1, dim])], axis: 1)
                .reshaped(b, -1, dim)
            x = aggregator(inputEmbeddings: x).reshaped(-1, patchSize + 1, dim)
            x = x[0..., patchSize..<(patchSize + 1), 0...].reshaped(b, -1, dim)
        }
        return fc3(x)
    }
}
public final class MingAudioDecoder: Module {
    public let decoder: Qwen2ModelInner
    @ModuleInfo var fc1: Linear
    public let head: MingISTFTHead
    let patchSize: Int
    public init(_ config: MingAudioBackboneConfiguration, patchSize: Int) {
        self.patchSize = patchSize
        decoder = Qwen2ModelInner(config.backbone, slidingWindow: config.useSlidingWindow ? config.slidingWindow : nil,
                                 slidingLayerStart: config.maxWindowLayers)
        _fc1.wrappedValue = Linear(config.latentDim, config.backbone.hiddenSize)
        head = MingISTFTHead(dim: config.backbone.hiddenSize, nFFT: config.outputDim * 4, hop: config.outputDim)
    }
    public func upsample(_ x: MLXArray) -> MLXArray {
        guard patchSize > 0 else { return x }
        let index = (MLXArray(0..<(x.dim(1) * patchSize)).asType(.float32) + 0.5) / Float(patchSize) - 0.5
        let left = floor(index).asType(.int32), right = floor(index).asType(.int32) + 1
        let weight = (index - left.asType(.float32))[.newAxis, 0..., .newAxis]
        let xf = x.asType(.float32)
        return take(xf, clip(left, min: 0, max: x.dim(1) - 1), axis: 1) * (1 - weight)
            + take(xf, clip(right, min: 0, max: x.dim(1) - 1), axis: 1) * weight
    }
    public func callAsFunction(_ latent: MLXArray) -> MLXArray {
        head(decoder(inputEmbeddings: upsample(fc1(latent))))
    }
}

/// Transformer audio VAE used by Ming dense, Tiny, and the older Omni family.
public final class MingAudioVAE: Module {
    public let encoder: MingAudioEncoder
    public let decoder: MingAudioDecoder
    public let config: MingAudioVAEConfiguration
    public init(_ config: MingAudioVAEConfiguration) {
        self.config = config
        encoder = MingAudioEncoder(config.encoder, patchSize: config.patchSize)
        decoder = MingAudioDecoder(config.decoder, patchSize: config.patchSize)
    }
    public func encode(_ waveform: MLXArray, noise: MLXArray? = nil) -> MLXArray {
        let h = encoder(waveform), dim = config.encoder.latentDim
        let mu = h[0..., 0..., ..<dim], logvar = h[0..., 0..., dim...]
        // Python draws [B,C,T], then transposes to the external [B,T,C] layout.
        let epsilon = noise ?? MLXRandom.normal([mu.dim(0), dim, mu.dim(1)], dtype: mu.dtype).transposed(0, 2, 1)
        return mu + exp(0.5 * clip(logvar, min: -30, max: 20)) * epsilon
    }
    public func decode(_ latent: MLXArray) -> MLXArray { decoder(latent) }
}

/// Per-utterance state for the legacy Ming decoder. It holds one patch of interpolation lookahead.
public final class MingAudioDecoderState {
    var previous: MLXArray?
    var historyLast: MLXArray?
    var cache: [KVCache]
    var audioOverlap: [[Float]]?
    var windowOverlap: [[Float]]?
    public init(configuration: MingAudioVAEConfiguration) {
        cache = (0..<configuration.decoder.backbone.hiddenLayers).map { _ in KVCacheSimple() }
    }
}

extension MingISTFTHead {
    func streaming(_ x: MLXArray, state: MingAudioDecoderState, lastChunk: Bool) -> MLXArray {
        let projected = out(x).asType(.float32), half = nFFT / 2 + 1
        let magnitude = minimum(exp(projected[0..., 0..., ..<half]), 100)
        let phase = projected[0..., 0..., half...]
        let spectrum = magnitude * cos(phase) + MLXArray(real: Float(0), imaginary: Float(1)) * magnitude * sin(phase)
        let frames = MLXFFT.irfft(spectrum, n: nFFT, axis: -1) * istft.window
        let count = frames.dim(1), length = (count - 1) * hop + nFFT
        let values = frames.asArray(Float.self), window = istft.window.asArray(Float.self)
        let overlap = nFFT - hop, pad = overlap / 2
        var result: [MLXArray] = [], nextAudio: [[Float]] = [], nextWindow: [[Float]] = []
        for batch in 0..<x.dim(0) {
            var audio = [Float](repeating: 0, count: length), envelope = audio
            for t in 0..<count {
                for j in 0..<nFFT {
                    audio[t * hop + j] += values[(batch * count + t) * nFFT + j]
                    envelope[t * hop + j] += window[j] * window[j]
                }
            }
            if let previous = state.audioOverlap, let previousWindow = state.windowOverlap {
                for j in 0..<overlap { audio[j] += previous[batch][j]; envelope[j] += previousWindow[batch][j] }
            } else {
                audio.removeFirst(pad); envelope.removeFirst(pad)
            }
            nextAudio.append(Array(audio.suffix(overlap))); nextWindow.append(Array(envelope.suffix(overlap)))
            let emitted = audio.count - (lastChunk ? pad : overlap)
            for j in 0..<emitted { audio[j] /= max(envelope[j], 1e-11) }
            result.append(MLXArray(Array(audio.prefix(emitted))))
        }
        state.audioOverlap = nextAudio; state.windowOverlap = nextWindow
        return stacked(result)
    }
}

extension MingAudioVAE {
    /// Decode successive latent patches with native causal caches and overlap-add state.
    /// Pass `lastChunk` for the final patch to flush interpolation lookahead and the waveform tail.
    public func decode(_ latent: MLXArray, state: MingAudioDecoderState, lastChunk: Bool) -> MLXArray {
        let x = decoder.fc1(latent)
        var pieces: [MLXArray] = []
        if let previous = state.previous {
            let lookahead = x[0..., ..<1, 0...]
            let joined: MLXArray
            let start: Int
            if let history = state.historyLast {
                joined = concatenated([history, previous, lookahead], axis: 1); start = decoder.patchSize
            } else { joined = concatenated([previous, lookahead], axis: 1); start = 0 }
            pieces.append(decoder.upsample(joined)[0..., start..<(start + previous.dim(1) * decoder.patchSize), 0...])
            state.historyLast = previous[0..., (previous.dim(1) - 1)..., 0...]
            state.previous = x
            if lastChunk {
                let final = decoder.upsample(concatenated([state.historyLast!, x], axis: 1))
                pieces.append(final[0..., decoder.patchSize..., 0...])
                state.previous = nil
            }
        } else if lastChunk { pieces = [decoder.upsample(x)] }
        else { state.previous = x }
        guard !pieces.isEmpty else { return .zeros([latent.dim(0), 0]) }
        let embedded = pieces.count == 1 ? pieces[0] : concatenated(pieces, axis: 1)
        return decoder.head.streaming(decoder.decoder(inputEmbeddings: embedded, cache: state.cache), state: state, lastChunk: lastChunk)
    }
}
