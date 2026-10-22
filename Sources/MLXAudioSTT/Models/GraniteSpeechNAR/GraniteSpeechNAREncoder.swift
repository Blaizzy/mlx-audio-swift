import MLX
import MLXNN

struct GraniteSpeechNAREncoderOutput {
    let charLogits: MLXArray
    let bpeLogits: MLXArray
    let hiddenStates: [MLXArray]
}

class GraniteSpeechNAREncoder: Module {
    let config: GraniteSpeechNAREncoderConfig
    let layerIndices: [Int]
    @ModuleInfo(key: "input_linear") var inputLinear: Linear
    let layers: [GraniteSpeechConformerBlock]
    let out: Linear
    @ModuleInfo(key: "out_mid") var outMid: Linear
    @ModuleInfo(key: "out_bpe") var outBPE: Linear
    private let _attentionDists: MLXArray

    init(_ config: GraniteSpeechNAREncoderConfig, layerIndices: [Int]) {
        self.config = config
        self.layerIndices = layerIndices
        let c = config.conformer
        _inputLinear.wrappedValue = Linear(c.inputDim, c.hiddenDim)
        layers = (0 ..< c.numLayers).map { _ in
            GraniteSpeechConformerBlock(c, nonAutoregressive: true)
        }
        out = Linear(c.hiddenDim, c.outputDim)
        _outMid.wrappedValue = Linear(c.outputDim, c.hiddenDim)
        _outBPE.wrappedValue = Linear(c.hiddenDim, config.bpeOutputDim)
        let positions = arange(c.contextSize).asType(.int32)
        _attentionDists =
            clip(
                positions.expandedDimensions(axis: 1) - positions.expandedDimensions(axis: 0),
                min: -c.contextSize, max: c.contextSize) + c.maxPosEmb
    }

    func callAsFunction(_ features: MLXArray) -> GraniteSpeechNAREncoderOutput {
        var h = inputLinear(features)
        var states = [h]
        var charLogits: MLXArray!
        var blankProbabilities: MLXArray!
        for (i, layer) in layers.enumerated() {
            h = layer(h, attentionDists: _attentionDists)
            if i + 1 == config.selfConditioningLayer {
                charLogits = out(h)
                let probabilities = softmax(charLogits.asType(.float32), axis: -1)
                blankProbabilities = probabilities[0..., 0..., 0]
                h = h + outMid(probabilities.asType(h.dtype))
            }
            states.append(h)
        }
        let pooled = Self.posteriorWeightedPool(
            h, blankProbabilities: blankProbabilities, window: config.bpePoolingWindow)
        return GraniteSpeechNAREncoderOutput(
            charLogits: charLogits, bpeLogits: outBPE(pooled).asType(features.dtype),
            hiddenStates: layerIndices.map { states[$0 < 0 ? states.count + $0 : $0] })
    }

    static func posteriorWeightedPool(_ h: MLXArray, blankProbabilities: MLXArray, window: Int)
        -> MLXArray
    {
        let b = h.dim(0)
        let length = h.dim(1)
        let channels = h.dim(2)
        let padding = (window - length % window) % window
        let states = padded(h, widths: [.init((0, 0)), .init((0, padding)), .init((0, 0))])
        let importance = padded(
            1 - blankProbabilities, widths: [.init((0, 0)), .init((0, padding))]
        )
        .reshaped(b, -1, window)
        let weights = importance / maximum(importance.sum(axis: -1, keepDims: true), 1e-6)
        return (states.reshaped(b, -1, window, channels) * weights.expandedDimensions(axis: -1))
            .sum(axis: 2)
    }
}
