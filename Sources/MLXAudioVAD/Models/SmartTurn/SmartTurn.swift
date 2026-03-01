import Foundation
import HuggingFace
import MLX
import MLXAudioCore
import MLXLMCommon
import MLXNN

public struct SmartTurnEndpointOutput: Sendable {
    public let prediction: Int
    public let probability: Float

    public init(prediction: Int, probability: Float) {
        self.prediction = prediction
        self.probability = probability
    }
}

public enum SmartTurnModelError: Error, LocalizedError {
    case invalidRepositoryID(String)

    public var errorDescription: String? {
        switch self {
        case .invalidRepositoryID(let repoID):
            return "Invalid repository ID: \(repoID)"
        }
    }
}

private class SmartTurnWhisperAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scaling: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ config: SmartTurnEncoderConfig) {
        numHeads = config.encoderAttentionHeads
        headDim = config.dModel / config.encoderAttentionHeads
        scaling = sqrt(Float(headDim))

        self._qProj.wrappedValue = Linear(config.dModel, config.dModel, bias: true)
        self._kProj.wrappedValue = Linear(config.dModel, config.dModel, bias: config.kProjBias)
        self._vProj.wrappedValue = Linear(config.dModel, config.dModel, bias: true)
        self._outProj.wrappedValue = Linear(config.dModel, config.dModel, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let bsz = x.dim(0)
        let seqLen = x.dim(1)

        var q = qProj(x).reshaped(bsz, seqLen, numHeads, headDim)
        var k = kProj(x).reshaped(bsz, seqLen, numHeads, headDim)
        var v = vProj(x).reshaped(bsz, seqLen, numHeads, headDim)

        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 3, 1)
        v = v.transposed(0, 2, 1, 3)

        var attn = MLX.matmul(q, k) / scaling
        attn = softmax(attn, axis: -1)

        var out = MLX.matmul(attn, v)
        out = out.transposed(0, 2, 1, 3).reshaped(bsz, seqLen, numHeads * headDim)
        return outProj(out)
    }
}

private class SmartTurnWhisperEncoderLayer: Module {
    @ModuleInfo(key: "self_attn_layer_norm") var selfAttnLayerNorm: LayerNorm
    @ModuleInfo(key: "self_attn") var selfAttn: SmartTurnWhisperAttention
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    init(_ config: SmartTurnEncoderConfig) {
        self._selfAttnLayerNorm.wrappedValue = LayerNorm(dimensions: config.dModel)
        self._selfAttn.wrappedValue = SmartTurnWhisperAttention(config)
        self._fc1.wrappedValue = Linear(config.dModel, config.encoderFfnDim, bias: true)
        self._fc2.wrappedValue = Linear(config.encoderFfnDim, config.dModel, bias: true)
        self._finalLayerNorm.wrappedValue = LayerNorm(dimensions: config.dModel)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var residual = x
        var h = selfAttnLayerNorm(x)
        h = selfAttn(h)
        h = h + residual

        residual = h
        h = finalLayerNorm(h)
        h = fc2(gelu(fc1(h)))
        h = h + residual
        return h
    }
}

private class SmartTurnWhisperEncoder: Module {
    let config: SmartTurnEncoderConfig

    @ModuleInfo(key: "conv1") var conv1: Conv1d
    @ModuleInfo(key: "conv2") var conv2: Conv1d
    @ModuleInfo(key: "embed_positions") var embedPositions: Embedding
    @ModuleInfo(key: "layers") var layers: [SmartTurnWhisperEncoderLayer]
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm

    init(_ config: SmartTurnEncoderConfig) {
        self.config = config

        self._conv1.wrappedValue = Conv1d(
            inputChannels: config.numMelBins,
            outputChannels: config.dModel,
            kernelSize: 3,
            padding: 1
        )
        self._conv2.wrappedValue = Conv1d(
            inputChannels: config.dModel,
            outputChannels: config.dModel,
            kernelSize: 3,
            stride: 2,
            padding: 1
        )
        self._embedPositions.wrappedValue = Embedding(
            embeddingCount: config.maxSourcePositions,
            dimensions: config.dModel
        )
        self._layers.wrappedValue = (0..<config.encoderLayers).map { _ in
            SmartTurnWhisperEncoderLayer(config)
        }
        self._layerNorm.wrappedValue = LayerNorm(dimensions: config.dModel)
    }

    func callAsFunction(_ inputFeatures: MLXArray) -> MLXArray {
        // Input follows HF convention: (batch, n_mels, n_frames)
        var x = inputFeatures.transposed(0, 2, 1)
        x = gelu(conv1(x))
        x = gelu(conv2(x))

        let positions = MLXArray(0..<x.dim(1))
        x = x + embedPositions(positions)

        for layer in layers {
            x = layer(x)
        }

        return layerNorm(x)
    }
}

public class SmartTurnModel: Module {
    public let config: SmartTurnConfig

    @ModuleInfo(key: "encoder") private var encoder: SmartTurnWhisperEncoder
    @ModuleInfo(key: "pool_attention_0") private var poolAttention0: Linear
    @ModuleInfo(key: "pool_attention_2") private var poolAttention2: Linear
    @ModuleInfo(key: "classifier_0") private var classifier0: Linear
    @ModuleInfo(key: "classifier_1") private var classifier1: LayerNorm
    @ModuleInfo(key: "classifier_4") private var classifier4: Linear
    @ModuleInfo(key: "classifier_6") private var classifier6: Linear

    public init(_ config: SmartTurnConfig) {
        self.config = config
        let dModel = config.encoderConfig.dModel

        self._encoder.wrappedValue = SmartTurnWhisperEncoder(config.encoderConfig)
        self._poolAttention0.wrappedValue = Linear(dModel, 256)
        self._poolAttention2.wrappedValue = Linear(256, 1)
        self._classifier0.wrappedValue = Linear(dModel, 256)
        self._classifier1.wrappedValue = LayerNorm(dimensions: 256)
        self._classifier4.wrappedValue = Linear(256, 64)
        self._classifier6.wrappedValue = Linear(64, 1)
    }

    public var modelDType: DType {
        config.dtype == "float16" ? .float16 : .float32
    }

    public func callAsFunction(_ inputFeatures: MLXArray, returnLogits: Bool = false) -> MLXArray {
        var features = inputFeatures
        if features.ndim == 2 {
            features = features.expandedDimensions(axis: 0)
        }

        let hidden = encoder(features.asType(modelDType))
        var attn = poolAttention2(tanh(poolAttention0(hidden)))
        attn = softmax(attn, axis: 1)
        let pooled = MLX.sum(hidden * attn, axis: 1)

        var x = classifier0(pooled)
        x = classifier1(x)
        x = gelu(x)
        x = classifier4(x)
        x = gelu(x)
        let logits = classifier6(x)

        if returnLogits {
            return logits
        }
        return sigmoid(logits)
    }

    func prepareAudioSamples(_ audio: MLXArray, sampleRate: Int? = nil) throws -> [Float] {
        try smartTurnPrepareAudioSamples(
            audio,
            sampleRate: sampleRate,
            processor: config.processorConfig
        )
    }

    public func prepareInputFeatures(_ audio: MLXArray, sampleRate: Int? = nil) throws -> MLXArray {
        let prepared = try prepareAudioSamples(audio, sampleRate: sampleRate)
        var mel = smartTurnLogMelSpectrogram(
            prepared,
            sampleRate: config.processorConfig.samplingRate,
            nFft: config.processorConfig.nFft,
            hopLength: config.processorConfig.hopLength,
            nMels: config.processorConfig.nMels
        )

        guard mel.ndim == 2 else {
            throw SmartTurnError.invalidFeatureShape(mel.shape)
        }

        if mel.dim(1) != config.processorConfig.nMels {
            mel = mel.transposed(1, 0)
        }

        let frames = mel.dim(0)
        let targetFrames = config.processorConfig.maxAudioSeconds
            * config.processorConfig.samplingRate
            / config.processorConfig.hopLength

        if frames > targetFrames {
            mel = mel[(frames - targetFrames)..., 0...]
        } else if frames < targetFrames {
            mel = MLX.padded(
                mel,
                widths: [.init((targetFrames - frames, 0)), .init((0, 0))]
            )
        }

        // Return HF-style shape: (n_mels, n_frames)
        return mel.transposed(1, 0).asType(modelDType)
    }

    public func prepareInputFeatures(audioURL: URL) throws -> MLXArray {
        let targetSampleRate = config.processorConfig.samplingRate
        let (_, audio) = try loadAudioArray(from: audioURL, sampleRate: targetSampleRate)
        return try prepareInputFeatures(audio, sampleRate: targetSampleRate)
    }

    public func predictEndpoint(
        _ audio: MLXArray,
        sampleRate: Int? = nil,
        threshold: Float? = nil
    ) throws -> SmartTurnEndpointOutput {
        let features = try prepareInputFeatures(audio, sampleRate: sampleRate)
        let probability = self(features)[0, 0].item(Float.self)
        let thresholdValue = threshold ?? config.processorConfig.threshold
        let prediction = probability > thresholdValue ? 1 : 0
        return SmartTurnEndpointOutput(prediction: prediction, probability: probability)
    }

    public func predictEndpoint(audioURL: URL, threshold: Float? = nil) throws -> SmartTurnEndpointOutput {
        let features = try prepareInputFeatures(audioURL: audioURL)
        let probability = self(features)[0, 0].item(Float.self)
        let thresholdValue = threshold ?? config.processorConfig.threshold
        let prediction = probability > thresholdValue ? 1 : 0
        return SmartTurnEndpointOutput(prediction: prediction, probability: probability)
    }

    private static func remapKey(_ key: String) -> String {
        var out = key
        if out.hasPrefix("inner.") {
            out.removeFirst("inner.".count)
        }

        out = out.replacingOccurrences(of: "pool_attention.0.", with: "pool_attention_0.")
        out = out.replacingOccurrences(of: "pool_attention.2.", with: "pool_attention_2.")
        out = out.replacingOccurrences(of: "classifier.0.", with: "classifier_0.")
        out = out.replacingOccurrences(of: "classifier.1.", with: "classifier_1.")
        out = out.replacingOccurrences(of: "classifier.4.", with: "classifier_4.")
        out = out.replacingOccurrences(of: "classifier.6.", with: "classifier_6.")
        return out
    }

    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()

        for (key, var value) in weights {
            if key.hasPrefix("val_") {
                continue
            }

            let targetKey = remapKey(key)

            if (targetKey == "encoder.conv1.weight" || targetKey == "encoder.conv2.weight"),
               value.ndim == 3 {
                value = value.transposed(0, 2, 1)
            }

            if targetKey.hasSuffix("fc1.weight"), value.ndim == 2, value.dim(0) < value.dim(1) {
                value = value.transposed(1, 0)
            }

            if targetKey.hasSuffix("fc2.weight"), value.ndim == 2, value.dim(0) > value.dim(1) {
                value = value.transposed(1, 0)
            }

            if targetKey == "pool_attention_0.weight", value.ndim == 2, value.dim(0) != 256 {
                value = value.transposed(1, 0)
            }

            if targetKey == "pool_attention_2.weight", value.ndim == 2, value.dim(0) != 1 {
                value = value.transposed(1, 0)
            }

            sanitized[targetKey] = value
        }

        return sanitized
    }

    public static func fromPretrained(_ repoId: String) async throws -> SmartTurnModel {
        guard let repoID = Repo.ID(rawValue: repoId) else {
            throw SmartTurnModelError.invalidRepositoryID(repoId)
        }

        let modelURL = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: "safetensors"
        )

        let configURL = modelURL.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(SmartTurnConfig.self, from: configData)

        let model = SmartTurnModel(config)
        let weightFiles = try FileManager.default.contentsOfDirectory(
            at: modelURL,
            includingPropertiesForKeys: nil
        ).filter { $0.pathExtension == "safetensors" }

        var allWeights = [String: MLXArray]()
        for file in weightFiles {
            let weights = try loadArrays(url: file)
            for (k, v) in weights {
                allWeights[k] = v
            }
        }

        let sanitized = sanitize(allWeights)
        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: .noUnusedKeys)
        eval(model.parameters())
        return model
    }
}
