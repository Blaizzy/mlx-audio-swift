import Foundation
import MLXAudioCore

/// Configuration for Voxtral's Mistral backbone, flow acoustic head and audio decoder.
public struct VoxtralTTSConfiguration: Decodable, Sendable {
    public var dim = 3072
    public var layers = 26
    public var headDim = 128
    public var hiddenDim = 9216
    public var heads = 32
    public var kvHeads = 8
    public var vocabularySize = 131072
    public var ropeTheta: Float = 1_000_000
    public var normEps: Float = 1e-5
    public var useBiases = false
    public var tiedEmbeddings = true
    public var maxPositionEmbeddings = 128000
    public var sampleRate = 24000
    public var semanticSize = 8192
    public var acousticSize = 21
    public var acousticCodebooks = 36
    public var frameRate: Float = 12.5
    public var acousticDim = 3072
    public var acousticLayers = 3
    public var acousticHeadDim = 128
    public var acousticHiddenDim = 9216
    public var acousticHeads = 32
    public var acousticKVHeads = 8
    public var sigmaMax: Float = 1
    public var tokenizerDim = 1024
    public var tokenizerHiddenDim = 4096
    public var tokenizerHeads = 8
    public var tokenizerKVHeads = 8
    public var tokenizerHeadDim = 128
    public var patchSize = 240
    public var patchKernel = 7
    public var semanticDim = 256
    public var tokenizerAcousticDim = 36
    public var tokenizerNormEps: Float = 0.01
    public var qkNormEps: Float = 1e-6
    public var qkNorm = true
    public var layerScale = true
    public var layerScaleInit: Float = 0.01
    public var decoderLengths = [2,2,2,2]
    public var decoderKernels = [3,4,4,4]
    public var decoderStrides = [1,2,2,2]
    public var bosID = 1
    public var audioID = 24
    public var beginAudioID = 25
    public var quantization: BaseConfiguration.PerLayerQuantization?

    public init() {}
    private struct Key: CodingKey {
        var stringValue: String
        var intValue: Int? { nil }
        init(_ name: String) { stringValue = name }
        init?(stringValue: String) { self.init(stringValue) }
        init?(intValue: Int) { return nil }
    }
    public init(from decoder: Decoder) throws {
        self.init()
        let c = try decoder.container(keyedBy: Key.self)
        func read<T: Decodable>(_ key: String, _ fallback: T, _ source: KeyedDecodingContainer<Key>? = nil) throws -> T {
            try (source ?? c).decodeIfPresent(T.self, forKey: Key(key)) ?? fallback
        }
        dim = try read("dim", dim); layers = try read("n_layers", layers)
        headDim = try read("head_dim", headDim); hiddenDim = try read("hidden_dim", hiddenDim)
        heads = try read("n_heads", heads); kvHeads = try read("n_kv_heads", kvHeads)
        vocabularySize = try read("vocab_size", vocabularySize); ropeTheta = try read("rope_theta", ropeTheta)
        normEps = try read("norm_eps", normEps); useBiases = try read("use_biases", useBiases)
        tiedEmbeddings = try read("tied_embeddings", tiedEmbeddings)
        maxPositionEmbeddings = try read("max_position_embeddings", maxPositionEmbeddings)
        let mm = try c.nestedContainerIfPresent(keyedBy: Key.self, forKey: Key("multimodal"))
        let audio = try mm?.nestedContainerIfPresent(keyedBy: Key.self, forKey: Key("audio_model_args"))
        let encoding = try audio?.nestedContainerIfPresent(keyedBy: Key.self, forKey: Key("audio_encoding_args"))
        let acoustic = try audio?.nestedContainerIfPresent(keyedBy: Key.self, forKey: Key("acoustic_transformer_args"))
        let tokenizer = try mm?.nestedContainerIfPresent(keyedBy: Key.self, forKey: Key("audio_tokenizer_args"))
        func nested<T: Decodable>(_ source: KeyedDecodingContainer<Key>?, _ key: String, _ flat: String, _ fallback: T) throws -> T {
            if let source { return try read(key, fallback, source) }
            return try read(flat, fallback)
        }
        sampleRate = try nested(encoding, "sampling_rate", "sample_rate", sampleRate)
        frameRate = try nested(encoding, "frame_rate", "frame_rate", frameRate)
        semanticSize = try nested(audio, "semantic_codebook_size", "semantic_codebook_size", semanticSize)
        acousticSize = try nested(audio, "acoustic_codebook_size", "acoustic_codebook_size", acousticSize)
        acousticCodebooks = try nested(audio, "n_acoustic_codebook", "n_acoustic_codebook", acousticCodebooks)
        acousticDim = try nested(acoustic, "dim", "acoustic_dim", acousticDim)
        acousticLayers = try nested(acoustic, "n_layers", "acoustic_n_layers", acousticLayers)
        acousticHeadDim = try nested(acoustic, "head_dim", "acoustic_head_dim", acousticHeadDim)
        acousticHiddenDim = try nested(acoustic, "hidden_dim", "acoustic_hidden_dim", acousticHiddenDim)
        acousticHeads = try nested(acoustic, "n_heads", "acoustic_n_heads", acousticHeads)
        acousticKVHeads = try nested(acoustic, "n_kv_heads", "acoustic_n_kv_heads", acousticKVHeads)
        sigmaMax = try nested(acoustic, "sigma_max", "acoustic_sigma_max", sigmaMax)
        tokenizerDim = try nested(tokenizer, "dim", "tokenizer_dim", tokenizerDim)
        tokenizerHiddenDim = try nested(tokenizer, "hidden_dim", "tokenizer_hidden_dim", tokenizerHiddenDim)
        tokenizerHeads = try nested(tokenizer, "n_heads", "tokenizer_n_heads", tokenizerHeads)
        tokenizerKVHeads = try nested(tokenizer, "n_kv_heads", "tokenizer_n_kv_heads", tokenizerKVHeads)
        tokenizerHeadDim = try nested(tokenizer, "head_dim", "tokenizer_head_dim", tokenizerHeadDim)
        patchSize = try nested(tokenizer, "pretransform_patch_size", "tokenizer_patch_size", patchSize)
        patchKernel = try nested(tokenizer, "patch_proj_kernel_size", "tokenizer_patch_proj_kernel_size", patchKernel)
        semanticDim = try nested(tokenizer, "semantic_dim", "tokenizer_semantic_dim", semanticDim)
        tokenizerAcousticDim = try nested(tokenizer, "acoustic_dim", "tokenizer_acoustic_dim", tokenizerAcousticDim)
        tokenizerNormEps = try nested(tokenizer, "norm_eps", "tokenizer_norm_eps", tokenizerNormEps)
        qkNormEps = try nested(tokenizer, "qk_norm_eps", "tokenizer_qk_norm_eps", qkNormEps)
        qkNorm = try nested(tokenizer, "qk_norm", "tokenizer_qk_norm", qkNorm)
        layerScale = try nested(tokenizer, "layer_scale", "tokenizer_layer_scale", layerScale)
        layerScaleInit = try nested(tokenizer, "layer_scale_init", "tokenizer_layer_scale_init", layerScaleInit)
        func list(_ key: String, _ fallback: [Int]) throws -> [Int] {
            let value: String = try nested(tokenizer, key, "tokenizer_" + key, fallback.map(String.init).joined(separator: ","))
            let parts = value.split(separator: ",", omittingEmptySubsequences: false)
            let parsed = parts.compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
            guard parsed.count == parts.count else { throw AudioGenerationError.invalidInput("Invalid Voxtral decoder list: \(key)") }
            return parsed
        }
        decoderLengths = try list("decoder_transformer_lengths_str", decoderLengths)
        decoderKernels = try list("decoder_convs_kernels_str", decoderKernels)
        decoderStrides = try list("decoder_convs_strides_str", decoderStrides)
        bosID = try nested(mm, "bos_token_id", "bos_token_id", bosID)
        audioID = try nested(audio, "audio_token_id", "audio_token_id", audioID)
        beginAudioID = try nested(audio, "begin_audio_token_id", "begin_audio_token_id", beginAudioID)
        quantization = try BaseConfiguration(from: decoder).perLayerQuantization
        if let q = quantization {
            quantization = .init(quantization: q.quantization, perLayerQuantization: Dictionary(uniqueKeysWithValues:
                q.perLayerQuantization.map { (String(VoxtralTTSModel.canonicalPath($0.key + ".weight").dropLast(7)), $0.value) }))
        }
    }
    func validate() throws {
        let positive = [dim,layers,headDim,hiddenDim,heads,kvHeads,vocabularySize,maxPositionEmbeddings,sampleRate,
                        semanticSize,acousticCodebooks,acousticDim,acousticLayers,acousticHeadDim,acousticHiddenDim,
                        acousticHeads,acousticKVHeads,tokenizerDim,tokenizerHiddenDim,tokenizerHeads,tokenizerKVHeads,
                        tokenizerHeadDim,patchSize,patchKernel,semanticDim]
        guard positive.allSatisfy({ $0 > 0 }), acousticSize > 1, heads % kvHeads == 0,
              acousticHeads % acousticKVHeads == 0, tokenizerHeads % tokenizerKVHeads == 0,
              tokenizerAcousticDim == acousticCodebooks, acousticDim % 2 == 0,
              headDim % 2 == 0, frameRate.isFinite, frameRate > 0,
              normEps > 0, tokenizerNormEps > 0, qkNormEps > 0, sigmaMax.isFinite, sigmaMax > 0,
              !decoderLengths.isEmpty, decoderLengths.count == decoderStrides.count,
              decoderLengths.count == decoderKernels.count, decoderLengths.allSatisfy({ $0 > 0 }),
              zip(decoderKernels, decoderStrides).allSatisfy({ $0 >= $1 && $1 > 0 }),
              [bosID,audioID,beginAudioID].allSatisfy({ $0 >= 0 && $0 < vocabularySize }) else {
            throw AudioGenerationError.invalidInput("Invalid Voxtral TTS architecture configuration")
        }
    }
    var samplesPerFrame: Int { decoderStrides.reduce(patchSize, *) }
    var audioEmbeddingCount: Int { (semanticSize / 128 + 1) * 128 + ((acousticSize * acousticCodebooks + 127) / 128) * 128 }
}

private extension KeyedDecodingContainer {
    func nestedContainerIfPresent<NestedKey: CodingKey>(keyedBy type: NestedKey.Type, forKey key: Key) throws -> KeyedDecodingContainer<NestedKey>? {
        contains(key) ? try nestedContainer(keyedBy: type, forKey: key) : nil
    }
}

public struct VoxtralTTSGenerationOptions: Sendable {
    public var denoisingSteps: Int
    public var guidanceScale: Float
    public var sanitizeText: Bool
    public init(denoisingSteps: Int = 8, guidanceScale: Float = 1.2, sanitizeText: Bool = true) {
        self.denoisingSteps = denoisingSteps; self.guidanceScale = guidanceScale; self.sanitizeText = sanitizeText
    }
}
