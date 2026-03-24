import MLX
import MLXNN

class OutputHead: Module {
    @ModuleInfo var linear: Linear

    init(config: T5Config) {
        self._linear.wrappedValue = Linear(config.dModel, config.vocabSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear(x)
    }
}

public class T5ForConditionalGeneration: Module {
    let config: T5Config
    @ModuleInfo var wte: Embedding
    @ModuleInfo var encoder: T5Encoder
    @ModuleInfo var decoder: T5Decoder
    @ModuleInfo(key: "lm_head") var lmHead: OutputHead?

    public init(config: T5Config) {
        self.config = config
        self._wte.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.dModel
        )
        self._encoder.wrappedValue = T5Encoder(config: config)
        self._decoder.wrappedValue = T5Decoder(config: config)

        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = OutputHead(config: config)
        }
    }

    public func encode(_ inputIds: MLXArray) -> MLXArray {
        encoder(wte(inputIds))
    }

    public func decode(
        _ decoderInputIds: MLXArray,
        encoderOutput: MLXArray,
        cache: [KVCache?]? = nil
    ) -> (MLXArray, [KVCache]) {
        let embeddings = wte(decoderInputIds)
        let (decoderOutput, newCaches) = decoder(
            embeddings, memory: encoderOutput, cache: cache
        )

        let logits: MLXArray
        if config.tieWordEmbeddings {
            let scale = 1.0 / Float(config.dModel).squareRoot()
            logits = wte.asLinear(decoderOutput * scale)
        } else {
            logits = lmHead!(decoderOutput)
        }

        return (logits, newCaches)
    }
}
