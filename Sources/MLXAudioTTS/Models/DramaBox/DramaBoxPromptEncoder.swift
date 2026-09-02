import Foundation
@preconcurrency import MLX
import MLXNN

/// Text → `a_ctx` `[B, 1024, 2048]`. Gemma is held outside the Module tree
/// so quantized 4-bit weights are not mixed into DiT/VAE `load_weights`.
final class DramaBoxPromptEncoder {
    let gemma: DramaBoxGemma3TextBackbone
    let tokenizer: LTXVGemmaTokenizer
    let processor: DramaBoxEmbeddingsProcessor

    init(
        gemma: DramaBoxGemma3TextBackbone,
        tokenizer: LTXVGemmaTokenizer,
        processor: DramaBoxEmbeddingsProcessor
    ) {
        self.gemma = gemma
        self.tokenizer = tokenizer
        self.processor = processor
    }

    func encode(_ text: String, maxLength: Int = 1024) -> MLXArray {
        let (inputIds, attentionMask) = tokenizer.encode(text, maxLength: maxLength)
        let gemmaOut = gemma(inputIds, attentionMask: attentionMask)
        return processor(gemmaOut.hiddenStates, attentionMask: attentionMask).audioEncoding
    }
}
