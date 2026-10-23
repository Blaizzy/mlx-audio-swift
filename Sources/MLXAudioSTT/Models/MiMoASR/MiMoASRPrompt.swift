import MLX
import Tokenizers

struct MiMoASRPrompt {
    let config: MiMoASRConfig
    let tokenizer: Tokenizer

    static func language(_ value: String?) -> String? {
        switch value?.lowercased() {
        case "en", "en-us", "english": "en"
        case "zh", "zh-cn", "chinese", "mandarin": "zh"
        default: nil
        }
    }
    func text(_ ids: [Int]) -> MLXArray {
        let c = config
        let length = ids.count * c.groupSize
        var rows = MLXArray(c.speechEmptyIds.map(Int32.init)).expandedDimensions(axis: -1)
        rows = broadcast(rows, to: [c.audioChannels, length])
        var tokens = MLXArray.full([length], values: MLXArray(Int32(-100)))
        tokens[.stride(by: c.groupSize)] = MLXArray(ids.map(Int32.init))
        return concatenated([tokens.expandedDimensions(axis: 0), rows], axis: 0)
    }
    func paddedCodes(_ codes: MLXArray) -> MLXArray {
        precondition(codes.ndim == 2 && codes.dim(0) == config.audioChannels && codes.dim(1) > 0)
        let remainder = codes.dim(1) % config.groupSize
        guard remainder != 0 else { return codes.asType(.int32) }
        return concatenated(
            [
                codes,
                broadcast(
                    codes[0..., (-1)...], to: [config.audioChannels, config.groupSize - remainder]),
            ], axis: 1
        ).asType(.int32)
    }
    func callAsFunction(codes: MLXArray, language: String?) -> MLXArray {
        let c = config
        let codes = paddedCodes(codes)
        let groups = codes.dim(1) / c.groupSize
        var speech = broadcast(
            MLXArray(c.speechEmptyIds.map(Int32.init)).expandedDimensions(axis: -1),
            to: [c.audioChannels, (groups + 2) * c.groupSize])
        speech[0..., c.groupSize ..< (c.groupSize + codes.dim(1))] = codes
        let tokens =
            [c.startOfSpeechId] + Array(repeating: c.emptyId, count: groups) + [c.endOfSpeechId]
        var audio = text(tokens)
        audio[1...] = speech
        func encode(_ text: String) -> [Int] {
            tokenizer.encode(text: text, addSpecialTokens: false)
        }
        let instruction = language == "en" ? "Please transcribe this audio file" : "将音频内容转换成文字格式"
        // Resolve language markers through the checkpoint's vocabulary.
        let suffix =
            language == nil
            ? " thinking\n\n response\n<chinese>"
            : "<think>\n\n</think>\n<\(language == "en" ? "english" : "chinese")>"
        return concatenated(
            [
                text(encode("<|im_start|>user\n")), audio, text(encode(instruction)),
                text(encode("<|im_end|>\n")), text(encode("<|im_start|>assistant\n")),
                text(encode(suffix)),
            ], axis: 1)
    }
}
