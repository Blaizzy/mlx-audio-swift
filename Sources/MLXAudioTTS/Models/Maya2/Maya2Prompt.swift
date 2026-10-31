import Foundation
import MLX
import MLXAudioCore

struct Maya2Prompt {
    var tokens: [Int] = []
    var kinds: [Int] = []
    var codes: [[Int32]] = []
    var sentinels: [Int] = []

    static func build(text: String, language: String, voice: String?, referenceText: String?,
                      referenceCodes: MLXArray?, referenceLanguage: String?, numCodebooks: Int,
                      encode: (String) -> [Int], tokenID: (String) -> Int?) throws -> Self {
        guard Maya2Model.languages.contains(language),
              referenceLanguage.map({ Maya2Model.languages.contains($0) }) ?? true else {
            throw AudioGenerationError.invalidInput("Supported Maya-2 languages: \(Maya2Model.languages.joined(separator: ", "))")
        }
        let text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { throw AudioGenerationError.invalidInput("Speech text must not be empty") }
        var p = Self()
        func control(_ name: String) throws -> Int {
            guard let id = tokenID(name), id >= 0 else {
                throw AudioGenerationError.invalidInput("Maya-2 tokenizer is missing \(name)")
            }
            return id
        }
        func span(_ content: String, _ lang: String?) throws {
            let ids = (try lang.map { [try control("<|lang:\($0)|>")] } ?? [])
                + encode(content) + [try control("<|audio|>")]
            p.tokens += ids; p.kinds += Array(repeating: 0, count: ids.count)
            p.codes += Array(repeating: Array(repeating: -1, count: numCodebooks), count: ids.count)
            p.sentinels.append(p.tokens.count - 1)
        }
        let targetLanguage: String?
        if let referenceCodes {
            guard referenceCodes.ndim == 2, referenceCodes.dim(0) == numCodebooks,
                  referenceCodes.dim(1) > 0 else { throw AudioGenerationError.invalidInput("Invalid reference code shape") }
            guard let referenceText, !referenceText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw AudioGenerationError.invalidInput("refText is required for in-context voice cloning")
            }
            let refLang = referenceLanguage ?? language
            try span(referenceText, refLang)
            let n = referenceCodes.dim(1)
            let flat = referenceCodes.transposed().asArray(Int32.self)
            p.tokens += Array(repeating: 0, count: n + 1)
            p.kinds += Array(repeating: 1, count: n) + [3]
            for i in 0..<n { p.codes.append(Array(flat[(i * numCodebooks)..<((i + 1) * numCodebooks)])) }
            p.codes.append(Array(repeating: -1, count: numCodebooks))
            targetLanguage = refLang == language ? nil : language
        } else { targetLanguage = language }
        try span(voice.map { "{\($0)}: \(text)" } ?? text, targetLanguage)
        return p
    }
}
