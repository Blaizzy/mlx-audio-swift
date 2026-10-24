import Foundation
import Hub
import Tokenizers

func loadMossMusicTokenizer(from directory: URL) async throws -> Tokenizer {
    if FileManager.default.fileExists(
        atPath: directory.appendingPathComponent("tokenizer.json").path)
    {
        return try await loadSTTTokenizer(from: directory)
    }
    func json(_ name: String) throws -> [String: Any] {
        guard
            let dictionary = try JSONSerialization.jsonObject(
                with: Data(contentsOf: directory.appendingPathComponent(name))) as? [String: Any]
        else { throw TokenizerError.missingConfig }
        return dictionary
    }
    let configuration = try json("tokenizer_config.json")
    let vocab = try json("vocab.json")
    let merges = try String(
        contentsOf: directory.appendingPathComponent("merges.txt"), encoding: .utf8
    )
    .components(separatedBy: .newlines).filter { !$0.isEmpty && !$0.hasPrefix("#version:") }
    var added: [[String: Any]] = []
    for (key, var token) in configuration["added_tokens_decoder"] as? [String: [String: Any]] ?? [:]
    {
        guard let id = Int(key) else { continue }
        token["id"] = id
        added.append(token)
    }
    let addedPath = directory.appendingPathComponent("added_tokens.json")
    if FileManager.default.fileExists(atPath: addedPath.path) {
        for (token, id) in try json("added_tokens.json") {
            guard let id = id as? Int, !added.contains(where: { $0["id"] as? Int == id }) else {
                continue
            }
            added.append(["id": id, "content": token, "special": false, "normalized": false])
        }
    }
    added.sort { ($0["id"] as! Int) < ($1["id"] as! Int) }
    // Some Qwen2 releases distribute the equivalent vocabulary and merge files only.
    let byteLevel: [String: Any] = [
        "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false,
    ]
    let pattern =
        #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#
    let data: [String: Any] = [
        "added_tokens": added, "normalizer": ["type": "NFC"],
        "pre_tokenizer": [
            "type": "Sequence",
            "pretokenizers": [
                [
                    "type": "Split", "pattern": ["Regex": pattern], "behavior": "Isolated",
                    "invert": false,
                ], byteLevel,
            ],
        ],
        "post_processor": byteLevel, "decoder": byteLevel,
        "model": [
            "type": "BPE", "vocab": vocab, "merges": merges,
            "unk_token": configuration["unk_token"] ?? NSNull(), "byte_fallback": false,
            "fuse_unk": false, "ignore_merges": false,
        ],
    ]
    func config(_ dictionary: [String: Any]) -> Config {
        Config(Dictionary(uniqueKeysWithValues: dictionary.map { ($0.key as NSString, $0.value) }))
    }
    return try AutoTokenizer.from(
        tokenizerConfig: config(configuration), tokenizerData: config(data))
}
