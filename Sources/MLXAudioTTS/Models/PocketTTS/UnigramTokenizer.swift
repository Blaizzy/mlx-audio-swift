struct TokenLattice {
    let sentence: String
    let bosTokenId: Int
    let eosTokenId: Int

    var nodes: [TokenLatticeNode] = []
    var beginNodes: [[TokenLatticeNode]]
    var endNodes: [[TokenLatticeNode]]

    var count: Int { sentence.count }

    init(sentence: String, bosTokenId: Int, eosTokenId: Int) {
        self.sentence = sentence
        self.bosTokenId = bosTokenId
        self.eosTokenId = eosTokenId

        beginNodes = Array(repeating: [], count: sentence.count + 1)
        endNodes = Array(repeating: [], count: sentence.count + 1)

        let bos = TokenLatticeNode(tokenId: bosTokenId, startOffset: 0, length: 0, score: 0)
        let eos = TokenLatticeNode(tokenId: eosTokenId, startOffset: sentence.count, length: 0, score: 0)

        nodes.append(bos)
        nodes.append(eos)

        beginNodes[sentence.count].append(eos)
        endNodes[0].append(bos)
    }
}

extension TokenLattice {
    mutating func insert(startOffset: Int, length: Int, score: Float, tokenId: Int) {
        let node = TokenLatticeNode(tokenId: tokenId, startOffset: startOffset, length: length, score: score)
        beginNodes[startOffset].append(node)
        endNodes[startOffset + length].append(node)
        nodes.append(node)
    }
}

extension TokenLattice {
    func viterbi() -> [TokenLatticeNode] {
        for offset in 0...count {
            guard beginNodes[offset].count > 0 else { return [] }

            for rnode in beginNodes[offset] {
                rnode.prev = nil
                var bestScore: Float = 0
                var bestNode: TokenLatticeNode?
                for lnode in endNodes[offset] {
                    let score = lnode.backtraceScore + rnode.score
                    if bestNode == nil || score > bestScore {
                        bestNode = lnode.clone()
                        bestScore = score
                    }
                }

                if bestNode != nil {
                    rnode.prev = bestNode
                    rnode.backtraceScore = bestScore
                }
            }
        }

        let root = beginNodes[count][0]
        guard let prev = root.prev else { return [] }

        var result: [TokenLatticeNode] = []
        var node = prev
        while node.prev != nil {
            result.append(node.clone())
            node = node.prev!
        }
        return result.reversed()
    }

    func piece(_ node: TokenLatticeNode) -> any StringProtocol {
        let start = sentence.index(sentence.startIndex, offsetBy: node.startOffset)
        let end = sentence.index(start, offsetBy: node.length)
        return sentence[start..<end]
    }
}

final class TokenLatticeNode {
    let tokenId: Int
    let startOffset: Int
    let length: Int
    let score: Float

    var prev: TokenLatticeNode?
    var backtraceScore: Float = 0

    init(tokenId: Int, startOffset: Int, length: Int, score: Float, prev: TokenLatticeNode? = nil, backtraceScore: Float = 0) {
        self.tokenId = tokenId
        self.startOffset = startOffset
        self.length = length
        self.score = score
        self.prev = prev
        self.backtraceScore = backtraceScore
    }

    func clone() -> TokenLatticeNode {
        TokenLatticeNode(tokenId: tokenId, startOffset: startOffset, length: length, score: score, prev: prev, backtraceScore: backtraceScore)
    }
}

// MARK: -

final class TrieNode {
    var children: [Character: TrieNode] = [:]
    var isEnd: Bool = false
}

final class Trie {
    private let root = TrieNode()

    func append(contentsOf tokens: [String]) {
        for token in tokens { insert(token) }
    }

    func insert(_ token: String) {
        var node = root
        for ch in token {
            if node.children[ch] == nil { node.children[ch] = TrieNode() }
            node = node.children[ch]!
        }
        node.isEnd = true
    }

    func commonPrefixSearchIterator(_ substring: Substring) -> [String] {
        var results: [String] = []
        var node = root
        var current = ""
        for ch in substring {
            guard let next = node.children[ch] else { break }
            current.append(ch)
            node = next
            if node.isEnd {
                results.append(current)
            }
        }
        return results
    }
}

// MARK: -

struct SentencePieceToken {
    let token: String
    let score: Float
}

enum TokenizerError: Error, CustomStringConvertible {
    case invalidJSON(String)
    case missingField(String)

    var description: String {
        switch self {
        case .invalidJSON(let msg): return "Invalid JSON: \(msg)"
        case .missingField(let msg): return "Missing field: \(msg)"
        }
    }
}

public final class UnigramTokenizer {
    let vocab: [SentencePieceToken]
    let unknownTokenId: Int
    let unknownTokenScore: Float
    let tokensToIds: [String: Int]
    let trie: Trie

    init(tokenizerJSON: [String: Any]) throws {
        guard let model = tokenizerJSON["model"] as? [String: Any] else {
            throw TokenizerError.missingField("model")
        }
        guard let unkId = model["unk_id"] as? Int else {
            throw TokenizerError.missingField("model.unk_id")
        }
        guard let vocabList = model["vocab"] as? [[Any]] else {
            throw TokenizerError.missingField("model.vocab")
        }

        var pieces: [SentencePieceToken] = []
        pieces.reserveCapacity(vocabList.count)
        for entry in vocabList {
            guard entry.count == 2,
                  let token = entry[0] as? String,
                  let score = entry[1] as? Double else {
                throw TokenizerError.invalidJSON("model.vocab entry malformed")
            }
            pieces.append(SentencePieceToken(token: token, score: Float(score)))
        }

        vocab = pieces
        unknownTokenId = unkId
        let minScore = vocab.reduce(Float.greatestFiniteMagnitude) { min($0, $1.score) }
        unknownTokenScore = minScore - 10

        var mapping: [String: Int] = [:]
        mapping.reserveCapacity(vocab.count)
        for (i, tok) in vocab.enumerated() { mapping[tok.token] = i }
        tokensToIds = mapping

        trie = Trie()
        trie.append(contentsOf: vocab.map { $0.token })
    }

    func encodeWithByteFallback(_ text: String) -> [Int] {
        let pre = applyMetaspace(text)
        var lattice = TokenLattice(sentence: pre, bosTokenId: unknownTokenId, eosTokenId: unknownTokenId)

        let sentence = lattice.sentence
        var beginPos = 0
        while beginPos < sentence.count {
            let mblen = 1
            var hasSingleNode = false

            let beginIndex = sentence.index(sentence.startIndex, offsetBy: beginPos)
            for token in trie.commonPrefixSearchIterator(sentence[beginIndex...]) {
                guard let tokenId = tokensToIds[token] else { continue }
                let tokenScore = vocab[tokenId].score
                lattice.insert(startOffset: beginPos, length: token.count, score: tokenScore, tokenId: tokenId)
                if !hasSingleNode, token.count == mblen { hasSingleNode = true }
            }

            if !hasSingleNode {
                lattice.insert(startOffset: beginPos, length: mblen, score: unknownTokenScore, tokenId: unknownTokenId)
            }
            beginPos += mblen
        }

        let path = lattice.viterbi()
        var ids: [Int] = []
        for node in path {
            if node.tokenId == unknownTokenId {
                let piece = lattice.piece(node)
                for b in piece.utf8 {
                    ids.append(byteMap[b] ?? unknownTokenId)
                }
            } else {
                ids.append(node.tokenId)
            }
        }
        return ids
    }

    func decode(_ ids: [Int]) -> String {
        var bytes: [UInt8] = []
        var pieces: [String] = []
        for id in ids {
            guard id >= 0 && id < vocab.count else { continue }
            let tok = vocab[id].token
            if tok.hasPrefix("<0x"), tok.hasSuffix(">"), tok.count == 6 {
                let hex = tok.dropFirst(3).dropLast(1)
                if let b = UInt8(hex, radix: 16) {
                    bytes.append(b)
                }
                continue
            }
            if !bytes.isEmpty {
                if let str = String(bytes: bytes, encoding: .utf8) {
                    pieces.append(str)
                }
                bytes.removeAll()
            }
            pieces.append(tok)
        }
        if !bytes.isEmpty, let str = String(bytes: bytes, encoding: .utf8) {
            pieces.append(str)
        }
        let joined = pieces.joined()
        let restored = joined.replacingOccurrences(of: "▁", with: " ")
        return restored.trimmingCharacters(in: .whitespaces)
    }

    // MARK: - Helpers

    private lazy var byteMap: [UInt8: Int] = {
        var map: [UInt8: Int] = [:]
        for (i, tok) in vocab.enumerated() {
            let piece = tok.token
            if piece.hasPrefix("<0x"), piece.hasSuffix(">"), piece.count == 6 {
                let hex = piece.dropFirst(3).dropLast(1)
                if let b = UInt8(hex, radix: 16) { map[b] = i }
            }
        }
        return map
    }()

    private func applyMetaspace(_ text: String) -> String {
        // From tokenizer.json: replacement="▁", prepend_scheme="always"
        let replaced = text.replacingOccurrences(of: " ", with: "▁")
        return "▁" + replaced
    }
}
