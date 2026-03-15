import Foundation

enum SentencePiecePieceType: Int {
    case normal = 1
    case unknown = 2
    case control = 3
    case userDefined = 4
    case unused = 5
    case byte = 6
}

enum SentencePieceModelError: Error, CustomStringConvertible {
    case malformedVarint
    case truncatedField
    case unsupportedWireType(UInt64)
    case missingVocabulary

    var description: String {
        switch self {
        case .malformedVarint:
            "Malformed SentencePiece protobuf varint"
        case .truncatedField:
            "Truncated SentencePiece protobuf field"
        case .unsupportedWireType(let wireType):
            "Unsupported SentencePiece protobuf wire type: \(wireType)"
        case .missingVocabulary:
            "SentencePiece model did not contain any vocabulary pieces"
        }
    }
}

struct SentencePieceProtobufReader {
    let data: Data
    var index: Data.Index

    init(_ data: Data) {
        self.data = data
        self.index = data.startIndex
    }

    var isAtEnd: Bool { index >= data.endIndex }

    mutating func readVarint() throws -> UInt64 {
        var value: UInt64 = 0
        var shift: UInt64 = 0

        while index < data.endIndex, shift < 64 {
            let byte = data[index]
            index = data.index(after: index)
            value |= UInt64(byte & 0x7f) << shift
            if byte & 0x80 == 0 {
                return value
            }
            shift += 7
        }

        throw SentencePieceModelError.malformedVarint
    }

    mutating func readLengthDelimited() throws -> Data {
        let length = Int(try readVarint())
        let end = data.index(index, offsetBy: length, limitedBy: data.endIndex)
        guard let end else {
            throw SentencePieceModelError.truncatedField
        }
        let slice = data[index..<end]
        index = end
        return Data(slice)
    }

    mutating func readFixed32() throws -> UInt32 {
        let end = data.index(index, offsetBy: 4, limitedBy: data.endIndex)
        guard let end else {
            throw SentencePieceModelError.truncatedField
        }
        let slice = data[index..<end]
        index = end
        return slice.enumerated().reduce(UInt32(0)) { partial, entry in
            partial | (UInt32(entry.element) << (entry.offset * 8))
        }
    }

    mutating func skipField(wireType: UInt64) throws {
        switch wireType {
        case 0:
            _ = try readVarint()
        case 1:
            guard let end = data.index(index, offsetBy: 8, limitedBy: data.endIndex) else {
                throw SentencePieceModelError.truncatedField
            }
            index = end
        case 2:
            _ = try readLengthDelimited()
        case 5:
            _ = try readFixed32()
        default:
            throw SentencePieceModelError.unsupportedWireType(wireType)
        }
    }
}

struct SentencePieceModelParser {
    static func parsePieces(from data: Data) throws -> (pieces: [SentencePieceToken], unknownTokenId: Int) {
        var reader = SentencePieceProtobufReader(data)
        var pieces: [SentencePieceToken] = []
        var unknownTokenId: Int?

        while !reader.isAtEnd {
            let key = try reader.readVarint()
            let fieldNumber = Int(key >> 3)
            let wireType = key & 0x7

            if fieldNumber == 1, wireType == 2 {
                let pieceData = try reader.readLengthDelimited()
                if let piece = try parsePiece(from: pieceData) {
                    if piece.type == .unknown, unknownTokenId == nil {
                        unknownTokenId = pieces.count
                    }
                    pieces.append(piece)
                }
            } else {
                try reader.skipField(wireType: wireType)
            }
        }

        guard !pieces.isEmpty else {
            throw SentencePieceModelError.missingVocabulary
        }

        let resolvedUnknownId = unknownTokenId
            ?? pieces.firstIndex(where: { $0.token == "<unk>" })
            ?? 0
        return (pieces, resolvedUnknownId)
    }

    private static func parsePiece(from data: Data) throws -> SentencePieceToken? {
        var reader = SentencePieceProtobufReader(data)
        var token: String?
        var score: Float = 0
        var type: SentencePiecePieceType = .normal

        while !reader.isAtEnd {
            let key = try reader.readVarint()
            let fieldNumber = Int(key >> 3)
            let wireType = key & 0x7

            switch (fieldNumber, wireType) {
            case (1, 2):
                let tokenData = try reader.readLengthDelimited()
                token = String(decoding: tokenData, as: UTF8.self)
            case (2, 5):
                score = Float(bitPattern: try reader.readFixed32())
            case (3, 0):
                let rawType = Int(try reader.readVarint())
                type = SentencePiecePieceType(rawValue: rawType) ?? .normal
            default:
                try reader.skipField(wireType: wireType)
            }
        }

        guard let token else { return nil }
        return SentencePieceToken(token: token, score: score, type: type)
    }
}

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

    init(
        tokenId: Int,
        startOffset: Int,
        length: Int,
        score: Float,
        prev: TokenLatticeNode? = nil,
        backtraceScore: Float = 0
    ) {
        self.tokenId = tokenId
        self.startOffset = startOffset
        self.length = length
        self.score = score
        self.prev = prev
        self.backtraceScore = backtraceScore
    }

    func clone() -> TokenLatticeNode {
        TokenLatticeNode(
            tokenId: tokenId,
            startOffset: startOffset,
            length: length,
            score: score,
            prev: prev,
            backtraceScore: backtraceScore
        )
    }
}

final class TrieNode {
    var children: [Character: TrieNode] = [:]
    var isEnd = false
}

final class Trie {
    private let root = TrieNode()

    func append(contentsOf tokens: [String]) {
        for token in tokens {
            insert(token)
        }
    }

    func insert(_ token: String) {
        var node = root
        for ch in token {
            if node.children[ch] == nil {
                node.children[ch] = TrieNode()
            }
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

struct SentencePieceToken {
    let token: String
    let score: Float
    let type: SentencePiecePieceType

    init(token: String, score: Float, type: SentencePiecePieceType = .normal) {
        self.token = token
        self.score = score
        self.type = type
    }
}

enum TokenizerError: Error, CustomStringConvertible {
    case invalidJSON(String)
    case missingField(String)

    var description: String {
        switch self {
        case .invalidJSON(let msg):
            return "Invalid JSON: \(msg)"
        case .missingField(let msg):
            return "Missing field: \(msg)"
        }
    }
}

public final class UnigramTokenizer {
    let vocab: [SentencePieceToken]
    let unknownTokenId: Int
    let unknownTokenScore: Float
    let tokensToIds: [String: Int]
    let trie: Trie

    private init(vocab: [SentencePieceToken], unknownTokenId: Int) {
        self.vocab = vocab
        self.unknownTokenId = unknownTokenId
        let minScore = vocab.reduce(Float.greatestFiniteMagnitude) { min($0, $1.score) }
        self.unknownTokenScore = minScore - 10

        var mapping: [String: Int] = [:]
        mapping.reserveCapacity(vocab.count)
        for (i, tok) in vocab.enumerated() {
            mapping[tok.token] = i
        }
        self.tokensToIds = mapping

        self.trie = Trie()
        self.trie.append(contentsOf: vocab.map { $0.token })
    }

    public convenience init(tokenizerJSON: [String: Any]) throws {
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

        self.init(vocab: pieces, unknownTokenId: unkId)
    }

    public convenience init(tokenizerJSONData: Data) throws {
        let json = try JSONSerialization.jsonObject(with: tokenizerJSONData) as? [String: Any] ?? [:]
        try self.init(tokenizerJSON: json)
    }

    public convenience init(sentencePieceModelData: Data) throws {
        let parsed = try SentencePieceModelParser.parsePieces(from: sentencePieceModelData)
        self.init(vocab: parsed.pieces, unknownTokenId: parsed.unknownTokenId)
    }

    public static func from(tokenizerJSONURL: URL) throws -> UnigramTokenizer {
        try UnigramTokenizer(tokenizerJSONData: Data(contentsOf: tokenizerJSONURL))
    }

    public static func from(sentencePieceModelURL: URL) throws -> UnigramTokenizer {
        try UnigramTokenizer(sentencePieceModelData: Data(contentsOf: sentencePieceModelURL))
    }

    public func encodeWithByteFallback(_ text: String) -> [Int] {
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
                if !hasSingleNode, token.count == mblen {
                    hasSingleNode = true
                }
            }

            if !hasSingleNode {
                lattice.insert(
                    startOffset: beginPos,
                    length: mblen,
                    score: unknownTokenScore,
                    tokenId: unknownTokenId
                )
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

    public func decode(_ ids: [Int]) -> String {
        var bytes: [UInt8] = []
        var pieces: [String] = []
        for id in ids {
            guard id >= 0 && id < vocab.count else { continue }
            let token = vocab[id]
            if token.type == .control || token.type == .unused {
                continue
            }
            let tok = token.token
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

    private lazy var byteMap: [UInt8: Int] = {
        var map: [UInt8: Int] = [:]
        for (i, tok) in vocab.enumerated() {
            let piece = tok.token
            if piece.hasPrefix("<0x"), piece.hasSuffix(">"), piece.count == 6 {
                let hex = piece.dropFirst(3).dropLast(1)
                if let b = UInt8(hex, radix: 16) {
                    map[b] = i
                }
            }
        }
        return map
    }()

    private func applyMetaspace(_ text: String) -> String {
        let replaced = text.replacingOccurrences(of: " ", with: "▁")
        return "▁" + replaced
    }
}
