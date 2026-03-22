import Foundation

public enum CMUDictLoader {

    public static func load(from directory: URL) throws -> InMemoryLexicon {
        let url = directory.appendingPathComponent("cmudict.dict")

        guard FileManager.default.fileExists(atPath: url.path) else {
            throw G2PError.resourceLoadFailed(
                name: "cmudict.dict",
                reason: "File not found at \(url.path)"
            )
        }

        let data = try Data(contentsOf: url)

        guard let text = String(data: data, encoding: .isoLatin1)
            ?? String(data: data, encoding: .utf8) else {
            throw G2PError.resourceLoadFailed(
                name: "cmudict.dict",
                reason: "Unable to decode file content"
            )
        }

        let rawEntries = CMUDictParser.parse(text: text, primaryOnly: true)

        let lexiconEntries = rawEntries.map { raw in
            LexiconEntry(
                grapheme: raw.word,
                phonemes: ARPAbetMapper.convertSequence(raw.arpabet)
            )
        }

        return InMemoryLexicon(entries: lexiconEntries)
    }
}
