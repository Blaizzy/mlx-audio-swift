import Foundation

public enum CMUDictLoader {

    public static func loadFromBundle() throws -> InMemoryLexicon {
        guard let url = Bundle.module.url(
            forResource: "cmudict",
            withExtension: "dict",
            subdirectory: "CMUdict"
        ) else {
            throw G2PError.resourceLoadFailed(
                name: "cmudict.dict",
                reason: "File not found in bundle"
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
