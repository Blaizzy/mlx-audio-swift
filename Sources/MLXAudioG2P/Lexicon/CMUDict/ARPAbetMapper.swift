public enum ARPAbetMapper {

    public static func toIPA(_ arpabet: String) -> String? {
        guard !arpabet.isEmpty else { return nil }

        let lastChar = arpabet.last!
        let isStressed = lastChar >= "0" && lastChar <= "2"
        let base = isStressed ? String(arpabet.dropLast()) : arpabet
        let stress = isStressed ? Int(String(lastChar)) : nil

        switch base {
        case "AH":
            return stress == 0 ? "ə" : "ʌ"
        case "ER":
            return stress == 0 ? "ɚ" : "ɝ"
        default:
            break
        }

        return mapping[base]
    }

    public static func convertSequence(_ arpabet: [String]) -> [String] {
        arpabet.compactMap { toIPA($0) }
    }

    private static let mapping: [String: String] = [
        "AA": "ɑ",
        "AE": "æ",
        "AO": "ɔ",
        "AW": "aʊ",
        "AY": "aɪ",
        "EH": "ɛ",
        "EY": "eɪ",
        "IH": "ɪ",
        "IY": "i",
        "OW": "oʊ",
        "OY": "ɔɪ",
        "UH": "ʊ",
        "UW": "u",
        "B": "b",
        "CH": "tʃ",
        "D": "d",
        "DH": "ð",
        "F": "f",
        "G": "ɡ",
        "HH": "h",
        "JH": "dʒ",
        "K": "k",
        "L": "l",
        "M": "m",
        "N": "n",
        "NG": "ŋ",
        "P": "p",
        "R": "ɹ",
        "S": "s",
        "SH": "ʃ",
        "T": "t",
        "TH": "θ",
        "V": "v",
        "W": "w",
        "Y": "j",
        "Z": "z",
        "ZH": "ʒ",
    ]
}
