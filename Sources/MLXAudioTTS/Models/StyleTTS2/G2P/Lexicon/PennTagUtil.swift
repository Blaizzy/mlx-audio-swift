import NaturalLanguage

/// Maps Apple's NLTag (lexicalClass) to a Penn Treebank POS tag string.
/// `token` is optional but might enable some heuristics.
func pennTag(for nlTag: NLTag, token: String? = nil) -> String {
    let t = token?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    let lower = t.lowercased()

    if nlTag == .punctuation || nlTag == .sentenceTerminator || nlTag == .otherPunctuation {
        switch t {
        case ",": return ","
        case ".", "!", "?": return "."
        case ":", ";": return ":"
        case "``", "“", "„", "\"": return "``"
        case "''", "”": return "''"
        case "(", "[" , "{": return "("
        case ")", "]" , "}": return ")"
        case "$": return "$"
        case "#": return "#"
        case "-", "–", "—": return ":"
        default: break
        }
    }
  
    if nlTag == .openQuote { return "``" }
    if nlTag == .closeQuote { return "''" }
    if nlTag == .openParenthesis { return "(" }
    if nlTag == .closeParenthesis { return ")" }

    let whDeterminers: Set<String> = ["which", "whatever", "whichever"]
    let whPronouns: Set<String>    = ["who", "whom", "whose", "whoever", "whomever", "what", "whatever", "which", "whichever"]
    let whAdverbs: Set<String>     = ["when", "where", "why", "how"]
    let possessivePronouns: Set<String> = ["my","your","his","her","its","our","their"]
    let auxBe: Set<String>   = ["am","is","are","was","were","be","been","being"]
    let auxDo: Set<String>   = ["do","does","did"]
    let auxHave: Set<String> = ["have","has","had"]
    let subordinatingConjunctions: Set<String> = [
        "because","although","though","if","while","when","whenever","before","after","since","unless","until","that","whether","as"
    ]

    func looksPlural(_ s: String) -> Bool {
        let l = s.lowercased()
        guard l.count > 2 else { return false }
        if l.hasSuffix("ss") || l.hasSuffix("'s") || l.hasSuffix("’s") { return false }
        return l.hasSuffix("s")
    }
  
    func isCapitalizedWord(_ s: String) -> Bool {
        guard let first = s.first else { return false }
        return String(first) == String(first).uppercased()
    }

    switch nlTag {
    case .noun:
        if !t.isEmpty {
            if isCapitalizedWord(t) && !looksPlural(t) { return "NNP" }   // rough proper-noun guess
            if isCapitalizedWord(t) && looksPlural(t)  { return "NNPS" }
            if looksPlural(t)                           { return "NNS" }
        }
        return "NN"

    case .verb:
        // Handful of common auxiliaries + superficial morphology
        if auxBe.contains(lower) { return lower == "being" ? "VBG" : (lower == "been" ? "VBN" : "VB") }
        if auxDo.contains(lower) { return ["does"].contains(lower) ? "VBZ" : (lower == "did" ? "VBD" : "VB") }
        if auxHave.contains(lower) { return ["has"].contains(lower) ? "VBZ" : (lower == "had" ? "VBD" : "VB") }
        if lower.hasSuffix("ing") { return "VBG" }
        if lower.hasSuffix("ed")  { return "VBD" }
        if lower.hasSuffix("en")  { return "VBN" }
        if lower.hasSuffix("s")   { return "VBZ" }
        return "VB"

    case .adjective:
        if lower.hasSuffix("er") { return "JJR" }
        if lower.hasSuffix("est") { return "JJS" }
        return "JJ"

    case .adverb:
        if whAdverbs.contains(lower) { return "WRB" }
        if lower.hasSuffix("er") { return "RBR" }
        if lower.hasSuffix("est") { return "RBS" }
        return "RB"

    case .pronoun:
        if lower == "'s" || lower == "’s" { return "POS" }
        if whPronouns.contains(lower) {
            // simple possessive wh- detection
            if lower == "whose" { return "WP$" }
            return "WP"
        }
        if possessivePronouns.contains(lower) { return "PRP$" }
        return "PRP"

    case .determiner:
        if whDeterminers.contains(lower) { return "WDT" }
        if lower == "that" { return "DT" }
        return "DT"

    case .preposition:
        if lower == "to" { return "TO" }
        return "IN"

    case .conjunction:
        if subordinatingConjunctions.contains(lower) { return "IN" }
        return "CC"

    case .number:
        return "CD"

    case .interjection:
        return "UH"

    case .particle:
        if lower == "to" { return "TO" }
        return "RP"

    case .word, .otherWord:
        // could be symbol/foreign; fall back:
        return "FW"

    case .punctuation, .sentenceTerminator, .openQuote, .closeQuote,
         .openParenthesis, .closeParenthesis, .otherPunctuation:
        // already handled above; if we get here, use generic punctuation class
        return "."

    case .whitespace, .paragraphBreak, .wordJoiner:
        return "XX"

    // Name types (when using NLTagScheme.nameType)
    case .personalName, .organizationName, .placeName:
        // Usually proper nouns in PTB
        return "NNP"

    // Less common NLTag cases
    case .classifier, .idiom, .dash:
        return "FW"

    default:
        return "XX"
    }
}

func isPersonalPronoun(tag: NLTag, token: String) -> Bool {
  let personalPronouns: Set<String> = [
    "i", "me", "my", "mine", "myself",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "it", "its", "itself",
    "we", "us", "our", "ours", "ourselves",
    "they", "them", "their", "theirs", "themselves"
  ]
  
  return tag == .pronoun && personalPronouns.contains(token.lowercased())
}
