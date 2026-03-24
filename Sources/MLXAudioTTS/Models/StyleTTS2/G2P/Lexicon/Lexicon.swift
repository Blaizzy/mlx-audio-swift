import Foundation
import NaturalLanguage


final class Lexicon {
  static let usVocab: Set<Character> = Set("AIOWYbdfhijklmnpstuvwzæðŋɑɔəɛɜɡɪɹɾʃʊʌʒʤʧˈˌθᵊᵻʔ")
  static let gbVocab: Set<Character> = Set("AIQWYabdfhijklmnpstuvwzðŋɑɒɔəɛɜɡɪɹʃʊʌʒʤʧˈˌːθᵊ")
  static let lexiconOrdinals: [Int] = [39, 45] + Array(65...90) + Array(97...122)
  static let ordinals: Set<String> = Set(["st", "nd", "rd", "th"])

  static let addSymbols: [String: String] = [".": "dot", "/": "slash"]
  static let primaryStress: Character = "ˈ"
  static let secondaryStress: Character = "ˌ"
  static let vowelSet: Set<Character> = Set("AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ")
  static let symbolSet: [String: String] = ["%": "percent", "&": "and", "+": "plus", "@": "at"]
  static let usTaus: Set<Character> = Set("AIOWYiuæɑəɛɪɹʊʌ")
  static let currencies: [String: (String, String)] = [
      "$": ("dollar", "cent"),
      "£": ("pound", "pence"),
      "€": ("euro", "cent")
  ]
  
  private let british: Bool
  private let capStresses: (Double, Double) = (0.5, 2.0)
  private let num2Words = EnglishNum2Word()

  // Gold and silver dictionaries
  private let golds: [String: Any]
  private let silvers: [String: Any]
  private let vocab: Set<Character>

  init(british: Bool, directory: URL) {
    self.british = british
    // Load and grow dictionaries
    let rawGolds = DataResourcesUtil.loadGold(british: british, directory: directory)
    let rawSilvers = DataResourcesUtil.loadSilver(british: british, directory: directory)
    self.golds = Lexicon.growDictionary(rawGolds)
    self.silvers = Lexicon.growDictionary(rawSilvers)
  
    self.vocab = british ? Lexicon.gbVocab : Lexicon.usVocab
  }
    
  /// Grows a dictionary by adding capitalized / lowercase variants of existing word keys
  private static func growDictionary(_ d: [String: Any]) -> [String: Any] {
    // "Inefficient but correct."
    var e: [String: Any] = [:]
    
    for (k, v) in d {
      if k.count < 2 {
          continue
      }
      
      if k == k.lowercased() {
        if k != k.capitalized {
            e[k.capitalized] = v
        }
      } else if k == k.lowercased().capitalized {
        e[k.lowercased()] = v
      }
    }
    
    // Merge the new dictionary with the original, giving priority to original
    return e.merging(d) { (_, original) in original }
  }
  
  /// Applies stress modifications to phonetic strings
  static func applyStress(_ phoneticString: String?, stress: Double?) -> String? {
    func restress(_ ps: String) -> String {
      let characters = Array(ps)
      var indexedChars: [(Double, Character)] = characters.enumerated().map { (Double($0), $1) }
      
      // Find stress positions and their corresponding vowel positions
      var stressToVowel: [Int: Int] = [:]
      for (i, char) in characters.enumerated() {
        if stresses.contains(char) {
          // Find next vowel after this stress marker
          for j in (i + 1)..<characters.count {
            if Lexicon.vowelSet.contains(characters[j]) {
              stressToVowel[i] = j
              break
            }
          }
        }
      }
      
      // Reposition stress markers
      for (stressIndex, vowelIndex) in stressToVowel {
        let stressChar = indexedChars[stressIndex].1
        indexedChars[stressIndex] = (Double(vowelIndex) - 0.5, stressChar)
      }
      
      // Sort by position and extract characters
      return String(indexedChars.sorted { $0.0 < $1.0 }.map { $0.1 })
    }
    
    guard let phoneticString else { return nil }
    guard let stress else { return phoneticString }
    
    let stresses = Set<Character>([Lexicon.primaryStress, Lexicon.secondaryStress])
          
    if stress < -1 {
      return phoneticString.replacingOccurrences(of: String(Lexicon.primaryStress), with: "")
                           .replacingOccurrences(of: String(Lexicon.secondaryStress), with: "")
    } else if stress == -1 || (stress == 0 || stress == -0.5) && phoneticString.contains(Lexicon.primaryStress) {
      return phoneticString.replacingOccurrences(of: String(Lexicon.secondaryStress), with: "")
                            .replacingOccurrences(of: String(Lexicon.primaryStress), with: String(Lexicon.secondaryStress))
    } else if (stress == 0 || stress == 0.5 || stress == 1) && !phoneticString.contains(where: { stresses.contains($0) }) {
      if !phoneticString.contains(where: { Lexicon.vowelSet.contains($0) }) {
        return phoneticString
      }
      return restress(String(Lexicon.secondaryStress) + phoneticString)
    } else if stress >= 1 && !phoneticString.contains(Lexicon.primaryStress) && phoneticString.contains(Lexicon.secondaryStress) {
      return phoneticString.replacingOccurrences(of: String(Lexicon.secondaryStress), with: String(Lexicon.primaryStress))
    } else if stress > 1 && !phoneticString.contains(where: { stresses.contains($0) }) {
      if !phoneticString.contains(where: { Lexicon.vowelSet.contains($0) }) {
        return phoneticString
      }
      return restress(String(Lexicon.primaryStress) + phoneticString)
    }
    
    return phoneticString
  }
  
  func transcribe(_ token: MToken, ctx: TokenContext) -> (String?, Int?) {
    var word = token.text
    if let alias = token.`_`.alias { word = alias }
    word = word.replacingOccurrences(of: String(UnicodeScalar(8216)!), with: "'")
               .replacingOccurrences(of: String(UnicodeScalar(8217)!), with: "'")
    word = word.precomposedStringWithCompatibilityMapping
    
    word = String(word.map { unicodeNumericIfNeeded($0) } )
    
    let stress: Double? = (word == word.lowercased() ? nil : (word == word.uppercased() ? capStresses.1 : capStresses.0))
    let res = getWord(word, tag: token.tag, stress: stress, ctx: ctx)
    if let phoneme = res.phoneme {
      return (Lexicon.applyStress(appendCurrency(phoneme, currency: token.`_`.currency), stress: token.`_`.stress), res.rating)
    } else if isNumber(word: word, is_head: token.`_`.is_head) {
      let num = getNumber(word, currency: token.`_`.currency, is_head: token.`_`.is_head, num_flags: token.`_`.num_flags)
      return (Lexicon.applyStress(num.0, stress: token.`_`.stress), num.1)
    } else if !word.unicodeScalars.allSatisfy({ Lexicon.lexiconOrdinals.contains(Int($0.value)) }) {
      return (nil, nil)
    }
    
    return (nil, nil)
  }
  
  /// Converts Unicode digits to ASCII digits if needed
  private func unicodeNumericIfNeeded(_ c: Character) -> Character {
    guard c.isNumber else { return c }
    
    if let numericValue = c.wholeNumberValue {
      if numericValue >= 0 && numericValue <= 9 {
        return Character("\(numericValue)")
      }
    }
    
    return c
  }
    
  private func getWord(_ word: String, tag: NLTag?, stress: Double?, ctx: TokenContext) -> (phoneme: String?, rating: Int?) {
    let sc = getSpecialCase(word, tag: tag, stress: stress, ctx: ctx)
    if sc.phoneme != nil { return sc }
    var candidate = word
    let wl = word.lowercased()
    
    if word.count > 1,
       word.replacingOccurrences(of: "'", with: "").allSatisfy({ $0.isLetter }),
       word != word.lowercased(),
       (!(tag?.isProperNoun ?? false) || word.count > 7),
       golds[word] == nil, silvers[word] == nil,
       (word == word.uppercased() || word.dropFirst().lowercased() == word.dropFirst()),
        (golds[wl] != nil || silvers[wl] != nil || [stem_s, stem_ed, stem_ing].contains(where: { fn in fn(wl, tag, stress, ctx).0 != nil })) {
      candidate = wl
    }
    
    if isKnown(candidate) {
      return lookup(candidate, tag: tag, stress: stress, ctx: ctx)
    } else if candidate.hasSuffix("s'"), isKnown(String(candidate.dropLast(2)) + "'s") {
      return lookup(String(candidate.dropLast(2)) + "'s", tag: tag, stress: stress, ctx: ctx)
    } else if candidate.hasSuffix("'"), isKnown(String(candidate.dropLast())) {
      return lookup(String(candidate.dropLast()), tag: tag, stress: stress, ctx: ctx)
    }
    
    let s = stem_s(candidate, tag: tag, stress: stress, ctx: ctx)
    if s.phoneme != nil { return s }
    
    let ed = stem_ed(candidate, tag: tag, stress: stress, ctx: ctx)
    if ed.phoneme != nil { return ed }
    
    let ing = stem_ing(candidate, tag: tag, stress: (stress == nil ? 0.5 : stress), ctx: ctx)
    if ing.phoneme != nil { return ing }
    
    return (nil, nil)
  }
  
  private func getSpecialCase(_ word: String, tag: NLTag?, stress: Double?, ctx: TokenContext) -> (phoneme: String?, rating: Int?) {
    if tag == .punctuation, let target = Lexicon.addSymbols[word] {
      return lookup(target, tag: nil, stress: -0.5, ctx: ctx)
    } else if let sym = Lexicon.symbolSet[word] {
      return lookup(sym, tag: nil, stress: nil, ctx: ctx)
    } else if word.trimmingCharacters(in: CharacterSet(charactersIn: ".")).contains(".") {
      let parts = word.split(separator: ".")
      if parts.map({ $0.count }).max() ?? 0 < 3 {
        return getNNP(word)
      }
    } else if word == "a" || word == "A" {
      if tag == .determiner { return ("ɐ", 4) }
      return ("ˈA", 4)
    } else if ["am", "Am", "AM"].contains(word) {
      if let t = tag, pennTag(for: t, token: word).hasPrefix("NN") {
        return getNNP(word)
      }
      
      if ctx.futureVowel == nil || word != "am" || (stress != nil && stress! > 0) {
        if let v = golds["am"] as? String { return (v, 4) }
      }
      return ("ɐm", 4)
    } else if ["an", "An", "AN"].contains(word) {
      if word == "AN", let t = tag, pennTag(for: t, token: word).hasPrefix("NN") {
        return getNNP(word)
      }
      return ("ɐn", 4)
    } else if word == "I", let tag, isPersonalPronoun(tag: tag, token: word) {
      return (String(Lexicon.secondaryStress) + "I", 4)
    } else if ["by", "By", "BY"].contains(word), getParentTag(tag, token: word) == "ADV" {
      return ("bˈI", 4)
    } else if ["to", "To"].contains(word) || (word == "TO" && tag == .preposition) {
      let chosen: String
      if ctx.futureVowel == nil {
        chosen = (golds["to"] as? String) ?? "to"
      } else if ctx.futureVowel == false {
        chosen = "tə"
      } else {
        chosen = "tʊ"
      }
      return (chosen, 4)
    } else if ["in", "In"].contains(word) || (word == "IN" && !(tag?.isProperNoun ?? false)) {
      let s = (ctx.futureVowel == nil || tag != .preposition) ? String(Lexicon.primaryStress) : ""
      return (s + "ɪn", 4)
    } else if ["the", "The"].contains(word) || (word == "THE" && tag == .determiner) {
      return (ctx.futureVowel == true ? "ði" : "ðə", 4)
    } else if tag == .preposition, word.range(of: "(?i)vs\\.?$", options: .regularExpression) != nil {
      return lookup("versus", tag: nil, stress: nil, ctx: ctx)
    } else if ["used", "Used", "USED"].contains(word) {
      if (tag == .verb || tag == .adjective) && ctx.futureTo {
        if let m = golds["used"] as? [String: String?], let v = m["VBD"] as? String { return (v, 4) }
      }
      if let m = golds["used"] as? [String: String?], let v = m["DEFAULT"] as? String { return (v, 4) }
    }
    
    return (nil, nil)
  }
    
  private func lookup(_ w: String, tag: NLTag?, stress: Double?, ctx: TokenContext?) -> (phoneme: String?, rating: Int?) {
    var word = w
    var isNNP: Bool? = nil
    if word == word.uppercased(), golds[word] == nil {
      word = word.lowercased()
      isNNP = tag?.isProperNoun
    }
    var phoneticString: Any? = golds[word]
    var rating = 4
    if phoneticString == nil, isNNP != true {
      phoneticString = silvers[word]
      rating = 3
    }
    
    if let phonemeDict = phoneticString as? [String: String?] {
      var t = getParentTag(tag, token: w)
      if let ctx = ctx, ctx.futureVowel == nil, phonemeDict["None"] != nil {
        t = "XX"
      }
      phoneticString = phonemeDict[t ?? "DEFAULT"] ?? phonemeDict["DEFAULT"] ?? nil
    }
    
    if phoneticString == nil || (isNNP == true && !(phoneticString as? String ?? "").contains(Lexicon.primaryStress)) {
      let nn = getNNP(word)
      if nn.phoneme != nil { return nn }
    }
    
    let applied = Lexicon.applyStress(phoneticString as? String, stress: stress)
    return (applied, rating)
  }
  
  private func getParentTag(_ tag: NLTag?, token: String?) -> String? {
    guard let tag = tag else { return "XX" }
    let pennTag = pennTag(for: tag, token: token)
    if pennTag.hasPrefix("VB") { return "VERB" }
    if pennTag.hasPrefix("NN") { return "NOUN" }
    if pennTag.hasPrefix("ADV") || pennTag.hasPrefix("RB") { return "ADV" }
    if pennTag.hasPrefix("ADJ") || pennTag.hasPrefix("JJ") { return "ADJ" }
    return "XX"
  }
  
  /// Spells out acronyms, abbreviations and proper nouns letter-by-letter
  private func getNNP(_ word: String) -> (phoneme: String?, rating: Int?) {
    let pieces: [String?] = word.compactMap { ch in
      if ch.isLetter {
        let s = String(ch).uppercased()
        if let v = golds[s] as? String { return v }
      }
      return nil
    }
    
    if pieces.contains(where: { $0 == nil }) { return (nil, nil) }
    
    let joined = Lexicon.applyStress(pieces.compactMap{ $0 }.joined(separator: ""), stress: 0)
    if let joined {
      let ps = joined.replacingLastOccurrence(of: Lexicon.secondaryStress, with: Lexicon.primaryStress)
      return (ps, 3)
    }
  
    return (nil, nil)
  }
  
  private func isKnown(_ word: String) -> Bool {
    if golds[word] != nil || Lexicon.symbolSet[word] != nil || silvers[word] != nil { return true }
    
    if !word.allSatisfy({ ch in
      if let v = ch.unicodeScalars.first?.value {
        return Lexicon.lexiconOrdinals.contains(Int(v))
      }
      return false
    }) {
      return false
    }
    
    if word.count == 1 { return true }
    if word == word.uppercased(), golds[word.lowercased()] != nil { return true }
    let idx = word.index(after: word.startIndex)
    return word[idx...].uppercased() == word[idx...]
  }
  
  private func stem_s(_ word: String, tag: NLTag?, stress: Double?, ctx: TokenContext?) -> (phoneme: String?, rating: Int?) {
    guard word.count >= 3, word.hasSuffix("s") else { return (nil, nil) }
    var stem: String?
    
    if !word.hasSuffix("ss"), isKnown(String(word.dropLast())) {
      stem = String(word.dropLast())
    } else if (word.hasSuffix("'s") || (word.count > 4 && word.hasSuffix("es") && !word.hasSuffix("ies"))), isKnown(String(word.dropLast(2))) {
      stem = String(word.dropLast(2))
    } else if word.count > 4 && word.hasSuffix("ies"), isKnown(String(word.dropLast(3)) + "y") {
      stem = String(word.dropLast(3)) + "y"
    }
    
    guard let s = stem else { return (nil, nil) }
    let looked = lookup(s, tag: tag, stress: stress, ctx: ctx)
    return (pluralizeS(looked.0), looked.1)
  }
  
  private func pluralizeS(_ stem: String?) -> String? {
    guard let stem = stem, !stem.isEmpty else { return nil }
    if let last = stem.last, "ptkfθ".contains(last) { return stem + "s" }
    if let last = stem.last, "szʃʒʧʤ".contains(last) { return stem + (british ? "ɪ" : "ᵻ") + "z" }
    return stem + "z"
  }
  
  private func pastEd(_ stem: String?) -> String? {
    guard let stem = stem, !stem.isEmpty else { return nil }
    if let last = stem.last, "pkfθʃsʧ".contains(last) { return stem + "t" }
    if stem.hasSuffix("d") { return stem + (british ? "ɪ" : "ᵻ") + "d" }
    if !stem.hasSuffix("t") { return stem + "d" }
    if british || stem.count < 2 { return stem + "ɪd" }
    if let penult = stem.dropLast().last, Lexicon.usTaus.contains(penult) { return String(stem.dropLast()) + "ɾᵻd" }
    return stem + "ᵻd"
  }

  private func stem_ed(_ word: String, tag: NLTag?, stress: Double?, ctx: TokenContext?) -> (phoneme: String?, rating: Int?) {
    guard word.count >= 4, word.hasSuffix("d") else { return (nil, nil) }
    var stem: String?
    
    if !word.hasSuffix("dd"), isKnown(String(word.dropLast())) {
      stem = String(word.dropLast())
    } else if word.count > 4 && word.hasSuffix("ed") && !word.hasSuffix("eed"), isKnown(String(word.dropLast(2))) {
      stem = String(word.dropLast(2))
    }
    
    guard let s = stem else { return (nil, nil) }
    let looked = lookup(s, tag: tag, stress: stress, ctx: ctx)
    return (pastEd(looked.0), looked.1)
  }
  
  private func progIng(_ stem: String?) -> String? {
    guard let stem = stem, !stem.isEmpty else { return nil }
    
    if british {
      if let last = stem.last, "əː".contains(last) { return nil }
    } else {
      if stem.count > 1, stem.hasSuffix("t"), let penult = stem.dropLast().last, Lexicon.usTaus.contains(penult) {
        return String(stem.dropLast()) + "ɾɪŋ"
      }
    }
    
    return stem + "ɪŋ"
  }

  private func stem_ing(_ word: String, tag: NLTag?, stress: Double?, ctx: TokenContext?) -> (phoneme: String?, rating: Int?) {
    guard word.count >= 5, word.hasSuffix("ing") else { return (nil, nil) }
    var stem: String?
    
    if word.count > 5, isKnown(String(word.dropLast(3))) {
      stem = String(word.dropLast(3))
    } else if isKnown(String(word.dropLast(3)) + "e") {
      stem = String(word.dropLast(3)) + "e"
    } else if word.count > 5, word.range(of: #"([bcdgklmnprstvxz])\1ing$|cking$"#, options: .regularExpression) != nil, isKnown(String(word.dropLast(4))) {
      stem = String(word.dropLast(4))
    }
    
    guard let s = stem else { return (nil, nil) }
    let looked = lookup(s, tag: tag, stress: stress, ctx: ctx)
    return (progIng(looked.phoneme), looked.rating)
  }
  
  private func isCurrency(_ word: String) -> Bool {
    if !word.contains(".") { return true }
    if word.filter({ $0 == "." }).count > 1 { return false }
    if let cents = word.split(separator: ".").last { return cents.count < 3 || Set(cents) == Set(["0"]) }
    
    return false
  }
  
  private func appendCurrency(_ phoneme: String?, currency: String?) -> String? {
    guard let phoneme, let currency else { return phoneme }
    
    if let pair = Lexicon.currencies[currency] {
      if let plural = stem_s(pair.0 + "s", tag: nil, stress: nil, ctx: nil).phoneme {
        return phoneme + " " + plural
      }
    }
    
    return phoneme
  }
  
  private func isNumber(word: String, is_head: Bool) -> Bool {
    if word.allSatisfy({ !$0.isNumber }) { return false }    
    let suffixes: [String] = ["ing", "'d", "ed", "'s"] + Lexicon.ordinals + ["s"]
    var core = word
    for s in suffixes {
      if core.hasSuffix(s) {
        core = String(core.dropLast(s.count))
        break
      }
    }
    
    return core.enumerated().allSatisfy { (i, c) in
        return c.isNumber || c == "," || c == "." || (is_head && i == 0 && c == "-")
    }
  }
    
  /// Helper function to check if a string contains only digits
  private func isPlainDigits(_ string: String) -> Bool {
    return !string.isEmpty && string.allSatisfy { $0.isNumber }
  }
  
  private func getNumber(_ input: String, currency: String?, is_head: Bool, num_flags: String) -> (String?, Int?) {
    var result: [(String, Int)] = []

    func appendLookup(_ w: String, s: Double?) {
      let looked = lookup(w, tag: nil, stress: s, ctx: nil)
      if let p = looked.0, let r = looked.1 { result.append((p, r)) }
    }
    
    func extend_num(_ num: String, first: Bool = true, escape: Bool = false) {
      let splits: [String]
      if escape {
        splits = num.split(whereSeparator: { !$0.isLetter }).map(String.init)
      } else {
        if let val = Decimal(string: num) {
          splits = num2Words.convert(val).split(separator: " ").map(String.init)
        } else {
          splits = num.split(whereSeparator: { !$0.isLetter }).map(String.init)
        }
      }
      
      for (i, w) in splits.enumerated() {
        if w != "and" || num_flags.contains("&") {
          if first && i == 0 && splits.count > 1 && w == "one" && num_flags.contains("a") {
            result.append(("ə", 4))
          } else {
            let s = (w == "point") ? -2.0 : nil
            appendLookup(w, s: s)
          }
        } else if w == "and" && num_flags.contains("n") && !result.isEmpty {
          let last = result.removeLast()
          result.append((last.0 + "ən", last.1))
        }
      }
    }
    
    var word = input
    var suffix: String? = nil
    if let m = word.range(of: "[a-z']+$", options: .regularExpression) {
      suffix = String(word[m])
      word.removeSubrange(m)
    }
        
    if word.hasPrefix("-") {
      appendLookup("minus", s: nil)
      word.removeFirst()
    }
    
    if isPlainDigits(word), let sf = suffix, Lexicon.ordinals.contains(sf) {
      if let n = Int(word) {
        extend_num(num2Words.convert(Decimal(n), to: .ordinal), escape: true)
      }
    } else if result.isEmpty, word.count == 4, !Lexicon.currencies.contains(where: { currency == $0.key }), isPlainDigits(word) {
      if let n = Int(word) {
        extend_num(num2Words.convert(Decimal(n), to: .year), escape: true)
      }
    } else if !is_head && !word.contains(".") {
      let num = word.replacingOccurrences(of: ",", with: "")
      if num.first == "0" || num.count > 3 {
        for n in num { extend_num(String(n), first: false) }
      } else if num.count == 3 && !num.hasSuffix("00") {
        extend_num(String(num.first!))
        if num[num.index(num.startIndex, offsetBy: 1)] == "0" {
          appendLookup("O", s: -2)
          extend_num(String(num.last!), first: false)
        } else {
          extend_num(String(num.suffix(2)), first: false)
        }
      } else {
        extend_num(num)
      }
    } else if word.filter({ $0 == "." }).count > 1 || !is_head {
      var first = true
      for num in word.replacingOccurrences(of: ",", with: "").split(separator: ".").map(String.init) {
        if num.isEmpty {}
        else if num.first == "0" || (num.count != 2 && num.dropFirst().contains(where: { $0 != "0" })) {
          for n in num {
            extend_num(String(n), first: false)
          }
        } else {
          extend_num(num, first: first)
        }
        first = false
      }
    } else if let curr = currency, let units = Lexicon.currencies[curr], isCurrency(word) {
          var pairs: [(Int, String)] = []
          let parts = word.replacingOccurrences(of: ",", with: "").split(separator: ".")
          let a = parts.indices.contains(0) ? Int(parts[0]) ?? 0 : 0
          let b = parts.indices.contains(1) ? Int(parts[1]) ?? 0 : 0
          pairs = [(a, units.0), (b, units.1)].filter { _ in true }
          if pairs.count > 1 {
              if pairs[1].0 == 0 { pairs = Array(pairs.prefix(1)) }
              else if pairs[0].0 == 0 { pairs = Array(pairs.suffix(1)) }
          }
      
          for (i, (num, unit)) in pairs.enumerated() {
              if i > 0 { appendLookup("and", s: nil) }
              extend_num(String(num), first: i == 0)
              if abs(num) != 1 && unit != "pence" {
                  if let s = stem_s(unit + "s", tag: nil, stress: nil, ctx: nil).0 { result.append((s, 4)) }
              } else {
                  appendLookup(unit, s: nil)
              }
          }
      } else {
        if isPlainDigits(word) {
          if let n = Int(word) { word = num2Words.convert(Decimal(n), to: .decimal) }
        } else if !word.contains(".") {
            let num = word.replacingOccurrences(of: ",", with: "")
          if let n = Int(num) {
            word = num2Words.convert(Decimal(n),
                                     to: (suffix != nil && Lexicon.ordinals.contains(suffix!)) ? .ordinal : .decimal)
          }
        } else {
            let num = word.replacingOccurrences(of: ",", with: "")
            if num.first == "." {
                let tail = num.dropFirst().compactMap { Int(String($0)) }.map { num2Words.convert(Decimal($0)) }.joined(separator: " ")
                word = "point " + tail
            } else {
              if let d = Double(num) { word = num2Words.convert(Decimal(d)) }
            }
        }
        
        extend_num(word, escape: true)
    }
  
    if result.isEmpty { return (nil, nil) }
    
    var text = result.map { $0.0 }.joined(separator: " ")
    let rating = result.map { $0.1 }.min() ?? 4
    
    if let s = suffix, s == "s" || s == "'s" {
      text = pluralizeS(text) ?? text
    } else if let s = suffix, s == "ed" || s == "'d" {
      text = pastEd(text) ?? text
    } else if suffix == "ing" {
      text = progIng(text) ?? text
    }
    
    return (text, rating)
  }
}
