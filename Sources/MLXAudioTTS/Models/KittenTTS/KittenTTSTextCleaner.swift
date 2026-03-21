import Foundation

enum KittenTTSTextCleaner {
    private static let pad = "$"
    private static let punctuation = ";:,.!?¡¿—…\"«»\u{201C}\u{201D} "
    private static let letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    private static let lettersIPA = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘\u{2018}\u{0329}\u{2019}ᵻ"

    static let symbolToIndex: [Character: Int] = {
        var map = [Character: Int]()
        var idx = 0
        for ch in pad { map[ch] = idx; idx += 1 }
        for ch in punctuation { map[ch] = idx; idx += 1 }
        for ch in letters { map[ch] = idx; idx += 1 }
        for ch in lettersIPA { map[ch] = idx; idx += 1 }
        return map
    }()

    static func cleanText(_ text: String) -> [Int] {
        text.compactMap { symbolToIndex[$0] }
    }
}
