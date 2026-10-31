import Foundation
import Testing
@testable import MLXAudioTTS

@Suite("English number words")
struct EnglishNum2WordTests {
    let words = EnglishNum2Word()

    @Test func twentiesKeepTheirTens() {
        #expect(words.convert(25) == "twenty-five")
        #expect(words.convert(221) == "two hundred and twenty-one")
        #expect(words.convert(21, to: .ordinal).lowercased() == "twenty-first")
        #expect(words.convert(2023, to: .year) == "twenty twenty-three")
        #expect(words.convert(1999, to: .year) == "nineteen ninety-nine")
        #expect(words.convert(20) == "twenty")
    }

    @Test func millionsUseTheirOwnWord() {
        #expect(words.convert(1_000_000) == "one million")
        #expect(words.convert(2_500_000) == "two million, five hundred thousand")
        #expect(words.convert(3_000_000_000) == "three billion")
    }
}
