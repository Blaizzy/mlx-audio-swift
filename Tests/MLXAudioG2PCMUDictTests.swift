import Foundation
import Testing
@testable import MLXAudioG2P

struct CMUDictParserTests {

    @Test func parsesBasicEntry() throws {
        let entry = try #require(CMUDictParser.parseLine("hello  HH AH0 L OW1"))
        #expect(entry.word == "hello")
        #expect(entry.arpabet == ["HH", "AH0", "L", "OW1"])
        #expect(entry.variant == nil)
    }

    @Test func parsesVariantEntry() throws {
        let entry = try #require(CMUDictParser.parseLine("the(2)  DH IY0"))
        #expect(entry.word == "the")
        #expect(entry.arpabet == ["DH", "IY0"])
        #expect(entry.variant == 2)
    }

    @Test func skipsCommentLines() {
        #expect(CMUDictParser.parseLine(";;; this is a comment") == nil)
    }

    @Test func skipsEmptyLines() {
        #expect(CMUDictParser.parseLine("") == nil)
        #expect(CMUDictParser.parseLine("   ") == nil)
    }

    @Test func handlesSinglePhoneme() throws {
        let entry = try #require(CMUDictParser.parseLine("a  AH0"))
        #expect(entry.word == "a")
        #expect(entry.arpabet == ["AH0"])
    }

    @Test func parsesBulkText() {
        let text = """
        ;;; comment
        hello  HH AH0 L OW1
        world  W ER1 L D
        the  DH AH0
        the(2)  DH IY0
        """
        let entries = CMUDictParser.parse(text: text)
        #expect(entries.count == 4)
        #expect(entries[0].word == "hello")
        #expect(entries[3].variant == 2)
    }

    @Test func filtersPrimaryOnly() {
        let text = """
        the  DH AH0
        the(2)  DH IY0
        hello  HH AH0 L OW1
        """
        let entries = CMUDictParser.parse(text: text, primaryOnly: true)
        #expect(entries.count == 2)
        #expect(entries.allSatisfy { $0.variant == nil })
    }
}

struct ARPAbetMapperTests {

    @Test func mapsConsonant() {
        #expect(ARPAbetMapper.toIPA("TH") == "θ")
        #expect(ARPAbetMapper.toIPA("SH") == "ʃ")
        #expect(ARPAbetMapper.toIPA("NG") == "ŋ")
        #expect(ARPAbetMapper.toIPA("HH") == "h")
        #expect(ARPAbetMapper.toIPA("CH") == "tʃ")
        #expect(ARPAbetMapper.toIPA("JH") == "dʒ")
        #expect(ARPAbetMapper.toIPA("ZH") == "ʒ")
    }

    @Test func mapsVowelStrippingStress() {
        #expect(ARPAbetMapper.toIPA("AH0") == "ə")
        #expect(ARPAbetMapper.toIPA("AH1") == "ʌ")
        #expect(ARPAbetMapper.toIPA("AH2") == "ʌ")
        #expect(ARPAbetMapper.toIPA("ER0") == "ɚ")
        #expect(ARPAbetMapper.toIPA("ER1") == "ɝ")
    }

    @Test func mapsRegularVowels() {
        #expect(ARPAbetMapper.toIPA("AA0") == "ɑ")
        #expect(ARPAbetMapper.toIPA("AA1") == "ɑ")
        #expect(ARPAbetMapper.toIPA("AE1") == "æ")
        #expect(ARPAbetMapper.toIPA("EY0") == "eɪ")
        #expect(ARPAbetMapper.toIPA("OW1") == "oʊ")
        #expect(ARPAbetMapper.toIPA("AW0") == "aʊ")
        #expect(ARPAbetMapper.toIPA("AY1") == "aɪ")
        #expect(ARPAbetMapper.toIPA("OY0") == "ɔɪ")
    }

    @Test func returnsNilForUnknown() {
        #expect(ARPAbetMapper.toIPA("XX") == nil)
        #expect(ARPAbetMapper.toIPA("") == nil)
    }

    @Test func convertsFullSequence() {
        let ipa = ARPAbetMapper.convertSequence(["HH", "AH0", "L", "OW1"])
        #expect(ipa == ["h", "ə", "l", "oʊ"])
    }

    @Test func convertsSequenceSkipsUnknown() {
        let ipa = ARPAbetMapper.convertSequence(["HH", "XX", "L"])
        #expect(ipa == ["h", "l"])
    }
}

struct CMUDictLoaderTests {

    private static var cmuDictDir: URL? {
        ProcessInfo.processInfo.environment["MLXAUDIO_CMUDICT_DIR"].map { URL(fileURLWithPath: $0) }
    }

    @Test func loadsFromDirectory() throws {
        guard let dir = Self.cmuDictDir else {
            print("Skipping: set MLXAUDIO_CMUDICT_DIR to cmudict directory")
            return
        }
        let lexicon = try CMUDictLoader.load(from: dir)
        #expect(lexicon.lookup("hello") != nil)
        #expect(lexicon.lookup("world") != nil)
        #expect(lexicon.lookup("the") != nil)
    }

    @Test func producesCorrectIPA() throws {
        guard let dir = Self.cmuDictDir else { return }
        let lexicon = try CMUDictLoader.load(from: dir)
        let hello = try #require(lexicon.lookup("hello"))
        #expect(hello.phonemes == ["h", "ə", "l", "oʊ"])
    }

    @Test func handlesUppercaseQuery() throws {
        guard let dir = Self.cmuDictDir else { return }
        let lexicon = try CMUDictLoader.load(from: dir)
        #expect(lexicon.lookup("HELLO") != nil)
        #expect(lexicon.lookup("Hello") != nil)
    }

    @Test func returnsNilForNonsense() throws {
        guard let dir = Self.cmuDictDir else { return }
        let lexicon = try CMUDictLoader.load(from: dir)
        #expect(lexicon.lookup("xyzzyplugh") == nil)
    }

    @Test func hasReasonableCount() throws {
        guard let dir = Self.cmuDictDir else { return }
        let lexicon = try CMUDictLoader.load(from: dir)
        #expect(lexicon.lookup("phone") != nil)
        #expect(lexicon.lookup("knight") != nil)
        #expect(lexicon.lookup("psychology") != nil)
        #expect(lexicon.lookup("through") != nil)
    }

    @Test func fixesDigraphsCorrectly() throws {
        guard let dir = Self.cmuDictDir else { return }
        let lexicon = try CMUDictLoader.load(from: dir)

        let phone = try #require(lexicon.lookup("phone"))
        #expect(phone.phonemes.contains("f"))
        #expect(!phone.phonemes.contains("p"))

        let knight = try #require(lexicon.lookup("knight"))
        #expect(knight.phonemes.first == "n")
    }
}

