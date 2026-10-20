import Foundation
import XCTest
import MLXAudioCore
@testable import PhononTranslate

final class TranslationIntegrationTests: XCTestCase {
    /// Opt in explicitly: this loads the real models and may download weights.
    @MainActor
    func testEnglishToFrenchAndBothVoiceModels() async throws {
        guard ProcessInfo.processInfo.environment["PHONON_TRANSLATE_MODEL_TESTS"] == "1" else {
            throw XCTSkip("Set PHONON_TRANSLATE_MODEL_TESTS=1 to run real model inference.")
        }
        let engine = TranslationEngine()
        let token = HuggingFaceToken.read()
        let source = "Hello! I would like a coffee, please."
        let translation = try await engine.translate(
            source, token: token, onModelStatus: { _, _ in }, onText: { _ in })
        XCTAssertTrue(translation.lowercased().contains("café"), "Expected a French translation, received: \(translation)")
        XCTAssertFalse(translation.lowercased().contains("coffee"))
        XCTAssertFalse(translation.contains("<think>"))
        XCTAssertTrue(["voudrais", "aimerais", "souhaite"].contains(where: translation.lowercased().contains),
                      "The translation must preserve the request for a coffee: \(translation)")
        XCTAssertFalse(translation.contains("?"), "The translation must not turn the request into a question.")
        print("French translation: \(translation)")

        let second = try await engine.translate(
            "The train leaves tomorrow morning.", token: token,
            onModelStatus: { _, _ in }, onText: { _ in })
        XCTAssertTrue(second.lowercased().contains("train"))
        XCTAssertTrue(second.lowercased().contains("demain"))
        XCTAssertTrue(second.lowercased().contains("matin"))
        XCTAssertFalse(second.lowercased().contains("café"), "Translations must not retain the previous phrase.")
        print("Second French translation: \(second)")

        for language in SpeechLanguage.allCases {
            let text = language == .english ? source : translation
            let stream = try await engine.speech(text, language: language, token: token, onModelStatus: { _, _ in })
            var samples: [Float] = []
            var chunks = 0
            for try await chunk in stream {
                samples += chunk
                chunks += 1
            }
            XCTAssertGreaterThan(chunks, 1)
            XCTAssertGreaterThan(samples.count, 12_000)
            XCTAssertTrue(samples.allSatisfy { $0.isFinite })
            XCTAssertGreaterThan(samples.map(abs).max() ?? 0, 0.001)
            print("\(language.title): \(samples.count) samples in \(chunks) streamed chunks")
            if let outputDirectory = ProcessInfo.processInfo.environment["PHONON_TRANSLATE_TEST_OUTPUT"] {
                let directory = URL(fileURLWithPath: outputDirectory)
                try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
                try AudioUtils.writeWavFile(
                    samples: samples, sampleRate: 24_000,
                    fileURL: directory.appendingPathComponent("\(language.rawValue).wav"))
                try translation.write(to: directory.appendingPathComponent("translation.txt"), atomically: true, encoding: .utf8)
            }
        }
    }
}
