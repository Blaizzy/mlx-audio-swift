import Foundation

enum SpeechLanguage: String, Sendable, CaseIterable {
    case english, french

    var title: String { self == .english ? "English" : "French" }
    var model: ShowcaseModel { self == .english ? .englishVoice : .frenchVoice }
}

enum ShowcaseModel: String, CaseIterable, Identifiable, Sendable {
    case translator, englishVoice, frenchVoice

    var id: String { rawValue }

    var repository: String {
        switch self {
        case .translator: "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
        case .englishVoice: "gradium/phonon-40m-en"
        case .frenchVoice: "gradium/phonon-40m-fr"
        }
    }

    var title: String {
        switch self {
        case .translator: "Qwen2.5 · 0.5B"
        case .englishVoice: "Phonon · 40M"
        case .frenchVoice: "Phonon · 40M"
        }
    }

    var role: String {
        switch self {
        case .translator: "English → French translation · 4-bit"
        case .englishVoice: "English speech · 24 kHz"
        case .frenchVoice: "French speech · 24 kHz"
        }
    }

    var approximateDownload: String {
        switch self {
        case .translator: "290 MB"
        case .englishVoice: "122 MB"
        case .frenchVoice: "122 MB"
        }
    }

    var pageURL: URL { URL(string: "https://huggingface.co/\(repository)")! }
}

enum ModelStatus: Equatable, Sendable {
    case notLoaded
    case loading(Double)
    case ready
    case failed

    var label: String {
        switch self {
        case .notLoaded: "Loads on first use"
        case .loading(let progress): "Loading · \(Int(progress * 100))%"
        case .ready: "Ready on this device"
        case .failed: "Could not load · retry an action"
        }
    }
}

typealias ModelStatusHandler = @MainActor @Sendable (ShowcaseModel, ModelStatus) -> Void

enum TranslationError: LocalizedError {
    case emptyInput, inputTooLong, emptyTranslation, emptyAudio

    var errorDescription: String? {
        switch self {
        case .emptyInput: "Type something in English first."
        case .inputTooLong: "Keep the English text under 1,000 characters."
        case .emptyTranslation: "The model did not return a translation. Please try again."
        case .emptyAudio: "The voice model did not return audio. Please try again."
        }
    }
}

enum TranslationText {
    static let characterLimit = 1_000

    static func validated(_ input: String) throws -> String {
        let text = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { throw TranslationError.emptyInput }
        guard text.count <= characterLimit else { throw TranslationError.inputTooLong }
        return text
    }

    static func cleaned(_ response: String) -> String {
        var text = response
        if let end = text.range(of: "</think>") { text = String(text[end.upperBound...]) }
        if text.contains("<think>") { return "" }
        return text.replacingOccurrences(of: "<|im_end|>", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
