import Foundation
import Observation

@MainActor
@Observable
final class Translator {
    enum Operation: Equatable {
        case translating
        case speaking(SpeechLanguage)
    }

    var english = "Hello! I would like a coffee and a croissant, please."
    var french = ""
    private(set) var translatedSource: String?
    private(set) var operation: Operation?
    private(set) var isCancelling = false
    private(set) var modelStatuses: [ShowcaseModel: ModelStatus] = [:]
    private(set) var translationDuration: TimeInterval?
    var errorMessage: String?
    var hasToken: Bool

    @ObservationIgnored private lazy var engine = TranslationEngine()
    @ObservationIgnored private lazy var player = SpeechPlayer()
    @ObservationIgnored private var operationTask: Task<Void, Never>?

    init(isPreview: Bool = false) {
        hasToken = isPreview ? false : HuggingFaceToken.read() != nil
    }

    var isBusy: Bool { operation != nil }
    var isTranslating: Bool { operation == .translating }
    var inputIsValid: Bool { (try? TranslationText.validated(english)) != nil }
    var translationIsCurrent: Bool {
        translatedSource == english.trimmingCharacters(in: .whitespacesAndNewlines) && !french.isEmpty
    }
    var canTranslate: Bool { inputIsValid && !isBusy }

    var statusText: String {
        if isCancelling { return "Stopping…" }
        if let loading = ShowcaseModel.allCases.first(where: {
            if case .loading = modelStatuses[$0] { return true }
            return false
        }), case .loading(let fraction) = modelStatuses[loading] {
            return "Loading \(loading.title) · \(Int(fraction * 100))%"
        }
        switch operation {
        case .translating: return "Translating on this device…"
        case .speaking(let language): return "Speaking \(language.title)…"
        case nil:
            if !french.isEmpty && !translationIsCurrent { return "English text changed. Translate again to update the French." }
            if let translationDuration, translationIsCurrent {
                return "Translated on device in \(translationDuration.formatted(.number.precision(.fractionLength(1))))s"
            }
            return "Models download on first use. Your text stays on this device."
        }
    }

    func canSpeak(_ language: SpeechLanguage) -> Bool {
        if operation == .speaking(language) { return !isCancelling }
        return !isBusy && (language == .english ? inputIsValid : translationIsCurrent)
    }

    func translate() {
        guard canTranslate else { return }
        let source = english.trimmingCharacters(in: .whitespacesAndNewlines)
        operation = .translating
        french = ""
        translatedSource = nil
        translationDuration = nil
        errorMessage = nil
        let token = HuggingFaceToken.read()
        operationTask = Task { [weak self] in
            guard let self else { return }
            defer { finishOperation() }
            let started = Date()
            do {
                let result = try await engine.translate(
                    source, token: token,
                    onModelStatus: { [weak self] kind, status in self?.updateStatus(kind, status) },
                    onText: { [weak self] text in self?.french = text })
                try Task.checkCancellation()
                french = result
                translatedSource = source
                translationDuration = Date().timeIntervalSince(started)
            } catch {
                if Task.isCancelled {
                    french = ""
                } else {
                    errorMessage = friendlyError(error, for: .translator)
                }
            }
        }
    }

    func speak(_ language: SpeechLanguage) {
        if operation == .speaking(language) { cancel(); return }
        guard canSpeak(language) else { return }
        let text = language == .english ? english.trimmingCharacters(in: .whitespacesAndNewlines) : french
        operation = .speaking(language)
        errorMessage = nil
        let token = HuggingFaceToken.read()
        operationTask = Task { [weak self] in
            guard let self else { return }
            defer { finishOperation() }
            do {
                let stream = try await engine.speech(
                    text, language: language, token: token,
                    onModelStatus: { [weak self] kind, status in self?.updateStatus(kind, status) })
                try Task.checkCancellation()
                try await player.play(stream)
            } catch {
                if !Task.isCancelled { errorMessage = friendlyError(error, for: language.model) }
            }
        }
    }

    func cancel() {
        guard isBusy else { return }
        isCancelling = true
        operationTask?.cancel()
        if case .speaking = operation { player.stop() }
        // Keep operations disabled until cancellation has unwound the producer.
    }

    private func updateStatus(_ kind: ShowcaseModel, _ status: ModelStatus) {
        if case .loading(let fraction) = status {
            guard isBusy else { return }
            if modelStatuses[kind] == .ready { return }
            if modelStatuses[kind] == .failed && fraction > 0 { return }
        }
        modelStatuses[kind] = status
    }

    private func finishOperation() {
        for (kind, status) in modelStatuses {
            if case .loading = status { modelStatuses[kind] = .notLoaded }
        }
        operation = nil
        isCancelling = false
        operationTask = nil
    }

    private func friendlyError(_ error: Error, for model: ShowcaseModel) -> String {
        let description = error.localizedDescription
        if model != .translator && ["401", "403", "gated", "unauthorized", "forbidden"].contains(where: {
            description.lowercased().contains($0)
        }) {
            let language = model == .englishVoice ? "English" : "French"
            return "To use the \(language) voice, accept the Phonon 40M access terms on Hugging Face and add a read token in Models & access."
        }
        return description
    }

    static func preview(translated: Bool = false, loading: Bool = false) -> Translator {
        let model = Translator(isPreview: true)
        if translated {
            model.french = "Bonjour ! Je voudrais un café et un croissant, s’il vous plaît."
            model.translatedSource = model.english
            model.translationDuration = 0.8
        }
        if loading {
            model.operation = .translating
            model.modelStatuses[.translator] = .loading(0.42)
        }
        return model
    }
}
