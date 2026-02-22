import AVFoundation
import FoundationModels
import MLX
import MLXAudioCore
import MLXAudioTTS
import MLXLMCommon

@MainActor
protocol ConversationControllerDelegate: AnyObject {
    func conversationControllerDidStartUserSpeech(_ controller: ConversationController)
    func conversationController(_ controller: ConversationController, didFinish transcription: String)
}

@MainActor
@Observable
final class ConversationController {
    private enum TurnMarker {
        case complete
        case incompleteShort
        case incompleteLong
    }

    private enum IncompleteTimeoutKind {
        case short
        case long

        var logLabel: String {
            switch self {
            case .short: "short"
            case .long: "long"
            }
        }
    }

    struct UserTurnCompletionConfig: Sendable {
        var instructions: String?
        var incompleteShortTimeout: TimeInterval = 3
        var incompleteLongTimeout: TimeInterval = 10
        var incompleteShortPrompt: String?
        var incompleteLongPrompt: String?

        var completionInstructions: String {
            instructions ?? ConversationController.defaultTurnCompletionInstructions
        }

        var shortPrompt: String {
            incompleteShortPrompt ?? ConversationController.defaultIncompleteShortPrompt
        }

        var longPrompt: String {
            incompleteLongPrompt ?? ConversationController.defaultIncompleteLongPrompt
        }
    }

    private nonisolated static let baseInstructions = "You are a helpful voice assistant. Your goal is to demonstrate your capabilities in a succinct way. Your output will be spoken aloud, so avoid special characters that can't easily be spoken, such as emojis or bullet points."

    private nonisolated static let defaultTurnCompletionInstructions = """
    CRITICAL INSTRUCTION - MANDATORY RESPONSE FORMAT:
    Every single response MUST begin with a turn completion indicator. This is not optional.

    TURN COMPLETION DECISION FRAMEWORK:
    Ask yourself: "Has the user finished speaking for this turn?"
    This is about conversational endpoint detection, not whether you already have every detail needed.

    Mark as COMPLETE (✓) when:
    - The user has completed a request, question, or statement
    - The utterance is syntactically and conversationally complete
    - You can naturally respond next, even if you need a follow-up clarification

    Mark as INCOMPLETE SHORT (○) when the user will likely continue soon:
    - The user was clearly cut off mid-sentence or mid-word
    - The user is in the middle of a thought that got interrupted
    - Brief technical interruption (they'll resume in a few seconds)

    Mark as INCOMPLETE LONG (◐) when the user needs more time:
    - The user explicitly asks for time: "let me think", "give me a minute", "hold on"
    - The user is clearly pondering or deliberating: "hmm", "well...", "that's a good question"
    - The user acknowledged but hasn't answered yet: "That's interesting..."
    - The response feels like a preamble before the actual answer

    IMPORTANT DEFAULT:
    - If uncertain, choose COMPLETE (✓).
    - A complete request with missing details is still COMPLETE. Ask clarifying questions in your ✓ response.
      Example: "Can you tell me what the weather is today?" -> `✓ Sure. What city are you in?`

    RESPOND in one of these three formats:
    1. If COMPLETE: `✓` followed by a space and your full substantive response
    2. If INCOMPLETE SHORT: ONLY the character `○` (user will continue in a few seconds)
    3. If INCOMPLETE LONG: ONLY the character `◐` (user needs more time to think)

    FORMAT REQUIREMENTS:
    - ALWAYS use single-character indicators: `✓` (complete), `○` (short wait), or `◐` (long wait)
    - For COMPLETE: `✓` followed by a space and your full response
    - For INCOMPLETE: ONLY the single character (`○` or `◐`) with absolutely nothing else
    - Your turn indicator must be the very first character in your response
    """

    private nonisolated static let defaultIncompleteShortPrompt = """
    The user paused briefly. Generate a brief, natural prompt to encourage them to continue.

    IMPORTANT: You MUST respond with ✓ followed by your message. Do NOT output ○ or ◐ - the user has already been given time to continue.

    Your response should:
    - Be contextually relevant to what was just discussed
    - Sound natural and conversational
    - Be very concise (1 sentence max)
    - Gently prompt them to continue
    """

    private nonisolated static let defaultIncompleteLongPrompt = """
    The user has been quiet for a while. Generate a friendly check-in message.

    IMPORTANT: You MUST respond with ✓ followed by your message. Do NOT output ○ or ◐ - the user has already been given plenty of time.

    Your response should:
    - Acknowledge they might be thinking or busy
    - Offer to help or continue when ready
    - Be warm and understanding
    - Be brief (1 sentence)
    """

    @ObservationIgnored
    weak var delegate: ConversationControllerDelegate?

    private(set) var isActive: Bool = false
    private(set) var isDetectingSpeech = false
    private(set) var canSpeak: Bool = false
    private(set) var isSpeaking: Bool = false

    var isMicrophoneMuted: Bool {
        audioEngine.isMicrophoneMuted
    }

    @ObservationIgnored
    private let audioEngine: AudioEngine
    @ObservationIgnored
    private var configuredAudioEngine = false
    @ObservationIgnored
    private let vad: SemanticVAD
    @ObservationIgnored
    private var model: SpeechGenerationModel?
    @ObservationIgnored
    private var captureTask: Task<Void, Never>?
    @ObservationIgnored
    private var languageSession: LanguageModelSession?
    @ObservationIgnored
    private var incompleteTimeoutTask: Task<Void, Never>?
    @ObservationIgnored
    private var llmTurnTask: Task<Void, Never>?
    @ObservationIgnored
    private var incompleteTimeoutRevision: Int = 0
    @ObservationIgnored
    private var llmTurnRevision: Int = 0
    @ObservationIgnored
    private var turnCompletionConfig = UserTurnCompletionConfig()

    init(ttsRepoId: String = "Marvis-AI/marvis-tts-250m-v0.2-MLX-8bit") {
        self.audioEngine = AudioEngine(inputBufferSize: 1024)
        self.vad = SemanticVAD()
        audioEngine.delegate = self

        Task { @MainActor in
            do {
                print("Loading TTS model: \(ttsRepoId)")
                self.model = try await TTS.loadModel(modelRepo: ttsRepoId)
                print("Loaded TTS model.")
            } catch {
                print("Error loading model: \(error)")
            }
            self.canSpeak = model != nil
        }
    }

    func start() async throws {
#if os(iOS)
        let session = AVAudioSession.sharedInstance()
        try session.setActive(false)
        try session.setCategory(.playAndRecord, mode: .voiceChat, policy: .default, options: [.defaultToSpeaker])
        try session.setPreferredIOBufferDuration(0.02)
        try session.setActive(true)
#endif

        resetLanguageSession()
        try await ensureEngineStarted()
        startCaptureLoopIfNeeded()
        isActive = true
    }

    func stop() async throws {
        cancelTurnHandling()
        languageSession = nil
        stopCaptureLoop()
        audioEngine.endSpeaking()
        audioEngine.stop()
        isDetectingSpeech = false
        await vad.reset()
#if os(iOS)
        try AVAudioSession.sharedInstance().setActive(false)
#endif
        isActive = false
    }

    func toggleInputMute(toMuted: Bool?) async {
        let currentMuted = audioEngine.isMicrophoneMuted
        let newMuted = toMuted ?? !currentMuted
        audioEngine.isMicrophoneMuted = newMuted

        if newMuted, isDetectingSpeech {
            cancelTurnHandling()
            await vad.reset()
            isDetectingSpeech = false
        }
    }

    func stopSpeaking() async {
        audioEngine.endSpeaking()
    }

    func setUserTurnCompletionConfig(_ config: UserTurnCompletionConfig) {
        turnCompletionConfig = config
    }

    func speak(text: String) async throws {
        guard let model else {
            print("Error: TTS model not yet loaded.")
            return
        }

        let audioStream = model.generatePCMBufferStream(
            text: text,
            voice: "conversational_a",
            refAudio: nil,
            refText: nil,
            language: "en",
            generationParameters: model.defaultGenerationParameters
        )
        try await ensureEngineStarted()

        audioEngine.speak(buffersStream: audioStream)
    }

    private func resetLanguageSession() {
        let instructions = Self.baseInstructions + "\n\n" + turnCompletionConfig.completionInstructions
        languageSession = LanguageModelSession(instructions: instructions)
    }

    private func ensureEngineStarted() async throws {
        if !configuredAudioEngine {
            try audioEngine.setup()
            configuredAudioEngine = true
            print("Configured audio engine.")
        }
        try audioEngine.start()
        audioEngine.isMicrophoneMuted = false
        print("Started audio engine.")
    }

    private func startCaptureLoopIfNeeded() {
        guard captureTask == nil else { return }

        let stream = audioEngine.capturedChunks
        let vad = vad
        captureTask = Task(priority: .userInitiated) { [weak self] in
            await Self.runCaptureLoop(stream: stream, vad: vad) { event in
                await MainActor.run {
                    self?.handleVADEvent(event)
                }
            }
        }
    }

    private func stopCaptureLoop() {
        captureTask?.cancel()
        captureTask = nil
    }

    private func handleVADEvent(_ event: SemanticVAD.Event) {
        switch event {
        case .started:
            isDetectingSpeech = true
            cancelIncompleteTimeout()
            cancelLLMTurnTask()
            delegate?.conversationControllerDidStartUserSpeech(self)
        case let .stopped(transcription):
            isDetectingSpeech = false
            let cleaned = transcription?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            guard !cleaned.isEmpty else { return }
            delegate?.conversationController(self, didFinish: cleaned)
            handleCompletedUserTranscript(cleaned)
        }
    }

    private func cancelTurnHandling() {
        incompleteTimeoutRevision += 1
        incompleteTimeoutTask?.cancel()
        incompleteTimeoutTask = nil
        cancelLLMTurnTask()
    }

    private func cancelLLMTurnTask() {
        llmTurnRevision += 1
        llmTurnTask?.cancel()
        llmTurnTask = nil
    }

    private func cancelIncompleteTimeout() {
        incompleteTimeoutRevision += 1
        incompleteTimeoutTask?.cancel()
        incompleteTimeoutTask = nil
    }

    private func scheduleIncompleteTimeout(_ kind: IncompleteTimeoutKind) {
        cancelIncompleteTimeout()
        let revision = incompleteTimeoutRevision
        let timeout: TimeInterval
        let prompt: String
        switch kind {
        case .short:
            timeout = turnCompletionConfig.incompleteShortTimeout
            prompt = turnCompletionConfig.shortPrompt
        case .long:
            timeout = turnCompletionConfig.incompleteLongTimeout
            prompt = turnCompletionConfig.longPrompt
        }
        let delayNanos = UInt64(timeout * 1_000_000_000)

        print("Turn marked incomplete (\(kind.logLabel)); scheduling reprompt in \(timeout)s")

        incompleteTimeoutTask = Task { @MainActor [weak self] in
            do {
                try await Task.sleep(nanoseconds: delayNanos)
            } catch {
                return
            }
            guard let self else { return }
            guard revision == self.incompleteTimeoutRevision else { return }
            self.incompleteTimeoutTask = nil
            self.startLLMTurnTask(
                prompt: prompt,
                source: "incomplete_\(kind.logLabel)_timeout",
                originalTranscript: nil
            )
        }
    }

    private func startLLMTurnTask(
        prompt: String,
        source: String,
        originalTranscript: String?
    ) {
        cancelLLMTurnTask()
        let revision = llmTurnRevision
        llmTurnTask = Task { @MainActor [weak self] in
            guard let self else { return }
            await self.requestTurnAwareResponse(
                prompt: prompt,
                source: source,
                originalTranscript: originalTranscript
            )
            guard revision == self.llmTurnRevision else { return }
            self.llmTurnTask = nil
        }
    }

    private func requestTurnAwareResponse(
        prompt: String,
        source: String,
        originalTranscript: String? = nil
    ) async {
        guard let session = languageSession else { return }

        do {
            print("Using LLM prompt:\n\(prompt)")
            let response = try await session.respond(to: prompt)
            print("LLM turn response [\(source)]: \(response.content)")
            try await handleTurnResponse(
                response.content,
                source: source,
                originalTranscript: originalTranscript
            )
        } catch is CancellationError {
            // no-op
        } catch {
            print("Turn-aware response failed [\(source)]: \(error)")
        }
    }

    private func handleTurnResponse(
        _ text: String,
        source: String,
        originalTranscript: String?
    ) async throws {
        guard let (marker, payload) = parseTurnResponse(text) else {
            let fallback = text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !fallback.isEmpty else { return }
            print("Warning: Missing turn marker; speaking raw output.")
            try await speak(text: fallback)
            return
        }

        switch marker {
        case .complete:
            cancelIncompleteTimeout()
            let spoken = payload.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !spoken.isEmpty else { return }
            try await speak(text: spoken)
        case .incompleteShort:
            if source == "user_transcript",
               let originalTranscript,
               isLikelyCompleteUtterance(originalTranscript) {
                print("Guardrail: Model returned incomplete for a likely complete transcript; forcing immediate response.")
                await requestForcedCompleteResponse(for: originalTranscript)
                return
            }
            scheduleIncompleteTimeout(.short)
        case .incompleteLong:
            if source == "user_transcript",
               let originalTranscript,
               isLikelyCompleteUtterance(originalTranscript) {
                print("Guardrail: Model returned incomplete for a likely complete transcript; forcing immediate response.")
                await requestForcedCompleteResponse(for: originalTranscript)
                return
            }
            scheduleIncompleteTimeout(.long)
        }
    }

    private func requestForcedCompleteResponse(for transcript: String) async {
        let prompt = """
        The user has completed their turn. Respond now with a helpful assistant reply.

        User transcript:
        \(transcript)

        IMPORTANT: Your response MUST start with ✓ followed by your response text.
        Do NOT output ○ or ◐.
        """
        await requestTurnAwareResponse(
            prompt: prompt,
            source: "forced_complete_guardrail",
            originalTranscript: nil
        )
    }

    private func isLikelyCompleteUtterance(_ transcript: String) -> Bool {
        let trimmed = transcript.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return false }

        if let last = trimmed.last, [".", "?", "!"].contains(last) {
            return true
        }
        return false
    }

    private func parseTurnResponse(_ text: String) -> (TurnMarker, String)? {
        let matches: [(marker: TurnMarker, index: String.Index)] = [
            (.complete, text.firstIndex(of: "✓")),
            (.incompleteShort, text.firstIndex(of: "○")),
            (.incompleteLong, text.firstIndex(of: "◐")),
        ].compactMap { marker, index in
            guard let index else { return nil }
            return (marker, index)
        }

        guard let firstMatch = matches.min(by: { $0.index < $1.index }) else { return nil }

        switch firstMatch.marker {
        case .complete:
            var withoutMarker = text
            withoutMarker.remove(at: firstMatch.index)
            let content = withoutMarker.trimmingCharacters(in: .whitespacesAndNewlines)
            return (.complete, content)
        case .incompleteShort:
            return (.incompleteShort, "")
        case .incompleteLong:
            return (.incompleteLong, "")
        }
    }

    private func handleCompletedUserTranscript(_ transcription: String) {
        guard !isSpeaking, transcription.count > 1 else { return }
        let prompt = """
        Determine whether the user has completed their turn, then respond in the required marker format.

        User transcript:
        \(transcription)

        IMPORTANT:
        - Your response MUST begin with exactly one turn marker: ✓, ○, or ◐.
        - If COMPLETE, respond with ✓ followed by a space and your response text.
        - If INCOMPLETE, respond with ONLY ○ or ONLY ◐ and no additional text.
        """
        startLLMTurnTask(
            prompt: prompt,
            source: "user_transcript",
            originalTranscript: transcription
        )
    }

    @concurrent
    private static func runCaptureLoop(
        stream: AsyncStream<AudioChunk>,
        vad: SemanticVAD,
        onEvent: @escaping @Sendable (SemanticVAD.Event) async -> Void
    ) async {
        for await chunk in stream {
            if Task.isCancelled { break }
            if let event = await vad.process(chunk: chunk) {
                await onEvent(event)
            }
        }
    }
}

// MARK: - AudioEngineDelegate

extension ConversationController: @MainActor AudioEngineDelegate {
    func audioCaptureEngine(_ engine: AudioEngine, isSpeakingDidChange speaking: Bool) {
        isSpeaking = speaking
    }
}
