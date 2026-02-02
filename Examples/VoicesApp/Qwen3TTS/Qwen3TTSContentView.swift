//
//  Qwen3TTSContentView.swift
//  Qwen3TTS
//
//  Main UI for Qwen3-TTS demo app with voice selection and text input.
//

import SwiftUI
import UniformTypeIdentifiers

struct Qwen3TTSContentView: View {
    @StateObject private var viewModel = Qwen3TTSViewModel()
    @State private var inputText = "Hello! Thanks for calling today. I'm Vivian, your support agent. Let's take a look at what's going on with your account."
    @State private var showSettings = false
    @State private var showFilePicker = false
    @FocusState private var isTextEditorFocused: Bool

    private static let customVoiceDefaultText = "Hello! Thanks for calling today. I'm Vivian, your support agent. Let's take a look at what's going on with your account."
    private static let voiceDesignDefaultText = "It's in the top drawer... wait, it's empty? No way, that's impossible! I'm sure I put it there."
    private static let voiceCloningDefaultText = "Hey, it's Jerry. I'm calling to sell you nothing! That's right. Nothing at all."

    var body: some View {
        NavigationStack {
            ZStack {
                #if os(iOS)
                Color(.systemBackground)
                    .ignoresSafeArea()
                #endif

                ScrollView(showsIndicators: false) {
                    VStack(spacing: 20) {
                        // Status Header
                        statusHeaderView

                        // Model Selection
                        modelSelectionView

                        // Voice Selection (for CustomVoice)
                        if viewModel.selectedModel.usesVoiceNames {
                            voiceSelectionView
                        }

                        // Reference Audio (for Base model voice cloning)
                        if viewModel.selectedModel.supportsVoiceCloning {
                            referenceAudioView
                        }

                        // Voice Description (for CustomVoice and VoiceDesign models, not Base)
                        if !viewModel.selectedModel.supportsVoiceCloning {
                            voiceInstructView
                        }

                        // Text Input
                        textInputView

                        // Settings (collapsible)
                        settingsView

                        // Action Buttons
                        actionButtonsView
                    }
                    .padding()
                }
            }
            .navigationTitle("Qwen3 TTS")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.large)
            #endif
            .contentShape(Rectangle())
            .onTapGesture {
                isTextEditorFocused = false
            }
            .task {
                await viewModel.loadModel()
            }
            .onChange(of: viewModel.selectedModel) { _, newModel in
                // Update default text based on model type
                if newModel.supportsVoiceCloning {
                    inputText = Self.voiceCloningDefaultText
                } else if newModel.usesVoiceNames {
                    inputText = Self.customVoiceDefaultText
                } else {
                    inputText = Self.voiceDesignDefaultText
                }
                // Clear reference audio when switching away from Base model
                if !newModel.supportsVoiceCloning {
                    viewModel.clearReferenceAudio()
                }
            }
            .fileImporter(
                isPresented: $showFilePicker,
                allowedContentTypes: [UTType.audio, UTType.wav],
                allowsMultipleSelection: false
            ) { result in
                switch result {
                case .success(let urls):
                    if let url = urls.first {
                        // Start accessing security-scoped resource
                        if url.startAccessingSecurityScopedResource() {
                            viewModel.loadReferenceAudio(from: url)
                            url.stopAccessingSecurityScopedResource()
                        } else {
                            viewModel.loadReferenceAudio(from: url)
                        }
                    }
                case .failure(let error):
                    print("File picker error: \(error)")
                }
            }
        }
    }

    // MARK: - View Components

    private var statusHeaderView: some View {
        HStack {
            stateIndicator
            Spacer()
            // Show stats in header when idle and stats are available
            if case .idle = viewModel.state, viewModel.generationTime > 0 {
                statsContent
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.secondary.opacity(0.1))
        )
    }

    @ViewBuilder
    private var stateIndicator: some View {
        HStack(spacing: 8) {
            switch viewModel.state {
            case .idle:
                if viewModel.isModelLoaded {
                    if viewModel.generationTime > 0 {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                    } else {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                        Text("Ready")
                    }
                } else {
                    Text("Not loaded")
                }

            case .loading:
                ProgressView()
                    .controlSize(.small)
                Text("Loading model...")

            case .generating(let step, let codes):
                ProgressView()
                    .controlSize(.small)
                Text("Generating: step \(step), \(codes) codes")

            case .decoding:
                ProgressView()
                    .controlSize(.small)
                Text("Decoding audio...")

            case .playing:
                Image(systemName: "speaker.wave.2.fill")
                    .foregroundStyle(.blue)
                    .symbolEffect(.pulse)
                Text("Playing audio")

            case .error(let message):
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(.red)
                Text(message)
                    .lineLimit(2)
            }
        }
        .font(.subheadline)
    }

    @ViewBuilder
    private var statsContent: some View {
        HStack(spacing: 12) {
            if viewModel.generationTime > 0 {
                Label(String(format: "%.1fs", viewModel.generationTime), systemImage: "cpu")
            }
            if viewModel.audioDuration > 0 {
                Label(String(format: "%.1fs", viewModel.audioDuration), systemImage: "waveform")
            }
            if viewModel.generationTime > 0 && viewModel.audioDuration > 0 {
                let rtf = viewModel.audioDuration / viewModel.generationTime
                Label(String(format: "%.2fx", rtf), systemImage: "speedometer")
            }
        }
        .font(.caption)
        .foregroundStyle(.secondary)
    }

    private var modelSelectionView: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Model")
                .font(.subheadline)
                .foregroundStyle(.secondary)

            Menu {
                ForEach(Qwen3Model.allCases) { model in
                    Button {
                        if viewModel.selectedModel != model {
                            viewModel.selectedModel = model
                            Task {
                                await viewModel.loadModel()
                            }
                        }
                    } label: {
                        HStack {
                            Text(model.displayName)
                            if viewModel.selectedModel == model {
                                Image(systemName: "checkmark")
                            }
                        }
                    }
                }
            } label: {
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text(viewModel.selectedModel.displayName)
                            .foregroundStyle(.primary)
                        Spacer()
                        Image(systemName: "chevron.up.chevron.down")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Text(viewModel.selectedModel.description)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }
                .padding()
                .background(
                    RoundedRectangle(cornerRadius: 10)
                        .fill(Color.secondary.opacity(0.1))
                )
            }
            .disabled(viewModel.state.isActive)
        }
    }

    private var voiceSelectionView: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Voice")
                .font(.subheadline)
                .foregroundStyle(.secondary)

            Menu {
                ForEach(Qwen3Voice.allCases) { voice in
                    Button {
                        viewModel.selectedVoice = voice
                    } label: {
                        HStack {
                            Text(voice.displayName)
                            if viewModel.selectedVoice == voice {
                                Image(systemName: "checkmark")
                            }
                        }
                    }
                }
            } label: {
                HStack {
                    Text(viewModel.selectedVoice.displayName)
                        .foregroundStyle(.primary)
                    Spacer()
                    Image(systemName: "chevron.up.chevron.down")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding()
                .background(
                    RoundedRectangle(cornerRadius: 10)
                        .fill(Color.secondary.opacity(0.1))
                )
            }
            .disabled(viewModel.state.isActive)
        }
    }

    private var referenceAudioView: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Reference Audio")
                .font(.subheadline)
                .foregroundStyle(.secondary)

            HStack {
                Button {
                    showFilePicker = true
                } label: {
                    HStack {
                        Image(systemName: "waveform.badge.plus")
                        Text(viewModel.referenceAudioURL?.lastPathComponent ?? "Select Audio File")
                            .lineLimit(1)
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(
                        RoundedRectangle(cornerRadius: 10)
                            .fill(Color.secondary.opacity(0.1))
                    )
                }
                .buttonStyle(.plain)
                .disabled(viewModel.state.isActive)

                if viewModel.referenceAudioURL != nil {
                    Button {
                        viewModel.clearReferenceAudio()
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                }
            }

            Text("WAV file at 24kHz for voice cloning")
                .font(.caption)
                .foregroundStyle(.tertiary)
        }
    }

    private var voiceInstructView: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Voice Description")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                if viewModel.selectedModel.usesVoiceNames {
                    Text("(Optional)")
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                }
            }

            TextField("Describe the voice style...", text: $viewModel.voiceInstruct, axis: .vertical)
                .lineLimit(2...4)
                .textFieldStyle(.plain)
                .padding()
                .background(
                    RoundedRectangle(cornerRadius: 10)
                        .fill(Color.secondary.opacity(0.1))
                )
                .disabled(viewModel.state.isActive)

            Text("E.g., \"A calm, clear female voice with a professional tone.\"")
                .font(.caption)
                .foregroundStyle(.tertiary)
        }
    }

    private var textInputView: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Text Input")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)

                Spacer()

                if !inputText.isEmpty {
                    Button {
                        inputText = ""
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundStyle(.secondary)
                    }
                }
            }

            ZStack(alignment: .topLeading) {
                TextEditor(text: $inputText)
                    .font(.body)
                    .frame(minHeight: 150)
                    .scrollContentBackground(.hidden)
                    .padding()
                    .background(
                        RoundedRectangle(cornerRadius: 12)
                            .fill(Color.secondary.opacity(0.1))
                    )
                    .focused($isTextEditorFocused)
                    .disabled(viewModel.state.isActive)

                if inputText.isEmpty {
                    Text("Enter text to synthesize...")
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 20)
                        .padding(.top, 25)
                        .allowsHitTesting(false)
                }
            }

            Text("\(inputText.count) characters")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var settingsView: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header with toggle
            Button {
                withAnimation(.easeInOut(duration: 0.2)) {
                    showSettings.toggle()
                }
            } label: {
                HStack {
                    Image(systemName: "slider.horizontal.3")
                        .foregroundStyle(.secondary)
                    Text("Generation Settings")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Image(systemName: showSettings ? "chevron.up" : "chevron.down")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .buttonStyle(.plain)

            if showSettings {
                VStack(spacing: 16) {
                    // Temperature
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text("Temperature")
                                .font(.caption)
                            Spacer()
                            Text(String(format: "%.2f", viewModel.temperature))
                                .font(.caption.monospacedDigit())
                                .foregroundStyle(.secondary)
                        }
                        Slider(value: $viewModel.temperature, in: 0.1...2.0, step: 0.05)
                            .tint(.blue)
                    }

                    // Top-K
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text("Top-K")
                                .font(.caption)
                            Spacer()
                            Text("\(viewModel.topK)")
                                .font(.caption.monospacedDigit())
                                .foregroundStyle(.secondary)
                        }
                        Slider(value: Binding(
                            get: { Double(viewModel.topK) },
                            set: { viewModel.topK = Int($0) }
                        ), in: 1...100, step: 1)
                            .tint(.blue)
                    }

                    // Top-P
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text("Top-P")
                                .font(.caption)
                            Spacer()
                            Text(String(format: "%.2f", viewModel.topP))
                                .font(.caption.monospacedDigit())
                                .foregroundStyle(.secondary)
                        }
                        Slider(value: $viewModel.topP, in: 0.1...1.0, step: 0.05)
                            .tint(.blue)
                    }

                    // Repetition Penalty
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text("Repetition Penalty")
                                .font(.caption)
                            Spacer()
                            Text(String(format: "%.2f", viewModel.repetitionPenalty))
                                .font(.caption.monospacedDigit())
                                .foregroundStyle(.secondary)
                        }
                        Slider(value: $viewModel.repetitionPenalty, in: 1.0...2.0, step: 0.05)
                            .tint(.blue)
                    }

                    // Max Tokens
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text("Max Tokens")
                                .font(.caption)
                            Spacer()
                            Text("\(viewModel.maxTokens)")
                                .font(.caption.monospacedDigit())
                                .foregroundStyle(.secondary)
                        }
                        Slider(value: Binding(
                            get: { Double(viewModel.maxTokens) },
                            set: { viewModel.maxTokens = Int($0) }
                        ), in: 100...4000, step: 100)
                            .tint(.blue)
                    }

                    // Reset button
                    Button {
                        viewModel.temperature = 0.3
                        viewModel.topK = 50
                        viewModel.topP = 0.95
                        viewModel.repetitionPenalty = 1.05
                        viewModel.maxTokens = 2000
                    } label: {
                        HStack {
                            Image(systemName: "arrow.counterclockwise")
                            Text("Reset to Defaults")
                        }
                        .font(.caption)
                    }
                    .buttonStyle(.bordered)
                }
                .padding()
                .background(
                    RoundedRectangle(cornerRadius: 10)
                        .fill(Color.secondary.opacity(0.05))
                )
                .disabled(viewModel.state.isActive)
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.secondary.opacity(0.1))
        )
    }

    private var actionButtonsView: some View {
        HStack(spacing: 16) {
            // Generate / Stop button
            Button {
                isTextEditorFocused = false
                if viewModel.state.isGenerating {
                    viewModel.stopGeneration()
                } else {
                    viewModel.startGeneration(text: inputText)
                }
            } label: {
                ZStack {
                    Circle()
                        .fill(viewModel.state.isGenerating ? Color.red : Color.blue)
                        .frame(width: 56, height: 56)

                    if viewModel.state.isGenerating {
                        Image(systemName: "stop.fill")
                            .font(.title2)
                            .foregroundStyle(.white)
                    } else if case .loading = viewModel.state {
                        ProgressView()
                            .tint(.white)
                    } else {
                        Image(systemName: "waveform")
                            .font(.title2)
                            .foregroundStyle(.white)
                    }
                }
            }
            .buttonStyle(.plain)
            .disabled(!viewModel.isModelLoaded || viewModel.state == .playing || viewModel.state == .loading || (!viewModel.state.isGenerating && inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty))

            // Play / Stop button
            Button {
                if case .playing = viewModel.state {
                    viewModel.stopPlayback()
                } else {
                    Task {
                        await viewModel.playAudio()
                    }
                }
            } label: {
                ZStack {
                    Circle()
                        .fill(viewModel.state == .playing ? Color.red : (viewModel.hasGeneratedAudio ? Color.orange : Color.secondary.opacity(0.3)))
                        .frame(width: 56, height: 56)

                    Image(systemName: viewModel.state == .playing ? "stop.fill" : "play.fill")
                        .font(.title2)
                        .foregroundStyle(viewModel.hasGeneratedAudio ? .white : .secondary)
                }
            }
            .buttonStyle(.plain)
            .disabled(!viewModel.hasGeneratedAudio || viewModel.state.isGenerating || viewModel.state == .loading)

            // Share button
            if let audioURL = viewModel.lastAudioURL, viewModel.hasGeneratedAudio && !viewModel.state.isActive {
                ShareLink(item: audioURL) {
                    ZStack {
                        Circle()
                            .fill(Color.green)
                            .frame(width: 56, height: 56)

                        Image(systemName: "square.and.arrow.up")
                            .font(.title2)
                            .foregroundStyle(.white)
                    }
                }
                .buttonStyle(.plain)
            } else {
                // Placeholder when share is disabled
                ZStack {
                    Circle()
                        .fill(Color.secondary.opacity(0.3))
                        .frame(width: 56, height: 56)

                    Image(systemName: "square.and.arrow.up")
                        .font(.title2)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .padding(.vertical, 8)
    }
}

// MARK: - Preview

#Preview {
    Qwen3TTSContentView()
}
