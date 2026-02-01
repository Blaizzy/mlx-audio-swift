//
//  Qwen3TTSContentView.swift
//  Qwen3TTS
//
//  Main UI for Qwen3-TTS demo app with voice selection and text input.
//

import SwiftUI

struct Qwen3TTSContentView: View {
    @StateObject private var viewModel = Qwen3TTSViewModel()
    @State private var inputText = "This is a demo of the Qwen3 TTS model running on Apple Silicon."
    @State private var showSettings = false
    @FocusState private var isTextEditorFocused: Bool

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

                        // Voice Selection (for CustomVoice) or Instruct (for VoiceDesign/Base)
                        if viewModel.selectedModel.usesVoiceNames {
                            voiceSelectionView
                        } else {
                            voiceInstructView
                        }

                        // Text Input
                        textInputView

                        // Settings (collapsible)
                        settingsView

                        // Generation Stats
                        if viewModel.generationTime > 0 || viewModel.audioDuration > 0 {
                            statsView
                        }

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
        }
    }

    // MARK: - View Components

    private var statusHeaderView: some View {
        VStack(spacing: 8) {
            HStack {
                stateIndicator
                Spacer()
            }

            // Download progress
            if case .loading = viewModel.state {
                VStack(alignment: .leading, spacing: 4) {
                    ProgressView(value: viewModel.downloadProgress)
                        .tint(.blue)

                    Text("Downloading model: \(Int(viewModel.downloadProgress * 100))%")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
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
                Image(systemName: "checkmark.circle.fill")
                    .foregroundStyle(.green)
                Text(viewModel.isModelLoaded ? "Ready" : "Not loaded")

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
                                await viewModel.loadModel(forceReload: true)
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

    private var voiceInstructView: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Voice Description")
                .font(.subheadline)
                .foregroundStyle(.secondary)

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
                .foregroundStyle(.secondary)
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

    private var statsView: some View {
        HStack(spacing: 16) {
            if viewModel.generationTime > 0 {
                Qwen3StatView(
                    title: "Generation",
                    value: String(format: "%.1fs", viewModel.generationTime),
                    icon: "cpu"
                )
            }

            if viewModel.audioDuration > 0 {
                Qwen3StatView(
                    title: "Duration",
                    value: String(format: "%.1fs", viewModel.audioDuration),
                    icon: "waveform"
                )
            }

            if viewModel.generationTime > 0 && viewModel.audioDuration > 0 {
                let rtf = viewModel.audioDuration / viewModel.generationTime
                Qwen3StatView(
                    title: "RTF",
                    value: String(format: "%.2fx", rtf),
                    icon: "speedometer"
                )
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(Color.secondary.opacity(0.05))
        )
    }

    private var actionButtonsView: some View {
        VStack(spacing: 12) {
            // Primary Generate button (full width)
            Button {
                isTextEditorFocused = false
                if viewModel.state.isGenerating {
                    viewModel.stopGeneration()
                } else {
                    viewModel.startGeneration(text: inputText)
                }
            } label: {
                HStack(spacing: 8) {
                    if viewModel.state.isGenerating {
                        Image(systemName: "stop.fill")
                        Text("Stop")
                    } else if case .loading = viewModel.state {
                        ProgressView()
                            .controlSize(.small)
                            #if os(iOS)
                            .tint(.white)
                            #endif
                        Text("Loading...")
                    } else {
                        Image(systemName: "waveform")
                        Text("Generate Speech")
                    }
                }
                .font(.headline)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 16)
            }
            .buttonStyle(.borderedProminent)
            .tint(viewModel.state.isGenerating ? .red : .blue)
            .disabled(!viewModel.isModelLoaded || viewModel.state == .playing || viewModel.state == .loading || (!viewModel.state.isGenerating && inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty))

            // Secondary buttons row
            HStack(spacing: 12) {
                // Play / Stop Playback button
                Button {
                    if case .playing = viewModel.state {
                        viewModel.stopPlayback()
                    } else {
                        Task {
                            await viewModel.playAudio()
                        }
                    }
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: viewModel.state == .playing ? "stop.fill" : "play.fill")
                        Text(viewModel.state == .playing ? "Stop" : "Play")
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 12)
                }
                .buttonStyle(.bordered)
                .tint(viewModel.state == .playing ? .red : .primary)
                .disabled(!viewModel.hasGeneratedAudio || viewModel.state.isGenerating || viewModel.state == .loading)

                // Share button
                ShareLink(item: viewModel.lastAudioURL ?? URL(fileURLWithPath: "/")) {
                    HStack(spacing: 6) {
                        Image(systemName: "square.and.arrow.up")
                        Text("Share")
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 12)
                }
                .buttonStyle(.bordered)
                .tint(.primary)
                .disabled(!viewModel.hasGeneratedAudio || viewModel.state.isActive)
            }
        }
    }
}

// MARK: - Stat View

struct Qwen3StatView: View {
    let title: String
    let value: String
    let icon: String

    var body: some View {
        VStack(spacing: 2) {
            Image(systemName: icon)
                .font(.caption)
                .foregroundStyle(.tertiary)

            Text(value)
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)

            Text(title)
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
        .frame(maxWidth: .infinity)
    }
}

// MARK: - Preview

#Preview {
    Qwen3TTSContentView()
}
