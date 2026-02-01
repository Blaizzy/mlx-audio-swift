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

    private var statsView: some View {
        HStack(spacing: 20) {
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
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.secondary.opacity(0.1))
        )
    }

    private var actionButtonsView: some View {
        HStack(spacing: 12) {
            // Generate button
            Button {
                isTextEditorFocused = false
                Task {
                    await viewModel.generate(text: inputText)
                }
            } label: {
                HStack {
                    if viewModel.state.isActive && viewModel.state != .playing {
                        ProgressView()
                            .controlSize(.small)
                            #if os(iOS)
                            .tint(.white)
                            #endif
                    } else {
                        Image(systemName: "play.fill")
                    }
                    Text(generateButtonTitle)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 14)
            }
            .buttonStyle(.borderedProminent)
            .disabled(!viewModel.isModelLoaded || viewModel.state.isActive || inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

            // Stop button
            Button {
                viewModel.stopPlayback()
            } label: {
                HStack {
                    Image(systemName: "stop.fill")
                    Text("Stop")
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 14)
            }
            .buttonStyle(.borderedProminent)
            .tint(.red)
            .disabled(viewModel.state != .playing)

            // Share button
            if let audioURL = viewModel.lastAudioURL {
                ShareLink(item: audioURL) {
                    HStack {
                        Image(systemName: "square.and.arrow.up")
                        Text("Share")
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 14)
                }
                .buttonStyle(.borderedProminent)
                .tint(.green)
                .disabled(viewModel.state.isActive)
            }
        }
    }

    private var generateButtonTitle: String {
        switch viewModel.state {
        case .loading:
            return "Loading..."
        case .generating:
            return "Generating..."
        case .decoding:
            return "Decoding..."
        default:
            return "Generate"
        }
    }
}

// MARK: - Stat View

struct Qwen3StatView: View {
    let title: String
    let value: String
    let icon: String

    var body: some View {
        VStack(spacing: 4) {
            Image(systemName: icon)
                .font(.title3)
                .foregroundStyle(.secondary)

            Text(value)
                .font(.headline)

            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
    }
}

// MARK: - Preview

#Preview {
    Qwen3TTSContentView()
}
