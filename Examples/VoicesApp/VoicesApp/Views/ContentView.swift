import SwiftUI

// MARK: - Compact Slider (iOS)

#if os(iOS)
struct CompactToggle: View {
    let label: String
    @Binding var isOn: Bool
    var font: Font = .footnote
    var toggleWidth: CGFloat = 40
    var toggleHeight: CGFloat = 24
    var thumbSize: CGFloat = 20
    var tint: Color = .blue

    var body: some View {
        HStack {
            Text(label)
                .font(font)
            Spacer()
            ZStack(alignment: isOn ? .trailing : .leading) {
                Capsule()
                    .fill(isOn ? tint : Color.gray.opacity(0.3))
                    .frame(width: toggleWidth, height: toggleHeight)

                Circle()
                    .fill(.white)
                    .shadow(color: .black.opacity(0.15), radius: 1, x: 0, y: 1)
                    .frame(width: thumbSize, height: thumbSize)
                    .padding(2)
            }
            .animation(.easeInOut(duration: 0.15), value: isOn)
            .onTapGesture {
                isOn.toggle()
            }
        }
    }
}

struct CompactSlider: View {
    @Binding var value: Double
    let range: ClosedRange<Double>
    var step: Double? = nil
    var thumbSize: CGFloat = 16
    var trackHeight: CGFloat = 4
    var tint: Color = .blue

    var body: some View {
        GeometryReader { geometry in
            let width = geometry.size.width
            let thumbOffset = (value - range.lowerBound) / (range.upperBound - range.lowerBound) * (width - thumbSize)

            ZStack(alignment: .leading) {
                // Track background
                Capsule()
                    .fill(Color.gray.opacity(0.3))
                    .frame(height: trackHeight)

                // Track fill
                Capsule()
                    .fill(tint)
                    .frame(width: thumbOffset + thumbSize / 2, height: trackHeight)

                // Thumb
                Circle()
                    .fill(.white)
                    .shadow(color: .black.opacity(0.15), radius: 2, x: 0, y: 1)
                    .frame(width: thumbSize, height: thumbSize)
                    .offset(x: thumbOffset)
                    .gesture(
                        DragGesture(minimumDistance: 0)
                            .onChanged { gesture in
                                let newValue = range.lowerBound + (gesture.location.x / width) * (range.upperBound - range.lowerBound)
                                let clampedValue = min(max(newValue, range.lowerBound), range.upperBound)
                                if let step = step {
                                    value = (clampedValue / step).rounded() * step
                                } else {
                                    value = clampedValue
                                }
                            }
                    )
            }
        }
        .frame(height: thumbSize)
    }
}
#endif

struct ContentView: View {
    @Environment(\.scenePhase) private var scenePhase
    @State private var viewModel = TTSViewModel()
    @State private var textInput = ""
    @State private var selectedVoice: Voice?
    @State private var showVoices = false
    @State private var showSettings = false
    @State private var showVoiceDesign = false
    @State private var showVoiceCloning = false
    @State private var recentlyUsed: [Voice] = Voice.samples
    @State private var customVoices: [Voice] = Voice.customVoices
    @State private var clonedVoices: [Voice] = []
    @State private var selectedClonedVoice: Voice?

    #if os(iOS)
    private let buttonHeight: CGFloat = 44
    private let buttonFont: Font = .callout
    private let inputFont: Font = .body
    #else
    private let buttonHeight: CGFloat = 44
    private let buttonFont: Font = .subheadline
    private let inputFont: Font = .title2
    #endif

    var body: some View {
        VStack(spacing: 0) {
            // Main text input area - fills available space
            TextEditor(text: $textInput)
                .font(inputFont)
                .scrollContentBackground(.hidden)
                .background(.clear)
                .disabled(viewModel.isGenerating)
                .padding(.horizontal, 12)
                .padding(.top, 4)
                .overlay(alignment: .topLeading) {
                    if textInput.isEmpty {
                        Text("Start typing here...")
                            .font(inputFont)
                            .foregroundStyle(.tertiary)
                            .padding(.horizontal, 16)
                            .padding(.top, 12)
                            .allowsHitTesting(false)
                    }
                }

            // Bottom content (status, player)
            VStack(spacing: 4) {
                // Status/Progress
                if !viewModel.generationProgress.isEmpty {
                    HStack(spacing: 6) {
                        ProgressView()
                            .scaleEffect(0.6)
                        Text(viewModel.generationProgress)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal)
                }

                // Error message
                if let error = viewModel.errorMessage {
                    Text(error)
                        .font(.caption2)
                        .foregroundStyle(.red)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal)
                }

                // Audio player
                if viewModel.audioURL != nil {
                    CompactAudioPlayer(
                        isPlaying: viewModel.isPlaying,
                        currentTime: viewModel.currentTime,
                        duration: viewModel.duration,
                        onPlayPause: { viewModel.togglePlayPause() },
                        onSeek: { viewModel.seek(to: $0) }
                    )
                    .padding(.horizontal)
                }
            }
            .padding(.bottom, 4)

            // Bottom bar
            HStack(spacing: 8) {
                // Voice controls - adapt based on model variant
                voiceControlChip

                // Settings button
                Button(action: { showSettings = true }) {
                    #if os(iOS)
                    Image(systemName: "slider.horizontal.3")
                        .font(buttonFont)
                        .foregroundStyle(.primary)
                        .frame(width: buttonHeight, height: buttonHeight)
                        .background(Color.gray.opacity(0.2))
                        .clipShape(Capsule())
                    #else
                    Label("Settings", systemImage: "slider.horizontal.3")
                    #endif
                }
                #if os(iOS)
                .buttonStyle(.plain)
                #else
                .buttonStyle(.bordered)
                #endif

                Spacer()

                // Generate / Stop button
                if viewModel.isGenerating {
                    Button(action: {
                        viewModel.stop()
                    }) {
                        #if os(iOS)
                        HStack(spacing: 6) {
                            ProgressView()
                                .tint(.white)
                                .scaleEffect(0.7)
                            Text("Stop")
                                .font(buttonFont)
                                .fontWeight(.medium)
                        }
                        .foregroundStyle(.white)
                        .frame(height: buttonHeight)
                        .padding(.horizontal, 16)
                        .background(Color.red)
                        .clipShape(Capsule())
                        #else
                        HStack(spacing: 6) {
                            ProgressView()
                                .controlSize(.small)
                                .tint(.white)
                            Text("Generating…")
                        }
                        #endif
                    }
                    #if os(iOS)
                    .buttonStyle(.plain)
                    #else
                    .buttonStyle(.borderedProminent)
                    .tint(.orange)
                    #endif
                } else {
                    Button(action: {
                        let voice: Voice? = viewModel.voiceMode == .voiceList ? selectedVoice : nil
                        viewModel.startSynthesis(text: textInput, voice: voice)
                        if let voice = selectedVoice, viewModel.voiceMode == .voiceList {
                            recentlyUsed.removeAll { $0.id == voice.id }
                            recentlyUsed.insert(voice, at: 0)
                        }
                    }) {
                        #if os(iOS)
                        Text("Generate")
                            .font(buttonFont)
                            .fontWeight(.medium)
                            .foregroundStyle(canGenerate ? .white : .secondary)
                            .frame(height: buttonHeight)
                            .padding(.horizontal, 16)
                            .background(canGenerate ? Color.blue : Color.gray.opacity(0.2))
                            .clipShape(Capsule())
                        #else
                        Label("Generate", systemImage: "waveform")
                        #endif
                    }
                    #if os(iOS)
                    .buttonStyle(.plain)
                    #else
                    .buttonStyle(.borderedProminent)
                    #endif
                    .disabled(!canGenerate)
                }
            }
            .padding(.horizontal)
            .padding(.vertical, 8)
            #if os(iOS)
            .background(Color(uiColor: .systemBackground).opacity(0.95))
            #else
            .background(.bar)
            #endif
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .sheet(isPresented: $showVoices) {
            VoicesView(
                recentlyUsed: $recentlyUsed,
                customVoices: $customVoices
            ) { voice in
                selectedVoice = voice
                showVoices = false
            }
            #if os(iOS)
            .presentationDetents([.large])
            .presentationDragIndicator(.visible)
            #endif
        }
        .sheet(isPresented: $showSettings) {
            SettingsView(viewModel: viewModel)
                #if os(iOS)
                .presentationDetents([.large])
                .presentationDragIndicator(.visible)
                #endif
        }
        .sheet(isPresented: $showVoiceDesign) {
            VoiceDesignSheet(voiceInstruct: $viewModel.voiceInstruct)
                #if os(iOS)
                .presentationDetents([.medium])
                .presentationDragIndicator(.visible)
                #else
                .frame(minWidth: 400, minHeight: 300)
                #endif
        }
        .sheet(isPresented: $showVoiceCloning) {
            ClonedVoicesView(
                clonedVoices: $clonedVoices,
                selectedVoice: selectedClonedVoice
            ) { voice in
                selectedClonedVoice = voice
                viewModel.referenceAudioURL = voice.audioFileURL
                viewModel.referenceTranscription = voice.transcription ?? ""
                showVoiceCloning = false
            }
            #if os(iOS)
            .presentationDetents([.large])
            .presentationDragIndicator(.visible)
            #endif
        }
        .onChange(of: scenePhase) { _, phase in
            switch phase {
            case .background:
                viewModel.pause()
                viewModel.stop()
            default:
                break
            }
        }
        .task {
            await viewModel.loadModel()
        }
    }

    private var canGenerate: Bool {
        !textInput.isEmpty && !viewModel.isGenerating && viewModel.isModelLoaded
    }

    // MARK: - Voice Control Chip

    @ViewBuilder
    private var voiceControlChip: some View {
        switch viewModel.voiceMode {
        case .voiceList:
            // VyvoTTS: Voice browser button
            Button(action: { showVoices = true }) {
                HStack(spacing: 6) {
                    if let voice = selectedVoice {
                        VoiceAvatar(color: voice.color, size: 20)
                        Text("\(voice.name)")
                            .lineLimit(1)
                    } else {
                        Image(systemName: "waveform")
                        Text("Voice")
                    }
                }
                .font(buttonFont)
                #if os(iOS)
                .foregroundStyle(.primary)
                .frame(height: buttonHeight)
                .padding(.horizontal, 12)
                .background(Color.gray.opacity(0.2))
                .clipShape(Capsule())
                #endif
            }
            #if os(iOS)
            .buttonStyle(.plain)
            #else
            .buttonStyle(.bordered)
            #endif

        case .presetVoices:
            // Qwen3 CustomVoice: Menu picker + optional style
            Menu {
                ForEach(TTSViewModel.qwen3Voices, id: \.id) { voice in
                    Button(action: { viewModel.selectedQwen3Voice = voice.id }) {
                        HStack {
                            Text(voice.name)
                            if viewModel.selectedQwen3Voice == voice.id {
                                Image(systemName: "checkmark")
                            }
                        }
                    }
                }
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: "person.wave.2")
                    Text(TTSViewModel.qwen3Voices.first { $0.id == viewModel.selectedQwen3Voice }?.name ?? "Voice")
                        .lineLimit(1)
                }
                .font(buttonFont)
                #if os(iOS)
                .foregroundStyle(.primary)
                .frame(height: buttonHeight)
                .padding(.horizontal, 12)
                .background(Color.gray.opacity(0.2))
                .clipShape(Capsule())
                #endif
            }

            // Optional style instruction
            Button(action: { showVoiceDesign = true }) {
                HStack(spacing: 6) {
                    Image(systemName: "theatermasks")
                    #if os(macOS)
                    Text(viewModel.voiceInstruct.isEmpty ? "Style" : "Styled")
                    #endif
                }
                .font(buttonFont)
                #if os(iOS)
                .foregroundStyle(.primary)
                .frame(width: buttonHeight, height: buttonHeight)
                .background(Color.gray.opacity(0.2))
                .clipShape(Capsule())
                #endif
            }
            #if os(iOS)
            .buttonStyle(.plain)
            #else
            .buttonStyle(.bordered)
            #endif

        case .voiceCloning:
            // Qwen3 Base: Cloned voices browser
            Button(action: { showVoiceCloning = true }) {
                HStack(spacing: 6) {
                    Image(systemName: "mic.badge.plus")
                    if let voice = selectedClonedVoice {
                        Text(voice.name)
                            .lineLimit(1)
                    } else {
                        Text("Clone")
                    }
                }
                .font(buttonFont)
                #if os(iOS)
                .foregroundStyle(.primary)
                .frame(height: buttonHeight)
                .padding(.horizontal, 12)
                .background(Color.gray.opacity(0.2))
                .clipShape(Capsule())
                #endif
            }
            #if os(iOS)
            .buttonStyle(.plain)
            #else
            .buttonStyle(.bordered)
            #endif

        case .voiceDesign:
            // Qwen3 VoiceDesign: Style description button
            Button(action: { showVoiceDesign = true }) {
                HStack(spacing: 6) {
                    Image(systemName: "theatermasks")
                    Text(viewModel.voiceInstruct.isEmpty ? "Style" : "Styled")
                }
                .font(buttonFont)
                #if os(iOS)
                .foregroundStyle(.primary)
                .frame(height: buttonHeight)
                .padding(.horizontal, 12)
                .background(Color.gray.opacity(0.2))
                .clipShape(Capsule())
                #endif
            }
            #if os(iOS)
            .buttonStyle(.plain)
            #else
            .buttonStyle(.bordered)
            #endif
        }
    }
}

// MARK: - Voice Selector Button

struct VoiceSelectorButton: View {
    let selectedVoice: Voice?
    let isLoading: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 12) {
                if let voice = selectedVoice {
                    VoiceAvatar(color: voice.color, size: 44)

                    VStack(alignment: .leading, spacing: 2) {
                        Text(voice.name)
                            .font(.body)
                            .fontWeight(.medium)
                            .foregroundStyle(.primary)

                        Text(voice.description.isEmpty ? voice.language : voice.description)
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }
                } else {
                    ZStack {
                        Circle()
                            .fill(Color.gray.opacity(0.2))
                            .frame(width: 44, height: 44)

                        Image(systemName: "waveform")
                            .font(.title3)
                            .foregroundStyle(.secondary)
                    }

                    VStack(alignment: .leading, spacing: 2) {
                        Text("Select a voice")
                            .font(.body)
                            .fontWeight(.medium)
                            .foregroundStyle(.primary)

                        Text("Tap to browse voices")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }
                }

                Spacer()

                Image(systemName: "chevron.right")
                    .font(.body)
                    .foregroundStyle(.secondary)
            }
            .padding()
            .background(Color.gray.opacity(0.15))
            .clipShape(RoundedRectangle(cornerRadius: 16))
        }
        .buttonStyle(.plain)
        .disabled(isLoading)
    }
}

// MARK: - Text Input Section

struct TextInputSection: View {
    @Binding var text: String
    let isGenerating: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Enter text to synthesize")
                .font(.subheadline)
                .foregroundStyle(.secondary)

            TextField("Type something here...", text: $text, axis: .vertical)
                .textFieldStyle(.plain)
                .lineLimit(5...15)
                .padding(12)
                .background(Color.gray.opacity(0.15))
                .clipShape(RoundedRectangle(cornerRadius: 12))
                .disabled(isGenerating)

            HStack {
                Spacer()
                Text("\(text.count) characters")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }
}

// MARK: - Status View

struct StatusView: View {
    let message: String
    let tokensPerSecond: Double

    var body: some View {
        HStack(spacing: 12) {
            ProgressView()
                .scaleEffect(0.8)

            VStack(alignment: .leading, spacing: 2) {
                Text(message)
                    .font(.subheadline)
                    .foregroundStyle(.primary)

                if tokensPerSecond > 0 {
                    Text(String(format: "%.1f tokens/sec", tokensPerSecond))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            Spacer()
        }
        .padding()
        .background(Color.blue.opacity(0.1))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }
}

// MARK: - Error View

struct ErrorView: View {
    let message: String

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.red)

            Text(message)
                .font(.subheadline)
                .foregroundStyle(.primary)

            Spacer()
        }
        .padding()
        .background(Color.red.opacity(0.1))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }
}

// MARK: - Compact Audio Player

struct CompactAudioPlayer: View {
    let isPlaying: Bool
    let currentTime: TimeInterval
    let duration: TimeInterval
    let onPlayPause: () -> Void
    let onSeek: (TimeInterval) -> Void

    #if os(iOS)
    private let playButtonSize: CGFloat = 32
    private let spacing: CGFloat = 8
    private let padding: CGFloat = 8
    #else
    private let playButtonSize: CGFloat = 44
    private let spacing: CGFloat = 12
    private let padding: CGFloat = 12
    #endif

    var body: some View {
        HStack(spacing: spacing) {
            // Play/Pause button
            Button(action: onPlayPause) {
                Image(systemName: isPlaying ? "pause.circle.fill" : "play.circle.fill")
                    .font(.system(size: playButtonSize))
                    .foregroundStyle(.blue)
            }
            .buttonStyle(.plain)

            VStack(spacing: 2) {
                // Progress bar
                #if os(iOS)
                CompactSlider(
                    value: Binding(
                        get: { currentTime },
                        set: { onSeek($0) }
                    ),
                    range: 0...max(duration, 0.01)
                )
                #else
                Slider(
                    value: Binding(
                        get: { currentTime },
                        set: { onSeek($0) }
                    ),
                    in: 0...max(duration, 0.01)
                )
                .tint(.blue)
                #endif

                // Time labels
                HStack {
                    Text(formatTime(currentTime))
                        .font(.caption2)
                        .foregroundStyle(.secondary)

                    Spacer()

                    Text(formatTime(duration))
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .padding(padding)
        .background(Color.gray.opacity(0.1))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }

    private func formatTime(_ time: TimeInterval) -> String {
        let minutes = Int(time) / 60
        let seconds = Int(time) % 60
        return String(format: "%d:%02d", minutes, seconds)
    }
}

// MARK: - Bottom Action Bar

struct BottomActionBar: View {
    let isGenerating: Bool
    let isModelLoaded: Bool
    let canGenerate: Bool
    let onGenerate: () -> Void

    var body: some View {
        Button(action: onGenerate) {
            HStack(spacing: 8) {
                if isGenerating {
                    ProgressView()
                        .tint(.white)
                } else {
                    Image(systemName: "waveform")
                }

                Text(buttonTitle)
                    .fontWeight(.semibold)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 16)
            .background(canGenerate ? Color.blue : Color.gray)
            .foregroundStyle(.white)
            .clipShape(RoundedRectangle(cornerRadius: 14))
        }
        .buttonStyle(.plain)
        .disabled(!canGenerate)
        .padding()
    }

    private var buttonTitle: String {
        if !isModelLoaded {
            return "Loading Model..."
        } else if isGenerating {
            return "Generating..."
        } else {
            return "Generate Speech"
        }
    }
}

// MARK: - Voice Design Sheet

struct VoiceDesignSheet: View {
    @Environment(\.dismiss) private var dismiss
    @Binding var voiceInstruct: String

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextEditor(text: $voiceInstruct)
                        .frame(minHeight: 60, maxHeight: 120)
                        .scrollContentBackground(.hidden)
                } header: {
                    Text("Voice Description")
                } footer: {
                    Text("Describe the desired voice characteristics, e.g. \"A calm, clear female voice with a professional tone.\"")
                }
            }
            #if os(macOS)
            .formStyle(.grouped)
            .padding(.horizontal)
            #endif
            .navigationTitle("Voice Style")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}

#Preview {
    ContentView()
}
