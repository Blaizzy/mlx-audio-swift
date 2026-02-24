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
    @State private var selectedVoice: Voice? = Voice.samples.first(where: { $0.name == "Lily" }) ?? Voice.samples.first
    @State private var showVoices = false
    @State private var showSettings = false
    @State private var recentlyUsed: [Voice] = Voice.samples
    @State private var customVoices: [Voice] = Voice.customVoices

    private let buttonHeight: CGFloat = 44
    private let buttonFont: Font = .title3
    private let inputFont: Font = .title2

    var body: some View {
        VStack(spacing: 0) {
            // Main text input area - fills available space
            TextEditor(text: $textInput)
                .font(inputFont)
                .scrollContentBackground(.hidden)
                .background(.clear)
                .disabled(viewModel.isGenerating)
                .overlay(alignment: .topLeading) {
                    if textInput.isEmpty {
                        Text("Start typing here...")
                            .font(inputFont)
                            .foregroundStyle(.tertiary)
                            .padding(.horizontal, 4)
                            .allowsHitTesting(false)
                    }
                }
                .padding(.horizontal, 16)
                .padding(.top, 16)

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
                    HStack(spacing: 8) {
                        CompactAudioPlayer(
                            isPlaying: viewModel.isPlaying,
                            currentTime: viewModel.currentTime,
                            duration: viewModel.duration,
                            onPlayPause: { viewModel.togglePlayPause() },
                            onSeek: { viewModel.seek(to: $0) }
                        )

                        // Download button
                        Button(action: { viewModel.saveAudioFile() }) {
                            Image(systemName: "arrow.down.circle.fill")
                                .font(.system(size: 32))
                                .foregroundStyle(.blue)
                        }
                        .buttonStyle(.plain)
                        .help("Save audio file")
                        .padding(.trailing, 12)
                    }
                    .background(Color.gray.opacity(0.1))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                    .padding(.horizontal)
                }
            }
            .padding(.bottom, 4)

            // Bottom bar
            HStack(spacing: 8) {
                // Voice selector chip
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
                    .foregroundStyle(.primary)
                    .frame(height: buttonHeight)
                    .padding(.horizontal, 12)
                    .background(Color.gray.opacity(0.2))
                    .clipShape(Capsule())
                }
                .buttonStyle(.plain)

                // Settings button
                Button(action: { showSettings = true }) {
                    Image(systemName: "slider.horizontal.3")
                        .font(buttonFont)
                        .foregroundStyle(.primary)
                        .frame(width: buttonHeight, height: buttonHeight)
                        .background(Color.gray.opacity(0.2))
                        .clipShape(Capsule())
                }
                .buttonStyle(.plain)

                Spacer()

                // Generate / Stop button
                if viewModel.isGenerating {
                    Button(action: {
                        viewModel.stop()
                    }) {
                        Text("Stop")
                            .font(buttonFont)
                            .fontWeight(.medium)
                            .foregroundStyle(.white)
                            .frame(height: buttonHeight)
                            .padding(.horizontal, 16)
                            .background(Color.red)
                            .clipShape(Capsule())
                    }
                    .buttonStyle(.plain)
                } else {
                    Button(action: {
                        viewModel.startSynthesis(text: textInput, voice: selectedVoice)
                        if let voice = selectedVoice {
                            recentlyUsed.removeAll { $0.id == voice.id }
                            recentlyUsed.insert(voice, at: 0)
                        }
                    }) {
                        Text("Generate")
                            .font(buttonFont)
                            .fontWeight(.medium)
                            .foregroundStyle(canGenerate ? .white : .secondary)
                            .frame(height: buttonHeight)
                            .padding(.horizontal, 16)
                            .background(canGenerate ? Color.blue : Color.gray.opacity(0.2))
                            .clipShape(Capsule())
                    }
                    .buttonStyle(.plain)
                    .disabled(!canGenerate)
                    .keyboardShortcut(.return, modifiers: [.command])
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
            .presentationDetents([.large])
            .presentationDragIndicator(.visible)
        }
        .sheet(isPresented: $showSettings) {
            SettingsView(viewModel: viewModel)
                .presentationDetents([.large])
                .presentationDragIndicator(.visible)
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

    private let playButtonSize: CGFloat = 44
    private let spacing: CGFloat = 12
    private let padding: CGFloat = 12

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
                        .font(.body)
                        .foregroundStyle(.secondary)

                    Spacer()

                    Text(formatTime(duration))
                        .font(.body)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .padding(padding)
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

#Preview("CompactAudioPlayer") {
    CompactAudioPlayer(isPlaying: true, currentTime: 1.0, duration: 5.0, onPlayPause: {}, onSeek: { _ in })
}

#Preview("ContentView") {
    ContentView()
}
