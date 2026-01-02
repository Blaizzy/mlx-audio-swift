import SwiftUI

struct ContentView: View {
    @State private var viewModel = TTSViewModel()
    @State private var textInput = ""
    @State private var selectedVoice: Voice?
    @State private var showVoices = false
    @State private var showSettings = false
    @State private var recentlyUsed: [Voice] = Voice.samples
    @State private var customVoices: [Voice] = Voice.customVoices

    var body: some View {
        VStack(spacing: 0) {
            // Main text input area
            VStack(alignment: .leading) {
                TextField("Start typing here...", text: $textInput, axis: .vertical)
                    .font(.title)
                    .textFieldStyle(.plain)
                    .disabled(viewModel.isGenerating)
                    .padding(.top, 20)

                Spacer()

                // Status/Progress
                if !viewModel.generationProgress.isEmpty {
                    HStack(spacing: 8) {
                        ProgressView()
                            .scaleEffect(0.8)
                        Text(viewModel.generationProgress)
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }
                    .padding(.bottom, 8)
                }

                // Error message
                if let error = viewModel.errorMessage {
                    Text(error)
                        .font(.subheadline)
                        .foregroundStyle(.red)
                        .padding(.bottom, 8)
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
                    .padding(.bottom, 16)
                }
            }
            .padding(.horizontal)

            // Bottom bar
            HStack(spacing: 12) {
                // Voice selector chip
                Button(action: { showVoices = true }) {
                    HStack(spacing: 8) {
                        if let voice = selectedVoice {
                            VoiceAvatar(color: voice.color, size: 24)
                            Text("\(voice.name)")
                                .lineLimit(1)
                        } else {
                            Image(systemName: "waveform")
                            Text("Voice")
                        }
                    }
                    .font(.subheadline)
                    .foregroundStyle(.primary)
                    .frame(height: 44)
                    .padding(.horizontal, 16)
                    .background(Color.gray.opacity(0.2))
                    .clipShape(Capsule())
                }
                .buttonStyle(.plain)

                // Settings button
                Button(action: { showSettings = true }) {
                    Image(systemName: "slider.horizontal.3")
                        .font(.body)
                        .foregroundStyle(.primary)
                        .frame(width: 44, height: 44)
                        .background(Color.gray.opacity(0.2))
                        .clipShape(Capsule())
                }
                .buttonStyle(.plain)

                // Generate button
                Button(action: {
                    Task {
                        await viewModel.synthesize(text: textInput, voice: selectedVoice)
                        if let voice = selectedVoice {
                            recentlyUsed.removeAll { $0.id == voice.id }
                            recentlyUsed.insert(voice, at: 0)
                        }
                    }
                }) {
                    Text("Generate")
                        .font(.subheadline)
                        .fontWeight(.medium)
                        .foregroundStyle(canGenerate ? .white : .secondary)
                        .frame(height: 44)
                        .padding(.horizontal, 20)
                        .background(canGenerate ? Color.blue : Color.gray.opacity(0.2))
                        .clipShape(Capsule())
                }
                .buttonStyle(.plain)
                .disabled(!canGenerate)
            }
            .padding()
        }
        .sheet(isPresented: $showVoices) {
            VoicesView(
                recentlyUsed: $recentlyUsed,
                customVoices: $customVoices,
                collections: VoiceCollection.samples
            ) { voice in
                selectedVoice = voice
                showVoices = false
            }
        }
        .sheet(isPresented: $showSettings) {
            SettingsView(viewModel: viewModel)
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

    var body: some View {
        HStack(spacing: 12) {
            // Play/Pause button
            Button(action: onPlayPause) {
                Image(systemName: isPlaying ? "pause.circle.fill" : "play.circle.fill")
                    .font(.system(size: 44))
                    .foregroundStyle(.blue)
            }
            .buttonStyle(.plain)

            VStack(spacing: 4) {
                // Progress bar
                Slider(
                    value: Binding(
                        get: { currentTime },
                        set: { onSeek($0) }
                    ),
                    in: 0...max(duration, 0.01)
                )
                .tint(.blue)

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
        .padding(12)
        .background(Color.gray.opacity(0.1))
        .clipShape(RoundedRectangle(cornerRadius: 12))
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

#Preview {
    ContentView()
}
