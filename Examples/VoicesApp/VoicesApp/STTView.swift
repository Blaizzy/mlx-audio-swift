import SwiftUI
import UniformTypeIdentifiers

struct STTView: View {
    @Environment(\.scenePhase) private var scenePhase
    @State private var viewModel = STTViewModel()
    @State private var showFileImporter = false
    @State private var showSettings = false

    #if os(iOS)
    private let buttonHeight: CGFloat = 44
    private let buttonFont: Font = .callout
    private let bodyFont: Font = .body
    #else
    private let buttonHeight: CGFloat = 44
    private let buttonFont: Font = .subheadline
    private let bodyFont: Font = .title3
    #endif

    var body: some View {
        VStack(spacing: 0) {
            // Transcription result area
            ScrollViewReader { proxy in
                ScrollView {
                    if viewModel.transcriptionText.isEmpty && !viewModel.isGenerating && !viewModel.isRecording {
                        VStack(spacing: 12) {
                            Spacer(minLength: 80)
                            Image(systemName: "waveform.badge.mic")
                                .font(.system(size: 48))
                                .foregroundStyle(.tertiary)
                            Text("Import or record audio to transcribe")
                                .font(bodyFont)
                                .foregroundStyle(.tertiary)
                            Spacer()
                        }
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                    } else {
                        Text(viewModel.transcriptionText)
                            .font(bodyFont)
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding()

                        Color.clear
                            .frame(height: 1)
                            .id("bottom")
                    }
                }
                .onChange(of: viewModel.transcriptionText) {
                    withAnimation(.easeOut(duration: 0.15)) {
                        proxy.scrollTo("bottom", anchor: .bottom)
                    }
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            // Recording indicator
            if viewModel.isRecording {
                RecordingIndicator(
                    duration: viewModel.recordingDuration,
                    audioLevel: viewModel.audioLevel
                )
                .padding(.horizontal)
                .padding(.bottom, 4)
            }

            // Audio file info + player (hidden while recording)
            if viewModel.selectedAudioURL != nil && !viewModel.isRecording {
                VStack(spacing: 4) {
                    // File name
                    if let fileName = viewModel.audioFileName {
                        HStack(spacing: 6) {
                            Image(systemName: "doc.fill")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Text(fileName)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                            Spacer()
                        }
                        .padding(.horizontal)
                    }

                    // Audio player
                    CompactAudioPlayer(
                        isPlaying: viewModel.isPlaying,
                        currentTime: viewModel.currentTime,
                        duration: viewModel.duration,
                        onPlayPause: { viewModel.togglePlayPause() },
                        onSeek: { viewModel.seek(to: $0) }
                    )
                    .padding(.horizontal)
                }
                .padding(.bottom, 4)
            }

            // Status/Progress
            VStack(spacing: 4) {
                if !viewModel.generationProgress.isEmpty && !viewModel.isRecording {
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

                if let error = viewModel.errorMessage {
                    Text(error)
                        .font(.caption2)
                        .foregroundStyle(.red)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal)
                }
            }
            .padding(.bottom, 4)

            // Bottom bar
            HStack(spacing: 8) {
                if !viewModel.isRecording {
                    // File import button
                    Button(action: { showFileImporter = true }) {
                        ViewThatFits(in: .horizontal) {
                            HStack(spacing: 6) {
                                Image(systemName: "doc.badge.plus")
                                Text("Import")
                            }
                            .font(buttonFont)
                            .foregroundStyle(.primary)
                            .frame(height: buttonHeight)
                            .padding(.horizontal, 12)
                            .background(Color.gray.opacity(0.2))
                            .clipShape(Capsule())

                            Image(systemName: "doc.badge.plus")
                                .font(buttonFont)
                                .foregroundStyle(.primary)
                                .frame(width: buttonHeight, height: buttonHeight)
                                .background(Color.gray.opacity(0.2))
                                .clipShape(Capsule())
                        }
                    }
                    .buttonStyle(.plain)
                    .disabled(viewModel.isGenerating)
                }

                // Record button
                Button(action: {
                    if viewModel.isRecording {
                        viewModel.stopRecording()
                    } else {
                        viewModel.startRecording()
                    }
                }) {
                    ViewThatFits(in: .horizontal) {
                        HStack(spacing: 6) {
                            Image(systemName: viewModel.isRecording ? "stop.circle.fill" : "mic.fill")
                            Text(viewModel.isRecording ? "Stop" : "Record")
                        }
                        .font(buttonFont)
                        .fontWeight(viewModel.isRecording ? .medium : .regular)
                        .foregroundStyle(viewModel.isRecording ? .white : .red)
                        .frame(height: buttonHeight)
                        .padding(.horizontal, 12)
                        .background(viewModel.isRecording ? Color.red : Color.gray.opacity(0.2))
                        .clipShape(Capsule())

                        Image(systemName: viewModel.isRecording ? "stop.circle.fill" : "mic.fill")
                            .font(buttonFont)
                            .foregroundStyle(viewModel.isRecording ? .white : .primary)
                            .frame(width: buttonHeight, height: buttonHeight)
                            .background(viewModel.isRecording ? Color.red : Color.gray.opacity(0.2))
                            .clipShape(Capsule())
                    }
                }
                .buttonStyle(.plain)
                .disabled(!viewModel.isModelLoaded)

                if !viewModel.isRecording {
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

                    // Copy button (when transcription exists)
                    if !viewModel.transcriptionText.isEmpty {
                        Button(action: { viewModel.copyTranscription() }) {
                            Image(systemName: "doc.on.doc")
                                .font(buttonFont)
                                .foregroundStyle(.primary)
                                .frame(width: buttonHeight, height: buttonHeight)
                                .background(Color.gray.opacity(0.2))
                                .clipShape(Capsule())
                        }
                        .buttonStyle(.plain)
                    }

                    // Stats after generation
                    if !viewModel.isGenerating && viewModel.tokensPerSecond > 0 {
                        HStack(spacing: 8) {
                            Label(
                                String(format: "%.1f tok/s", viewModel.tokensPerSecond),
                                systemImage: "speedometer"
                            )
                            Label(
                                String(format: "%.1f GB", viewModel.peakMemory),
                                systemImage: "memorychip"
                            )
                        }
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .monospacedDigit()
                    }
                } else {
                    // Cancel button while recording
                    Button(action: { viewModel.cancelRecording() }) {
                        Text("Cancel")
                            .font(buttonFont)
                            .foregroundStyle(.secondary)
                            .frame(height: buttonHeight)
                            .padding(.horizontal, 12)
                            .background(Color.gray.opacity(0.2))
                            .clipShape(Capsule())
                    }
                    .buttonStyle(.plain)

                    // Stats during recording
                    if viewModel.tokensPerSecond > 0 {
                        HStack(spacing: 8) {
                            Label(
                                String(format: "%.1f tok/s", viewModel.tokensPerSecond),
                                systemImage: "speedometer"
                            )
                            Label(
                                String(format: "%.1f GB", viewModel.peakMemory),
                                systemImage: "memorychip"
                            )
                        }
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .monospacedDigit()
                    }
                }

                Spacer()

                // Transcribe / Stop button (for file transcription, not shown during recording)
                if !viewModel.isRecording {
                    if viewModel.isGenerating {
                        Button(action: {
                            viewModel.stop()
                        }) {
                            ViewThatFits(in: .horizontal) {
                                Text("Stop")
                                    .font(buttonFont)
                                    .fontWeight(.medium)
                                    .foregroundStyle(.white)
                                    .frame(height: buttonHeight)
                                    .padding(.horizontal, 16)
                                    .background(Color.red)
                                    .clipShape(Capsule())

                                Image(systemName: "stop.fill")
                                    .font(buttonFont)
                                    .foregroundStyle(.white)
                                    .frame(width: buttonHeight, height: buttonHeight)
                                    .background(Color.red)
                                    .clipShape(Capsule())
                            }
                        }
                        .buttonStyle(.plain)
                    } else {
                        Button(action: {
                            viewModel.startTranscription()
                        }) {
                            Image(systemName: "waveform.badge.mic")
                                .font(buttonFont)
                                .foregroundStyle(canTranscribe ? .white : .secondary)
                                .frame(width: buttonHeight, height: buttonHeight)
                                    .background(canTranscribe ? Color.blue : Color.gray.opacity(0.2))
                                    .clipShape(Capsule())
                        }
                        .buttonStyle(.plain)
                        .disabled(!canTranscribe)
                    }
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
        .fileImporter(
            isPresented: $showFileImporter,
            allowedContentTypes: [.audio, .wav, .mp3, .mpeg4Audio, .aiff],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let urls):
                if let url = urls.first {
                    if url.startAccessingSecurityScopedResource() {
                        viewModel.selectAudioFile(url)
                    }
                }
            case .failure(let error):
                viewModel.errorMessage = "File import failed: \(error.localizedDescription)"
            }
        }
        .sheet(isPresented: $showSettings) {
            STTSettingsView(viewModel: viewModel)
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

    private var canTranscribe: Bool {
        viewModel.selectedAudioURL != nil && !viewModel.isGenerating && viewModel.isModelLoaded
    }
}

// MARK: - Recording Indicator

private struct RecordingIndicator: View {
    let duration: TimeInterval
    let audioLevel: Float

    @State private var isPulsing = false

    var body: some View {
        HStack(spacing: 10) {
            // Pulsing red dot
            Circle()
                .fill(Color.red)
                .frame(width: 10, height: 10)
                .scaleEffect(isPulsing ? 1.3 : 1.0)
                .opacity(isPulsing ? 0.7 : 1.0)
                .animation(.easeInOut(duration: 0.8).repeatForever(autoreverses: true), value: isPulsing)
                .onAppear { isPulsing = true }

            // Duration
            Text(formatDuration(duration))
                .font(.caption)
                .monospacedDigit()
                .foregroundStyle(.secondary)

            Spacer()
        }
        .padding(.vertical, 6)
    }

    private func formatDuration(_ duration: TimeInterval) -> String {
        let minutes = Int(duration) / 60
        let seconds = Int(duration) % 60
        return String(format: "%d:%02d", minutes, seconds)
    }
}

#Preview {
    STTView()
}
