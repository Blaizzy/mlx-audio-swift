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
                    if viewModel.transcriptionText.isEmpty && !viewModel.isGenerating {
                        VStack(spacing: 12) {
                            Spacer(minLength: 80)
                            Image(systemName: "waveform.badge.mic")
                                .font(.system(size: 48))
                                .foregroundStyle(.tertiary)
                            Text("Import an audio file to transcribe")
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

            // Audio file info + player
            if viewModel.selectedAudioURL != nil {
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

                Spacer()

                // Transcribe / Stop button
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
                        ViewThatFits(in: .horizontal) {
                            Text("Transcribe")
                                .font(buttonFont)
                                .fontWeight(.medium)
                                .foregroundStyle(canTranscribe ? .white : .secondary)
                                .frame(height: buttonHeight)
                                .padding(.horizontal, 16)
                                .background(canTranscribe ? Color.blue : Color.gray.opacity(0.2))
                                .clipShape(Capsule())

                            Image(systemName: "waveform.badge.mic")
                                .font(buttonFont)
                                .foregroundStyle(canTranscribe ? .white : .secondary)
                                .frame(width: buttonHeight, height: buttonHeight)
                                .background(canTranscribe ? Color.blue : Color.gray.opacity(0.2))
                                .clipShape(Capsule())
                        }
                    }
                    .buttonStyle(.plain)
                    .disabled(!canTranscribe)
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

#Preview {
    STTView()
}
