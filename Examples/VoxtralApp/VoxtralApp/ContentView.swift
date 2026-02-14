import SwiftUI

struct ContentView: View {
    @State private var viewModel = TranscriptionViewModel()
    @State private var showSettings = false

    var body: some View {
        VStack(spacing: 0) {
            // Status bar
            statusBar
                .padding(.horizontal)
                .padding(.vertical, 10)

            Divider()

            // Transcription area
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 12) {
                        // Streaming current text
                        if !viewModel.currentText.isEmpty {
                            currentTranscriptionView
                                .id("current")
                        }

                        // History
                        ForEach(viewModel.transcriptions) { transcription in
                            transcriptionRow(transcription)
                        }
                    }
                    .padding()
                }
                .onChange(of: viewModel.currentText) {
                    withAnimation {
                        proxy.scrollTo("current", anchor: .top)
                    }
                }
            }

            Divider()

            // Bottom toolbar
            toolbar
                .padding(.horizontal)
                .padding(.vertical, 10)
        }
        .frame(minWidth: 400, minHeight: 300)
        .task {
            await viewModel.loadModel()
        }
        .popover(isPresented: $showSettings, arrowEdge: .bottom) {
            settingsPopover
        }
    }

    // MARK: - Status Bar

    private var statusBar: some View {
        HStack {
            Circle()
                .fill(statusColor)
                .frame(width: 8, height: 8)

            Text(statusText)
                .font(.subheadline.weight(.medium))
                .foregroundStyle(.secondary)

            Spacer()

            if viewModel.state == .loading {
                ProgressView()
                    .controlSize(.small)
            }
        }
    }

    private var statusText: String {
        switch viewModel.state {
        case .loading: "Loading model…"
        case .ready: "Ready"
        case .listening: "Listening…"
        case .transcribing: "Transcribing…"
        }
    }

    private var statusColor: Color {
        switch viewModel.state {
        case .loading: .orange
        case .ready: .gray
        case .listening: .green
        case .transcribing: .blue
        }
    }

    // MARK: - Current Transcription

    private var currentTranscriptionView: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 4) {
                Text(viewModel.currentText)
                    .font(.body)

                // Blinking cursor
                Rectangle()
                    .fill(.primary)
                    .frame(width: 2, height: 16)
                    .opacity(cursorOpacity)
                    .animation(
                        .easeInOut(duration: 0.6).repeatForever(autoreverses: true),
                        value: cursorOpacity
                    )
            }

            if viewModel.tokensPerSecond > 0 {
                Text("\(viewModel.tokensPerSecond, specifier: "%.1f") tok/s")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.blue.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))
    }

    @State private var cursorOpacity: Double = 1.0

    // MARK: - Transcription Row

    private func transcriptionRow(_ transcription: Transcription) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(transcription.text)
                .font(.body)
                .textSelection(.enabled)

            HStack(spacing: 8) {
                Text(transcription.timestamp, style: .time)
                Text("·")
                Text("\(transcription.tokensPerSecond, specifier: "%.1f") tok/s")
                Text("·")
                Text("\(transcription.duration, specifier: "%.1f")s")
            }
            .font(.caption2)
            .foregroundStyle(.tertiary)
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 8))
    }

    // MARK: - Toolbar

    private var toolbar: some View {
        HStack {
            Button {
                showSettings.toggle()
            } label: {
                Image(systemName: "slider.horizontal.3")
            }

            Spacer()

            if viewModel.peakMemory > 0 {
                Text("\(viewModel.peakMemory / 1024 / 1024, specifier: "%.0f") MB")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
                    .monospacedDigit()
            }

            Spacer()

            Button {
                if viewModel.state == .listening || viewModel.state == .transcribing {
                    viewModel.stopListening()
                } else {
                    viewModel.startListening()
                }
            } label: {
                Label(
                    isActive ? "Stop" : "Start",
                    systemImage: isActive ? "stop.circle.fill" : "mic.circle.fill"
                )
                .font(.subheadline.weight(.medium))
            }
            .disabled(viewModel.state == .loading)
            .tint(isActive ? .red : .blue)

            if let error = viewModel.errorMessage {
                Text(error)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .lineLimit(1)
            }
        }
    }

    private var isActive: Bool {
        viewModel.state == .listening || viewModel.state == .transcribing
    }

    // MARK: - Settings Popover

    private var settingsPopover: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("VAD Settings")
                .font(.headline)

            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text("Energy Threshold")
                        .font(.subheadline)
                    Spacer()
                    Text("\(viewModel.energyThreshold, specifier: "%.3f")")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                Slider(value: $viewModel.energyThreshold, in: 0.001...0.1)
            }

            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text("Hang Time")
                        .font(.subheadline)
                    Spacer()
                    Text("\(viewModel.hangTime, specifier: "%.1f")s")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                Slider(value: $viewModel.hangTime, in: 0.5...5.0, step: 0.1)
            }
        }
        .padding()
        .frame(width: 280)
    }
}
