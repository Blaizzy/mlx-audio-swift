//
//  STSView.swift
//  VoicesApp
//
//  Created by Alpay Calalli on 25.02.26.
//

import SwiftUI
import UniformTypeIdentifiers

struct STSView: View {
    @State private var viewModel = STSViewModel()
    @State private var isFilePickerPresented = false
    @State private var selectedInputURL: URL?
    @State private var isDraggingOver = false

    var body: some View {
        VStack(spacing: 0) {
            headerBar

            ScrollView {
                VStack(spacing: 20) {
                    modelSection
                    inputSection
                    settingsSection

                    if viewModel.isModelLoaded && selectedInputURL != nil {
                        separateButton
                    }

                    if let error = viewModel.errorMessage {
                        errorBanner(error)
                    }

                    if viewModel.audioURL != nil {
                        audioPlayerSection
                    }
                }
                .padding(20)
            }
        }
   }

    // MARK: - Header

    private var headerBar: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text("Audio Separation")
                    .font(.headline)
                Text("Source separation using SAMAudio")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            if viewModel.isGenerating {
                Button("Cancel", role: .destructive) {
                    viewModel.stop()
                }
                .buttonStyle(.bordered)
            }
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
        .background(.bar)
    }

    // MARK: - Model Section

    private var modelSection: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Label("Model", systemImage: "cpu")
                        .font(.subheadline.weight(.semibold))
                    Spacer()
                    modelStatusBadge
                }

                TextField("Model ID", text: $viewModel.modelId)
                    .textFieldStyle(.roundedBorder)
                    .font(.system(.caption, design: .monospaced))
                    .disabled(viewModel.isLoading || viewModel.isGenerating)

                if viewModel.isLoading {
                    HStack(spacing: 8) {
                        ProgressView()
                            .controlSize(.small)
                        Text(viewModel.generationProgress)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                } else {
                    HStack(spacing: 8) {
                        Button(viewModel.isModelLoaded ? "Reload" : "Load Model") {
                            Task { await viewModel.reloadModel() }
                        }
                        .buttonStyle(.bordered)
                        .disabled(viewModel.isLoading)
                    }
                }
            }
        }
    }

    private var modelStatusBadge: some View {
        HStack(spacing: 4) {
            Circle()
                .fill(viewModel.isModelLoaded ? Color.green : Color.orange)
                .frame(width: 7, height: 7)
            Text(viewModel.isModelLoaded ? "Ready" : "Not loaded")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    // MARK: - Input Section

    private var inputSection: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 12) {
                Label("Input Audio", systemImage: "waveform")
                    .font(.subheadline.weight(.semibold))

                // Drop zone / file picker
                ZStack {
                    RoundedRectangle(cornerRadius: 10)
                        .strokeBorder(
                            isDraggingOver ? Color.accentColor : Color.secondary.opacity(0.3),
                            style: StrokeStyle(lineWidth: 2, dash: [6])
                        )
                        .background(
                            RoundedRectangle(cornerRadius: 10)
                                .fill(isDraggingOver
                                    ? Color.accentColor.opacity(0.05)
                                    : Color.secondary.opacity(0.03))
                        )
                        .frame(height: 100)

                    VStack(spacing: 8) {
                        if let url = selectedInputURL {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundStyle(.green)
                                .font(.title2)
                            Text(url.lastPathComponent)
                                .font(.subheadline.weight(.medium))
                            Text("Click to change")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        } else {
                            Image(systemName: "arrow.up.doc")
                                .font(.title2)
                                .foregroundStyle(.secondary)
                            Text("Drop audio file or click to browse")
                                .font(.subheadline)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
                .onTapGesture { isFilePickerPresented = true }
                .onDrop(of: [.audio, .wav, .mp3, .aiff], isTargeted: $isDraggingOver) { providers in
                    handleDrop(providers: providers)
                }
                .fileImporter(
                    isPresented: $isFilePickerPresented,
                    allowedContentTypes: [.audio, .wav, .mp3, .aiff, .mpeg4Audio],
                    allowsMultipleSelection: false
                ) { result in
                    if case .success(let urls) = result, let url = urls.first {
                        // Security-scoped resource access for sandboxed apps
                        if url.startAccessingSecurityScopedResource() {
                            selectedInputURL = url
                        }
                    }
                }
            }
        }
    }

    // MARK: - Settings Section

    private var settingsSection: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 14) {
                Label("Separation Settings", systemImage: "slider.horizontal.3")
                    .font(.subheadline.weight(.semibold))

                // Description
                VStack(alignment: .leading, spacing: 6) {
                    Text("What to isolate")
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    HStack(spacing: 8) {
                        ForEach(["speech", "music", "drums", "bass"], id: \.self) { preset in
                            Button(preset) {
                                viewModel.separationDescription = preset
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.small)
                            .tint(viewModel.separationDescription == preset ? .accentColor : .secondary)
                        }
                    }

                    TextField("Custom description", text: $viewModel.separationDescription)
                        .textFieldStyle(.roundedBorder)
                        .disabled(viewModel.isGenerating)
                }

                Divider()

                // Target vs Residual toggle
                VStack(alignment: .leading, spacing: 6) {
                    Text("Output")
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    Picker("Output", selection: $viewModel.useResidual) {
                        Text("Target (isolated source)").tag(false)
                        Text("Residual (everything else)").tag(true)
                    }
                    .pickerStyle(.segmented)
                    .disabled(viewModel.isGenerating)

                    Text(viewModel.useResidual
                         ? "Returns the background after removing the target"
                         : "Returns only the isolated \"\(viewModel.separationDescription)\"")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    // MARK: - Separate Button

    private var separateButton: some View {
        VStack(spacing: 10) {
            if viewModel.isGenerating {
                VStack(spacing: 8) {
                    ProgressView()
                    Text(viewModel.generationProgress)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity)
                .padding()
                .background(Color.secondary.opacity(0.05), in: RoundedRectangle(cornerRadius: 12))
            } else {
                Button {
                    guard let url = selectedInputURL else { return }
                    viewModel.startSeparation(inputURL: url)
                } label: {
                    Label("Separate Audio", systemImage: "arrow.triangle.branch")
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 4)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .disabled(!viewModel.isModelLoaded)
            }
        }
    }

    // MARK: - Audio Player Section

    private var audioPlayerSection: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 14) {
                HStack {
                    Label("Output", systemImage: "waveform.badge.checkmark")
                        .font(.subheadline.weight(.semibold))
                    Spacer()
                    Button {
                        viewModel.saveAudioFile()
                    } label: {
                        Label("Save", systemImage: "square.and.arrow.down")
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                }

                // Scrubber
                VStack(spacing: 4) {
                    Slider(
                        value: Binding(
                            get: { viewModel.currentTime },
                            set: { viewModel.seek(to: $0) }
                        ),
                        in: 0...max(viewModel.duration, 1)
                    )

                    HStack {
                        Text(formatTime(viewModel.currentTime))
                        Spacer()
                        Text(formatTime(viewModel.duration))
                    }
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                }

                // Playback controls
                HStack(spacing: 20) {
                    Spacer()
                    Button {
                        viewModel.seek(to: 0)
                    } label: {
                        Image(systemName: "backward.end.fill")
                            .font(.title3)
                    }
                    .buttonStyle(.plain)

                    Button {
                        viewModel.togglePlayPause()
                    } label: {
                        Image(systemName: viewModel.isPlaying ? "pause.circle.fill" : "play.circle.fill")
                            .font(.system(size: 44))
                            .foregroundStyle(.primary)
                    }
                    .buttonStyle(.plain)

                    Button {
                        viewModel.stop()
                    } label: {
                        Image(systemName: "stop.fill")
                            .font(.title3)
                    }
                    .buttonStyle(.plain)

                    Spacer()
                }
            }
        }
    }

    // MARK: - Error Banner

    private func errorBanner(_ message: String) -> some View {
        HStack(spacing: 10) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.red)
            Text(message)
                .font(.caption)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(Color.red.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))
    }

    // MARK: - Helpers

    private func handleDrop(providers: [NSItemProvider]) -> Bool {
        guard let provider = providers.first else { return false }
        provider.loadFileRepresentation(forTypeIdentifier: UTType.audio.identifier) { url, _ in
            guard let url else { return }
           
            let dest = FileManager.default.temporaryDirectory
                .appendingPathComponent(url.lastPathComponent)
            try? FileManager.default.copyItem(at: url, to: dest)
            DispatchQueue.main.async {
                selectedInputURL = dest
            }
        }
        return true
    }

    private func formatTime(_ time: TimeInterval) -> String {
        let minutes = Int(time) / 60
        let seconds = Int(time) % 60
        return String(format: "%d:%02d", minutes, seconds)
    }
}

#Preview {
    STSView()
        .frame(width: 480, height: 700)
}
