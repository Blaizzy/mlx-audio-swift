import SwiftUI
import UniformTypeIdentifiers
import AVFoundation

struct AddVoiceView: View {
    @Environment(\.dismiss) private var dismiss

    @State private var voiceName = ""
    @State private var voiceDescription = ""
    @State private var selectedLanguage = "English"
    @State private var selectedColor: Color = .blue

    // Voice cloning
    @State private var audioFileURL: URL?
    @State private var transcription = ""
    @State private var showFilePicker = false
    @State private var audioDuration: TimeInterval = 0
    @State private var isPlayingPreview = false
    @State private var audioPlayer: AVAudioPlayer?

    let languages = ["English", "Spanish", "French", "German", "Italian", "Portuguese", "Japanese", "Chinese"]
    let colorOptions: [Color] = [
        .blue, .purple, .pink, .red, .orange, .yellow, .green, .teal, .cyan, .indigo
    ]

    var onSave: ((Voice) -> Void)?

    private var isVoiceCloning: Bool {
        audioFileURL != nil
    }

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextField("Voice name", text: $voiceName)
                    TextField("Description (optional)", text: $voiceDescription)
                } header: {
                    Text("Basic Info")
                }

                // Voice Cloning Section
                Section {
                    // Audio file picker
                    Button(action: { showFilePicker = true }) {
                        HStack {
                            Image(systemName: audioFileURL != nil ? "waveform" : "plus.circle")
                                .foregroundStyle(audioFileURL != nil ? .green : .blue)
                            if let url = audioFileURL {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(url.lastPathComponent)
                                        .foregroundStyle(.primary)
                                        .lineLimit(1)
                                    Text(formatDuration(audioDuration))
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                            } else {
                                Text("Select audio file")
                                    .foregroundStyle(.primary)
                            }
                            Spacer()
                            if audioFileURL != nil {
                                Button(action: { playPreview() }) {
                                    Image(systemName: isPlayingPreview ? "stop.circle.fill" : "play.circle.fill")
                                        .font(.title2)
                                        .foregroundStyle(.blue)
                                }
                                .buttonStyle(.plain)

                                Button(action: { clearAudio() }) {
                                    Image(systemName: "xmark.circle.fill")
                                        .font(.title2)
                                        .foregroundStyle(.secondary)
                                }
                                .buttonStyle(.plain)
                            }
                        }
                    }
                    .buttonStyle(.plain)

                    if audioFileURL != nil {
                        TextField("Transcription (what is said in the audio)", text: $transcription, axis: .vertical)
                            .lineLimit(3...6)
                    }
                } header: {
                    Text("Voice Cloning (Optional)")
                } footer: {
                    if audioFileURL != nil && transcription.isEmpty {
                        Text("Enter the exact text spoken in the audio file for best results.")
                            .foregroundStyle(.orange)
                    } else if audioFileURL == nil {
                        Text("Upload a 5-15 second audio sample to clone a voice. The model will attempt to match the voice characteristics.")
                    }
                }

                Section {
                    Picker("Language", selection: $selectedLanguage) {
                        ForEach(languages, id: \.self) { language in
                            Text(language).tag(language)
                        }
                    }
                } header: {
                    Text("Language")
                }

                Section {
                    LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 5), spacing: 12) {
                        ForEach(colorOptions, id: \.self) { color in
                            ColorButton(
                                color: color,
                                isSelected: selectedColor == color
                            ) {
                                selectedColor = color
                            }
                        }
                    }
                    .padding(.vertical, 8)
                } header: {
                    Text("Color")
                }

                Section {
                    HStack {
                        Spacer()
                        ZStack {
                            VoiceAvatar(color: selectedColor.opacity(0.5), size: 80)
                            if isVoiceCloning {
                                Image(systemName: "waveform")
                                    .font(.title2)
                                    .foregroundStyle(.white)
                            }
                        }
                        Spacer()
                    }
                    .padding(.vertical, 20)

                    HStack {
                        Spacer()
                        VStack(spacing: 4) {
                            Text(voiceName.isEmpty ? "Voice Name" : voiceName)
                                .font(.headline)
                            Text(voiceDescription.isEmpty ? (isVoiceCloning ? "Cloned Voice" : selectedLanguage) : voiceDescription)
                                .font(.subheadline)
                                .foregroundStyle(.secondary)
                            if isVoiceCloning {
                                Label("Voice Clone", systemImage: "waveform")
                                    .font(.caption)
                                    .foregroundStyle(.blue)
                            }
                        }
                        Spacer()
                    }
                } header: {
                    Text("Preview")
                }
            }
            #if os(macOS)
            .formStyle(.grouped)
            .padding(.horizontal)
            #endif
            .navigationTitle("Add Voice")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") {
                        dismiss()
                    }
                }

                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") {
                        saveVoice()
                    }
                    .disabled(voiceName.isEmpty || (audioFileURL != nil && transcription.isEmpty))
                }
            }
            .fileImporter(
                isPresented: $showFilePicker,
                allowedContentTypes: [.audio, .wav, .mp3, .aiff],
                allowsMultipleSelection: false
            ) { result in
                handleFileSelection(result)
            }
            .onDisappear {
                audioPlayer?.stop()
            }
        }
        #if os(macOS)
        .frame(minWidth: 400, minHeight: 500)
        #endif
    }

    private func saveVoice() {
        // Copy audio file to app's documents directory if provided
        var savedAudioURL: URL? = nil
        if let sourceURL = audioFileURL {
            savedAudioURL = copyAudioToDocuments(sourceURL)
        }

        let newVoice = Voice(
            name: voiceName,
            description: voiceDescription.isEmpty ? (isVoiceCloning ? "Cloned Voice" : "") : voiceDescription,
            language: selectedLanguage,
            color: selectedColor.opacity(0.3),
            isCustom: true,
            audioFileURL: savedAudioURL,
            transcription: transcription.isEmpty ? nil : transcription
        )
        onSave?(newVoice)
        dismiss()
    }

    private func copyAudioToDocuments(_ sourceURL: URL) -> URL? {
        let fileManager = FileManager.default
        guard let documentsDir = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first else {
            return nil
        }

        let voicesDir = documentsDir.appendingPathComponent("ClonedVoices", isDirectory: true)
        try? fileManager.createDirectory(at: voicesDir, withIntermediateDirectories: true)

        let destURL = voicesDir.appendingPathComponent("\(UUID().uuidString)_\(sourceURL.lastPathComponent)")

        do {
            // Start accessing security-scoped resource
            guard sourceURL.startAccessingSecurityScopedResource() else {
                return nil
            }
            defer { sourceURL.stopAccessingSecurityScopedResource() }

            try fileManager.copyItem(at: sourceURL, to: destURL)
            return destURL
        } catch {
            print("Failed to copy audio file: \(error)")
            return nil
        }
    }

    private func handleFileSelection(_ result: Result<[URL], Error>) {
        switch result {
        case .success(let urls):
            guard let url = urls.first else { return }

            // Start accessing security-scoped resource
            guard url.startAccessingSecurityScopedResource() else { return }
            defer { url.stopAccessingSecurityScopedResource() }

            audioFileURL = url
            loadAudioDuration(from: url)

        case .failure(let error):
            print("File selection error: \(error)")
        }
    }

    private func loadAudioDuration(from url: URL) {
        guard url.startAccessingSecurityScopedResource() else { return }
        defer { url.stopAccessingSecurityScopedResource() }

        do {
            let player = try AVAudioPlayer(contentsOf: url)
            audioDuration = player.duration
        } catch {
            print("Failed to load audio: \(error)")
            audioDuration = 0
        }
    }

    private func playPreview() {
        if isPlayingPreview {
            audioPlayer?.stop()
            isPlayingPreview = false
            return
        }

        guard let url = audioFileURL else { return }

        guard url.startAccessingSecurityScopedResource() else { return }

        do {
            audioPlayer = try AVAudioPlayer(contentsOf: url)
            audioPlayer?.play()
            isPlayingPreview = true

            // Stop after playback
            DispatchQueue.main.asyncAfter(deadline: .now() + audioDuration + 0.1) {
                self.isPlayingPreview = false
                url.stopAccessingSecurityScopedResource()
            }
        } catch {
            print("Failed to play audio: \(error)")
            url.stopAccessingSecurityScopedResource()
        }
    }

    private func clearAudio() {
        audioPlayer?.stop()
        audioFileURL = nil
        transcription = ""
        audioDuration = 0
        isPlayingPreview = false
    }

    private func formatDuration(_ duration: TimeInterval) -> String {
        let minutes = Int(duration) / 60
        let seconds = Int(duration) % 60
        return String(format: "%d:%02d", minutes, seconds)
    }
}

struct ColorButton: View {
    let color: Color
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            ZStack {
                Circle()
                    .fill(color.gradient)
                    .frame(width: 44, height: 44)

                if isSelected {
                    Circle()
                        .strokeBorder(Color.primary, lineWidth: 3)
                        .frame(width: 44, height: 44)

                    Image(systemName: "checkmark")
                        .font(.caption)
                        .fontWeight(.bold)
                        .foregroundStyle(.white)
                }
            }
        }
        .buttonStyle(.plain)
    }
}

#Preview {
    AddVoiceView()
}
