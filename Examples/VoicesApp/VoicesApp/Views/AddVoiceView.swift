import SwiftUI

struct AddVoiceView: View {
    @Environment(\.dismiss) private var dismiss

    @State private var voiceName = ""
    @State private var voiceDescription = ""
    @State private var selectedLanguage = "English"
    @State private var selectedColor: Color = .blue

    let languages = ["English", "Spanish", "French", "German", "Italian", "Portuguese", "Japanese", "Chinese"]
    let colorOptions: [Color] = [
        .blue, .purple, .pink, .red, .orange, .yellow, .green, .teal, .cyan, .indigo
    ]

    var onSave: ((Voice) -> Void)?

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextField("Voice name", text: $voiceName)
                    TextField("Description (optional)", text: $voiceDescription)
                } header: {
                    Text("Basic Info")
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
                        VoiceAvatar(color: selectedColor.opacity(0.5), size: 80)
                        Spacer()
                    }
                    .padding(.vertical, 20)

                    HStack {
                        Spacer()
                        VStack(spacing: 4) {
                            Text(voiceName.isEmpty ? "Voice Name" : voiceName)
                                .font(.headline)
                            Text(voiceDescription.isEmpty ? selectedLanguage : voiceDescription)
                                .font(.subheadline)
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                    }
                } header: {
                    Text("Preview")
                }
            }
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
                        let newVoice = Voice(
                            name: voiceName,
                            description: voiceDescription,
                            language: selectedLanguage,
                            color: selectedColor.opacity(0.3),
                            isCustom: true
                        )
                        onSave?(newVoice)
                        dismiss()
                    }
                    .disabled(voiceName.isEmpty)
                }
            }
        }
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
