import SwiftUI

struct ClonedVoicesView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var searchText = ""
    @State private var showAddVoice = false

    @Binding var clonedVoices: [Voice]
    var selectedVoice: Voice?
    var onVoiceSelected: ((Voice) -> Void)?

    var filteredVoices: [Voice] {
        if searchText.isEmpty {
            return clonedVoices
        }
        return clonedVoices.filter {
            $0.name.localizedCaseInsensitiveContains(searchText) ||
            $0.description.localizedCaseInsensitiveContains(searchText)
        }
    }

    var body: some View {
        NavigationStack {
            List {
                // Add new voice button
                Section {
                    AddVoiceButton {
                        showAddVoice = true
                    }
                }
                .listRowBackground(Color.clear)

                // Cloned voices section
                if !filteredVoices.isEmpty {
                    Section {
                        ForEach(filteredVoices) { voice in
                            HStack {
                                VoiceRow(
                                    voice: voice,
                                    onTap: {
                                        onVoiceSelected?(voice)
                                        dismiss()
                                    }
                                )

                                if selectedVoice?.id == voice.id {
                                    Image(systemName: "checkmark")
                                        .foregroundStyle(.blue)
                                        .fontWeight(.semibold)
                                }
                            }
                            .swipeActions(edge: .trailing, allowsFullSwipe: true) {
                                Button(role: .destructive) {
                                    clonedVoices.removeAll { $0.id == voice.id }
                                } label: {
                                    Label("Delete", systemImage: "trash")
                                }
                            }
                        }
                    } header: {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Your Cloned Voices")
                            Text("Voices created from audio samples")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .textCase(nil)
                        }
                    }
                }
            }
            #if os(iOS)
            .listStyle(.insetGrouped)
            #endif
            .searchable(text: $searchText, prompt: "Search cloned voices")
            .navigationTitle("Voice Cloning")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
            .sheet(isPresented: $showAddVoice) {
                AddVoiceView { newVoice in
                    clonedVoices.append(newVoice)
                }
            }
            .overlay {
                if clonedVoices.isEmpty && searchText.isEmpty {
                    ContentUnavailableView {
                        Label("No Cloned Voices", systemImage: "mic.badge.plus")
                    } description: {
                        Text("Tap \"Add a new voice\" to create a cloned voice from an audio sample.")
                    }
                }
            }
        }
        #if os(macOS)
        .frame(minWidth: 400, minHeight: 400)
        #endif
    }
}

#Preview {
    ClonedVoicesView(
        clonedVoices: .constant([]),
        onVoiceSelected: { _ in }
    )
}
