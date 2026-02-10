import SwiftUI

struct VoicesView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var searchText = ""
    @State private var showAddVoice = false
    @State private var selectedVoice: Voice?

    @Binding var recentlyUsed: [Voice]
    @Binding var customVoices: [Voice]
    var onVoiceSelected: ((Voice) -> Void)?

    var filteredRecentlyUsed: [Voice] {
        if searchText.isEmpty {
            return recentlyUsed
        }
        return recentlyUsed.filter {
            $0.name.localizedCaseInsensitiveContains(searchText) ||
            $0.description.localizedCaseInsensitiveContains(searchText)
        }
    }

    var filteredCustomVoices: [Voice] {
        if searchText.isEmpty {
            return customVoices
        }
        return customVoices.filter {
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

                // Recently used section
                if !filteredRecentlyUsed.isEmpty {
                    Section {
                        ScrollView(.horizontal, showsIndicators: false) {
                            HStack(spacing: 6) {
                                ForEach(filteredRecentlyUsed) { voice in
                                    VoiceChip(voice: voice) {
                                        selectedVoice = voice
                                        onVoiceSelected?(voice)
                                    }
                                }
                            }
                        }
                    } header: {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Recently used")
                            Text("Voices you've used recently")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .textCase(nil)
                        }
                    }
                }

                // Your voices section
                if !filteredCustomVoices.isEmpty {
                    Section {
                        ForEach(filteredCustomVoices) { voice in
                            VoiceRow(
                                voice: voice,
                                onTap: {
                                    selectedVoice = voice
                                    onVoiceSelected?(voice)
                                }
                            )
                            .swipeActions(edge: .trailing, allowsFullSwipe: true) {
                                Button(role: .destructive) {
                                    customVoices.removeAll { $0.id == voice.id }
                                } label: {
                                    Label("Delete", systemImage: "trash")
                                }
                            }
                        }
                    } header: {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Your Voices")
                            Text("Voices you've created")
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
            .searchable(text: $searchText, prompt: "Search voices")
            .navigationTitle("Voices")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button(action: { dismiss() }) {
                        Image(systemName: "xmark.circle.fill")
                            .font(.body)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .sheet(isPresented: $showAddVoice) {
                AddVoiceView { newVoice in
                    customVoices.append(newVoice)
                }
            }
        }
        #if os(macOS)
        .frame(minWidth: 400, minHeight: 400)
        #endif
    }
}

// MARK: - Add Voice Button

struct AddVoiceButton: View {
    var action: () -> Void

    #if os(iOS)
    private let circleSize: CGFloat = 36
    private let iconFont: Font = .callout
    private let titleFont: Font = .footnote
    private let subtitleFont: Font = .caption
    #else
    private let circleSize: CGFloat = 50
    private let iconFont: Font = .title2
    private let titleFont: Font = .body
    private let subtitleFont: Font = .subheadline
    #endif

    var body: some View {
        Button(action: action) {
            HStack(spacing: 10) {
                ZStack {
                    Circle()
                        .fill(Color.black)
                        .frame(width: circleSize, height: circleSize)

                    Image(systemName: "plus")
                        .font(iconFont)
                        .fontWeight(.semibold)
                        .foregroundStyle(.white)
                }

                VStack(alignment: .leading, spacing: 1) {
                    Text("Add a new voice")
                        .font(titleFont)
                        .fontWeight(.medium)
                        .foregroundStyle(.primary)

                    Text("Create or clone a voice")
                        .font(subtitleFont)
                        .foregroundStyle(.secondary)
                }

                Spacer()
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}

#Preview {
    VoicesView(
        recentlyUsed: .constant(Voice.samples),
        customVoices: .constant(Voice.customVoices)
    )
}
