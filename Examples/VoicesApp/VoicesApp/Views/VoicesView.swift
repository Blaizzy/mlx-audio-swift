import SwiftUI

struct VoicesView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var searchText = ""
    @State private var showAddVoice = false
    @State private var selectedVoice: Voice?

    @Binding var recentlyUsed: [Voice]
    @Binding var customVoices: [Voice]
    let collections: [VoiceCollection]
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
            ScrollView {
                VStack(alignment: .leading, spacing: 24) {
                    // Search bar
                    SearchBar(text: $searchText)
                        .padding(.horizontal)

                    // Add new voice button
                    AddVoiceButton {
                        showAddVoice = true
                    }
                    .padding(.horizontal)

                    // Recently used section
                    if !filteredRecentlyUsed.isEmpty {
                        RecentlyUsedSection(
                            voices: filteredRecentlyUsed,
                            onVoiceTap: { voice in
                                selectedVoice = voice
                                onVoiceSelected?(voice)
                            }
                        )
                        .padding(.horizontal)
                    }

                    // Voice collections
                    if searchText.isEmpty {
                        CollectionsSection(collections: collections)
                            .padding(.horizontal)
                    }

                    // Your voices section
                    if !filteredCustomVoices.isEmpty {
                        YourVoicesSection(
                            voices: filteredCustomVoices,
                            onVoiceTap: { voice in
                                selectedVoice = voice
                                onVoiceSelected?(voice)
                            },
                            onDelete: { voice in
                                customVoices.removeAll { $0.id == voice.id }
                            }
                        )
                        .padding(.horizontal)
                    }
                }
                .padding(.vertical)
            }
            .navigationTitle("Voices")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.large)
            #endif
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button(action: { dismiss() }) {
                        Image(systemName: "xmark.circle.fill")
                            .font(.title2)
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
    }
}

// MARK: - Search Bar

struct SearchBar: View {
    @Binding var text: String

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "magnifyingglass")
                .foregroundStyle(.secondary)

            TextField("Search", text: $text)
                .textFieldStyle(.plain)

            if !text.isEmpty {
                Button(action: { text = "" }) {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.secondary)
                }
            }
        }
        .padding(12)
        .background(Color.gray.opacity(0.15))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }
}

// MARK: - Add Voice Button

struct AddVoiceButton: View {
    var action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 12) {
                ZStack {
                    Circle()
                        .fill(Color.black)
                        .frame(width: 50, height: 50)

                    Image(systemName: "plus")
                        .font(.title2)
                        .fontWeight(.semibold)
                        .foregroundStyle(.white)
                }

                VStack(alignment: .leading, spacing: 2) {
                    Text("Add a new voice")
                        .font(.body)
                        .fontWeight(.medium)
                        .foregroundStyle(.primary)

                    Text("Create or clone a voice in seconds")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }

                Spacer()
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Recently Used Section

struct RecentlyUsedSection: View {
    let voices: [Voice]
    var onVoiceTap: ((Voice) -> Void)?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Recently used")
                    .font(.title2)
                    .fontWeight(.bold)

                Text("Voices you've used recently")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(voices) { voice in
                        VoiceChip(voice: voice) {
                            onVoiceTap?(voice)
                        }
                    }
                }
            }
        }
    }
}

// MARK: - Your Voices Section

struct YourVoicesSection: View {
    let voices: [Voice]
    var onVoiceTap: ((Voice) -> Void)?
    var onDelete: ((Voice) -> Void)?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Your Voices")
                    .font(.title2)
                    .fontWeight(.bold)

                Text("Voices you've created")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }

            VStack(spacing: 8) {
                ForEach(voices) { voice in
                    VoiceRow(
                        voice: voice,
                        showDeleteButton: true,
                        onDelete: { onDelete?(voice) },
                        onTap: { onVoiceTap?(voice) }
                    )
                }
            }
        }
    }
}

#Preview {
    VoicesView(
        recentlyUsed: .constant(Voice.samples),
        customVoices: .constant(Voice.customVoices),
        collections: VoiceCollection.samples
    )
}
