import SwiftUI

struct SettingsView: View {
    @Environment(\.dismiss) private var dismiss
    @Bindable var viewModel: TTSViewModel

    var body: some View {
        NavigationStack {
            Form {
                modelSection
                generationSection
                chunkingSection
                resetSection
            }
            #if os(macOS)
            .formStyle(.grouped)
            .padding(.horizontal)
            #endif
            .navigationTitle("Settings")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
        #if os(macOS)
        .frame(minWidth: 550, minHeight: 500)
        #endif
    }

    // MARK: - Model

    private var modelSection: some View {
        Section {
            Picker("Model Type", selection: Binding(
                get: { viewModel.modelType },
                set: { viewModel.switchModelType(to: $0) }
            )) {
                ForEach(TTSModelType.allCases) { type in
                    Text(type.rawValue).tag(type)
                }
            }

            Picker("Variant", selection: $viewModel.modelId) {
                ForEach(viewModel.modelType.availableModels, id: \.id) { model in
                    Text(model.name).tag(model.id)
                }
            }
            .onChange(of: viewModel.modelId) { _, newValue in
                // Auto-reload when a known preset is selected via picker
                if viewModel.modelType.availableModels.contains(where: { $0.id == newValue }),
                   newValue != viewModel.loadedModelId {
                    Task { await viewModel.reloadModel() }
                }
            }

            HStack {
                TextField("Model ID", text: $viewModel.modelId)
                    .textFieldStyle(.plain)

                Button {
                    Task { await viewModel.reloadModel() }
                } label: {
                    Text("Load")
                        .fontWeight(.medium)
                }
                .buttonStyle(.bordered)
                .disabled(viewModel.isLoading)
            }
        } header: {
            Text("Model")
        }
    }

    // MARK: - Generation Parameters

    private var generationSection: some View {
        Section {
            HStack {
                Text("Max Tokens")
                Spacer()
                Text("\(viewModel.maxTokens)")
                    .foregroundStyle(.secondary)
            }
            Slider(
                value: Binding(
                    get: { Double(viewModel.maxTokens) },
                    set: { viewModel.maxTokens = Int($0) }
                ),
                in: 100...2000,
                step: 100
            )
            .tint(.blue)

            HStack {
                Text("Temperature")
                Spacer()
                Text(String(format: "%.2f", viewModel.temperature))
                    .foregroundStyle(.secondary)
            }
            Slider(
                value: Binding(
                    get: { Double(viewModel.temperature) },
                    set: { viewModel.temperature = Float($0) }
                ),
                in: 0.0...1.0,
                step: 0.05
            )
            .tint(.blue)

            HStack {
                Text("Top P")
                Spacer()
                Text(String(format: "%.2f", viewModel.topP))
                    .foregroundStyle(.secondary)
            }
            Slider(
                value: Binding(
                    get: { Double(viewModel.topP) },
                    set: { viewModel.topP = Float($0) }
                ),
                in: 0.0...1.0,
                step: 0.05
            )
            .tint(.blue)
        } header: {
            Text("Generation")
        }
    }

    // MARK: - Text Chunking

    private var chunkingSection: some View {
        Section {
            Toggle("Chunk Text", isOn: $viewModel.enableChunking)

            if viewModel.enableChunking {
                Toggle("Stream Audio", isOn: $viewModel.streamingPlayback)

                HStack {
                    Text("Max Chunk Length")
                    Spacer()
                    Text("\(viewModel.maxChunkLength)")
                        .foregroundStyle(.secondary)
                }
                Slider(
                    value: Binding(
                        get: { Double(viewModel.maxChunkLength) },
                        set: { viewModel.maxChunkLength = Int($0) }
                    ),
                    in: 100...500,
                    step: 50
                )
                .tint(.blue)

            }
        } header: {
            Text("Text Chunking")
        }
    }

    // MARK: - Reset

    private var resetSection: some View {
        Section {
            Button("Reset to Defaults") {
                viewModel.modelType = .qwen3TTS
                viewModel.modelId = viewModel.modelType.defaultModelId
                viewModel.maxTokens = 1200
                viewModel.temperature = 0.6
                viewModel.topP = 0.8
                viewModel.enableChunking = true
                viewModel.maxChunkLength = 200
                viewModel.splitPattern = "\n"
                viewModel.streamingPlayback = true
            }
        }
    }
}

#Preview {
    SettingsView(viewModel: TTSViewModel())
}
