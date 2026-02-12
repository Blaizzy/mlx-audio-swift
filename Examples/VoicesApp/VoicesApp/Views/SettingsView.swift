import SwiftUI

struct SettingsView: View {
    @Environment(\.dismiss) private var dismiss
    @Bindable var viewModel: TTSViewModel

    var body: some View {
        NavigationStack {
            ScrollView {
                settingsContent
            }
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
        .frame(minWidth: 450, minHeight: 700)
        #endif
    }

    #if os(iOS)
    private let sectionSpacing: CGFloat = 12
    private let labelFont: Font = .caption
    private let textFont: Font = .footnote
    private let horizontalPadding: CGFloat = 16
    #else
    private let sectionSpacing: CGFloat = 16
    private let labelFont: Font = .subheadline
    private let textFont: Font = .subheadline
    private let horizontalPadding: CGFloat = 20
    #endif

    private var settingsContent: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Model Section
            VStack(alignment: .leading, spacing: 2) {
                Text("Model")
                    .font(labelFont)
                    .foregroundStyle(.secondary)

                HStack(spacing: 6) {
                    TextField("Model ID", text: $viewModel.modelId)
                        .font(textFont)
                        .textFieldStyle(.plain)
                        .padding(8)
                        .background(Color.gray.opacity(0.15))
                        .clipShape(RoundedRectangle(cornerRadius: 6))

                    Button(action: {
                        Task {
                            await viewModel.reloadModel()
                        }
                    }) {
                        Text("Load")
                            .font(textFont)
                            .fontWeight(.medium)
                            .foregroundStyle(.white)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 8)
                            .background(Color.blue)
                            .clipShape(RoundedRectangle(cornerRadius: 6))
                    }
                    .buttonStyle(.plain)
                    .disabled(viewModel.isLoading)
                }
                .padding(.top, 4)
            }
            .padding(.bottom, sectionSpacing)

            // Voice Design Section
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text("Voice Design")
                        .font(labelFont)
                        .foregroundStyle(.secondary)

                    Spacer()

                    #if os(iOS)
                    CompactToggle(label: "", isOn: $viewModel.useVoiceDesign, font: textFont, toggleWidth: 36, toggleHeight: 20, thumbSize: 16)
                    #else
                    Toggle("", isOn: $viewModel.useVoiceDesign)
                        .labelsHidden()
                    #endif
                }

                // Info box about VoiceDesign
                VStack(alignment: .leading, spacing: 8) {
                    HStack(alignment: .top, spacing: 8) {
                        Image(systemName: "info.circle.fill")
                            .font(.caption)
                            .foregroundStyle(.blue)

                        VStack(alignment: .leading, spacing: 4) {
                            Text("VoiceDesign Feature")
                                .font(.caption)
                                .fontWeight(.medium)

                            Text("Describe the voice you want in natural language. Only works with VoiceDesign models.")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }

                    // Recommended models
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Recommended Models:")
                            .font(.caption2)
                            .fontWeight(.medium)
                            .foregroundStyle(.secondary)

                        VStack(alignment: .leading, spacing: 2) {
                            ModelHintRow(modelId: "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16")
                        }
                    }

                    // Example descriptions
                    if viewModel.useVoiceDesign {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Example Descriptions:")
                                .font(.caption2)
                                .fontWeight(.medium)
                                .foregroundStyle(.secondary)

                            VStack(alignment: .leading, spacing: 2) {
                                ExampleDescription(text: "A young woman with a bright, energetic voice")
                                ExampleDescription(text: "A deep, calm male narrator")
                                ExampleDescription(text: "An elderly person speaking slowly")
                            }
                        }
                    }
                }
                .padding(10)
                .background(Color.blue.opacity(0.08))
                .clipShape(RoundedRectangle(cornerRadius: 8))

                // Voice description input (only shown when enabled)
                if viewModel.useVoiceDesign {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Voice Description")
                            .font(.caption2)
                            .foregroundStyle(.secondary)

                        TextEditor(text: $viewModel.voiceDescription)
                            .font(textFont)
                            .frame(height: 80)
                            .scrollContentBackground(.hidden)
                            .padding(8)
                            .background(Color.gray.opacity(0.15))
                            .clipShape(RoundedRectangle(cornerRadius: 6))
                            .overlay(alignment: .topLeading) {
                                if viewModel.voiceDescription.isEmpty {
                                    Text("Describe the voice you want...")
                                        .font(textFont)
                                        .foregroundStyle(.tertiary)
                                        .padding(.horizontal, 12)
                                        .padding(.top, 16)
                                        .allowsHitTesting(false)
                                }
                            }
                    }
                }
            }
            .padding(.bottom, sectionSpacing)

            // Length Section
            VStack(alignment: .leading, spacing: 2) {
                Text("Length")
                    .font(labelFont)
                    .foregroundStyle(.secondary)

                HStack {
                    Text("Max Tokens")
                        .font(textFont)
                    Spacer()
                    Text("\(viewModel.maxTokens)")
                        .font(textFont)
                        .foregroundStyle(.secondary)
                }
                .padding(.top, 4)

                #if os(iOS)
                CompactSlider(
                    value: Binding(
                        get: { Double(viewModel.maxTokens) },
                        set: { viewModel.maxTokens = Int($0) }
                    ),
                    range: 100...2000,
                    step: 100
                )
                #else
                Slider(
                    value: Binding(
                        get: { Double(viewModel.maxTokens) },
                        set: { viewModel.maxTokens = Int($0) }
                    ),
                    in: 100...2000,
                    step: 100
                )
                .tint(.blue)
                #endif
            }
            .padding(.bottom, sectionSpacing)

            // Temperature Section
            VStack(alignment: .leading, spacing: 2) {
                Text("Temperature")
                    .font(labelFont)
                    .foregroundStyle(.secondary)

                HStack {
                    Text("Temperature")
                        .font(textFont)
                    Spacer()
                    Text(String(format: "%.2f", viewModel.temperature))
                        .font(textFont)
                        .foregroundStyle(.secondary)
                }
                .padding(.top, 4)

                #if os(iOS)
                CompactSlider(
                    value: Binding(
                        get: { Double(viewModel.temperature) },
                        set: { viewModel.temperature = Float($0) }
                    ),
                    range: 0.0...1.0,
                    step: 0.05
                )
                #else
                Slider(
                    value: Binding(
                        get: { Double(viewModel.temperature) },
                        set: { viewModel.temperature = Float($0) }
                    ),
                    in: 0.0...1.0,
                    step: 0.05
                )
                .tint(.blue)
                #endif
            }
            .padding(.bottom, sectionSpacing)

            // Top P Section
            VStack(alignment: .leading, spacing: 2) {
                Text("Top P")
                    .font(labelFont)
                    .foregroundStyle(.secondary)

                HStack {
                    Text("Top P")
                        .font(textFont)
                    Spacer()
                    Text(String(format: "%.2f", viewModel.topP))
                        .font(textFont)
                        .foregroundStyle(.secondary)
                }
                .padding(.top, 4)

                #if os(iOS)
                CompactSlider(
                    value: Binding(
                        get: { Double(viewModel.topP) },
                        set: { viewModel.topP = Float($0) }
                    ),
                    range: 0.0...1.0,
                    step: 0.05
                )
                #else
                Slider(
                    value: Binding(
                        get: { Double(viewModel.topP) },
                        set: { viewModel.topP = Float($0) }
                    ),
                    in: 0.0...1.0,
                    step: 0.05
                )
                .tint(.blue)
                #endif
            }
            .padding(.bottom, sectionSpacing)

            // Repetition Penalty Section
            VStack(alignment: .leading, spacing: 2) {
                Text("Repetition Penalty")
                    .font(labelFont)
                    .foregroundStyle(.secondary)

                HStack {
                    Text("Repetition Penalty")
                        .font(textFont)
                    Spacer()
                    Text(String(format: "%.2f", viewModel.repetitionPenalty))
                        .font(textFont)
                        .foregroundStyle(.secondary)
                }
                .padding(.top, 4)

                #if os(iOS)
                CompactSlider(
                    value: Binding(
                        get: { Double(viewModel.repetitionPenalty) },
                        set: { viewModel.repetitionPenalty = Float($0) }
                    ),
                    range: 1.0...2.0,
                    step: 0.05
                )
                #else
                Slider(
                    value: Binding(
                        get: { Double(viewModel.repetitionPenalty) },
                        set: { viewModel.repetitionPenalty = Float($0) }
                    ),
                    in: 1.0...2.0,
                    step: 0.05
                )
                .tint(.blue)
                #endif

                // Hint for VoiceDesign models
                HStack(spacing: 6) {
                    Image(systemName: "info.circle.fill")
                        .font(.caption2)
                        .foregroundStyle(.orange)

                    Text("For Qwen3-TTS VoiceDesign, use minimum 1.1 to reduce repetition")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .padding(.top, 4)
                .padding(8)
                .background(Color.orange.opacity(0.1))
                .clipShape(RoundedRectangle(cornerRadius: 6))
            }
            .padding(.bottom, sectionSpacing)

            // Text Chunking Section
            VStack(alignment: .leading, spacing: 2) {
                Text("Text Chunking")
                    .font(labelFont)
                    .foregroundStyle(.secondary)

                #if os(iOS)
                CompactToggle(label: "Chunk text", isOn: $viewModel.enableChunking, font: textFont)
                    .padding(.top, 4)
                #else
                Toggle("Chunk text", isOn: $viewModel.enableChunking)
                    .font(textFont)
                    .padding(.top, 4)
                #endif

                if viewModel.enableChunking {
                    #if os(iOS)
                    CompactToggle(label: "Stream audio", isOn: $viewModel.streamingPlayback, font: textFont)
                    #else
                    Toggle("Stream audio", isOn: $viewModel.streamingPlayback)
                        .font(textFont)
                    #endif

                    HStack {
                        Text("Max chunk length")
                            .font(textFont)
                        Spacer()
                        Text("\(viewModel.maxChunkLength)")
                            .font(textFont)
                            .foregroundStyle(.secondary)
                    }
                    .padding(.top, 4)

                    #if os(iOS)
                    CompactSlider(
                        value: Binding(
                            get: { Double(viewModel.maxChunkLength) },
                            set: { viewModel.maxChunkLength = Int($0) }
                        ),
                        range: 100...500,
                        step: 50
                    )
                    #else
                    Slider(
                        value: Binding(
                            get: { Double(viewModel.maxChunkLength) },
                            set: { viewModel.maxChunkLength = Int($0) }
                        ),
                        in: 100...500,
                        step: 50
                    )
                    .tint(.blue)
                    #endif

                    TextField("Split pattern", text: $viewModel.splitPattern)
                        .font(textFont)
                        .textFieldStyle(.plain)
                        .padding(8)
                        .background(Color.gray.opacity(0.15))
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                }
            }

            // Reset button
            Button(action: {
                viewModel.modelId = "mlx-community/VyvoTTS-EN-Beta-4bit"
                viewModel.resetGenerationParameterOverrides()
                viewModel.useVoiceDesign = false
                viewModel.voiceDescription = ""
                viewModel.enableChunking = true
                viewModel.maxChunkLength = 200
                viewModel.splitPattern = "\n"
                viewModel.streamingPlayback = true
            }) {
                Text("Reset to Defaults")
                    .font(textFont)
                    .fontWeight(.medium)
                    .foregroundStyle(.blue)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(Color.blue.opacity(0.15))
                    .clipShape(RoundedRectangle(cornerRadius: 6))
            }
            .buttonStyle(.plain)
            .padding(.top, 16)
            .padding(.bottom, 12)
        }
        .padding(.horizontal, horizontalPadding)
    }
}

// MARK: - Helper Views

struct ModelHintRow: View {
    let modelId: String

    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: "arrow.forward.circle.fill")
                .font(.caption2)
                .foregroundStyle(.blue.opacity(0.6))

            Text(modelId)
                .font(.system(.caption2, design: .monospaced))
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
                .lineLimit(1)
                .truncationMode(.middle)
                .help("Click to select and copy")
        }
    }
}

struct ExampleDescription: View {
    let text: String

    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: "quote.bubble.fill")
                .font(.caption2)
                .foregroundStyle(.blue.opacity(0.4))

            Text(text)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .italic()
        }
    }
}

#Preview {
    SettingsView(viewModel: TTSViewModel())
}
