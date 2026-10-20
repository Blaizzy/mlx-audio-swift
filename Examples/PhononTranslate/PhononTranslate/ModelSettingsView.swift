import SwiftUI

struct ModelSettingsView: View {
    @Bindable var translator: Translator
    @Environment(\.dismiss) private var dismiss
    @State private var token = ""
    @State private var tokenMessage: String?

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    ForEach(ShowcaseModel.allCases) { model in
                        VStack(alignment: .leading, spacing: 7) {
                            HStack {
                                Text(model.title).font(.headline)
                                Spacer()
                                Text("~\(model.approximateDownload)")
                                    .font(.caption).foregroundStyle(.secondary)
                            }
                            Text(model.role).font(.callout)
                            Link(model.repository, destination: model.pageURL)
                                .font(.caption)
                            status(model)
                        }
                        .padding(.vertical, 8)
                    }
                } header: {
                    Text("Three small models, one device")
                } footer: {
                    Text("Models download when first used, then stay cached for offline use. No text or audio is sent to a translation or speech service.")
                }

                Section {
                    Text("Both voices require accepting Gradium’s access terms and a Hugging Face read token.")
                        .font(.callout)
                    Link("Open English model & accept access terms", destination: ShowcaseModel.englishVoice.pageURL)
                    Link("Open French model & accept access terms", destination: ShowcaseModel.frenchVoice.pageURL)
                    Link("Get a Hugging Face read token", destination: URL(string: "https://huggingface.co/settings/tokens")!)
                    if translator.hasToken {
                        Label("A token is available", systemImage: "checkmark.shield")
                            .foregroundStyle(.green)
                    }
                    SecureField("Hugging Face token (hf_…)", text: $token)
                        .textContentType(.password)
                        .privacySensitive()
                        .accessibilityIdentifier("huggingFaceToken")
                    Button("Save token securely") { saveToken() }
                        .disabled(token.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || translator.isBusy)
                    if let tokenMessage {
                        Text(tokenMessage).font(.caption).foregroundStyle(.secondary)
                    }
                } header: {
                    Text("Voice model access")
                } footer: {
                    Text("The token is stored in this device’s Keychain. A token supplied with HF_TOKEN is also supported.")
                }
            }
            .formStyle(.grouped)
            .navigationTitle("Models & access")
            .toolbar {
                ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } }
            }
        }
        .tint(TranslateStyle.coral)
        #if os(macOS)
        .frame(width: 540, height: 660)
        #endif
    }

    @ViewBuilder private func status(_ model: ShowcaseModel) -> some View {
        let state = translator.modelStatuses[model] ?? .notLoaded
        HStack(spacing: 6) {
            if case .loading(let fraction) = state {
                ProgressView(value: fraction).frame(maxWidth: 120)
            } else {
                Image(systemName: state == .ready ? "checkmark.circle.fill" : state == .failed ? "exclamationmark.circle" : "arrow.down.circle")
                    .foregroundStyle(state == .ready ? .green : .secondary)
            }
            Text(state.label)
        }
        .font(.caption)
        .foregroundStyle(.secondary)
    }

    private func saveToken() {
        do {
            try HuggingFaceToken.save(token)
            token = ""
            translator.hasToken = HuggingFaceToken.read() != nil
            tokenMessage = "Token saved. You can now try both voices."
        } catch {
            tokenMessage = error.localizedDescription
        }
    }
}

#Preview { ModelSettingsView(translator: .preview()) }
