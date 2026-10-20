import SwiftUI

enum TranslateStyle {
    static let coral = Color(red: 0.85, green: 0.29, blue: 0.20)
    static let blue = Color(red: 0.22, green: 0.39, blue: 0.65)
}

struct ContentView: View {
    @Bindable var translator: Translator
    @Environment(\.colorScheme) private var colorScheme
    @State private var showingModels = false
    @FocusState private var editingEnglish: Bool

    private var background: Color {
        colorScheme == .dark ? Color(red: 0.10, green: 0.11, blue: 0.13) : Color(red: 0.97, green: 0.96, blue: 0.93)
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 28) {
                header
                introduction
                ViewThatFits(in: .horizontal) {
                    HStack(alignment: .top, spacing: 18) {
                        englishCard.frame(minWidth: 300)
                        frenchCard.frame(minWidth: 300)
                    }
                    VStack(spacing: 18) {
                        englishCard
                        frenchCard
                    }
                }
                actionBar
                footer
            }
            .padding(.horizontal, 28)
            .padding(.vertical, 30)
            .frame(maxWidth: 1_040)
            .frame(maxWidth: .infinity)
        }
        .background(background)
        .tint(TranslateStyle.coral)
        .sheet(isPresented: $showingModels) {
            ModelSettingsView(translator: translator)
        }
        .onDisappear { translator.cancel() }
    }

    private var header: some View {
        HStack(spacing: 12) {
            Image(systemName: "waveform")
                .font(.title2)
                .foregroundStyle(.white)
                .frame(width: 44, height: 44)
                .background(TranslateStyle.coral.gradient, in: RoundedRectangle(cornerRadius: 14))
            VStack(alignment: .leading, spacing: 2) {
                Text("Phonon Translate").font(.headline)
                Text("A small-model showcase").font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            Button { showingModels = true } label: {
                Label("Models & access", systemImage: "slider.horizontal.3")
            }
            .buttonStyle(.bordered)
            .accessibilityIdentifier("modelSettings")
        }
    }

    private var introduction: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label("ENGLISH → FRENCH", systemImage: "globe.europe.africa")
                .font(.caption.weight(.semibold))
                .tracking(1.2)
                .foregroundStyle(TranslateStyle.coral)
            Text("Say it in French.")
                .font(.system(size: 38, weight: .semibold, design: .serif))
                .accessibilityAddTraits(.isHeader)
            Text("Write in English. Translate and listen in both languages, right on your device.")
                .font(.body)
                .foregroundStyle(.secondary)
            HStack(spacing: 8) {
                phraseButton("At the café", text: "Hello! I would like a coffee and a croissant, please.")
                phraseButton("A quick hello", text: "Hello, it's lovely to meet you. How are you today?")
                phraseButton("Travel plans", text: "Excuse me, how do I get to the train station?")
            }
            .padding(.top, 6)
        }
    }

    private func phraseButton(_ title: String, text: String) -> some View {
        Button(title) {
            translator.english = text
            editingEnglish = false
        }
        .font(.caption)
        .buttonStyle(.bordered)
        .disabled(translator.isBusy)
    }

    private var englishCard: some View {
        languageCard(language: .english, code: "EN", accent: TranslateStyle.coral) {
            ZStack(alignment: .topLeading) {
                if translator.english.isEmpty {
                    Text("Type something you’d like to say…")
                        .foregroundStyle(.tertiary)
                        .padding(.top, 8)
                        .padding(.leading, 5)
                        .allowsHitTesting(false)
                }
                TextEditor(text: $translator.english)
                    .scrollContentBackground(.hidden)
                    .focused($editingEnglish)
                    .disabled(translator.isBusy)
                    .accessibilityLabel("English text")
                    .accessibilityIdentifier("englishInput")
            }
            .font(.system(size: 20, design: .serif))
            .frame(height: 190)
            HStack {
                Text("Your words")
                Spacer()
                Text("\(translator.english.count) / \(TranslationText.characterLimit)")
                    .foregroundStyle(translator.english.count > TranslationText.characterLimit ? .red : .secondary)
                    .monospacedDigit()
            }
            .font(.caption)
            .foregroundStyle(.secondary)
        }
    }

    private var frenchCard: some View {
        languageCard(language: .french, code: "FR", accent: TranslateStyle.blue) {
            ScrollView {
                VStack(alignment: .leading) {
                    if translator.french.isEmpty {
                        Text(translator.isTranslating ? "Finding the right words…" : "Your French translation will appear here.")
                            .foregroundStyle(.tertiary)
                    } else {
                        Text(translator.french)
                            .textSelection(.enabled)
                            .foregroundStyle(translator.translationIsCurrent || translator.isTranslating ? .primary : .secondary)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .font(.system(size: 20, design: .serif))
            .frame(maxWidth: .infinity, alignment: .leading)
            .frame(height: 190)
            .accessibilityIdentifier("frenchTranslation")
            HStack {
                Text(translator.french.isEmpty ? "Ready when you are" : "Votre traduction")
                Spacer()
                if !translator.french.isEmpty {
                    ShareLink(item: translator.french) {
                        Image(systemName: "square.and.arrow.up")
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel("Share French translation")
                }
            }
            .font(.caption)
            .foregroundStyle(.secondary)
        }
    }

    private func languageCard<Content: View>(
        language: SpeechLanguage, code: String, accent: Color, @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack(spacing: 10) {
                Text(code)
                    .font(.caption.weight(.bold))
                    .foregroundStyle(accent)
                    .frame(width: 35, height: 30)
                    .background(accent.opacity(0.10), in: RoundedRectangle(cornerRadius: 8))
                Text(language.title).font(.headline)
                Spacer()
                if language == .french && translator.isTranslating {
                    ProgressView().controlSize(.small)
                }
            }
            content()
            Divider()
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text(language.model.title).font(.caption.weight(.semibold))
                    Text("\(language.title) voice")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button { translator.speak(language) } label: {
                    Label(
                        translator.operation == .speaking(language) ? "Stop" : "Speak",
                        systemImage: translator.operation == .speaking(language) ? "stop.fill" : "speaker.wave.2.fill")
                }
                .buttonStyle(.bordered)
                .tint(accent)
                .disabled(!translator.canSpeak(language))
                .accessibilityLabel(translator.operation == .speaking(language) ? "Stop speaking \(language.title)" : "Speak \(language.title)")
                .accessibilityIdentifier("speak\(language.title)")
            }
        }
        .padding(22)
        .background(colorScheme == .dark ? Color.white.opacity(0.045) : .white, in: RoundedRectangle(cornerRadius: 22))
        .overlay(RoundedRectangle(cornerRadius: 22).strokeBorder(.primary.opacity(0.07)))
    }

    private var actionBar: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(alignment: .center, spacing: 14) {
                Button {
                    editingEnglish = false
                    translator.translate()
                } label: {
                    Label("Translate to French", systemImage: "arrow.right")
                        .font(.headline)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 5)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .disabled(!translator.canTranslate)
                .keyboardShortcut(.return, modifiers: .command)
                .accessibilityIdentifier("translateButton")
                if translator.isBusy {
                    ProgressView().controlSize(.small)
                    Button("Cancel") { translator.cancel() }
                        .buttonStyle(.plain)
                        .disabled(translator.isCancelling)
                }
                Spacer(minLength: 0)
            }
            Text(translator.statusText)
                .font(.caption)
                .foregroundStyle(.secondary)
                .accessibilityIdentifier("operationStatus")
            if let error = translator.errorMessage {
                HStack(alignment: .top, spacing: 10) {
                    Image(systemName: "exclamationmark.circle.fill").foregroundStyle(TranslateStyle.coral)
                    Text(error).font(.callout).textSelection(.enabled)
                    Spacer(minLength: 0)
                    Button { translator.errorMessage = nil } label: { Image(systemName: "xmark") }
                        .buttonStyle(.plain)
                        .accessibilityLabel("Dismiss error")
                }
                .padding(14)
                .background(TranslateStyle.coral.opacity(0.08), in: RoundedRectangle(cornerRadius: 12))
                .accessibilityIdentifier("errorMessage")
            }
        }
    }

    private var footer: some View {
        HStack(spacing: 7) {
            Image(systemName: "desktopcomputer")
            Text("Qwen translates. Phonon speaks. MLX runs it all locally.")
        }
        .font(.caption)
        .foregroundStyle(.secondary)
        .frame(maxWidth: .infinity)
        .padding(.top, 2)
    }
}

#Preview("Ready") { ContentView(translator: .preview()).frame(width: 980, height: 850) }
#Preview("Translated") { ContentView(translator: .preview(translated: true)).frame(width: 980, height: 850) }
#Preview("Downloading") { ContentView(translator: .preview(loading: true)).frame(width: 390, height: 900) }
