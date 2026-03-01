import AVFoundation
import SwiftUI
#if canImport(UIKit)
import UIKit
#endif

@MainActor
@Observable
class ContentViewModel {
    var conversationController = ConversationController()

    init() {
        conversationController.delegate = self
    }

    func startConversation() async throws {
        print("Starting conversation...")
        try await conversationController.start()

#if canImport(UIKit)
        UIApplication.shared.isIdleTimerDisabled = true
#endif
    }

    func stopConversation() async throws {
        try await conversationController.stop()

#if canImport(UIKit)
        UIApplication.shared.isIdleTimerDisabled = false
#endif

        print("Stopped conversation.")
    }
}

@MainActor
extension ContentViewModel: ConversationControllerDelegate {
    func conversationControllerDidStartUserSpeech(_ controller: ConversationController) {
        // no-op
    }

    func conversationController(_ controller: ConversationController, didFinish transcription: String) {
        print("Got transcription: '\(transcription)'")
    }
}

@MainActor
struct ContentView: View {
    @State private var permissionStatus: AVAudioApplication.recordPermission = .undetermined
    @State private var viewModel = ContentViewModel()

    var body: some View {
        VStack {
            Spacer()

            assistantCircle

            Spacer()
        }
        .padding()
        .onChange(of: viewModel.conversationController.canSpeak) { _, newValue in
            if newValue {
                Task {
                    if !viewModel.conversationController.isActive {
                        try await viewModel.startConversation()
                    }
                }
            }
        }
    }

    @ViewBuilder
    private var assistantCircle: some View {
        let isActive = viewModel.conversationController.isActive
        let isSpeaking = viewModel.conversationController.isSpeaking

        ZStack {
            Button {
                if viewModel.conversationController.canSpeak {
                    Task {
                        if permissionStatus == .undetermined {
                            let granted = await AVAudioApplication.requestRecordPermission()
                            withAnimation {
                                permissionStatus = granted ? .granted : .denied
                            }
                        }

                        try await toggleConversation()
                    }
                }
            } label: {
                Circle()
                    .fill(isSpeaking ? .orange : .black.opacity(isActive ? 1.0 : 0.4))
                    .frame(maxWidth: .infinity)
                    .clipShape(Circle())
                    .contentShape(Circle())
            }
            .buttonStyle(.plain)
            .padding(64)
        }
        .scaleEffect(CGSize(width: isActive ? 1.0 : 0.7, height: isActive ? 1.0 : 0.7))
        .animation(.easeOut(duration: 0.2), value: viewModel.conversationController.isActive)
        .animation(.easeOut(duration: 0.4), value: viewModel.conversationController.isSpeaking)
    }

    private func toggleConversation() async throws {
        do {
            if !viewModel.conversationController.isActive {
                try await viewModel.startConversation()
            } else {
                try await viewModel.stopConversation()
            }
        } catch {
            print("Failed to start conversation: \(error)")
        }
    }
}

#Preview {
    ContentView()
}
