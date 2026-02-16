
import SwiftUI

@main
struct VoicesApp: App {
    var body: some Scene {
        WindowGroup {
            TabView {
                ContentView()
                    .tabItem {
                        Label("TTS", systemImage: "waveform")
                    }

                STTView()
                    .tabItem {
                        Label("STT", systemImage: "mic")
                    }
            }
        }
        #if os(macOS)
        .defaultSize(width: 500, height: 800)
        #endif
    }
}
