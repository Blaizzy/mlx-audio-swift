import SwiftUI

@main
struct VoicesApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        #if os(macOS)
        .windowStyle(.hiddenTitleBar)
        .defaultSize(width: 500, height: 800)
        #endif
    }
}
