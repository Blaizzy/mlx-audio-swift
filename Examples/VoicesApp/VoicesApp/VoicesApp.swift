
import SwiftUI

@main
struct VoicesApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        #if os(macOS)
        .defaultSize(width: 700, height: 600)
        #endif
    }
}
