import SwiftUI

@main
struct PhononTranslateApp: App {
    @State private var translator = Translator()

    var body: some Scene {
        WindowGroup {
            ContentView(translator: translator)
                #if os(macOS)
                .frame(minWidth: 720, minHeight: 660)
                #endif
        }
        #if os(macOS)
        .defaultSize(width: 980, height: 850)
        .windowStyle(.hiddenTitleBar)
        #endif
    }
}
