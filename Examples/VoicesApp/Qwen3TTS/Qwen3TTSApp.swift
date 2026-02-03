//
//  Qwen3TTSApp.swift
//  Qwen3TTS
//
//  Main entry point for the Qwen3-TTS demo app.
//

import SwiftUI

@main
struct Qwen3TTSApp: App {
    var body: some Scene {
        WindowGroup {
            Qwen3TTSContentView()
        }
        #if os(macOS)
        .windowStyle(.hiddenTitleBar)
        .defaultSize(width: 500, height: 800)
        #endif
    }
}
