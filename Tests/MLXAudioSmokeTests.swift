//
//  MLXAudioSmokeTests.swift
//  MLXAudioTests
//
//  End-to-end inference smoke tests that download models from HuggingFace and run generation.
//  These are intentionally separated from the fast unit tests so CI can skip them easily.
//
//  Run all smoke tests (serialized):
//    xcodebuild test \
//      -scheme MLXAudio-Package \
//      -destination 'platform=macOS' \
//      -only-testing:MLXAudioTests/Smoke \
//      CODE_SIGNING_ALLOWED=NO
//
//  Run a single category:
//    -only-testing:'MLXAudioTests/Smoke/CodecsTests'
//    -only-testing:'MLXAudioTests/Smoke/TTSTests'
//    -only-testing:'MLXAudioTests/Smoke/STTTests'
//    -only-testing:'MLXAudioTests/Smoke/VADTests'
//    -only-testing:'MLXAudioTests/Smoke/LIDTests'
//    -only-testing:'MLXAudioTests/Smoke/STSTests'
//
//  Run a single test (note the trailing parentheses for Swift Testing):
//    -only-testing:'MLXAudioTests/Smoke/STTTests/Qwen3ASRTests/qwen3ASRTranscribe()'
//
//  Filter test results:
//   2>&1 | grep --color=never -E '(^􀟈 |^􁁛 |^􀢄 |^\*\* TEST|\x1b\[1;35m|model loaded|Encoded to|Reconstructed audio|Generating audio|Generated audio|Generated [0-9]+ tokens|Streaming|Saved |Received final|Found [0-9]|Processing time|Streaming complete|Chunk [0-9]|  [Tt]ext:|  prompt_tokens|  generation_tokens|  total_tokens|  prompt_tps|  generation_tps|total_time| peak_memory|Peak Memory|Prompt:.*tokens/s|Prompt Tokens|Total Time|SPEAKER audio|Sortformer Output|Audio input shape|Loading.*model|Loaded audio|ForcedAligner|Running forced|\[.*s - .*s\])'

import Testing
import MLX
import MLXLMCommon
import Foundation

@testable import MLXAudioCore
@testable import MLXAudioCodecs
@testable import MLXAudioTTS
@testable import MLXAudioSTT
@testable import MLXAudioVAD
@testable import MLXAudioSTS
@testable import MLXAudioLID


// MARK: - Helpers

let delimiter = String(repeating: "=", count: 60)

func testHeader(_ name: String) {
    // Free memory left over from the previous test (locals are now out of scope)
    Memory.clearCache()
    GPU.resetPeakMemory()
    print("\n\u{001B}[1;35m\(delimiter)\u{001B}[0m")
    print("\u{001B}[1;35m  \(name)\u{001B}[0m")
    print("\u{001B}[1;35m\(delimiter)\u{001B}[0m")
}

func testCleanup(_ name: String) {
    let peak = Double(Memory.peakMemory) / 1_073_741_824
    print("\u{001B}[1;35m\(delimiter) \(name) done (peak: \(String(format: "%.2f", peak)) GB)\u{001B}[0m\n")
}

func temporaryTestOutputURL(_ fileName: String) -> URL {
    URL(fileURLWithPath: "/tmp", isDirectory: true)
        .appendingPathComponent("\(UUID().uuidString)-\(fileName)")
}

func removeTemporaryTestOutput(_ url: URL) {
    try? FileManager.default.removeItem(at: url)
}

// MARK: - Top-level serialized wrapper (all suites run sequentially)

@Suite("SmokeTests", .serialized)
struct SmokeTests {

// MARK: - Codecs Tests

@Suite("Codecs Tests", .serialized)
struct CodecsTests {}


// MARK: - TTS Tests

@Suite("TTS Tests", .serialized)
struct TTSTests {}


// MARK: - STT Tests

@Suite("STT Tests", .serialized)
struct STTTests {}


// MARK: - VAD Tests

@Suite("VAD Tests", .serialized)
struct VADTests {}

// MARK: - LID Tests

@Suite("LID Tests", .serialized)
struct LIDTests {}

// MARK: - STS Tests

@Suite("STS Tests", .serialized)
struct STSTests {}

} // end Smoke
