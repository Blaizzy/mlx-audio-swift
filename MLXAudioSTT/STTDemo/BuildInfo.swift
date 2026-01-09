// BuildInfo.swift - Static version info
// Update version manually when releasing

struct BuildInfo {
    static let version = "0.2.0"
    static let commit = "297b790"
    static let branch = "feat/streaming-stt"

    static var full: String {
        "\(version) (\(commit))"
    }
}
